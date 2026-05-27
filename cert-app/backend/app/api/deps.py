"""API dependencies."""
from fastapi import Depends, HTTPException, status, Request, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import Optional
from functools import lru_cache
import logging

from app.database import get_db
from app.config import get_settings
from app.redis_client import redis_client

settings = get_settings()
logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)


def get_db_session(db: Session = Depends(get_db)) -> Session:
    """Get database session."""
    return db


def verify_job_secret(x_job_secret: Optional[str] = Header(None)) -> bool:
    """Verify job secret for admin endpoints."""
    if not settings.JOB_SECRET or not settings.JOB_SECRET.strip():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="JOB_SECRET not configured. Set in .env for admin API."
        )
    if not x_job_secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-Job-Secret header"
        )
    if x_job_secret != settings.JOB_SECRET:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid job secret"
        )
    
    return True


def _extract_client_ip(request: Request) -> str:
    """클라이언트 IP 추출. Render 등 프록시 뒤에서는 XFF 마지막 값(프록시가 추가한 실제 IP) 사용.
    첫 번째 값은 클라이언트가 위조 가능하므로 Rate Limit 키로 신뢰하지 않는다."""
    xff = request.headers.get("X-Forwarded-For")
    if xff:
        # Rightmost = 프록시(Render)가 삽입한 실제 출처 IP
        return xff.split(",")[-1].strip()
    return getattr(request.client, "host", None) if request.client else "127.0.0.1"


def check_rate_limit(request: Request) -> None:
    """Check rate limit for request."""
    client_ip = _extract_client_ip(request)
    
    key = f"rate_limit:{client_ip}"
    
    allowed, remaining, reset_after = redis_client.check_rate_limit(
        key,
        settings.RATE_LIMIT_REQUESTS,
        settings.RATE_LIMIT_WINDOW
    )
    
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={
                "X-RateLimit-Limit": str(settings.RATE_LIMIT_REQUESTS),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(reset_after),
                "Retry-After": str(reset_after)
            }
        )
    
    # Add rate limit headers to response
    request.state.rate_limit_remaining = remaining


def check_ai_rate_limit(request: Request) -> None:
    """AI 엔드포인트 전용 Rate Limit — OpenAI 호출이 포함된 경로에만 적용."""
    client_ip = _extract_client_ip(request)
    key = f"rate_limit_ai:{client_ip}"
    allowed, remaining, reset_after = redis_client.check_rate_limit(
        key,
        settings.AI_RATE_LIMIT_REQUESTS,
        settings.AI_RATE_LIMIT_WINDOW,
    )
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="AI 추천 요청 한도를 초과했습니다. 잠시 후 다시 시도해 주세요.",
            headers={
                "X-RateLimit-Limit": str(settings.AI_RATE_LIMIT_REQUESTS),
                "X-RateLimit-Remaining": "0",
                "Retry-After": str(reset_after),
            },
        )
    request.state.rate_limit_remaining = remaining


def check_auth_rate_limit(request: Request) -> None:
    """Auth 전용 엄격 레이트 리밋 (send_code, login, password_reset 등). 분당 5회 등."""
    client_ip = _extract_client_ip(request)
    key = f"rate_limit_auth:{client_ip}"
    allowed, remaining, reset_after = redis_client.check_rate_limit(
        key,
        settings.AUTH_RATE_LIMIT_REQUESTS,
        settings.AUTH_RATE_LIMIT_WINDOW,
    )
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="요청 한도를 초과했습니다. 잠시 후 다시 시도해 주세요.",
            headers={
                "X-RateLimit-Limit": str(settings.AUTH_RATE_LIMIT_REQUESTS),
                "X-RateLimit-Remaining": "0",
                "Retry-After": str(reset_after),
            },
        )
    request.state.rate_limit_remaining = remaining


import base64
import time
from jose import jwt, JWTError
from app.models import Profile

import httpx
from jose import jwk

# JWKS TTL 캐시 (1시간). Supabase 키 로테이션 시 재시작 없이 신규 토큰 검증 가능.
_JWKS_CACHE: dict = {"data": None, "ts": 0.0}
_JWKS_TTL_SECONDS = 3600


def _get_supabase_jwks(url: str) -> dict:
    """Fetch and cache Supabase JWKS with TTL."""
    now = time.time()
    if _JWKS_CACHE["data"] is not None and (now - _JWKS_CACHE["ts"]) < _JWKS_TTL_SECONDS:
        return _JWKS_CACHE["data"]
    try:
        response = httpx.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        _JWKS_CACHE["data"] = data
        _JWKS_CACHE["ts"] = now
        return data
    except Exception as e:
        logger.error(f"Failed to fetch JWKS from {url}: {e}")
        if _JWKS_CACHE["data"] is not None:
            return _JWKS_CACHE["data"]
        return {}

def _decode_supabase_token(token: str) -> dict:
    """Supabase JWT 검증. SUPABASE_JWT_ALGORITHM 설정 시 해당 알고리즘만 허용(알고리즘 혼동 공격 방지)."""
    try:
        header = jwt.get_unverified_header(token)
        token_alg = header.get("alg")

        # 설정된 알고리즘이 있으면 토큰 헤더 alg와 반드시 일치해야 함.
        # 미설정 시 토큰 헤더 alg를 따르지만 경고 로그를 남긴다.
        expected_alg = (settings.SUPABASE_JWT_ALGORITHM or "").strip().upper()
        if expected_alg:
            if token_alg != expected_alg:
                raise Exception(
                    f"JWT algorithm mismatch: token={token_alg}, expected={expected_alg}"
                )
            alg = expected_alg
        else:
            logger.debug(
                "SUPABASE_JWT_ALGORITHM not set; trusting token alg=%s. "
                "Set SUPABASE_JWT_ALGORITHM in .env to prevent algorithm substitution attacks.",
                token_alg,
            )
            alg = token_alg

        if alg == "HS256":
            secret = settings.SUPABASE_JWT_SECRET
            if not secret:
                raise Exception("HS256 token received but SUPABASE_JWT_SECRET is not configured")
            if len(secret) > 20 and '=' in secret:
                try:
                    decoded_secret = base64.b64decode(secret)
                except Exception:
                    decoded_secret = secret
            else:
                decoded_secret = secret

            return jwt.decode(
                token,
                decoded_secret,
                algorithms=["HS256"],
                options={"verify_aud": False}
            )

        elif alg in ["RS256", "ES256"]:
            if not settings.SUPABASE_URL:
                raise Exception("SUPABASE_URL required for RS256/ES256 verification")

            jwks_url = f"{settings.SUPABASE_URL}/auth/v1/.well-known/jwks.json"
            try:
                jwks = _get_supabase_jwks(jwks_url)
            except Exception as e:
                logger.error(f"JWKS fetch failed: {e}")
                raise Exception("JWKS fetch failed")

            if not jwks or "keys" not in jwks:
                raise Exception("Empty or invalid JWKS response")

            kid = header.get("kid")
            key_data = None
            for k in jwks.get("keys", []):
                if k.get("kid") == kid:
                    key_data = k
                    break
            if not key_data:
                for k in jwks.get("keys", []):
                    if k.get("alg") == alg:
                        key_data = k
                        break

            if key_data:
                try:
                    key = jwk.construct(key_data)
                    return jwt.decode(
                        token,
                        key,
                        algorithms=[alg],
                        options={"verify_aud": False}
                    )
                except Exception as e:
                    raise Exception(f"Token signature verification failed: {e}")
            else:
                raise Exception(f"No matching key found in JWKS for alg={alg} kid={kid}")

        else:
            raise Exception(f"Unsupported algorithm: {alg}")

    except Exception as e:
        logger.error(f"Token decode error: {e}")
        raise e

def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db_session)
) -> Optional[str]:
    """Get optional user ID from JWT token."""
    if not credentials:
        return None
    
    try:
        payload = _decode_supabase_token(credentials.credentials)
        uuid_sub = payload.get("sub")
        if not uuid_sub:
            return None
            
        # UUID만 반환 (userid는 변경 가능하므로 키로 사용 불가)
        return uuid_sub
    except Exception:
        return None

def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db_session)
) -> str:
    """
    Get current user ID from JWT token (required).
    반환값: 항상 Supabase auth UUID (sub) 를 반환한다.
    - profile.userid(사용자 지정 문자 ID)는 변경 가능하므로 user_id 키로 사용하면
      userid 변경 시 favorites/acquired_certs가 유실되는 버그가 발생한다.
    - UUID는 계정 삭제 전까지 절대 변경되지 않으므로 안전한 FK 키다.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    try:
        payload = _decode_supabase_token(credentials.credentials)
        uuid_sub = payload.get("sub")
        if not uuid_sub:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload: missing sub"
            )
            
        # Ensure profile exists (Social Login Support)
        profile = db.query(Profile).filter(Profile.id == uuid_sub).first()
        if not profile:
            email = payload.get("email")
            meta = payload.get("user_metadata", {})
            name = meta.get("full_name") or meta.get("name")
            try:
                profile = Profile(
                    id=uuid_sub,
                    email=email,
                    name=name,
                    nickname=name or email.split("@")[0] if email else "New User",
                    userid=None
                )
                db.add(profile)
                db.commit()
                logger.info(f"Auto-created profile for social user: {uuid_sub}")
            except Exception as e:
                db.rollback()
                logger.warning(f"Failed to auto-create profile: {e}")
        
        # 항상 UUID 반환 — userid는 변경 가능하므로 user_id 키로 사용 불가
        return uuid_sub
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"JWT validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )
