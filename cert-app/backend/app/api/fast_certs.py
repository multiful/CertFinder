import os
import orjson
import redis.asyncio as aioredis
from fastapi import APIRouter
from fastapi.responses import Response
from sqlalchemy import text
from app.database import SessionLocal
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Redis 연결 오버헤드를 줄이기 위한 싱글톤 Connection Pool
redis_pool = None

@router.on_event("startup")
async def init_redis():
    global redis_pool
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    # decode_responses=False를 사용하여 직렬화 비용 없이 bytes 채로 반환받음
    redis_pool = aioredis.from_url(
        redis_url, 
        max_connections=50, 
        decode_responses=False
    )

@router.on_event("shutdown")
async def close_redis():
    global redis_pool
    if redis_pool:
        await redis_pool.aclose()

@router.get("/certs/{cert_id}/fast")
async def get_cert_fast(cert_id: int):
    """
    초저지연(Ultra-low Latency) 자격증 조회 시스템 - (안티그래비티 프로젝트 '무중력 속도')
    불필요한 미들웨어 생략 & orjson 직렬화 & aioredis 활용
    """
    if redis_pool:
        try:
            # 비동기 Redis에서 GET 단 한 번으로 조회 (인위적 지연 없음)
            async with aioredis.async_timeout.timeout(0.05):
                cached_data = await redis_pool.get(f"fastcert:{cert_id}")
            if cached_data:
                return Response(content=cached_data, media_type="application/json")
        except Exception as e:
            logger.warning("Redis get failed for cert %s, falling back to DB: %s", cert_id, e)

    # Redis 미연결/미적중/타임아웃 시 DB 폴백
    db = SessionLocal()
    try:
        row = db.execute(
            text("""
                SELECT qual_id, qual_name, qual_type, main_field, ncs_large, managing_body, grade_code
                FROM qualification
                WHERE qual_id = :id
            """),
            {"id": cert_id}
        ).mappings().first()
        
        if not row:
            # 군더더기 메시지 삭제하여 초경량화 응답
            return Response(
                content=b'{"status":"error","message":"Not found"}', 
                status_code=404, 
                media_type="application/json"
            )
            
        # JSON 규격에 맞게 딕셔너리로 감싸거나 orjson 가장 빠른 변환 시도
        payload = {"status": "success", "data": dict(row)}
        result_bytes = orjson.dumps(payload)
        
        # 여기서 생성된 bytes 캐싱은 스트림 파이프라인(Redis Sync)에서 담당하므로 생략하거나 최소 방어로 추가 가능
        # 초경량 응답
        return Response(content=result_bytes, media_type="application/json")
    finally:
        db.close()
