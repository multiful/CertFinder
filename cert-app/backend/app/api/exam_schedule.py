"""HRDK 시험 일정 조회 — 한국산업인력공단 공공데이터 포털 API 프록시."""
from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
import httpx
import logging
from datetime import datetime
import json

from app.api.deps import get_db_session, check_rate_limit
from app.config import get_settings
from app.redis_client import redis_client
from app.models import Qualification

settings = get_settings()
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/certs", tags=["exam-schedule"])

HRDK_EXAM_SCHD_URL = "https://apis.data.go.kr/B490007/qualExamSchd/getQualExamSchdList"


class ExamRound(BaseModel):
    year: int
    round: int  # 회차
    qual_name: str
    # 필기 일정
    doc_reg_start: Optional[str] = None   # 원서접수 시작
    doc_reg_end: Optional[str] = None     # 원서접수 마감
    doc_exam_start: Optional[str] = None  # 필기시험 시작
    doc_exam_end: Optional[str] = None    # 필기시험 종료
    doc_pass_dt: Optional[str] = None     # 필기합격자 발표
    # 실기 일정
    prac_reg_start: Optional[str] = None  # 실기원서접수 시작
    prac_reg_end: Optional[str] = None    # 실기원서접수 마감
    prac_exam_start: Optional[str] = None # 실기시험 시작
    prac_exam_end: Optional[str] = None   # 실기시험 종료
    prac_pass_dt: Optional[str] = None    # 최종합격자 발표


class ExamScheduleResponse(BaseModel):
    qual_id: int
    qual_name: str
    source: str  # "hrdk" | "none"
    schedules: list[ExamRound]
    fetched_at: str


def _parse_date(val: Optional[str]) -> Optional[str]:
    """YYYYMMDD → YYYY-MM-DD 변환. 없으면 None."""
    if not val or not str(val).strip() or str(val) == "0":
        return None
    v = str(val).strip()
    if len(v) == 8 and v.isdigit():
        return f"{v[:4]}-{v[4:6]}-{v[6:]}"
    return v


async def _fetch_hrdk_schedule(qual_name: str, year: int) -> list[dict]:
    """HRDK API 호출 — 자격명 + 연도로 시험일정 목록 반환."""
    key = settings.HRDK_API_KEY
    if not key:
        raise ValueError("HRDK_API_KEY not configured")

    params = {
        "serviceKey": key,
        "numOfRows": "50",
        "pageNo": "1",
        "dataType": "JSON",
        "implYy": str(year),
        "qualNm": qual_name,
    }

    async with httpx.AsyncClient(timeout=settings.HRDK_TIMEOUT) as client:
        resp = await client.get(HRDK_EXAM_SCHD_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

    # 공공데이터포털 공통 응답 구조 파싱
    body = data.get("response", {}).get("body", {})
    items = body.get("items", {})
    if not items:
        return []
    raw = items.get("item", [])
    if isinstance(raw, dict):
        raw = [raw]
    return raw


def _parse_items(qual_id: int, qual_name: str, items: list[dict], year: int) -> list[ExamRound]:
    rounds = []
    for it in items:
        impl_seq = it.get("implSeq", 0)
        round_val = int(impl_seq) if str(impl_seq).isdigit() else 0
        rounds.append(ExamRound(
            year=year,
            round=round_val,
            qual_name=it.get("qualNm", qual_name),
            doc_reg_start=_parse_date(it.get("docRegStartDt")),
            doc_reg_end=_parse_date(it.get("docRegEndDt")),
            doc_exam_start=_parse_date(it.get("docExamStartDt")),
            doc_exam_end=_parse_date(it.get("docExamEndDt")),
            doc_pass_dt=_parse_date(it.get("docPassDt")),
            prac_reg_start=_parse_date(it.get("pracRegStartDt")),
            prac_reg_end=_parse_date(it.get("pracRegEndDt")),
            prac_exam_start=_parse_date(it.get("pracExamStartDt")),
            prac_exam_end=_parse_date(it.get("pracExamEndDt")),
            prac_pass_dt=_parse_date(it.get("pracPassDt")),
        ))
    rounds.sort(key=lambda r: (r.year, r.round))
    return rounds


@router.get(
    "/{qual_id}/exam-schedule",
    response_model=ExamScheduleResponse,
    summary="시험 일정 조회",
    description="HRDK 공공데이터 API에서 해당 자격증의 시험 일정(원서접수·필기·실기·합격자 발표)을 가져옵니다.",
)
async def get_exam_schedule(
    qual_id: int,
    year: Optional[int] = Query(None, description="조회 연도 (기본: 현재 연도)"),
    db: Session = Depends(get_db_session),
    _: None = Depends(check_rate_limit),
) -> ExamScheduleResponse:
    # 자격증 이름 조회
    cert = db.query(Qualification).filter(Qualification.qual_id == qual_id).first()
    if not cert:
        raise HTTPException(status_code=404, detail="Certification not found")

    target_year = year or datetime.now().year
    qual_name = cert.qual_name

    # Redis 캐시 확인
    cache_key = f"exam_sched:v1:{qual_id}:{target_year}"
    cached = redis_client.get(cache_key) if redis_client else None
    if cached:
        try:
            data = json.loads(cached)
            return ExamScheduleResponse(**data)
        except Exception:
            pass

    if not settings.HRDK_API_KEY:
        return ExamScheduleResponse(
            qual_id=qual_id,
            qual_name=qual_name,
            source="none",
            schedules=[],
            fetched_at=datetime.now().isoformat(),
        )

    try:
        items = await _fetch_hrdk_schedule(qual_name, target_year)

        # 이름이 정확히 일치하는 항목만 필터 (API가 부분 검색 결과를 내보낼 수 있음)
        exact = [it for it in items if it.get("qualNm", "").strip() == qual_name.strip()]
        if not exact and items:
            # 정확 일치 없으면 포함 검색 결과 중 가장 유사한 것 사용
            exact = items

        schedules = _parse_items(qual_id, qual_name, exact, target_year)
        source = "hrdk" if schedules else "none"

        result = ExamScheduleResponse(
            qual_id=qual_id,
            qual_name=qual_name,
            source=source,
            schedules=schedules,
            fetched_at=datetime.now().isoformat(),
        )

        # 캐시 저장
        if redis_client and schedules:
            try:
                redis_client.set(cache_key, result.model_dump_json(), ex=settings.HRDK_CACHE_TTL)
            except Exception:
                pass

        return result

    except httpx.HTTPStatusError as e:
        logger.error(f"HRDK API HTTP error for qual_id={qual_id}: {e.response.status_code}")
        raise HTTPException(status_code=502, detail=f"HRDK API 오류: HTTP {e.response.status_code}")
    except httpx.TimeoutException:
        logger.warning(f"HRDK API timeout for qual_id={qual_id}")
        raise HTTPException(status_code=504, detail="HRDK API 응답 시간 초과")
    except ValueError as e:
        logger.warning(str(e))
        return ExamScheduleResponse(
            qual_id=qual_id,
            qual_name=qual_name,
            source="none",
            schedules=[],
            fetched_at=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Unexpected error fetching exam schedule: {e}")
        raise HTTPException(status_code=500, detail="시험 일정 조회 중 오류가 발생했습니다.")
