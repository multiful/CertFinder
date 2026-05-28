"""HRDK 시험 일정 조회 — 한국산업인력공단 공공데이터 포털 API 프록시.

전략:
  - jmCd(종목코드) 없이 연도+자격구분으로 전체 일정 조회
  - qualgbCd=T (기술자격): grade_code('기사','기능사' 등)로 description 매칭
  - qualgbCd=S (전문자격): grade_code 없거나 등급형(1급/2급 등) — 전체 반환
  - 같은 등급 자격증은 동일 시험 일정 창을 공유하므로 매칭 가능
"""
from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
import httpx
import logging
import json
import re
from datetime import datetime

from app.api.deps import get_db_session, check_rate_limit
from app.config import get_settings
from app.redis_client import redis_client
from app.models import Qualification

settings = get_settings()
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/certs", tags=["exam-schedule"])

HRDK_BASE = "http://apis.data.go.kr/B490007/qualExamSchd/getQualExamSchdList"
HRDK_INFO_BASE = "http://apis.data.go.kr/B490007/qualInfo/getQualInfoList"
# HRDK API는 numOfRows 최대 30 정도 안정적
PAGE_SIZE = 30

# 국가기술자격(T) grade_code → description 매칭 키워드
# description 예시: "국가기술자격 기사 (2026년도 제1회)"
TECH_GRADE_KEYWORDS: dict[str, str] = {
    "기사":    "기사",
    "산업기사": "산업기사",
    "기능사":  "기능사",
    "기능장":  "기능장",
    "기술사":  "기술사",
}
# 구 코드 호환
GRADE_KEYWORDS = TECH_GRADE_KEYWORDS


def _infer_qualgb(grade_code: Optional[str]) -> str:
    """grade_code로 자격구분코드(T/S) 추론.

    기술자격: 기사/산업기사/기능사/기능장/기술사 → T
    전문자격: 그 외(1급/2급/None 등) → S
    """
    if grade_code and grade_code in TECH_GRADE_KEYWORDS:
        return "T"
    return "S"


class ExamRound(BaseModel):
    year: int
    round: int
    description: str
    doc_reg_start: Optional[str] = None
    doc_reg_end: Optional[str] = None
    doc_exam_start: Optional[str] = None
    doc_exam_end: Optional[str] = None
    doc_pass_dt: Optional[str] = None
    prac_reg_start: Optional[str] = None
    prac_reg_end: Optional[str] = None
    prac_exam_start: Optional[str] = None
    prac_exam_end: Optional[str] = None
    prac_pass_dt: Optional[str] = None


class ExamScheduleResponse(BaseModel):
    qual_id: int
    qual_name: str
    grade_code: Optional[str]
    source: str  # "hrdk" | "none" | "no_key"
    year: int
    schedules: list[ExamRound]
    fetched_at: str


def _fmt_date(val: Optional[str]) -> Optional[str]:
    """YYYYMMDD → YYYY-MM-DD. 빈 문자열·없으면 None."""
    if not val or not str(val).strip():
        return None
    v = str(val).strip()
    if len(v) == 8 and v.isdigit():
        return f"{v[:4]}-{v[4:6]}-{v[6:]}"
    return v


def _item_to_round(it: dict) -> ExamRound:
    return ExamRound(
        year=int(it.get("implYy", 0)),
        round=int(it.get("implSeq", 0)),
        description=it.get("description", ""),
        doc_reg_start=_fmt_date(it.get("docRegStartDt")),
        doc_reg_end=_fmt_date(it.get("docRegEndDt")),
        doc_exam_start=_fmt_date(it.get("docExamStartDt")),
        doc_exam_end=_fmt_date(it.get("docExamEndDt")),
        doc_pass_dt=_fmt_date(it.get("docPassDt")),
        prac_reg_start=_fmt_date(it.get("pracRegStartDt")),
        prac_reg_end=_fmt_date(it.get("pracRegEndDt")),
        prac_exam_start=_fmt_date(it.get("pracExamStartDt")),
        prac_exam_end=_fmt_date(it.get("pracExamEndDt")),
        prac_pass_dt=_fmt_date(it.get("pracPassDt")),
    )


async def _fetch_all_schedules(year: int, qualgb: str) -> list[dict]:
    """HRDK API를 페이지네이션하며 전체 일정 목록 조회."""
    key = settings.HRDK_API_KEY
    if not key:
        raise ValueError("HRDK_API_KEY not configured")

    all_items: list[dict] = []
    page = 1

    async with httpx.AsyncClient(timeout=settings.HRDK_TIMEOUT) as client:
        while True:
            params = {
                "serviceKey": key,
                "dataFormat": "json",
                "implYy": str(year),
                "qualgbCd": qualgb,
                "numOfRows": str(PAGE_SIZE),
                "pageNo": str(page),
            }
            resp = await client.get(HRDK_BASE, params=params)
            resp.raise_for_status()
            data = resp.json()

            body = data.get("body", {})
            items = body.get("items", [])
            if isinstance(items, dict):
                items = [items]
            if not items:
                break

            all_items.extend(items)
            total = int(body.get("totalCount", 0))
            if len(all_items) >= total or len(items) < PAGE_SIZE:
                break
            page += 1

    return all_items


def _match_by_grade(items: list[dict], grade_code: Optional[str]) -> list[dict]:
    """grade_code에 해당하는 일정만 필터링.

    T 타입: description에서 등급 키워드로 정확 매칭
    S 타입(전문자격): description이 '전문자격 (...)'으로만 표시되어 키워드 매칭 불가
             → 전체 반환 (연도 필터는 API 호출 시 이미 적용됨)
    """
    qualgb = _infer_qualgb(grade_code)

    # 전문자격(S) 또는 분류 불가 → 그대로 반환
    if qualgb == "S":
        return items

    keyword = TECH_GRADE_KEYWORDS.get(grade_code or "")
    if not keyword:
        return items

    # description 예: "국가기술자격 기사 (2026년도 제1회)"
    # "기사"는 "산업기사" 안에도 포함 → 단어 경계로 매칭
    pattern = re.compile(r'(?<![가-힣])' + re.escape(keyword) + r'(?![가-힣])')
    matched = [it for it in items if pattern.search(it.get("description", ""))]

    # "기사"로 검색 시 "산업기사"가 매칭되면 제거
    if keyword == "기사":
        matched = [it for it in matched if "산업기사" not in it.get("description", "")]

    return matched


@router.get(
    "/{qual_id}/exam-schedule",
    response_model=ExamScheduleResponse,
    summary="시험 일정 조회 (HRDK 공공 API)",
    description="HRDK 공공 API에서 해당 자격증 등급의 시험 일정을 가져옵니다.",
)
async def get_exam_schedule(
    qual_id: int,
    year: Optional[int] = Query(None, description="조회 연도 (기본: 현재 연도)"),
    db: Session = Depends(get_db_session),
    _: None = Depends(check_rate_limit),
) -> ExamScheduleResponse:

    cert = db.query(Qualification).filter(Qualification.qual_id == qual_id).first()
    if not cert:
        raise HTTPException(status_code=404, detail="Certification not found")

    target_year = year or datetime.now().year
    grade_code = cert.grade_code

    # Redis 캐시 확인 (grade_code + year 기준)
    cache_key = f"exam_sched:v2:{grade_code or 'none'}:{target_year}"
    cached_raw = None
    try:
        cached_raw = redis_client.get(cache_key) if redis_client else None
    except Exception:
        pass

    all_items: list[dict] = []

    if cached_raw:
        try:
            all_items = json.loads(cached_raw)
        except Exception:
            cached_raw = None

    if not all_items and not cached_raw:
        if not settings.HRDK_API_KEY:
            return ExamScheduleResponse(
                qual_id=qual_id, qual_name=cert.qual_name, grade_code=grade_code,
                source="no_key", year=target_year, schedules=[],
                fetched_at=datetime.now().isoformat(),
            )

        try:
            qualgb = _infer_qualgb(grade_code)
            all_items = await _fetch_all_schedules(target_year, qualgb)

            # 캐시 저장 (grade+year 단위, TTL=6시간)
            if redis_client and all_items:
                try:
                    redis_client.set(cache_key, json.dumps(all_items), ex=settings.HRDK_CACHE_TTL)
                except Exception:
                    pass

        except httpx.HTTPStatusError as e:
            logger.error(f"HRDK HTTP {e.response.status_code} for qual_id={qual_id}")
            raise HTTPException(status_code=502, detail=f"HRDK API 오류: HTTP {e.response.status_code}")
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="HRDK API 응답 시간 초과")
        except ValueError as e:
            logger.warning(str(e))
            return ExamScheduleResponse(
                qual_id=qual_id, qual_name=cert.qual_name, grade_code=grade_code,
                source="no_key", year=target_year, schedules=[],
                fetched_at=datetime.now().isoformat(),
            )
        except Exception as e:
            logger.error(f"HRDK unexpected error: {e}")
            raise HTTPException(status_code=500, detail="시험 일정 조회 중 오류가 발생했습니다.")

    matched = _match_by_grade(all_items, grade_code)
    matched.sort(key=lambda x: int(x.get("implSeq", 0)))
    schedules = [_item_to_round(it) for it in matched]

    return ExamScheduleResponse(
        qual_id=qual_id,
        qual_name=cert.qual_name,
        grade_code=grade_code,
        source="hrdk" if schedules else "none",
        year=target_year,
        schedules=schedules,
        fetched_at=datetime.now().isoformat(),
    )


class QualInfoResponse(BaseModel):
    qual_id: int
    qual_name: str
    source: str  # "hrdk" | "none" | "no_key"
    managing_body: Optional[str] = None
    exam_method: Optional[str] = None
    eligibility: Optional[str] = None
    job_description: Optional[str] = None
    fetched_at: str


@router.get(
    "/{qual_id}/qual-info",
    response_model=QualInfoResponse,
    summary="자격 상세정보 조회 (HRDK)",
    description="HRDK 공공 API에서 자격증 시험 방법·응시 자격·직무 내용을 가져옵니다.",
)
async def get_qual_info(
    qual_id: int,
    db: Session = Depends(get_db_session),
    _: None = Depends(check_rate_limit),
) -> QualInfoResponse:

    cert = db.query(Qualification).filter(Qualification.qual_id == qual_id).first()
    if not cert:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Certification not found")

    cache_key = f"qual_info:v1:{qual_id}"
    try:
        cached_raw = redis_client.get(cache_key) if redis_client else None
        if cached_raw and isinstance(cached_raw, dict):
            return QualInfoResponse(**cached_raw)
    except Exception:
        pass

    if not settings.HRDK_API_KEY:
        return QualInfoResponse(
            qual_id=qual_id, qual_name=cert.qual_name,
            source="no_key", fetched_at=datetime.now().isoformat(),
        )

    try:
        params = {
            "serviceKey": settings.HRDK_API_KEY,
            "dataFormat": "json",
            "jmNm": cert.qual_name,
            "numOfRows": "5",
            "pageNo": "1",
        }
        async with httpx.AsyncClient(timeout=settings.HRDK_TIMEOUT) as client:
            resp = await client.get(HRDK_INFO_BASE, params=params)
            resp.raise_for_status()
            data = resp.json()

        body = data.get("body", {})
        items = body.get("items", [])
        if isinstance(items, dict):
            items = [items]

        if not items:
            return QualInfoResponse(
                qual_id=qual_id, qual_name=cert.qual_name,
                source="none", fetched_at=datetime.now().isoformat(),
            )

        it = items[0]
        result = QualInfoResponse(
            qual_id=qual_id,
            qual_name=cert.qual_name,
            source="hrdk",
            managing_body=it.get("insttNm") or it.get("mngInsttNm") or None,
            exam_method=it.get("examMthd") or None,
            eligibility=it.get("applcQual") or None,
            job_description=it.get("jbdcScop") or None,
            fetched_at=datetime.now().isoformat(),
        )

        try:
            if redis_client:
                redis_client.set(cache_key, result.model_dump(), ex=86400)  # 24시간
        except Exception:
            pass

        return result

    except Exception as e:
        logger.error("HRDK qual-info error for qual_id=%s: %s", qual_id, e)
        return QualInfoResponse(
            qual_id=qual_id, qual_name=cert.qual_name,
            source="none", fetched_at=datetime.now().isoformat(),
        )
