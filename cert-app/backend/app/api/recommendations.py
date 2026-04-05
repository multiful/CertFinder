"""Recommendation API routes."""
from __future__ import annotations

import logging
import math
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import or_, text
from sqlalchemy.orm import Session

from app.api.deps import get_db_session, check_rate_limit, get_current_user
from app.schemas import (
    RecommendationListResponse,
    RecommendationResponse,
    JobCertificationRecommendationResponse,
    RelatedJobResponse,
    AvailableMajorsResponse
)
from app.crud import major_map_crud
from app.redis_client import redis_client
from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


def get_cache_ttl() -> int:
    """Get cache TTL for recommendations."""
    return settings.CACHE_TTL_RECOMMENDATIONS


def _recommendation_fetch_limit(requested_limit: int) -> int:
    """DB에서 넉넉히 가져온 뒤 점수·min_score로 걸러 최대 requested_limit개까지 반환."""
    return min(200, max(60, requested_limit * 4))


def _compute_raw_recommendation_score(
    mapping,
    latest_stats: Optional[object],
    grade_year: Optional[int],
) -> float:
    """통계·난이도·학년 보정을 반영한 원시 점수 (필터·정렬용, 상한 완화)."""
    base_relevance = float(mapping.score)
    if base_relevance <= 5:
        base_relevance = base_relevance * 2

    candidates = latest_stats.candidate_cnt if latest_stats and latest_stats.candidate_cnt else 0
    demand_score = 0.0
    if candidates > 0:
        demand_score = min(10.0, math.log10(candidates) * 2)

    pass_rate = latest_stats.pass_rate if latest_stats and latest_stats.pass_rate else 0.0
    stability_score = (pass_rate / 100.0) * 10.0

    diff = 5.0
    if latest_stats and latest_stats.difficulty_score is not None:
        diff = float(latest_stats.difficulty_score)

    if latest_stats:
        final_score = (base_relevance * 0.58) + (demand_score * 0.19) + (stability_score * 0.19)
        final_score += (diff / 10.0) * 0.35
        if grade_year is not None:
            if grade_year <= 2:
                if diff <= 6:
                    final_score += 1.0
            else:
                if diff > 6:
                    final_score += 1.0
    else:
        final_score = base_relevance

    return float(min(11.0, max(0.0, final_score)))


def _apply_display_score_spread(raw_values: list[float]) -> list[float]:
    """
    상위권이 한데 몰려 9.0처럼 보일 때, 순위는 유지한 채 표시 구간을 넓힌다.
    """
    if not raw_values:
        return []
    if len(raw_values) == 1:
        return [min(9.9, max(5.0, raw_values[0]))]
    lo, hi = min(raw_values), max(raw_values)
    if hi - lo >= 0.85:
        return [min(9.9, max(5.0, v)) for v in raw_values]
    lo_d, hi_d = 5.5, 9.3
    out: list[float] = []
    for v in raw_values:
        t = (v - lo) / (hi - lo) if hi > lo else 0.5
        out.append(lo_d + t * (hi_d - lo_d))
    return out


def generate_recommendation_reason(
    mapping,
    latest_stats: Optional[object]
) -> str:
    """Generate recommendation reason text."""
    reasons = []
    
    # Use stored reason if available
    if mapping.reason:
        return mapping.reason
    
    # Generate based on data
    if mapping.score >= 8:
        reasons.append("전공과 높은 연관성")
    elif mapping.score >= 5:
        reasons.append("전공과 관련된 분야")
    else:
        reasons.append("보조적 자격증")
    
    if latest_stats:
        if latest_stats.pass_rate and latest_stats.pass_rate >= 70:
            reasons.append(f"합격률 {latest_stats.pass_rate:.1f}%로 안정적")
        elif latest_stats.pass_rate and latest_stats.pass_rate <= 30:
            reasons.append(f"경쟁률 높음 (합격률 {latest_stats.pass_rate:.1f}%)")
        
        if latest_stats.difficulty_score:
            if latest_stats.difficulty_score >= 7:
                reasons.append("고난이도 시험")
            elif latest_stats.difficulty_score <= 4:
                reasons.append("입문자에게 적합")
    
    return " / ".join(reasons) if reasons else "전공 기반 추천"


@router.get(
    "",
    response_model=RecommendationListResponse,
    summary="Get recommendations by major",
    description="Get certification recommendations based on major/field of study."
)
async def get_recommendations(
    major: str = Query(..., description="Major or field of study"),
    limit: int = Query(10, ge=1, le=50, description="Number of recommendations"),
    min_score: float = Query(
        0.0,
        ge=0.0,
        le=10.0,
        description="원시 추천점수가 이 값 미만인 항목은 제외 (0이면 필터 없음)",
    ),
    grade_year: Optional[int] = Query(None, description="Grade year (0-4)"),
    db: Session = Depends(get_db_session),
    _: None = Depends(check_rate_limit),
):
    """Get certification recommendations for a major."""
    fetch_n = _recommendation_fetch_limit(limit)
    cache_key = redis_client.make_cache_key(
        "recs:v7",
        major=major.lower().strip(),
        limit=limit,
        min_score=round(min_score, 2),
        grade=grade_year,
    )

    try:
        cached = redis_client.get(cache_key)
        if cached and isinstance(cached, dict):
            logger.debug("Cache hit for recommendations: %s", major)
            return RecommendationListResponse(**cached)
    except Exception as e:
        logger.warning("Cache read failed for recommendations: %s", e)

    search_major = major.strip()
    resolved_major = search_major
    mappings = major_map_crud.get_by_major_with_stats(db, search_major, fetch_n)

    if not mappings:
        clean_major = search_major
        for suffix in ["학부", "학과", "전공", "공학부"]:
            if clean_major.endswith(suffix):
                clean_major = clean_major[: -len(suffix)]
                break

        from app.models import MajorQualificationMap

        matched_map = (
            db.query(MajorQualificationMap.major)
            .filter(
                or_(
                    MajorQualificationMap.major.ilike(f"%{clean_major}%"),
                    text(":major ILIKE '%' || major || '%'"),
                )
            )
            .params(major=search_major)
            .first()
        )

        if not matched_map and len(clean_major) >= 3:
            short_major = clean_major[:2]
            matched_map = (
                db.query(MajorQualificationMap.major)
                .filter(MajorQualificationMap.major.ilike(f"%{short_major}%"))
                .first()
            )

        if matched_map:
            matched_major = matched_map[0]
            logger.info("Fuzzy match: '%s' -> '%s'", search_major, matched_major)
            mappings = major_map_crud.get_by_major_with_stats(db, matched_major, fetch_n)
            resolved_major = matched_major

    if not mappings:
        return RecommendationListResponse(items=[], major=search_major, total=0)

    rows: list[dict] = []
    for mapping in mappings:
        qual = mapping.qualification
        if not qual:
            continue
        latest_stats = None
        if qual.stats:
            latest_stats = max(qual.stats, key=lambda s: (s.year, s.exam_round))
        raw = _compute_raw_recommendation_score(mapping, latest_stats, grade_year)
        rows.append(
            {
                "mapping": mapping,
                "qual": qual,
                "latest_stats": latest_stats,
                "raw": raw,
            }
        )

    rows = [r for r in rows if r["raw"] + 1e-9 >= min_score]
    rows.sort(key=lambda r: r["raw"], reverse=True)
    rows = rows[:limit]

    raws = [float(r["raw"]) for r in rows]
    display_scores = _apply_display_score_spread(raws)

    recommendations = []
    for r, disp in zip(rows, display_scores, strict=True):
        qual = r["qual"]
        mapping = r["mapping"]
        latest_stats = r["latest_stats"]
        recommendations.append(
            RecommendationResponse(
                qual_id=qual.qual_id,
                qual_name=qual.qual_name,
                qual_type=qual.qual_type,
                main_field=qual.main_field,
                managing_body=qual.managing_body,
                score=round(float(disp), 1),
                reason=generate_recommendation_reason(mapping, latest_stats),
                latest_pass_rate=latest_stats.pass_rate if latest_stats else None,
            )
        )

    response = RecommendationListResponse(
        items=recommendations,
        major=resolved_major,
        total=len(recommendations),
    )

    redis_client.set(cache_key, response.model_dump(mode="json"), get_cache_ttl())

    return response


@router.get(
    "/me",
    response_model=RecommendationListResponse,
    summary="Get recommendations for current user",
    description="Get certification recommendations based on the logged-in user's major."
)
async def get_my_recommendations(
    limit: int = Query(10, ge=1, le=50),
    min_score: float = Query(0.0, ge=0.0, le=10.0),
    db: Session = Depends(get_db_session),
    user_id: str = Depends(get_current_user),
    _: None = Depends(check_rate_limit),
):
    """Get recommendations based on current user's profile major."""
    # 1. Get user's major and grade from profiles table
    row = db.execute(
        text("SELECT detail_major, grade_year FROM profiles WHERE id = :id"),
        {"id": user_id}
    ).mappings().first()
    
    if not row or not row["detail_major"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="프로필에 전공 정보가 없습니다. 프로필을 먼저 설정해주세요."
        )
    
    major = row["detail_major"]
    grade_year = row["grade_year"]
    
    return await get_recommendations(
        major=major,
        limit=limit,
        min_score=min_score,
        grade_year=grade_year,
        db=db,
        _=None,
    )


@router.get(
    "/majors",
    response_model=AvailableMajorsResponse,
    summary="Get available majors",
    description="Get list of all available majors for recommendations."
)
async def get_available_majors(
    db: Session = Depends(get_db_session),
    _: None = Depends(check_rate_limit)
):
    """Get list of available majors."""
    cache_key = "recs:majors:v5"
    
    try:
        cached = redis_client.get(cache_key)
        if cached and isinstance(cached, list):
            return {"majors": cached}
    except Exception as e:
        logger.warning(f"Cache read failed for majors list: {e}")
    
    majors = major_map_crud.get_majors_list(db)
    # Ensure it's a list before returning/caching
    if not isinstance(majors, list):
        majors = list(majors) if majors else []
        
    redis_client.set(cache_key, majors, get_cache_ttl())
    
    return {"majors": majors}


@router.get(
    "/popular-majors",
    response_model=AvailableMajorsResponse,
    summary="Get popular majors by user count",
    description="전공 기반 자격증 추천용. profiles.detail_major 집계 후 상위 N개 반환.",
)
async def get_popular_majors(
    limit: int = Query(12, ge=1, le=20),
    db: Session = Depends(get_db_session),
    _: None = Depends(check_rate_limit),
):
    """사용자들이 설정한 전공(detail_major)을 카운팅해 인기 전공 목록 반환 (정규화 없이 DB 값 그대로)."""
    cache_key = f"recs:popular_majors:v4:{limit}"
    try:
        cached = redis_client.get(cache_key)
        if cached and isinstance(cached, list):
            return {"majors": cached}
    except Exception as e:
        logger.warning("Cache read failed for popular majors: %s", e)

    try:
        rows = db.execute(
            text("""
                SELECT detail_major AS major, COUNT(*) AS cnt
                FROM profiles
                WHERE detail_major IS NOT NULL AND TRIM(detail_major) != ''
                GROUP BY detail_major
                ORDER BY cnt DESC
                LIMIT :limit
            """),
            {"limit": limit},
        ).mappings().fetchall()
        majors = [str(r["major"]) for r in (rows or []) if r and r.get("major")]
    except Exception as e:
        logger.error("popular_majors DB query failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="인기 전공 조회 중 오류가 발생했습니다.",
        ) from e
    try:
        redis_client.set(cache_key, majors, get_cache_ttl())
    except Exception:
        pass
    return {"majors": majors}


@router.get(
    "/jobs/{job_id}/certifications",
    response_model=list[JobCertificationRecommendationResponse],
    summary="Get certifications for a job",
    description="Get recommended certifications required or helpful for a specific job goal."
)
async def get_certifications_for_job(
    job_id: int,
    db: Session = Depends(get_db_session),
    _: None = Depends(check_rate_limit)
):
    """
    Get certifications for a specific job.
    SQL Design:
    SELECT 
        q.qual_name, q.main_field, j.job_name,
        j.entry_salary, j.outlook_summary
    ...
    """
    query = text("""
        SELECT 
            q.qual_id,
            q.qual_name,
            q.main_field,
            j.job_name,
            j.entry_salary,
            j.outlook_summary
        FROM qualification q
        JOIN qualification_job_map qjm ON q.qual_id = qjm.qual_id
        JOIN job j ON qjm.job_id = j.job_id
        WHERE j.job_id = :job_id
        ORDER BY q.qual_type ASC
    """)
    
    results = db.execute(query, {"job_id": job_id}).mappings().all()
    
    return [JobCertificationRecommendationResponse(**row) for row in results]


@router.get(
    "/certifications/{qual_id}/jobs",
    response_model=list[RelatedJobResponse],
    summary="Get related jobs for a certification",
    description="Get jobs that can be pursued with a specific certification."
)
async def get_jobs_for_certification(
    qual_id: int,
    db: Session = Depends(get_db_session),
    _: None = Depends(check_rate_limit)
):
    """
    Get related jobs for a certification.
    SQL Design:
    SELECT 
        j.job_name, j.reward, j.stability, j.development
    ...
    """
    query = text("""
        SELECT 
            j.job_id,
            j.job_name,
            j.reward AS salary_score,
            j.stability AS stability_score,
            j.development AS growth_score
        FROM job j
        JOIN qualification_job_map qjm ON j.job_id = qjm.job_id
        WHERE qjm.qual_id = :qual_id
        ORDER BY j.reward DESC
        LIMIT 5
    """)
    
    results = db.execute(query, {"qual_id": qual_id}).mappings().all()
    
    return [RelatedJobResponse(**row) for row in results]
