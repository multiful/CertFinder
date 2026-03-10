"""
Hybrid 검색: BM25 + Vector를 RRF(Reciprocal Rank Fusion)로 병합 → 메타데이터 필터 → (선택) Cross-Encoder rerank.
Query Routing: 짧은 키워드/약어 쿼리는 BM25 중심 + Vector 게이팅.
RRF: score(d) = w_b * 1/(k+rank_bm25) + w_v * 1/(k+rank_vector).
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session
from sqlalchemy import text

from app.rag.config import get_rag_index_dir, get_rag_settings
from app.rag.eval.query_type import classify_query_type
from app.rag.index.bm25_index import BM25Index
from app.rag.index.vector_index import get_vector_search
from app.rag.utils.query_processor import expand_query_single_string, expand_query
from app.rag.utils.dense_query_rewrite import rewrite_for_dense, extract_slots_for_dense, UserProfile
from app.rag.utils.hyde import generate_hyde_document
from app.rag.utils.cot_query import expand_query_cot, stepback_query
from app.rag.retrieve.metadata_soft_score import (
    compute_metadata_soft_score,
    fetch_qual_metadata_bulk,
)
from app.rag.retrieve.personalized_soft_score import (
    compute_personalized_soft_score,
    merge_difficulty_into_metadata,
)
from app.rag.retrieve.contrastive_retriever import contrastive_search

logger = logging.getLogger(__name__)
# RRF_K는 get_rag_settings().RAG_RRF_K 사용 (기본 28). 하위 호환용 상수 유지.
RRF_K = 28


def _rrf_k() -> int:
    """설정에서 RRF 상수 조회. env RAG_RRF_K로 튜닝 가능."""
    return getattr(get_rag_settings(), "RAG_RRF_K", 28)


def _vector_gating_suspicious(
    bm25_list: List[Tuple[str, float]],
    vector_list: List[Tuple[str, float]],
    bm25_top_n: int = 20,
    vec_min_score: float = 0.55,
    vec_gap_min: float = 0.02,
) -> bool:
    """
    짧은 쿼리에서 Vector Top1이 오탐인지 판단.
    - 조건 A: vec_top1이 bm25_top_n 안에 없으면 suspicious
    - 조건 B: vec_top1_score < vec_min_score 이면 suspicious
    - 조건 C: (vec_top1 - vec_top2) < vec_gap_min 이면 suspicious
    반환: True면 Vector 가중치를 낮추거나 제외해야 함.
    """
    if not vector_list:
        return False
    vec_top1_id = vector_list[0][0]
    vec_top1_score = vector_list[0][1]
    vec_top2_score = vector_list[1][1] if len(vector_list) >= 2 else 0.0

    bm25_top_ids = {cid for cid, _ in bm25_list[:bm25_top_n]}
    if vec_top1_id not in bm25_top_ids and bm25_list:
        return True  # A: Vector 1위가 BM25 상위에 없음
    if vec_top1_score < vec_min_score:
        return True  # B: Vector 확신 낮음
    if (vec_top1_score - vec_top2_score) < vec_gap_min:
        return True  # C: 1·2위 격차 작음
    return False


def _is_short_query(query: str) -> bool:
    """토큰 수가 3 이하이면 짧은 키워드 쿼리로 간주."""
    return len((query or "").strip().split()) <= 3


def _query_weights_for_rrf(query: str) -> Tuple[float, float]:
    """
    질의 타입에 따라 BM25/Vector 가중치 반환.
    짧은 키워드: 설정값 RAG_HYBRID_SHORT_* 사용 (게이팅 후 w_vec 조정은 hybrid_retrieve에서).
    긴 쿼리: RAG_HYBRID_LONG_* 사용.
    """
    settings = get_rag_settings()
    if _is_short_query((query or "").strip()):
        w_b = getattr(settings, "RAG_HYBRID_SHORT_W_BM25", 1.0)
        w_v = getattr(settings, "RAG_HYBRID_SHORT_W_VEC", 0.2)
        return w_b, w_v
    w_b = getattr(settings, "RAG_HYBRID_LONG_W_BM25", 0.7)
    w_v = getattr(settings, "RAG_HYBRID_LONG_W_VEC", 1.0)
    return w_b, w_v


# query_type별 fusion 가중치 (자격증명 포함→BM25 강화, 자연어→Vector 강화). RAG_QUERY_TYPE_WEIGHTS_ENABLE 시 사용.
QUERY_TYPE_RRF_WEIGHTS: Dict[str, Tuple[float, float]] = {
    "cert_name_included": (0.40, 0.60),  # 자격증명·키워드 성향 → BM25 비중 상향
    "natural": (0.26, 0.74),              # 자연어 문장 → Vector 비중 상향
    "keyword": (0.36, 0.64),
    "major+job": (0.30, 0.70),
    "purpose_only": (0.32, 0.68),
    "roadmap": (0.34, 0.66),
    "comparison": (0.38, 0.62),
    "mixed": (0.30, 0.70),
}
# 비IT 쿼리 전용: BM25 강화(확장 골든 평가에서 비IT는 BM25가 유리). RAG_DOMAIN_AWARE_WEIGHTS_ENABLE 시 사용.
NON_IT_RRF_WEIGHTS: Tuple[float, float] = (0.58, 0.42)

# query_type별 Contrastive 가중치 multiplier (3-way RRF 시 사용).
CONTRASTIVE_QUERY_TYPE_WEIGHTS: Dict[str, float] = {
    # 자연어/의도형 쿼리에서는 Contrastive 비중을 강화
    "natural": 1.4,
    "purpose_only": 1.4,
    "roadmap": 1.3,
    "comparison": 1.3,
    "profile_personalized": 1.4,
    # 키워드/자격증명 위주 쿼리에서는 Contrastive 비중을 약화
    "keyword": 0.5,
    "cert_name_included": 0.3,
}


def _three_way_weights_by_query_type(query: str, query_type: str, settings: Any) -> Tuple[float, float, float]:
    """3-way RRF용 query_type·도메인 반영 가중치 (RAG_QUERY_TYPE_WEIGHTS_ENABLE 시). (w_bm25, w_dense, w_contrastive)."""
    if getattr(settings, "RAG_DOMAIN_AWARE_WEIGHTS_ENABLE", False) and not _query_suggests_it(query):
        b_ratio, v_ratio = NON_IT_RRF_WEIGHTS
    else:
        b_ratio, v_ratio = QUERY_TYPE_RRF_WEIGHTS.get(query_type, (0.30, 0.70))
    c_mult = CONTRASTIVE_QUERY_TYPE_WEIGHTS.get(query_type, 1.0)
    base_b = getattr(settings, "RAG_RRF_W_BM25", 1.0)
    base_v = getattr(settings, "RAG_RRF_W_DENSE1536", 1.0)
    base_c = getattr(settings, "RAG_RRF_W_CONTRASTIVE768", 1.2)
    # b_ratio, v_ratio는 2-way용 비율(합 1). 0.5,0.5 기준으로 스케일해 3-way base에 반영
    w_b = base_b * (b_ratio * 2.0)
    w_v = base_v * (v_ratio * 2.0)
    w_c = base_c * c_mult
    return (w_b, w_v, w_c)


def _query_suggests_it(query: str) -> bool:
    """쿼리가 IT 도메인으로 보이면 True. 도메인 가중치/도메인 불일치 감점에 사용."""
    try:
        from app.rag.utils.dense_query_rewrite import extract_slots_for_dense, _query_suggests_it_domain
        slots = extract_slots_for_dense(query)
        return _query_suggests_it_domain(slots, query)
    except Exception:
        return True  # 실패 시 IT로 간주(기존 동작 유지)


def _query_weights_by_type(query: str) -> Tuple[float, float]:
    """query_type별 BM25/Vector 가중치. RAG_QUERY_TYPE_WEIGHTS_ENABLE 시 기존 short/long 대신 사용."""
    qt = classify_query_type(query, from_golden=None)
    return QUERY_TYPE_RRF_WEIGHTS.get(qt, (0.30, 0.70))


def _dedup_per_cert(candidates: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """자격증(qual_id)당 최고점 청크 1개만 유지 후 점수 기준 재정렬. 상위 목록 다양화."""
    by_qual: Dict[int, Tuple[str, float]] = {}
    for cid, score in candidates:
        if ":" in cid:
            try:
                qid = int(cid.split(":")[0])
                if qid not in by_qual or score > by_qual[qid][1]:
                    by_qual[qid] = (cid, score)
            except ValueError:
                continue
    out = list(by_qual.values())
    out.sort(key=lambda x: -x[1])
    return out


def _rrf_merge(
    bm25_list: List[Tuple[str, float]],
    vector_list: List[Tuple[str, float]],
    w_bm25: float = 0.5,
    w_vector: float = 0.5,
    rrf_k: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """RRF: score(d) = w_b * 1/(k + rank_bm25) + w_v * 1/(k + rank_vector). k는 RAG_RRF_K 또는 28."""
    k = rrf_k if rrf_k is not None else _rrf_k()
    rank_bm25 = {cid: i + 1 for i, (cid, _) in enumerate(bm25_list)}
    rank_vec = {cid: i + 1 for i, (cid, _) in enumerate(vector_list)}
    all_ids = set(rank_bm25) | set(rank_vec)
    scores = [
        (cid, w_bm25 / (k + rank_bm25.get(cid, 9999)) + w_vector / (k + rank_vec.get(cid, 9999)))
        for cid in all_ids
    ]
    scores.sort(key=lambda x: -x[1])
    return scores


def _rrf_merge_n(
    lists: List[List[Tuple[str, float]]],
    weights: Optional[List[float]] = None,
    rrf_k: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """N-way RRF: 여러 순위 리스트를 가중 RRF로 병합. weights 미지정 시 동일 가중치."""
    if not lists:
        return []
    k = rrf_k if rrf_k is not None else _rrf_k()
    n = len(lists)
    w = weights if weights is not None else [1.0 / n] * n
    if len(w) != n:
        w = [1.0 / n] * n
    rank_maps = [{cid: i + 1 for i, (cid, _) in enumerate(lst)} for lst in lists]
    all_ids = set()
    for rm in rank_maps:
        all_ids |= set(rm.keys())
    scores = [
        (cid, sum(wi / (k + rm.get(cid, 9999)) for wi, rm in zip(w, rank_maps)))
        for cid in all_ids
    ]
    scores.sort(key=lambda x: -x[1])
    return scores


def _rrf_merge_3(
    list_a: List[Tuple[str, float]],
    list_b: List[Tuple[str, float]],
    list_c: List[Tuple[str, float]],
    w_a: float = 1.0 / 3,
    w_b: float = 1.0 / 3,
    w_c: float = 1.0 / 3,
    rrf_k: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """3-way RRF: score(d) = w_a/(k+rank_a) + w_b/(k+rank_b) + w_c/(k+rank_c)."""
    k = rrf_k if rrf_k is not None else _rrf_k()
    rank_a = {cid: i + 1 for i, (cid, _) in enumerate(list_a)}
    rank_b = {cid: i + 1 for i, (cid, _) in enumerate(list_b)}
    rank_c = {cid: i + 1 for i, (cid, _) in enumerate(list_c)}
    all_ids = set(rank_a) | set(rank_b) | set(rank_c)
    scores = [
        (
            cid,
            w_a / (k + rank_a.get(cid, 9999))
            + w_b / (k + rank_b.get(cid, 9999))
            + w_c / (k + rank_c.get(cid, 9999)),
        )
        for cid in all_ids
    ]
    scores.sort(key=lambda x: -x[1])
    return scores


def _linear_merge_3(
    list_a: List[Tuple[str, float]],
    list_b: List[Tuple[str, float]],
    list_c: List[Tuple[str, float]],
    w_a: float = 1.0 / 3,
    w_b: float = 1.0 / 3,
    w_c: float = 1.0 / 3,
) -> List[Tuple[str, float]]:
    """3-way Convex Combination: min-max 정규화 후 S = w_a*norm_a + w_b*norm_b + w_c*norm_c."""
    sa = {cid: s for cid, s in list_a}
    sb = {cid: s for cid, s in list_b}
    sc = {cid: s for cid, s in list_c}
    all_ids = set(sa) | set(sb) | set(sc)
    if not all_ids:
        return []
    vals_a = [sa[c] for c in all_ids if c in sa]
    vals_b = [sb[c] for c in all_ids if c in sb]
    vals_c = [sc[c] for c in all_ids if c in sc]
    def _norm(vals, d):
        if not vals:
            return 0.0, 1.0
        mn, mx = min(vals), max(vals)
        r = mx - mn if mx > mn else 1.0
        return mn, r
    min_a, r_a = _norm(vals_a, sa)
    min_b, r_b = _norm(vals_b, sb)
    min_c, r_c = _norm(vals_c, sc)
    scores = []
    for cid in all_ids:
        na = (sa.get(cid, 0) - min_a) / r_a if cid in sa else 0.0
        nb = (sb.get(cid, 0) - min_b) / r_b if cid in sb else 0.0
        nc = (sc.get(cid, 0) - min_c) / r_c if cid in sc else 0.0
        scores.append((cid, w_a * na + w_b * nb + w_c * nc))
    scores.sort(key=lambda x: -x[1])
    return scores


def _linear_merge(
    bm25_list: List[Tuple[str, float]],
    vector_list: List[Tuple[str, float]],
    w_bm25: float = 0.5,
    w_vector: float = 0.5,
) -> List[Tuple[str, float]]:
    """
    Convex Combination: min-max 정규화 후 S = w_bm25 * norm_bm25(d) + w_vector * norm_vector(d).
    한쪽 리스트에만 있는 문서는 해당 채널만 반영(반대쪽 0). 스케일 불일치 완화.
    """
    bm25_scores = {cid: s for cid, s in bm25_list}
    vec_scores = {cid: s for cid, s in vector_list}
    all_ids = set(bm25_scores) | set(vec_scores)
    if not all_ids:
        return []

    bm25_vals = [bm25_scores[cid] for cid in all_ids if cid in bm25_scores]
    vec_vals = [vec_scores[cid] for cid in all_ids if cid in vec_scores]
    min_b = min(bm25_vals) if bm25_vals else 0.0
    max_b = max(bm25_vals) if bm25_vals else 1.0
    min_v = min(vec_vals) if vec_vals else 0.0
    max_v = max(vec_vals) if vec_vals else 1.0
    range_b = max_b - min_b if max_b > min_b else 1.0
    range_v = max_v - min_v if max_v > min_v else 1.0

    scores = []
    for cid in all_ids:
        s_b = bm25_scores.get(cid)
        s_v = vec_scores.get(cid)
        norm_b = (s_b - min_b) / range_b if s_b is not None else 0.0
        norm_v = (s_v - min_v) / range_v if s_v is not None else 0.0
        combined = w_bm25 * norm_b + w_vector * norm_v
        scores.append((cid, combined))
    scores.sort(key=lambda x: -x[1])
    return scores


def hybrid_retrieve(
    db: Session,
    query: str,
    top_k: int = 5,
    alpha: Optional[float] = None,
    filters: Optional[Dict[str, Any]] = None,
    bm25_index_path: Optional[Path] = None,
    use_query_weights: bool = False,
    use_reranker: Optional[bool] = None,
    user_profile: Optional[UserProfile] = None,
    rrf_w_bm25: Optional[float] = None,
    rrf_w_dense1536: Optional[float] = None,
    rrf_w_contrastive768: Optional[float] = None,
    rrf_k_override: Optional[int] = None,
    top_n_candidates_override: Optional[int] = None,
    dedup_per_cert_override: Optional[bool] = None,
    bm25_top_n_override: Optional[int] = None,
    vector_top_n_override: Optional[int] = None,
    contrastive_top_n_override: Optional[int] = None,
    vector_threshold_override: Optional[float] = None,
    channels_override: Optional[List[str]] = None,
    force_reranker: bool = False,
) -> List[Tuple[str, float]]:
    """
    BM25 + Vector를 RRF로 병합.
    - use_query_weights=True: 쿼리 타입별 가중치(w_bm25/w_vector).
    - alpha 지정 시: BM25=alpha, Vector=1-alpha.
    - use_reranker: None이면 RAG_USE_CROSS_ENCODER_RERANKER 설정 따름, True/False면 강제.
    - force_reranker: True면 질의 타입·점수 게이팅 무시하고 항상 리랭커 API 호출 (평가/디버깅용).
    - user_profile: 있으면 RAG_PERSONALIZED_* 설정 시 개인화 rewrite/soft score 적용. 없으면 기존 경로.
    - rrf_w_bm25 / rrf_w_dense1536 / rrf_w_contrastive768: 3-way RRF 시 가중치 오버라이드(None이면 설정값 사용).
    - channels_override: 채널 제한. ["bm25"], ["vector"], ["contrastive"] 또는 조합. None이면 3채널 모두 사용.
    filters 있으면 메타데이터 필터. 반환: [(chunk_id, score), ...]
    """
    settings = get_rag_settings()
    channels_set = (channels_override or [])
    use_bm25 = len(channels_set) == 0 or "bm25" in channels_set
    use_vector = len(channels_set) == 0 or "vector" in channels_set
    use_contrastive_ch = len(channels_set) == 0 or "contrastive" in channels_set

    top_n = (top_n_candidates_override if top_n_candidates_override is not None else settings.RAG_TOP_N_CANDIDATES)
    # 채널별 후보 수(N): 오버라이드 있으면 우선, 없으면 설정값 또는 top_n
    bm25_top_n = bm25_top_n_override if bm25_top_n_override is not None else (getattr(settings, "RAG_BM25_TOP_N", None) or top_n)
    if isinstance(bm25_top_n, float):
        bm25_top_n = int(bm25_top_n)
    vec_top_k = vector_top_n_override if vector_top_n_override is not None else (getattr(settings, "RAG_VECTOR_TOP_N_OVERRIDE", None) or top_n)
    if isinstance(vec_top_k, float):
        vec_top_k = int(vec_top_k)
    contrastive_top_n = contrastive_top_n_override if contrastive_top_n_override is not None else (getattr(settings, "RAG_CONTRASTIVE_TOP_N", None) or top_n)
    if isinstance(contrastive_top_n, float):
        contrastive_top_n = int(contrastive_top_n)
    vec_threshold = vector_threshold_override if vector_threshold_override is not None else settings.RAG_VECTOR_THRESHOLD
    index_dir = bm25_index_path or (get_rag_index_dir() / "bm25.pkl")
    short_keyword = _is_short_query((query or "").strip())
    # 질의 타입은 BM25 쿼리 확장·가중치·게이팅·contrastive/reranker 사용 여부에 공통으로 활용
    query_type = classify_query_type(query, from_golden=None)

    # Vector (OpenAI embedding + pgvector). Dense query rewrite (개인화 시 profile 반영)
    vector_results: List[Tuple[str, float]] = []
    if use_vector:
        vector_query = query
        use_personalized_rewrite = (
            getattr(settings, "RAG_PERSONALIZED_DENSE_REWRITE_ENABLE", False)
            and user_profile is not None
        )
        if getattr(settings, "RAG_DENSE_USE_QUERY_REWRITE", True):
            try:
                rewritten = rewrite_for_dense(query, profile=user_profile if use_personalized_rewrite else None)
                if rewritten and rewritten.strip():
                    vector_query = rewritten
            except Exception as e:
                logger.debug("dense query rewrite failed, using original query: %s", e)
                if getattr(settings, "RAG_DENSE_QUERY_REWRITE_FALLBACK", True):
                    vector_query = query
        if getattr(settings, "RAG_DENSE_MULTI_QUERY_ENABLE", False):
            # Multi-query: 원본 + rewrite 각각 검색 후 RRF 병합 (diversity·recall 향상, Query expansion + multi-query 논문)
            vec_orig = get_vector_search(
                db, query, top_k=vec_top_k, threshold=vec_threshold, use_rewrite=False
            )
            vec_rewrite = get_vector_search(
                db, vector_query, top_k=vec_top_k, threshold=vec_threshold, use_rewrite=False
            )
            rrf_k_mq = _rrf_k()
            vector_results = _rrf_merge(vec_orig, vec_rewrite, w_bm25=0.5, w_vector=0.5, rrf_k=rrf_k_mq)
        else:
            vector_results = get_vector_search(
                db, vector_query, top_k=vec_top_k, threshold=vec_threshold, use_rewrite=False
            )

        # COT 쿼리 확장: 대안 검색 문구 생성 후 다중 벡터 검색 RRF (창의적 방법론)
        if getattr(settings, "RAG_COT_QUERY_EXPANSION_ENABLE", False):
            cot_alts = expand_query_cot(query, max_alternatives=getattr(settings, "RAG_COT_EXPANSION_MAX", 2))
            if cot_alts:
                cot_lists: List[List[Tuple[str, float]]] = []
                for alt in cot_alts:
                    try:
                        lst = get_vector_search(
                            db, alt, top_k=vec_top_k, threshold=vec_threshold, use_rewrite=False
                        )
                        if lst:
                            cot_lists.append(lst)
                    except Exception as e:
                        logger.debug("COT vector search for alt failed: %s", e)
                        continue
                if cot_lists:
                    vector_results = _rrf_merge_n([vector_results] + cot_lists, rrf_k=_rrf_k())

        # Step-back 메타 쿼리: 상위 목표 한 문장 추출 후 추가 벡터 검색, RRF 병합
        if getattr(settings, "RAG_STEPBACK_QUERY_ENABLE", False):
            stepback_q = stepback_query(query)
            if stepback_q:
                try:
                    vec_sb = get_vector_search(
                        db, stepback_q, top_k=vec_top_k, threshold=settings.RAG_VECTOR_THRESHOLD, use_rewrite=False
                    )
                    if vec_sb:
                        vector_results = _rrf_merge(
                            vector_results, vec_sb, w_bm25=0.5, w_vector=0.5, rrf_k=_rrf_k()
                        )
                except Exception as e:
                    logger.debug("stepback vector search failed: %s", e)

    # HyDE: 가상 문서 생성 후 벡터 검색, 3-way 병합 (방법론 확장). LONG_QUERY_ONLY면 짧은 쿼리(≤3단어)에서는 생략.
    hyde_results: List[Tuple[str, float]] = []
    if use_vector and getattr(settings, "RAG_HYDE_ENABLE", False):
        use_hyde = not (getattr(settings, "RAG_HYDE_LONG_QUERY_ONLY", True) and short_keyword)
        if use_hyde:
            hyde_doc = generate_hyde_document(query)
        else:
            hyde_doc = None
        if hyde_doc:
            try:
                hyde_results = get_vector_search(
                    db, hyde_doc, top_k=vec_top_k, threshold=vec_threshold, use_rewrite=False
                )
            except Exception as e:
                logger.debug("HyDE vector search failed: %s", e)
                hyde_results = []

    # BM25: single expansion 또는 multi-expansion(여러 확장 쿼리 검색 후 RRF). 선택 시 PRF.
    bm25_scores: List[Tuple[str, float]] = []
    if use_bm25 and Path(index_dir).exists():
        try:
            bm25 = BM25Index(index_path=Path(index_dir))
            bm25.load()
            if getattr(settings, "RAG_BM25_MULTI_EXPANSION_ENABLE", False):
                expansions = expand_query(query, max_expansions=4)
                if len(expansions) <= 1:
                    bm25_query = expand_query_single_string(query, for_recommendation=True, query_type=query_type)
                    bm25_scores = bm25.search(bm25_query, k=bm25_top_n)
                else:
                    bm25_lists = [bm25.search(q, k=bm25_top_n) for q in expansions[:4]]
                    bm25_scores = _rrf_merge_n(bm25_lists, rrf_k=_rrf_k())
            else:
                bm25_query = expand_query_single_string(query, for_recommendation=True, query_type=query_type)
                bm25_scores = bm25.search(bm25_query, k=bm25_top_n)

            if getattr(settings, "RAG_BM25_PRF_ENABLE", False) and bm25_scores:
                prf_top_k = getattr(settings, "RAG_BM25_PRF_TOP_K", 5)
                prf_n_terms = getattr(settings, "RAG_BM25_PRF_N_TERMS", 10)
                top_ids = [c[0] for c in bm25_scores[:prf_top_k]]
                contents = _fetch_contents_by_chunk_ids(db, top_ids)
                if contents:
                    terms = _extract_terms_for_prf(contents, query, n_terms=prf_n_terms)
                    if terms:
                        expanded_q = f"{bm25_query} {' '.join(terms)}"
                        bm25_second = bm25.search(expanded_q.strip(), k=top_n)
                        bm25_scores = _rrf_merge(
                            bm25_scores, bm25_second, w_bm25=0.5, w_vector=0.5, rrf_k=_rrf_k()
                        )
        except Exception as e:
            logger.debug("BM25 search failed (index or expansion): %s", e)

    # Contrastive 768 FAISS arm (별도 retriever, RRF로만 결합)
    contrastive_results: List[Tuple[str, float]] = []
    contrastive_enabled = getattr(settings, "RAG_CONTRASTIVE_ENABLE", False) and use_contrastive_ch
    if contrastive_enabled:
        # 단일 Contrastive 채널 평가(bm25_only/vector_only/contrastive_only 중 contrastive_only)는 게이팅을 끄고 항상 사용.
        single_contrastive_only = use_contrastive_ch and not use_bm25 and not use_vector and bool(channels_set)
        # 질의 타입 기반 Contrastive 게이팅: 자연어·복합 목적 질의 위주로만 사용해 비용·지연 절감
        allowed_types_raw = getattr(settings, "RAG_CONTRASTIVE_ALLOWED_QUERY_TYPES", "") or ""
        allowed_types = {t.strip() for t in allowed_types_raw.split(",") if t.strip()}
        if single_contrastive_only:
            use_contrastive_for_query = True
        else:
            use_contrastive_for_query = True
            if allowed_types:
                use_contrastive_for_query = query_type in allowed_types
            # 짧은 키워드·자격증명 위주 쿼리는 BM25+Vector로 충분한 경우가 많으므로 기본적으로 contrastive 비활성
            if short_keyword and query_type in ("cert_name_included", "keyword"):
                use_contrastive_for_query = False
        if use_contrastive_for_query:
            try:
                logger.debug(
                    "contrastive arm enabled (query_type=%s short=%s top_n=%d contrastive_top_n=%d single_only=%s)",
                    query_type,
                    short_keyword,
                    top_n,
                    contrastive_top_n,
                    single_contrastive_only,
                )
                contrastive_results = contrastive_search((query or "").strip(), top_k=contrastive_top_n)
            except Exception:
                logger.debug("contrastive_search failed (disabled or deps missing)", exc_info=True)
        else:
            logger.debug(
                "contrastive arm skipped by gating (query_type=%s short=%s enabled=%s single_only=%s)",
                query_type,
                short_keyword,
                contrastive_enabled,
                single_contrastive_only,
            )

    # Query Routing + Weighted RRF (쿼리 타입별·도메인별 가중치, 짧은 쿼리 시 Vector 게이팅)
    if use_query_weights or (alpha is None and getattr(settings, "RAG_ENHANCED_ALPHA", None) is None):
        if getattr(settings, "RAG_QUERY_TYPE_WEIGHTS_ENABLE", False):
            if getattr(settings, "RAG_DOMAIN_AWARE_WEIGHTS_ENABLE", False) and not _query_suggests_it(query):
                w_bm25, w_vector = NON_IT_RRF_WEIGHTS  # 비IT: BM25 강화
            else:
                w_bm25, w_vector = _query_weights_by_type(query)
        else:
            w_bm25, w_vector = _query_weights_for_rrf(query)
        if short_keyword and bm25_scores and vector_results:
            bm25_top_n = getattr(settings, "RAG_HYBRID_BM25_TOP_FOR_GATING", 20)
            vec_min = getattr(settings, "RAG_HYBRID_VEC_MIN_SCORE", 0.55)
            vec_gap = getattr(settings, "RAG_HYBRID_VEC_GAP_MIN", 0.02)
            suspicious = _vector_gating_suspicious(
                bm25_scores, vector_results,
                bm25_top_n=bm25_top_n, vec_min_score=vec_min, vec_gap_min=vec_gap,
            )
            if suspicious:
                w_vector = 0.0  # 게이팅 실패 시 Vector 반영 제외
            if getattr(settings, "RAG_HYBRID_DEBUG_LOG", False):
                logger.info(
                    "hybrid query=%r short_keyword=%s bm25_top10=%s vec_top10=%s w_bm25=%.2f w_vec=%.2f gating_suspicious=%s",
                    query, short_keyword,
                    [c[0] for c in bm25_scores[:10]],
                    [c[0] for c in vector_results[:10]],
                    w_bm25, w_vector, suspicious,
                )
    elif alpha is not None:
        a = alpha if 0 <= alpha <= 1 else getattr(settings, "RAG_ALPHA", 0.5)
        w_bm25, w_vector = a, 1.0 - a
    else:
        a = getattr(settings, "RAG_ENHANCED_ALPHA", None)
        if a is not None and 0 <= a <= 1:
            w_bm25, w_vector = a, 1.0 - a
        else:
            w_bm25, w_vector = 0.5, 0.5

    # Fusion: 2-way / 3-way(HyDE) / 3-way(BM25+dense1536+contrastive768). channels_override 시 요청 채널만 RRF.
    fusion_method = (getattr(settings, "RAG_FUSION_METHOD", None) or "rrf").strip().lower()
    rrf_k = rrf_k_override if rrf_k_override is not None else _rrf_k()
    if channels_set:
        lists_to_merge: List[List[Tuple[str, float]]] = []
        weights_to_merge: List[float] = []
        if use_bm25 and bm25_scores:
            lists_to_merge.append(bm25_scores)
            weights_to_merge.append(rrf_w_bm25 if rrf_w_bm25 is not None else getattr(settings, "RAG_RRF_W_BM25", 1.0))
        if use_vector and vector_results:
            lists_to_merge.append(vector_results)
            weights_to_merge.append(rrf_w_dense1536 if rrf_w_dense1536 is not None else getattr(settings, "RAG_RRF_W_DENSE1536", 1.0))
        if use_contrastive_ch and contrastive_results:
            base_wc = rrf_w_contrastive768 if rrf_w_contrastive768 is not None else getattr(
                settings, "RAG_RRF_W_CONTRASTIVE768", 1.2
            )
            if getattr(settings, "RAG_QUERY_TYPE_CONTRASTIVE_WEIGHTS_ENABLE", False):
                mul = CONTRASTIVE_QUERY_TYPE_WEIGHTS.get(query_type)
                if mul is not None:
                    base_wc *= mul
            lists_to_merge.append(contrastive_results)
            weights_to_merge.append(base_wc)
        if len(lists_to_merge) == 0:
            combined: List[Tuple[str, float]] = []
        elif len(lists_to_merge) == 1:
            combined = lists_to_merge[0][: top_n * 2]
        else:
            combined = _rrf_merge_n(lists_to_merge, weights=weights_to_merge, rrf_k=rrf_k)
    elif getattr(settings, "RAG_CONTRASTIVE_ENABLE", False) and contrastive_results:
        w_b = rrf_w_bm25 if rrf_w_bm25 is not None else getattr(settings, "RAG_RRF_W_BM25", 1.0)
        w_v = rrf_w_dense1536 if rrf_w_dense1536 is not None else getattr(settings, "RAG_RRF_W_DENSE1536", 1.0)
        w_c = rrf_w_contrastive768 if rrf_w_contrastive768 is not None else getattr(settings, "RAG_RRF_W_CONTRASTIVE768", 1.2)
        if getattr(settings, "RAG_QUERY_TYPE_CONTRASTIVE_WEIGHTS_ENABLE", False):
            mul = CONTRASTIVE_QUERY_TYPE_WEIGHTS.get(query_type)
            if mul is not None:
                w_c *= mul
        combined = _rrf_merge_n(
            [bm25_scores, vector_results, contrastive_results],
            weights=[w_b, w_v, w_c],
            rrf_k=rrf_k,
        )
    elif hyde_results and getattr(settings, "RAG_HYDE_ENABLE", False):
        w_hyde = getattr(settings, "RAG_HYDE_WEIGHT", 0.2)
        total_bv = w_bm25 + w_vector
        if total_bv <= 0:
            total_bv = 1.0
        w_b = (1.0 - w_hyde) * (w_bm25 / total_bv)
        w_v = (1.0 - w_hyde) * (w_vector / total_bv)
        if fusion_method == "linear":
            combined = _linear_merge_3(
                bm25_scores, vector_results, hyde_results,
                w_a=w_b, w_b=w_v, w_c=w_hyde,
            )
        else:
            combined = _rrf_merge_3(
                bm25_scores, vector_results, hyde_results,
                w_a=w_b, w_b=w_v, w_c=w_hyde, rrf_k=rrf_k,
            )
    else:
        if fusion_method == "linear":
            combined = _linear_merge(bm25_scores, vector_results, w_bm25=w_bm25, w_vector=w_vector)
        else:
            combined = _rrf_merge(bm25_scores, vector_results, w_bm25=w_bm25, w_vector=w_vector, rrf_k=rrf_k)
    candidates = combined[: top_n * 2]

    # Metadata soft scoring (직무/전공 일치 가산, 분야 이탈 감점)
    if getattr(settings, "RAG_METADATA_SOFT_SCORE_ENABLE", False) and candidates:
        try:
            qual_ids_soft = []
            for cid, _ in candidates:
                if ":" in cid:
                    try:
                        qual_ids_soft.append(int(cid.split(":")[0]))
                    except ValueError:
                        pass
            if qual_ids_soft:
                query_slots = extract_slots_for_dense(query)
                meta = fetch_qual_metadata_bulk(db, qual_ids_soft)
                soft_config = {
                    "job_bonus": getattr(settings, "RAG_METADATA_SOFT_JOB_BONUS", 0.15),
                    "major_bonus": getattr(settings, "RAG_METADATA_SOFT_MAJOR_BONUS", 0.10),
                    "target_bonus": getattr(settings, "RAG_METADATA_SOFT_TARGET_BONUS", 0.10),
                    "field_penalty": getattr(settings, "RAG_METADATA_SOFT_FIELD_PENALTY", -0.20),
                }
                if getattr(settings, "RAG_METADATA_DOMAIN_MISMATCH_ENABLE", False):
                    soft_config["domain_mismatch_penalty"] = getattr(
                        settings, "RAG_METADATA_DOMAIN_MISMATCH_PENALTY", -0.35
                    )
                query_is_it = _query_suggests_it(query) if getattr(settings, "RAG_METADATA_DOMAIN_MISMATCH_ENABLE", False) else None
                scored = []
                for cid, base_score in candidates:
                    qid = int(cid.split(":")[0]) if ":" in cid else None
                    qual_meta = meta.get(qid, {}) if qid is not None else {}
                    soft = compute_metadata_soft_score(query_slots, qual_meta, soft_config, query_is_it=query_is_it)
                    scored.append((cid, base_score + soft))
                scored.sort(key=lambda x: -x[1])
                candidates = scored
        except Exception:
            pass

    # 자격증 단위 다양화: qual_id당 최고점 청크 1개만 유지 후 재정렬 (상위 목록이 서로 다른 자격증으로)
    dedup_per_cert = dedup_per_cert_override if dedup_per_cert_override is not None else getattr(settings, "RAG_DEDUP_PER_CERT", False)
    if dedup_per_cert and candidates:
        candidates = _dedup_per_cert(candidates)

    # 개인화 soft scoring (profile 있을 때만, 전공/즐겨찾기/취득/난이도 적합도)
    if (
        getattr(settings, "RAG_PERSONALIZED_SOFT_SCORE_ENABLE", False)
        and user_profile is not None
        and candidates
    ):
        try:
            qual_ids_pers = []
            for cid, _ in candidates:
                if ":" in cid:
                    try:
                        qual_ids_pers.append(int(cid.split(":")[0]))
                    except ValueError:
                        pass
            if qual_ids_pers:
                from app.crud import get_qualification_aggregated_stats_bulk
                meta_pers = fetch_qual_metadata_bulk(db, qual_ids_pers)
                stats_bulk = get_qualification_aggregated_stats_bulk(db, qual_ids_pers)
                diff_by_qual = {
                    qid: s["avg_difficulty"]
                    for qid, s in (stats_bulk or {}).items()
                    if s.get("avg_difficulty") is not None
                }
                merge_difficulty_into_metadata(meta_pers, diff_by_qual)
                query_slots = extract_slots_for_dense(query)
                personal_config = {
                    "major_bonus": getattr(settings, "RAG_PERSONALIZED_MAJOR_BONUS", 0.15),
                    "favorite_field_bonus": getattr(settings, "RAG_PERSONALIZED_FAVORITE_FIELD_BONUS", 0.10),
                    "acquired_penalty": getattr(settings, "RAG_PERSONALIZED_ACQUIRED_PENALTY", -1.0),
                    "grade_difficulty_bonus": getattr(settings, "RAG_PERSONALIZED_GRADE_DIFFICULTY_BONUS", 0.10),
                    "far_too_difficult_penalty": getattr(settings, "RAG_PERSONALIZED_FAR_TOO_DIFFICULT_PENALTY", -0.15),
                }
                scored_pers = []
                for cid, base_score in candidates:
                    qid = int(cid.split(":")[0]) if ":" in cid else None
                    qual_meta = meta_pers.get(qid, {}) if qid is not None else {}
                    personal = compute_personalized_soft_score(
                        query_slots, qual_meta, user_profile, personal_config
                    )
                    scored_pers.append((cid, base_score + personal))
                scored_pers.sort(key=lambda x: -x[1])
                candidates = scored_pers
        except Exception as e:
            logger.debug("personalized soft score failed: %s", e)

    # 메타데이터 필터
    if filters and candidates:
        candidates = _apply_metadata_filter(db, candidates, filters)

    # (선택) 경량 Cross-Encoder Reranker
    do_rerank = use_reranker if use_reranker is not None else getattr(settings, "RAG_USE_CROSS_ENCODER_RERANKER", False)
    if do_rerank:
        # 1) 질의 타입·길이 기반 리랭커 게이팅 (force_reranker=True면 스킵)
        if not force_reranker:
            allowed_types_raw = getattr(settings, "RAG_RERANK_ALLOWED_QUERY_TYPES", "") or ""
            allowed_types = {t.strip() for t in allowed_types_raw.split(",") if t.strip()}
            if allowed_types and query_type not in allowed_types:
                logger.info(
                    "reranker skipped by query_type gating (query_type=%s allowed=%s)",
                    query_type,
                    sorted(allowed_types),
                )
                return candidates[:top_k]
            if (
                short_keyword
                and query_type in ("cert_name_included", "keyword")
                and not getattr(settings, "RAG_RERANK_ALLOW_SHORT_KEYWORD", False)
            ):
                logger.info(
                    "reranker skipped for short keyword query (query_type=%s short=%s)",
                    query_type,
                    short_keyword,
                )
                return candidates[:top_k]

        # 2) 조건부 rerank: top1/격차 게이팅 (force_reranker=True면 스킵 → 항상 API 호출)
        if not force_reranker and settings.RAG_RERANK_GATING_ENABLE and len(candidates) >= 2:
            top1 = float(candidates[0][1])
            top2 = float(candidates[1][1])
            need_rerank = (
                top1 < settings.RAG_RERANK_GATING_TOP1_MIN_SCORE
                or (top1 - top2) < settings.RAG_RERANK_GATING_MIN_GAP
            )
            logger.debug(
                "rerank gating check (top1=%.6f top2=%.6f min_score=%.6f min_gap=%.6f need_rerank=%s)",
                top1,
                top2,
                settings.RAG_RERANK_GATING_TOP1_MIN_SCORE,
                settings.RAG_RERANK_GATING_MIN_GAP,
                need_rerank,
            )
            if not need_rerank:
                return candidates[:top_k]

        pool_size = getattr(settings, "RAG_RERANK_POOL_SIZE", 20)
        to_rerank = candidates[:pool_size]
        if to_rerank:
            from app.rag.rerank.cross_encoder import rerank_with_cross_encoder
            chunk_ids = [c[0] for c in to_rerank]
            add_qual_name = getattr(settings, "RAG_RERANK_INPUT_ADD_QUAL_NAME", True)
            if add_qual_name:
                contents, qual_names = _fetch_contents_and_qual_names_by_chunk_ids(db, chunk_ids)
            else:
                contents = _fetch_contents_by_chunk_ids(db, chunk_ids)
                qual_names = {}
            reranker_query = _build_reranker_query(query, user_profile) if getattr(settings, "RAG_RERANK_INPUT_ADD_CONTEXT", True) else query
            pairs = []
            for cid in chunk_ids:
                content = contents.get(cid, "") or ""
                if qual_names.get(cid):
                    content = f"자격증: {qual_names[cid]}. {content}"
                pairs.append((cid, content))
            reranked = rerank_with_cross_encoder(
                reranker_query, pairs, top_k=top_k
            )
            if reranked:
                return reranked

    return candidates[:top_k]


def _fetch_contents_and_qual_names_by_chunk_ids(
    db: Session, chunk_ids: List[str]
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """리랭커용: content와 qual_name을 한 번의 DB 조회로 가져와 지연·라운드트립 절감."""
    if not chunk_ids:
        return {}, {}
    qual_to_chunks: Dict[int, set] = {}
    for cid in chunk_ids:
        if ":" in cid:
            try:
                a, b = cid.split(":", 1)
                qid, cidx = int(a), int(b)
                qual_to_chunks.setdefault(qid, set()).add(cidx)
            except ValueError:
                continue
    if not qual_to_chunks:
        return {}, {}
    qual_ids = list(qual_to_chunks.keys())
    try:
        sql = text("""
            SELECT v.qual_id, COALESCE(v.chunk_index, 0) AS chunk_index, v.content, q.qual_name
            FROM certificates_vectors v
            LEFT JOIN qualification q ON q.qual_id = v.qual_id
            WHERE v.qual_id = ANY(:ids)
        """)
        rows = db.execute(sql, {"ids": qual_ids}).fetchall()
    except Exception:
        return _fetch_contents_by_chunk_ids(db, chunk_ids), _fetch_qual_names_for_chunk_ids(db, chunk_ids)
    contents: Dict[str, str] = {}
    qual_names: Dict[str, str] = {}
    for r in rows:
        qid = int(getattr(r, "qual_id"))
        cidx = int(getattr(r, "chunk_index"))
        if cidx not in qual_to_chunks.get(qid, ()):
            continue
        cid = f"{qid}:{cidx}"
        content = getattr(r, "content", None)
        if content:
            contents[cid] = content
        qual_names[cid] = (getattr(r, "qual_name", None) or "").strip()
    return contents, qual_names


def _fetch_qual_names_for_chunk_ids(db: Session, chunk_ids: List[str]) -> Dict[str, str]:
    """chunk_id(qual_id:chunk_index) 목록에 대해 qual_id → qual_name 맵을 구한 뒤 chunk_id → qual_name 반환."""
    if not chunk_ids:
        return {}
    qual_ids: List[int] = []
    for cid in chunk_ids:
        if ":" in cid:
            try:
                qual_ids.append(int(cid.split(":", 1)[0]))
            except ValueError:
                pass
    if not qual_ids:
        return {}
    qual_ids = list(dict.fromkeys(qual_ids))
    try:
        rows = db.execute(
            text("SELECT qual_id, qual_name FROM qualification WHERE qual_id = ANY(:ids)"),
            {"ids": qual_ids},
        ).fetchall()
        qid_to_name = {r.qual_id: (r.qual_name or "").strip() for r in rows}
    except Exception:
        return {}
    out: Dict[str, str] = {}
    for cid in chunk_ids:
        if ":" in cid:
            try:
                qid = int(cid.split(":", 1)[0])
                out[cid] = qid_to_name.get(qid, "")
            except ValueError:
                pass
    return out


def _build_reranker_query(query: str, user_profile: Optional[UserProfile] = None) -> str:
    """§2-9 추천 적합도: 전공·목적·직무 문맥을 붙인 리랭커용 쿼리 문자열."""
    parts: List[str] = []
    if user_profile and user_profile.get("major"):
        parts.append(f"전공: {user_profile['major']}")
    slots = extract_slots_for_dense(query or "")
    if slots.get("목적"):
        parts.append(f"목적: {slots['목적']}")
    if slots.get("희망직무"):
        parts.append(f"직무: {slots['희망직무']}")
    if parts:
        return " ".join(parts) + " 질의: " + (query or "").strip()
    return (query or "").strip()


def _extract_terms_for_prf(
    contents: Dict[str, str],
    query: str,
    n_terms: int = 10,
) -> List[str]:
    """상위 문서 content에서 빈도 기반 확장어 추출. 쿼리 토큰·1글자 제외."""
    import re
    query_tokens = set(re.findall(r"[가-힣a-zA-Z0-9]+", (query or "").lower()))
    counter: Dict[str, int] = {}
    for cid, text in contents.items():
        for t in re.findall(r"[가-힣a-zA-Z0-9]+", (text or "").lower()):
            if len(t) >= 2 and t not in query_tokens:
                counter[t] = counter.get(t, 0) + 1
    sorted_terms = sorted(counter.items(), key=lambda x: -x[1])
    return [t for t, _ in sorted_terms[:n_terms]]


def _fetch_contents_by_chunk_ids(db: Session, chunk_ids: List[str]) -> Dict[str, str]:
    """chunk_id(qual_id:chunk_index) 목록에 대해 content 맵 반환.

    N+1 쿼리 대신, qual_id 단위로 한 번에 가져온 뒤 파이썬에서 chunk_index로 필터링한다.
    """
    if not chunk_ids:
        return {}

    qual_to_chunks: Dict[int, set[int]] = {}
    for cid in chunk_ids:
        if ":" in cid:
            try:
                a, b = cid.split(":", 1)
                qid = int(a)
                cidx = int(b)
                qual_to_chunks.setdefault(qid, set()).add(cidx)
            except ValueError:
                continue
    if not qual_to_chunks:
        return {}

    qual_ids = list(qual_to_chunks.keys())
    try:
        sql = text(
            """
            SELECT qual_id, COALESCE(chunk_index, 0) AS chunk_index, content
            FROM certificates_vectors
            WHERE qual_id = ANY(:ids)
            """
        )
        rows = db.execute(sql, {"ids": qual_ids}).fetchall()
    except Exception:
        return {}

    out: Dict[str, str] = {}
    for r in rows:
        qid = int(getattr(r, "qual_id"))
        cidx = int(getattr(r, "chunk_index"))
        if cidx in qual_to_chunks.get(qid, ()):
            content = getattr(r, "content", None)
            if content:
                out[f"{qid}:{cidx}"] = content
    return out


def _apply_metadata_filter(
    db: Session,
    candidates: List[Tuple[str, float]],
    filters: Dict[str, Any],
) -> List[Tuple[str, float]]:
    """qual_id가 filters(cert_name, category)에 맞는 qualification에 속하면 유지."""
    from sqlalchemy import text
    if not candidates:
        return []
    cert_name = filters.get("cert_name")
    category = filters.get("category") or filters.get("main_field")
    if not cert_name and not category:
        return candidates
    try:
        conditions = []
        params = {}
        if cert_name:
            conditions.append("qual_name ILIKE :name")
            params["name"] = f"%{cert_name}%"
        if category:
            conditions.append("(main_field = :cat OR ncs_large = :cat)")
            params["cat"] = category
        sql = text("SELECT qual_id FROM qualification WHERE " + " AND ".join(conditions))
        rows = db.execute(sql, params).fetchall()
        allowed_qual_ids = {r.qual_id for r in rows}
    except Exception:
        return candidates
    if not allowed_qual_ids:
        return []
    out = []
    for cid, score in candidates:
        if ":" in cid:
            try:
                qid = int(cid.split(":")[0])
                if qid in allowed_qual_ids:
                    out.append((cid, score))
            except ValueError:
                out.append((cid, score))
        else:
            out.append((cid, score))
    return out if out else candidates
