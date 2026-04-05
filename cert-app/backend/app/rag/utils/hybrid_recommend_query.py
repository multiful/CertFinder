"""AI 하이브리드 추천 API와 동일한 RAG 검색 질의 문자열 (운영·오프라인 평가 공통)."""
import re
from typing import Optional


def _compact_alnum(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").casefold())


def _interest_suggests_data_analysis(interest: str) -> bool:
    """짧은 직무 표현(예: '데이터 분석 직무')에서 BM25·밀집 검색용 앵커를 붙일지 판단."""
    c = _compact_alnum(interest)
    if not c:
        return False
    needles = (
        "데이터분석",
        "데이터사이언스",
        "빅데이터",
        "머신러닝",
        "기계학습",
        "sqld",
        "sqlp",
        "adsp",
        "adp",
        "분석가",
        "analyst",
    )
    return any(n in c for n in needles)


# 자격증명·도메인 토큰을 한 줄에 넣어 짧은 관심 문구의 RAG 매칭을 보강 (본문 의미는 사용자 입력이 우선)
_DATA_ANALYSIS_SEARCH_ANCHOR = (
    "빅데이터분석기사 데이터분석준전문가(ADsP) SQL개발자(SQLD) SQL전문가(SQLP) "
    "데이터분석전문가(ADP) AICE 통계 머신러닝 데이터사이언스"
)


def build_expanded_interest_for_hybrid(major: str, interest: Optional[str]) -> str:
    """
    hybrid_retrieve에 넣는 확장 질의.
    관심사(질의)를 앞에 두고 전공은 배경으로 둔다.
    """
    m = (major or "").strip()
    intr = (interest or "").strip()
    if intr:
        lines = [
            intr,
            f"(전공·배경: {m})",
            "위 관심·진로에 맞는 국가기술자격·자격증",
        ]
        if _interest_suggests_data_analysis(intr):
            lines.append(f"(관련 검색 보강: {_DATA_ANALYSIS_SEARCH_ANCHOR})")
        return "\n".join(lines).strip()
    return m
