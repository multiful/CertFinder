"""AI 하이브리드 추천 API와 동일한 RAG 검색 질의 문자열 (운영·오프라인 평가 공통)."""
from typing import Optional


def build_expanded_interest_for_hybrid(major: str, interest: Optional[str]) -> str:
    """
    hybrid_retrieve에 넣는 확장 질의.
    관심사(질의)를 앞에 두고 전공은 배경으로 둔다.
    """
    m = (major or "").strip()
    intr = (interest or "").strip()
    if intr:
        return (
            f"{intr}\n"
            f"(전공·배경: {m})\n"
            f"위 관심·진로에 맞는 국가기술자격·자격증"
        ).strip()
    return m
