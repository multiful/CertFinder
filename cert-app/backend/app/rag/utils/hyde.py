"""
HyDE (Hypothetical Document Embeddings): 질의에 대한 가상 답변 문서를 생성해 벡터 검색에 활용.
논문: Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels", 2022.
사용자 프로필(전공·학년·취득자격증)을 포함해 개인화된 가상 문서를 생성함으로써
쿼리-문서 semantic gap 해소 및 pool miss 감소.
"""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

HYDE_SYSTEM = (
    "당신은 자격증·취업 로드맵 안내 전문가입니다. "
    "주어진 사용자 정보와 질문을 바탕으로, 관련 자격증 설명문이나 취업 가이드 문단처럼 읽히는 "
    "짧은 문장 1~2개를 작성하세요. "
    "자격증명(정보처리기사, SQLD, 산업안전기사 등), 직무, 관련 키워드를 실제로 있을 법한 표현으로 포함하세요. "
    "질문을 그대로 반복하지 말고, 추천 문서처럼 서술하세요."
)

_PROFILE_TEMPLATE = (
    "전공: {major}\n"
    "학년: {grade}\n"
    "기취득 자격증: {acquired}\n"
    "질문: {query}\n\n"
    "위 사용자에게 맞는 자격증·취업 관련 설명 문단 1~2문장을 작성하세요 (검색용 가상 문서):"
)

_QUERY_ONLY_TEMPLATE = (
    "질문: {query}\n\n"
    "위 질문에 답하는 것처럼, 자격증·취업 관련 설명 문단 1~2문장을 작성하세요 (검색용 가상 문서):"
)


def _build_hyde_user_prompt(query: str, user_profile: Optional[Dict[str, Any]]) -> str:
    if not user_profile:
        return _QUERY_ONLY_TEMPLATE.format(query=query.strip())

    major = user_profile.get("major", "")
    grade = user_profile.get("grade_level")
    acquired: List[str] = (
        list(user_profile.get("acquired_cert_names") or [])
        or [str(q) for q in (user_profile.get("acquired_qual_ids") or [])]
    )

    if not major and grade is None and not acquired:
        return _QUERY_ONLY_TEMPLATE.format(query=query.strip())

    grade_str = f"{grade}학년" if grade else "미상"
    acquired_str = ", ".join(acquired) if acquired else "없음"
    return _PROFILE_TEMPLATE.format(
        major=major or "미상",
        grade=grade_str,
        acquired=acquired_str,
        query=query.strip(),
    )


def generate_hyde_document(
    query: str,
    max_tokens: int = 80,
    user_profile: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    질의+사용자 프로필에 대한 가상 답변 문서(1~2문장)를 LLM으로 생성.
    user_profile: major, grade_level, acquired_cert_names 등 UserProfile 필드.
    실패 시 None 반환.
    """
    if not (query or "").strip():
        return None
    try:
        from app.config import get_settings
        from app.rag.config import get_rag_settings
        from openai import OpenAI

        settings = get_settings()
        if not getattr(settings, "OPENAI_API_KEY", None):
            return None
        rag = get_rag_settings()
        temperature = getattr(rag, "RAG_HYDE_TEMPERATURE", 0.3)
        client = OpenAI(api_key=settings.OPENAI_API_KEY)

        user_prompt = _build_hyde_user_prompt(query, user_profile)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": HYDE_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text or len(text) < 10:
            return None
        return text[:800]
    except Exception as e:
        logger.warning("HyDE document generation failed: %s", e)
        return None
