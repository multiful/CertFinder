"""
Cohere Rerank API 래퍼.

model="rerank-v3.5" (다국어 지원 · 한국어 OK)
POST https://api.cohere.com/v2/rerank
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def rerank_with_cohere(
    query: str,
    pairs: List[Tuple[str, str]],
    api_key: str,
    top_k: Optional[int] = None,
    model: str = "rerank-v3.5",
) -> List[Tuple[str, float]]:
    """
    Cohere Rerank v2 API로 후보를 재정렬.

    pairs: [(chunk_id, text), ...]
    반환: [(chunk_id, relevance_score), ...] 내림차순. 실패 시 빈 리스트.
    """
    if not query or not pairs:
        return []
    try:
        import cohere
        co = cohere.ClientV2(api_key=api_key)
        documents = [text for _, text in pairs]
        resp = co.rerank(
            model=model,
            query=query,
            documents=documents,
            top_n=top_k or len(documents),
        )
        results = []
        for item in resp.results:
            chunk_id = pairs[item.index][0]
            results.append((chunk_id, float(item.relevance_score)))
        results.sort(key=lambda x: -x[1])
        if top_k:
            results = results[:top_k]
        return results
    except Exception as e:
        logger.warning("Cohere rerank 실패: %s", e)
        return []
