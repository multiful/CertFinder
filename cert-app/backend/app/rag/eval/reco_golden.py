"""
추천 골든셋(reco) 형식 지원: query_text + gold_ranked [{cert_name, relevance}]
→ question + gold_chunk_ids [qual_id:0] 로 변환하여 기존 RAG 평가에 사용.
cert_name → qual_id: (1) 별칭 리다이렉트 (2) DB qual_name 정확 일치 (3) 공백·구두점 무시 일치
(4) 골든 문자열이 qual_name 부분문자열 (5) compact 키가 qual_name compact에 포함 (긴 키만)
(6) DB qual_aliases 컬럼 참조 (약어·준말 → 정식 자격증명).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from sqlalchemy.orm import Session
from sqlalchemy import text


def _compact_match_key(s: str) -> str:
    """공백·마침표·중점 등 제거한 비교 키 (한글 lower 무의미)."""
    if not s:
        return ""
    return re.sub(r"[\s\.・·]+", "", (s or "").strip())


# 골든 expected_certs 표기 → DB qualification.qual_name 과 동일한 문자열로 통일
# 약어·준말·이전 표기 → 현행 자격증명 매핑 (정적 목록; DB qual_aliases 와 이중 보호)
RECO_GOLDEN_EXPECTED_CERT_ALIASES: Dict[str, str] = {
    # IT/정보처리
    "정처기": "정보처리기사",
    "정보처리": "정보처리기사",
    "정처산기": "정보처리산업기사",
    "정보처리산업": "정보처리산업기사",
    "정처기능사": "정보처리기능사",
    "정보처리기능": "정보처리기능사",
    # 컴퓨터활용능력
    "컴활": "컴퓨터활용능력 1급",
    "컴활 1급": "컴퓨터활용능력 1급",
    "컴활1급": "컴퓨터활용능력 1급",
    "컴퓨터활용1급": "컴퓨터활용능력 1급",
    "컴활 2급": "컴퓨터활용능력 2급",
    "컴활2급": "컴퓨터활용능력 2급",
    "컴퓨터활용2급": "컴퓨터활용능력 2급",
    # 빅데이터
    "빅분기": "빅데이터분석기사",
    "빅데이터기사": "빅데이터분석기사",
    "빅데이터분석": "빅데이터분석기사",
    # SQL
    "SQLD": "SQL개발자(SQLD)",
    "sqld": "SQL개발자(SQLD)",
    "SQL개발자": "SQL개발자(SQLD)",
    "sql개발자": "SQL개발자(SQLD)",
    "SQLP": "SQL전문가(SQLP)",
    "sqlp": "SQL전문가(SQLP)",
    "SQL전문가": "SQL전문가(SQLP)",
    "sql전문가": "SQL전문가(SQLP)",
    # 데이터분석
    "ADsP": "데이터분석준전문가(ADsP)",
    "adsp": "데이터분석준전문가(ADsP)",
    "데이터분석준전문가": "데이터분석준전문가(ADsP)",
    "데분준": "데이터분석준전문가(ADsP)",
    "ADP": "데이터분석전문가(ADP)",
    "adp": "데이터분석전문가(ADP)",
    "데이터분석전문가": "데이터분석전문가(ADP)",
    "데분전": "데이터분석전문가(ADP)",
    # 보안
    "보안기사": "정보보안기사",
    "정보보안": "정보보안기사",
    "보안산업기사": "정보보안산업기사",
    "정보보안산업": "정보보안산업기사",
    # 네트워크
    "네관사": "네트워크관리사",
    "네트워크관리": "네트워크관리사",
    # 리눅스
    "리마1급": "리눅스마스터 1급",
    "리눅스1급": "리눅스마스터 1급",
    "리눅스마스터1급": "리눅스마스터 1급",
    "리마2급": "리눅스마스터 2급",
    "리눅스2급": "리눅스마스터 2급",
    "리눅스마스터2급": "리눅스마스터 2급",
}


def _qual_id_to_name_map(db: Session) -> Dict[int, str]:
    rows = db.execute(text("SELECT qual_id, qual_name FROM qualification")).fetchall()
    return {r.qual_id: (r.qual_name or "").strip() for r in rows}


def _build_alias_to_id_map(db: Session) -> Dict[str, int]:
    """DB qual_aliases 컬럼에서 {alias → qual_id} 역방향 맵 생성."""
    rows = db.execute(
        text("SELECT qual_id, qual_aliases FROM qualification WHERE qual_aliases IS NOT NULL AND qual_aliases != ''")
    ).fetchall()
    out: Dict[str, int] = {}
    for r in rows:
        for alias in (r.qual_aliases or "").split(","):
            alias = alias.strip()
            if alias:
                out[alias] = r.qual_id
                out[alias.lower()] = r.qual_id
                out[re.sub(r"[\s\.・·]+", "", alias).lower()] = r.qual_id
    return out


def _cert_name_to_qual_ids(
    cert_name: str,
    qual_id_to_name: Dict[int, str],
    alias_to_id: Optional[Dict[str, int]] = None,
) -> List[int]:
    """
    cert_name에 해당하는 qual_id 목록.
    정확 일치 → compact 일치 → 부분문자열(짧은 qual_name 우선) → compact 부분포함(키 길이 하한)
    → DB qual_aliases 역방향 맵 (약어/준말 최종 보완).
    """
    c = (cert_name or "").strip()
    if not c or not qual_id_to_name:
        return []

    # 정적 별칭 리다이렉트 (약어 → 정식 자격증명)
    c = RECO_GOLDEN_EXPECTED_CERT_ALIASES.get(c, c)
    # 소문자 변형도 시도
    c = RECO_GOLDEN_EXPECTED_CERT_ALIASES.get(c.lower(), c)

    exact = [qid for qid, qname in qual_id_to_name.items() if (qname or "").strip() == c]
    if exact:
        # 동일 qual_name이 DB에 중복 저장된 경우가 있어 gold 확장(요구 항목 증가)으로 이어질 수 있음.
        # 평가에서는 "해당 cert_name에 대해 가장 일관된 1개"만 골라야 Success@4가 과도하게 깎이지 않는다.
        return [sorted(exact)[0]]

    c_key = _compact_match_key(c)
    if c_key and len(c_key) >= 2:
        exact_c = [
            qid
            for qid, qname in qual_id_to_name.items()
            if _compact_match_key(qname or "") == c_key
        ]
        if exact_c:
            return [sorted(exact_c)[0]]

    candidates = [(qid, qname) for qid, qname in qual_id_to_name.items() if c in (qname or "")]
    if not candidates and c_key and len(c_key) >= 6:
        for qid, qname in qual_id_to_name.items():
            qn = (qname or "").strip()
            if not qn:
                continue
            qk = _compact_match_key(qn)
            if c_key in qk:
                candidates.append((qid, qn))
    if candidates:
        candidates.sort(key=lambda x: len(x[1]))
        min_len = len(candidates[0][1])
        # min_len 후보 중에서도 중복 qual_id가 여러 개면 gold를 확장시키지 않도록 1개만 선택
        best = sorted([qid for qid, qname in candidates if len(qname) == min_len])[0]
        return [best]

    # DB qual_aliases 역방향 맵으로 최종 보완 (약어·준말 매핑)
    if alias_to_id:
        orig = (cert_name or "").strip()
        for lookup in (orig, orig.lower(), _compact_match_key(orig).lower()):
            if lookup and lookup in alias_to_id:
                return [alias_to_id[lookup]]

    return []


def cert_names_to_gold_chunk_ids(
    db: Session,
    gold_ranked: List[Dict[str, Any]],
    min_relevance: int = 1,
    qual_id_to_name: Dict[int, str] | None = None,
    alias_to_id: Optional[Dict[str, int]] = None,
) -> Set[str]:
    """
    gold_ranked [{cert_name, relevance}, ...] → 정답 qual_id들의 청크 id 집합 {"qual_id:0", ...}.
    relevance >= min_relevance 인 cert_name만 사용. DB에 없는 자격증명/선택 항목은 무시.
    alias_to_id: DB qual_aliases 역방향 맵 (미제공 시 자동 조회).
    """
    if qual_id_to_name is None:
        qual_id_to_name = _qual_id_to_name_map(db)
    if alias_to_id is None:
        alias_to_id = _build_alias_to_id_map(db)
    out: Set[str] = set()
    for g in gold_ranked or []:
        if int(g.get("relevance", 0)) < min_relevance:
            continue
        cert_name = (g.get("cert_name") or "").strip()
        if not cert_name:
            continue
        qids = _cert_name_to_qual_ids(cert_name, qual_id_to_name, alias_to_id)
        for qid in qids:
            out.add(f"{qid}:0")
    return out


def normalize_reco_golden(golden: List[Dict[str, Any]], db: Session) -> List[Dict[str, Any]]:
    """
    reco 형식 행(query_text, gold_ranked)이 있으면 question, gold_chunk_ids로 변환한 새 리스트 반환.
    profile-aware 확장: expected_certs(자격증명 리스트)만 있으면 gold_ranked로 변환 후 동일 처리.
    이미 gold_chunk_ids가 있는 행은 그대로 유지.
    """
    qual_id_to_name = _qual_id_to_name_map(db)
    alias_to_id = _build_alias_to_id_map(db)
    out: List[Dict[str, Any]] = []
    for row in golden:
        r = dict(row)
        gold_ranked = row.get("gold_ranked")
        if gold_ranked is None and row.get("expected_certs"):
            # profile-aware 포맷: expected_certs → gold_ranked
            expected = row["expected_certs"]
            if isinstance(expected, list):
                gold_ranked = [{"cert_name": (c if isinstance(c, str) else c.get("cert_name", "")), "relevance": 1} for c in expected]
            else:
                gold_ranked = []
            r["gold_ranked"] = gold_ranked
        if (r.get("gold_ranked") is not None) and not r.get("gold_chunk_ids"):
            r["gold_chunk_ids"] = list(cert_names_to_gold_chunk_ids(
                db, r["gold_ranked"], min_relevance=1,
                qual_id_to_name=qual_id_to_name,
                alias_to_id=alias_to_id,
            ))
            r["question"] = r.get("query_text") or r.get("question") or ""
        out.append(r)
    return out
