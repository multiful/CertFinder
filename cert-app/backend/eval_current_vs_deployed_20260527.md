# RAG 비교 평가 리포트 — 2026-05-27 (current vs enhanced_reranker)

## 평가 목적
내가 수정한 `enhanced_rag_03_hybrid` (`current` 파이프라인, 2채널: Dense+Sparse+쿼리확장+OR FTS)가
현재 웹에 배포된 3-way RAG(`hybrid_retrieve`, BM25+Vector+Contrastive)보다 성능이 좋은지 검증.

> 리랭커(Cross-Encoder HF Space) 미사용 조건으로 실행 (`--no-rerank`).
> `enhanced_reranker`는 `hybrid_retrieve`(BM25+Vector+Contrastive RRF) 결과 그대로 평가.

## 파이프라인 정의

| Pipeline | 엔드포인트 | 채널 | 설명 |
|----------|-----------|------|------|
| **current** | `GET /api/v1/recommendations/ai/hybrid-recommendation` | Dense+Sparse(2채널) | `enhanced_rag_03_hybrid` — 쿼리 확장 + OR FTS(내 수정) |
| **enhanced_reranker** | `GET /api/v1/certs/search/rag` | BM25+Vector+Contrastive(3채널) | `hybrid_retrieve` — 3-way RRF, 배포된 RAG 검색 |

## 결과 (골든셋 25개 질의)

| 지표 | current | enhanced_reranker | 차이 |
|------|---------|-------------------|------|
| Recall@5 | 0.647 | **0.750** | +15.9% |
| Recall@10 | 0.687 | **0.810** | +17.9% |
| Precision@5 | 0.216 | **0.288** | +33.3% |
| Precision@10 | 0.124 | **0.164** | +32.3% |
| MRR | 0.707 | **0.766** | +8.3% |
| NDCG@5 | 0.644 | **0.727** | +12.9% |
| NDCG@10 | 0.664 | **0.755** | +13.7% |
| MAP | 0.626 | **0.705** | +12.6% |
| F1@5 | 0.312 | **0.396** | +26.9% |
| F1@10 | 0.203 | **0.263** | +29.6% |
| Hit@5 | 0.720 | **0.880** | +22.2% |
| Hit@10 | 0.760 | **0.960** | +26.3% |
| avg latency | **133ms** | 1363ms | -90.2% (10x 빠름) |
| p95 latency | **193ms** | 1840ms | -89.5% (9.5x 빠름) |

## 해석

### 품질: enhanced_reranker(3-way)가 모든 지표에서 우위
- **Hit@10 +26.3%**: 10개 결과 안에 정답이 있는 비율이 0.76 → 0.96으로 대폭 상승.
- **Recall@10 +17.9%**: Contrastive(768-dim 도메인 특화 임베딩)가 1536-dim OpenAI 벡터가 놓치는 자격증을 보완.
- **MRR +8.3%**: 정답이 첫 화면에 보이는 빈도 차이. 이미 내 current가 0.707로 높은 값이라 격차가 작음.

### 속도: current(2채널)가 10배 빠름
- current 133ms vs enhanced_reranker 1363ms — 10배 차이.
- enhanced_reranker의 latency는 Contrastive FAISS(768-dim, HF Space 호출 포함) + BM25 인덱스 로드 때문.

### current 파이프라인의 개선 확인
이전 A/B 테스트(2026-05-27 `eval_ab_hybrid_rag_20260527.md`)에서 `current`는 `baseline`(벡터만) 대비:
- MRR +35.7%, avg_latency -61%

이번 테스트에서 `current`는 `enhanced_reranker` 대비 품질이 낮지만(MRR -8.3%), 속도는 10배 우위.
내 수정(쿼리 확장 + OR FTS)으로 `current`가 3-way RAG에 근접했음을 확인.

## 결론

**enhanced_reranker(3-way)가 품질 기준 우위.** current는 10배 빠르지만 Hit@10 기준 24포인트 낮음.

| 구분 | 추천 파이프라인 | 이유 |
|------|---------------|------|
| AI 추천 (ai_recommendations.py) | enhanced_reranker로 마이그레이션 권장 | 품질 중요, latency 1.3s 허용 가능 |
| 현행 유지 시 | current 유지 | 133ms 저지연 필요하거나 OpenAI 비용 절감 우선 |

## 환경 변수 권장 (enhanced_reranker 전환 시)

```env
RAG_CONTRASTIVE_ENABLE=true
RAG_CONTRASTIVE_INDEX_DIR=data/contrastive_index
RAG_TOP_N_CANDIDATES=88
RAG_BM25_TOP_N=88
RAG_CONTRASTIVE_TOP_N=76
```

## 원본 CSV

`ab_vs_deployed_20260527_161653.csv`

```
pipeline,Recall@5,Recall@10,MRR,NDCG@10,MAP,Hit@10,Avg_Latency_ms
current,0.647,0.687,0.707,0.664,0.626,0.760,133
enhanced_reranker,0.750,0.810,0.766,0.755,0.705,0.960,1363
```
