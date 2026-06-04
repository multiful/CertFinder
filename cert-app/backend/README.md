# CertFinder Backend

**CertFinder** 자격증 검색·추천 API 백엔드. 고성능 비동기 서버로, **Hybrid RAG**(BM25 + Vector + Contrastive 3-way RRF + Reranker) 기반 검색·추천과 Redis 기반 초고속 조회를 제공합니다.

---

## 목차

- [프로젝트 구조](#-프로젝트-구조)
- [주요 기술 스택](#-주요-기술-스택)
- [RAG 파이프라인 개요](#-rag-파이프라인-개요)
- [RAG 기능 전체 목록 (문서)](#-rag-기능-전체-목록-문서)
- [RAG 고도화 방식 및 성과](#-rag-고도화-방식-및-성과)
- [실행 방법](#-실행-방법)
- [평가 및 골든셋](#-평가-및-골든셋)
- [데이터·스크립트](#-데이터스크립트)
- [참고 문서](#-참고-문서)

---

## 🏗 프로젝트 구조

```text
backend/
├── app/
│   ├── api/
│   │   ├── auth.py           # 사용자 인증·프로필
│   │   ├── certs.py          # 자격증 조회·검색 (Standard)
│   │   ├── fast_certs.py     # Redis 기반 초고속 조회
│   │   ├── ai_recommendations.py  # AI 추천 (Hybrid RAG 연동)
│   │   ├── recommendations.py    # 전공·AI 기반 추천
│   │   ├── jobs.py, majors.py, favorites.py, acquired_certs.py, admin.py, contact.py
│   │   └── deps.py
│   ├── rag/                  # RAG 검색·생성
│   │   ├── api/routes.py     # RAG HTTP 라우트
│   │   ├── config.py         # RAG 하이퍼파라미터 (RAGSettings)
│   │   ├── retrieve/hybrid.py      # BM25+Vector+Contrastive 융합, 라우팅, soft, 리랭커
│   │   ├── retrieve/metadata_soft_score.py   # 메타데이터 soft (옵션)
│   │   ├── retrieve/personalized_soft_score.py  # 개인화 soft (프로필 시)
│   │   ├── rerank/cross_encoder.py # Reranker HF Space API (CertFinder Reranker)
│   │   ├── generate/evidence_first.py, gating.py
│   │   ├── index/            # BM25·Vector 인덱스 빌드
│   │   ├── eval/             # Recall@k, MRR, nDCG, 골든 로더
│   │   └── utils/            # Dense rewrite, HyDE, CoT, domain_tokens, …
│   ├── services/
│   │   ├── vector_service.py # OpenAI Embedding
│   │   ├── fast_sync_service.py  # Redis 벌크 동기화
│   │   └── data_loader.py
│   ├── redis_client.py, database.py, models.py, schemas/, config.py
│   └── utils/ai.py, auth.py, stream_producer.py
├── scripts/                     # 평가 스크립트
│   └── ab_test_golden_mapping.py        # 골든셋 A/B 매핑
├── data/                     # 골든셋, 코퍼스, contrastive 학습 데이터
├── main.py
├── requirements.txt
└── .env.example
```

---

## ⚡ 주요 기술 스택

| 영역 | 내용 |
|------|------|
| **API** | FastAPI, Pydantic, SQLAlchemy, Supabase(PostgreSQL) |
| **캐시·속도** | Redis (orjson 직렬화), FastSyncService 부팅 시 전체 인덱스 로드, StreamProducer Pub/Sub |
| **RAG** | BM25 + Vector + Contrastive 3-way RRF/Linear Fusion, Query Routing, Dense Query Rewrite, Reranker(HF Space API), Metadata·개인화 Soft Score |
| **배포** | Render(API, `api.certfinder.cloud`), Vercel(프론트, `www.certfinder.cloud`), UptimeRobot 등 `/health` 모니터링 |

---

## 🔍 RAG 파이프라인 개요

> **인덱싱·재색인**: `RAG_Indexing.md` · **문서 기준**: `RAG_LATEST_DOCS_RECORD.md` · **개선 이력**: `RAG_IMPROVEMENT.md`

1. **질의 처리**  
   Dense Query Rewrite(전공·학년·북마크·취득 반영), 짧은 쿼리 보조 키워드, Query Type 분류(DB 라벨 또는 폴백), 식별자 위주 질의 시 확장·rewrite 스킵, pre-retrieval 예산(옵션).

2. **검색**  
   - **BM25**: 디스크 `bm25.pkl`, 한글 2-gram, 자격명 부스팅, 쿼리 확장 규칙.  
   - **계층 BM25(옵션)**: `content` 문단 단위 child 검색 → `qual_id` 환원 후 BM25 채널과 블렌드(`RAG_HIERARCHICAL_*`).  
   - **Vector**: `certificates_vectors` pgvector, 원문+rewrite 다중 검색·키워드 확장 벡터(설정 시).  
   - **Contrastive**: 768-dim FAISS/원격 임베딩, Redis 캐시.  
   - **Query Routing**: 짧은 키워드형에서 BM25 비중·벡터 게이팅.  
   - **Contrastive 게이팅**: `RAG_CONTRASTIVE_ALLOWED_QUERY_TYPES` — DB 라벨이 없을 때 폴백 타입만 허용되도록 목록을 맞출 것.

3. **융합**  
   **Linear**(기본) / CombSum / CombMNZ, `RAG_RRF_K`·채널 가중·쿼리타입별 linear 3-way 가중. 후보 풀: `RAG_TOP_N_CANDIDATES` 등(기본 88 전후, `.env`로 조정).

4. **메타데이터·개인화**  
   **메타 soft**(옵션, 기본 OFF): 직무·전공·도메인 가산/감점. **개인화 soft**(프로필 시): 전공·즐겨찾기 분야·취득 감점·학년-난이도 등. **취득 hard exclude** 옵션.

5. **리랭커**  
   Cross-Encoder HF Space API(기본 OFF). 풀 크기·게이팅·입력 보강·pair 캐시.

6. **생성**  
   Evidence-first 프롬프트·Gating(근거 부족 응답).

---

## 📑 RAG 기능 전체 목록 (문서)

- **`app/rag/config.py`** — 구현된 `RAG_*` 스위치·채널·하이퍼파라미터 기준. 코드가 항상 최신 사실 소스.

---

## 📈 RAG 고도화 방식 및 성과

기준: **골든셋**(예: `reco_golden_recommendation_18.jsonl` 등) 기준 Recall@k, Hit@k, MRR@k, nDCG@k.  
베이스라인 대비 아래 조합으로 단계적 개선을 적용했고, 수치는 동일 골든·환경에서의 상대적 변화를 반영합니다.

### 적용한 고도화 방식

| 구분 | 고도화 방식 | 설명 |
|------|-------------|------|
| **1. Hybrid 검색** | BM25 + Vector 병합 | 단일 벡터 검색 대비 키워드형·자연어형 모두 대응, Recall·Hit 상승. |
| **2. RRF / Linear Fusion** | RRF K=60, 또는 Linear(λ*BM25+(1-λ)*Vector) | R@20·Hit@20·MRR@4 극대화를 위해 K·가중치 튜닝. Linear 적용 시 지표 추가 상승. |
| **3. Vector 임계값** | `RAG_VECTOR_THRESHOLD=0.02` | 저유사도 노이즈 제거. RRF 구간에서 MRR·nDCG 상승. |
| **4. 후보 풀 확대** | RRF Top-N 95 → 110 | MRR +3% 상승, 지연 +5% 수준(평가 스윕 기준). |
| **5. BM25 파라미터** | `b=0.5`, 한글 n-gram, 자격명 부스팅 | Hit@4·nDCG@20·nDCG@4 상승. |
| **6. Dense Query Rewrite** | 전공·학년·북마크·취득 반영 재질의 | Vector 채널 정확도·추천 적합도 향상. |
| **7. Query Routing + Gating** | 짧은 쿼리 BM25 강화, Vector min_score/gap 게이팅 | 짧은 키워드에서 Vector 오탐 억제, 상위 순위 보존. |
| **8. Reranker** | HF Space API(CertFinder Reranker), 풀 30→Top 4 | 최종 노출 순위 품질 향상. Reranker Gating으로 확신 높은 질의는 스킵해 지연 절감. |
| **9. Metadata Soft Score** | 직무·전공·목적 가산, 분야 이탈 감점 | RRF 후보 내 추천 적합 자격 상위 이동. |
| **10. 리랭커 입력 보강** | 쿼리에 전공·목적·직무, passage에 자격명 접두사 | Reranker가 문맥을 반영해 재정렬. |
| **11. Contrastive 게이팅** | `RAG_CONTRASTIVE_ALLOWED_QUERY_TYPES` (natural, purpose_only, roadmap 등) | Contrastive를 잘 쓰는 타입만 arm 활성화, 키워드/자격명 쿼리에서는 비활성 → 비용·노이즈 절감. |
| **12. 쿼리 타입별 Contrastive RRF 가중치** | `RAG_QUERY_TYPE_CONTRASTIVE_WEIGHTS_ENABLE` + `CONTRASTIVE_QUERY_TYPE_WEIGHTS` | 자연어·로드맵 계열은 Contrastive 비중 강화, 키워드/자격명은 약화 → 3-way RRF 품질 유지·Contrastive 단일 성능 크게 상승. |

### 저장된 고도화 기준 및 채널 Ablation

- **채널별 기여도**: BM25 only / Vector only / Contrastive only vs 3-way(enhanced_reranker) 비교. Contrastive는 게이팅·타입별 가중치 적용 후 단일 Recall@5·MRR이 크게 상승.
- **고도화 전략**: BM25·Vector·Contrastive **단일 성능**을 올린 뒤, **weight/게이팅 재튜닝**을 병행해야 RRF가 이득을 제대로 반영. 단일 악화 또는 RRF 악화 시 해당 변경은 적용하지 않음.

### Current 모델 대비 성장

- **Current 모델**: 이전 실서비스 기준 — 벡터 단일 검색(certificates_vectors) + 임계값 0.4. BM25·RRF·리랭커 미적용.
- **고도화 모델(Enhanced)**: Hybrid RAG — BM25 + Vector + **Contrastive** 3-way RRF + Dense Rewrite + Metadata Soft Score. **리랭커 미적용**으로 측정.

동일 골든셋·동일 환경에서 **리랭커 없이** 측정한 Current 대비 성장은 아래와 같다.

| 지표 | Current (벡터 단일, 임계값 0.4) | 고도화 (Hybrid, 리랭커 미적용) | 성장 |
|------|---------------------------------|---------------------------------|------|
| **Recall@5** | 0.167 | 0.556 | **+233%** |
| **Recall@10** | 0.167 | 0.667 | **+300%** |
| **Recall@20** | 0.278 | 0.778 | **+180%** |
| **Hit@20** | 0.67 | 2.00 | **+200%** |
| **Success@4** | 0.333 | 0.667 | **+100%** |
| **MRR@4** | 0.083 | 0.667 | **+700%** |

- 위 수치는 동일 골든·환경에서의 측정 결과이며, 골든셋·질의 수에 따라 수치가 달라질 수 있다.  
- **최신 수치**: `data/eval_three_models_8_report.md`, `data/eval_three_models_8.csv` (전체 골든 기준) 참고.

### 선택 적용·Ablation

- **HyDE**: Ablation에서 베이스라인 대비 상승했으나 운영에서는 오탐·비용 고려로 **기본 OFF**.
- **CoT / Step-back / BM25 PRF**: 방법론 확장용 옵션, 기본 OFF. 필요 시 `app/rag/config.py` 및 평가 스크립트로 비교 가능.

### 캐싱 전략 (Redis + LRU, RAG 전용)

RAG는 외부 API 호출(OpenAI, HF Space)과 대형 인덱스(FAISS, PostgreSQL)를 동시에 사용하기 때문에 **캐시 계층 설계가 곧 성능·비용·처리량 설계**입니다.

- **1단계: Query → Embedding 캐시 (Contrastive)**  
  - 위치: `app/rag/retrieve/contrastive_retriever.py`  
  - 키: Redis `contrastive:q2v:v1:{hash(query_text)}`  
  - 값: 정규화된 768-dim 벡터(list[float])  
  - TTL: 기본 **12시간** (`_CONTRASTIVE_Q2V_CACHE_TTL_SECONDS`)  
  - 무효화: Contrastive 모델/HF Space 교체 시 prefix(`v1`)만 올려 전체 무효화  
  - 효과 (골든 상위 8개, `top_k=20`, contrastive-only):  
    - Recall@20 / MRR@20: **0.8542 / 0.7708** (캐시 전후 동일)  
    - Avg_ms: **836.0 → 67.1 ms**  
    - P95_ms: **1600.0 → 43.2 ms**  
    → 동일 질의 재사용 시, 임베딩/검색 지연이 **약 10~20배 감소**.

- **2단계: Query + top_k → Contrastive 결과 캐시**  
  - 위치: `contrastive_retriever.contrastive_search()`  
  - 키: Redis `contrastive:results:v1:{hash(query_text, top_k)}`  
  - 값: `[[chunk_id, score], ...]` 형식의 상위 결과 리스트  
  - TTL: 기본 **24시간** (`_CONTRASTIVE_RESULTS_CACHE_TTL_SECONDS`)  
  - 무효화: FAISS 인덱스 리빌드/교체 시 prefix만 올려 전체 무효화  
  - 의미: HF Space 임베딩 + FAISS 검색까지 포함한 **완성된 후보 리스트**를 캐시하여, 동일 query/top_k 재요청 시 사실상 Redis read 수준 지연으로 응답.

- **3단계: (query, chunk_id) → Reranker score 캐시 (LRU + Redis)**  
  - 위치: `app/rag/rerank/cache.py` (`RerankerCache`), `app/rag/rerank/cross_encoder.py` (`_rerank_via_api`)  
  - 키:  
    - 로컬 LRU: `sha256(query ||| doc_hash)[:32]`  
    - Redis: `rerank:v1:{sha256(query ||| doc_hash)[:32]}`  
    - `doc_hash = sha256(passage_text)[:16]`  
  - 값: 단일 float score (또는 `{ "score": float }`)  
  - TTL: 기본 **1시간** (`RAG_RERANK_CACHE_TTL`, `.env`로 조절)  
  - 무효화: Reranker 모델/HF Space 교체 시 prefix(`v1`)만 올려 전체 무효화  
  - 동작:  
    1. 로컬 LRU → Redis 순서로 `(query, passage)` pair score 조회  
    2. miss인 pair만 HF Space로 batch 호출  
    3. 응답 score를 LRU + Redis에 동시에 기록  
  - 효과 (골든 상위 8개, `enhanced_reranker`, 3-way RRF 기준 2회차):  
    - Baseline(2-way) Avg_Latency_ms: **558.5 ms**  
    - Current(3-way) Avg_Latency_ms: **609.5 ms**  
    → contrastive + reranker까지 포함한 3-way에서도 **Baseline 대비 추가 지연이 ~50 ms 수준**으로 수렴.

- **4단계: RAG 응답 캐시 (질의 전체 응답)**  
  - 위치: `app/rag/retrieve/cache.py`, `redis_client.rag_ask_cache_key()`  
  - 키: `rag:ask:v1:{hash(query, filters, top_k, baseline_id)}`  
  - 값: RAG 전체 응답(JSON 직렬화, 모델 출력 포함)  
  - TTL: `RAG_CACHE_TTL` (기본 600초)  
  - 용도: 동일 질의/필터 조합에 대해 **전체 RAG 파이프라인 실행 자체를 건너뛰는 계층**으로, 반복 질의가 많은 환경에서 처리량(throughput)을 크게 끌어올림.

정리하면,

- **Contrastive**: Query→Embedding + Query+top_k→Results 두 레이어 캐시로 HF Space 호출·FAISS 검색을 크게 줄이면서, 검색 품질(Recall/MRR)은 그대로 유지.  
- **Reranker**: (query, passage) pair-level 캐시(LRU+Redis)로 동일 passage 재랭킹 비용을 제거하고, 후보 풀 구성이 조금 바뀌어도 재사용.  
- **전체 RAG**: Query+Filters+top_k+baseline_id 단위 응답 캐시로, 자주 반복되는 추천 질의에 대해 end-to-end RAG 실행을 피함.

---

## 🆕 최근 변경사항 (2026-05-27)

### RAG 품질 개선 (4개 이슈)

| # | 파일 | 변경 내용 |
|---|------|-----------|
| 1 | `app/utils/rag_hybrid.py` | `classify_query_and_expand`: 3-case 하드코딩 → `expand_query_single_string` 호출. 60+ 동의어·직무·전공·구어 패턴 전체 활성화. |
| 2 | `app/utils/rag_hybrid.py` | `_sparse_rank_map_for_query(use_or=True)` 추가. 확장된 긴 쿼리에 `to_tsquery` OR 검색 적용 — AND 시 매칭 0건 문제 해결. |
| 3 | `main.py` | `_background_bm25_prebuild()`: 서버 기동 25초 후 `bm25.pkl` 없으면 DB에서 자동 빌드 (Render 재배포 후 ephemeral storage 대응). |
| 4 | `main.py` | Contrastive pre-warm 실패 시 `logger.debug` → `logger.warning` + `diagnose_contrastive_status()` 상세 원인 출력. |

### 보안 패치

| 취약점 | 위치 | 조치 |
|--------|------|------|
| JWT 알고리즘 혼동 공격 | `app/api/deps.py` | `SUPABASE_JWT_ALGORITHM` 설정 시 토큰 alg 강제 일치 검증. 현행: `ES256`. |
| 이메일 열거(find-userid) | `app/api/auth.py` | 미가입 이메일에도 200 반환 — 가입 여부 구분 불가. |
| 사용자 ID 노출 | `app/api/auth.py` | 회원가입 충돌 에러 메시지에서 userid 제거. |

### 데이터 품질 패치

- 트렌딩 자격증 쿼리에 `is_active = TRUE` 필터 추가 (`app/api/certs.py`)
- 전공 기반 추천 쿼리에 `is_active = TRUE` 필터 추가 (`app/crud.py`)

### A/B 테스트 결과 (2026-05-27)

골든셋: 25개 질의 (`golden_ab_test.jsonl`) — 키워드·자연어·로드맵·전공직무 혼합.

#### 1) baseline vs current (vs 벡터 단독)
결과 상세: `eval_ab_hybrid_rag_20260527.md`

| 지표 | baseline (벡터만) | current (Hybrid RAG) | 개선율 |
|------|-------------------|----------------------|--------|
| Recall@5 | 0.567 | **0.647** | +14.1% |
| Recall@10 | 0.587 | **0.687** | +17.0% |
| **MRR** | 0.521 | **0.707** | **+35.7%** |
| NDCG@5 | 0.512 | **0.644** | +25.8% |
| MAP | 0.487 | **0.626** | +28.5% |
| Hit@10 | 0.640 | **0.760** | +18.8% |
| avg latency | 423ms | **164ms** | **-61%** |

#### 2) current vs enhanced_reranker (배포된 3-way RAG)
결과 상세: `eval_current_vs_deployed_20260527.md` · `ab_vs_deployed_20260527_161653.csv`

| 지표 | current (2채널+쿼리확장) | enhanced_reranker (3-way) | 차이 |
|------|--------------------------|---------------------------|------|
| Recall@5 | 0.647 | **0.750** | +15.9% |
| Recall@10 | 0.687 | **0.810** | +17.9% |
| MRR | 0.707 | **0.766** | +8.3% |
| NDCG@10 | 0.664 | **0.755** | +13.7% |
| Hit@10 | 0.760 | **0.960** | +26.3% |
| avg latency | **133ms** | 1363ms | current 10배 빠름 |

> **결론**: 3-way(`enhanced_reranker`)가 품질 전지표 우위. `current`는 10배 빠른 저지연 경로.  
> AI 추천 엔드포인트를 `hybrid_retrieve`로 마이그레이션하면 Hit@10 +26.3%, MRR +8.3% 기대.

### RLS 보안 패치 (2026-05-27)

ML/참조 데이터 테이블 3개에 RLS 활성화 + public SELECT 정책 추가:
- `query_type_labels`, `certificates_vectors_contextual`, `intent_labels`

---

## 🛠 실행 방법

1. **환경**  
   `.env` 설정 (참고: `.env.example`).  
   Python: `uv` 사용 시 `uv run`으로 실행.

2. **의존성**  
   `pip install -r requirements.txt` 또는 `uv pip install -r requirements.txt`

3. **서버**  
   `cert-app/backend` 디렉터리에서 `uvicorn main:app --reload`. (uv 미사용 시 venv의 `python -m uvicorn main:app --reload`)

4. **RAG 인덱스**  
   BM25: `python -m app.rag index`. Vector: Supabase `certificates_vectors`. Contrastive: FAISS 인덱스(`data/contrastive_index/`)·Redis 캐시. `RAG_Indexing.md` 참고.

---

## 📊 평가 및 골든셋

- **A/B 골든셋**: `golden_ab_test.jsonl`, `golden_ab_test_v2.jsonl` (백엔드 루트)  
  25개 질의 — 키워드·자연어·로드맵·전공직무 혼합. 평가 결과: `eval_ab_hybrid_rag_20260527.md`, `eval_current_vs_deployed_20260527.md`.
- **3모델 비교 (리랭커 없음)**  
  `cd cert-app/backend` 후  
  `uv run python -m app.rag.eval [--max-queries N]`  
  → baseline(단일 Vector) / current(2-way) / enhanced_reranker(3-way BM25+Vector+Contrastive RRF+리랭커) 비교.
- **주요 파라미터 튜닝 환경변수** (`app/rag/config.py` 기준):  
  `RAG_FUSION_METHOD`(기본 `linear`), `RAG_LINEAR_BM25_RANK_PRIOR`(0.009), `RAG_EVAL_TOP_K`(기본 10),  
  `RAG_DUAL_VECTOR_RRF_WHEN`(기본 `divergence`), `RAG_POST_METADATA_BM25_RANK_PRIOR`(0.004).

---

## 📁 데이터·스크립트

| 용도 | 파일 |
|------|------|
| **A/B 골든셋** | `golden_ab_test.jsonl`, `golden_ab_test_v2.jsonl` (백엔드 루트) |
| **평가 결과 (2026-05-27)** | `eval_ab_hybrid_rag_20260527.md`, `eval_current_vs_deployed_20260527.md`, `ab_vs_deployed_20260527_161653.csv` |
| **Contrastive 인덱스** | `data/contrastive_index/` — `cert_index.faiss`, `cert_metadata.json` (FAISS 기반 768-dim) |
| **BM25 인덱스** | `data/rag_index/bm25.pkl` — 서버 기동 시 자동 빌드 (Render ephemeral storage 대응) |
| **도메인 토큰** | `data/domain_tokens_new_cert_full.json` |
| **평가 스크립트** | `scripts/ab_test_golden_mapping.py` |

---

## 📚 참고 문서

- **RAG 인덱싱·파라미터**: `RAG_Indexing.md`
- **최신 문서 기준**: `RAG_LATEST_DOCS_RECORD.md`
- **Contrastive 인덱스**: `app/rag/contrastive/README.md`, `data/contrastive_index/README.md`
- **RAG 설정 소스**: `app/rag/config.py` (RAGSettings — 모든 `RAG_*` 환경변수 정의)

---

## Reranker

- **CertFinder Reranker**: HF Hub 모델·Space API로 서빙. 환경변수 `RAG_RERANKER_MODEL_REPO_ID`, `RAG_RERANKER_SPACE_REPO_ID`, `RAG_RERANKER_API_URL` 참고.
- **역할**: RRF/융합 상위 풀(`RAG_RERANK_POOL_SIZE`, 기본 20 전후) 재정렬 → Top-k 선택. Gating 적용 시 확신 높은 질의는 스킵.
- **평가**: `uv run python -m app.rag.eval` — baseline/current/enhanced_reranker 4-way 비교. 리랭커 ON/OFF 차이는 `eval_current_vs_deployed_20260527.md` 참고.
- **캐시 설정**: `RAG_RERANK_CACHE_TTL`(기본 1시간), `RAG_RERANK_POOL_SIZE`(기본 20) — `app/rag/config.py`.
