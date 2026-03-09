# Contrastive (768-dim) Retriever

로컬 모델 파일은 제거됨. **Hub 모델** `multifuly/cert-constrative-embedding` 사용.

- **설정:** `.env` 에 `RAG_CONTRASTIVE_MODEL=multifuly/cert-constrative-embedding`, `RAG_CONTRASTIVE_INDEX_DIR=data/contrastive_index`
- **문서:** `backend/docs/CONTRASTIVE_HUGGINGFACE_UPLOAD.md`, `backend/docs/CONTRASTIVE_USAGE_LOCAL.md`

---

## 인덱스 (`data/contrastive_index/`)

- **삭제하면 안 됨.** Contrastive 검색은 이 인덱스에 의존함.
- `cert_index.faiss`, `cert_metadata.json` 이 없으면 `contrastive_search()` 가 빈 리스트를 반환하고, 3-way RRF에서 contrastive arm 이 동작하지 않음.
- 용량이 부족하면 `RAG_CONTRASTIVE_ENABLE=false` 로 끈 뒤 인덱스 폴더를 지울 수 있음. (나중에 다시 쓰려면 인덱스 재생성 필요.)
