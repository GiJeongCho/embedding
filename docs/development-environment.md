# Embedding & Reranker 서비스 - 개발 환경 가이드

> 백엔드 개발자 대상 문서.
> 본 서비스는 **하나의 컨테이너 안에서 임베딩(Embedding)과 리랭커(Reranker) 두 가지 AI 모델을 동시에 서빙**하는 **FastAPI 추론 API** 입니다.
> RAG 파이프라인의 **(1) 벡터 인덱싱용 임베딩** 과 **(2) 검색 결과 재정렬용 리랭킹** 을 한 엔드포인트 묶음에서 제공합니다.

---

## 1. 서비스 개요

| 항목 | 내용 |
|------|------|
| 서비스 이름 | Qwen3 Embedding & Reranker API |
| 코드상 클래스 | `EmbeddingService` / `RerankerService` (`src/v1/`) |
| 임베딩 모델 | **Qwen/Qwen3-Embedding-0.6B** (AutoModel, mean-pooling) |
| 리랭커 모델 | **Qwen/Qwen3-Reranker-0.6B** (AutoModelForCausalLM, yes/no logit 스코어링) |
| 정밀도 | `float16` (GPU) / `float32` (CPU) |
| 응답 | JSON (모든 엔드포인트 동기 처리) |
| 처리 패턴 | 동기 + 배치(`batch_size`) |
| 디바이스 | GPU(CUDA) 우선, CPU fallback |

### 1.1 두 모델 공존 요점

- 한 프로세스에서 **두 개의 독립적인 Transformer 모델**이 GPU 상에 동시 상주합니다.
- **메모리 분리** (`EmbeddingService.model`, `RerankerService.model`) → 한쪽이 죽어도 다른 쪽은 정상 동작 가능 (현재 lifespan 패턴이 이를 보장).
- 0.6B + 0.6B float16 ≈ **VRAM 3~4 GB** 정도 사용 (배치/길이에 따라 가변).
- 4B/8B 임베딩 모델로 교체 시 VRAM 요구량이 급증하므로 **단일 서비스 컨테이너에 함께 두는지 분리하는지 의사결정 필요**(섹션 9 트러블슈팅 참고).

---

## 2. 기술 스택 (AI / Framework)

### 2.1 런타임
- **Python**: `>=3.11` (`pyproject.toml`)
- **Docker base**: `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`
- **CUDA**: 12.4 / cuDNN 9
- **GPU**: float16 0.6B 기준 VRAM ≈ 2 GB(임베딩) + 2 GB(리랭커) + 활성화 메모리. 권장 8 GB+.

### 2.2 패키지 매니저
- **uv** + `pyproject.toml`.
- `[tool.uv.index].pytorch` 로 PyTorch CUDA 12.4 인덱스(`https://download.pytorch.org/whl/cu124`) 고정.

### 2.3 핵심 라이브러리

| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| `fastapi` | ≥ 0.109 | REST API |
| `uvicorn` | ≥ 0.27 | ASGI 서버 |
| `transformers` | ≥ 4.42.3 | `AutoTokenizer` / `AutoModel` / `AutoModelForCausalLM` |
| `accelerate` | ≥ 0.30 | (옵션) `device_map`/오프로딩 지원 |
| `torch` | ≥ 2.6 (cu124) | 추론 백엔드 |
| `numpy` | ≥ 1.26 | 임베딩 벡터 후처리 |
| `huggingface-hub` | (test 스크립트) | 모델 다운로드 |
| `pydantic` | (FastAPI 의존) | 요청/응답 모델 |

### 2.4 모델 로딩 방식

| 모델 | 로더 | 출력 사용 방식 |
|------|------|-----------------|
| 임베딩 | `AutoModel.from_pretrained(..., torch_dtype=fp16)` | `last_hidden_state` → **attention mask 기반 mean pooling** → (선택) L2 정규화 |
| 리랭커 | `AutoModelForCausalLM.from_pretrained(..., torch_dtype=fp16)` + `padding_side="left"` | chat-template 으로 `(query, document)` 평가 → 마지막 토큰의 **`yes`/`no` 로짓 softmax** → yes 확률 = score |

> 두 모델 모두 `trust_remote_code=True`, `local_files_only=True` 로 오프라인 로드.

---

## 3. 디렉토리 구조

```
embedding/
├── Dockerfile
├── pyproject.toml
├── README.md
├── src/
│   ├── api.py                              # FastAPI 엔트리포인트 (/embed, /rerank, /health)
│   ├── v1/
│   │   ├── service.py                      # EmbeddingService
│   │   ├── reranker.py                     # RerankerService
│   │   └── utils/
│   │       └── model_loader.py             # load_model / load_reranker
│   └── resources/
│       └── model/                          # 로컬 모델 가중치 (Git 미추적)
│           ├── embedding_qwen3_0_6b/
│           └── reranker_qwen3_0_6b/
├── test/
│   └── download_model.py                   # HF에서 임베딩/리랭커 모델 다운로드
└── docs/                                   # 본 문서 위치
```

---

## 4. 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `EMBEDDING_MODEL_PATH` | `src/resources/model/embedding_qwen3_0_6b` (자동 계산) | 임베딩 모델 로컬 경로 |
| `RERANKER_MODEL_PATH` | `src/resources/model/reranker_qwen3_0_6b` (자동 계산) | 리랭커 모델 로컬 경로 |
| `EMBEDDING_PORT` | `5000` (코드) / `6002` (README) | `__main__` 실행 시 포트. Docker 에서는 `APP_PORT` 사용 |
| `APP_PORT` | (compose에서 주입) | Docker `CMD` 가 사용 |

> **포트 규칙(`dev/docker.sh`)**: dev = 6000번대 / stg = 9000번대 / prd = 8000번대. embedding 서비스는 별도 자리(예: 6004) 할당이 표준.

---

## 5. 로컬 개발 환경 구축

### 5.1 사전 요구사항
- NVIDIA GPU + 드라이버(CUDA 12.4 호환, R545+)
- `uv` 또는 pip
- (선택) `huggingface-cli` 인증 (gated 모델은 아니지만, 사내 망에서 Hub 접근이 필요한 경우)

### 5.2 의존성 설치 & 모델 다운로드

```bash
cd /home/pps-nipa/jenkins/dev/embedding

# 1) 의존성 설치
uv sync                       # uv.lock 기반 재현
# 또는: pip install .

# 2) 모델 다운로드 (최초 1회)
python test/download_model.py --all     # 임베딩 + 리랭커 모두
#   --embedding   임베딩만
#   --reranker    리랭커만
#   --hf_token <TOKEN>  필요 시

# 결과:
# src/resources/model/embedding_qwen3_0_6b/
# src/resources/model/reranker_qwen3_0_6b/
```

### 5.3 실행

```bash
# 개발 모드 (uv)
PYTHONPATH=. uv run uvicorn src.api:app --host 0.0.0.0 --port 6002 --reload

# 또는 일반 venv
PYTHONPATH=. uvicorn src.api:app --host 0.0.0.0 --port 6002 --reload
```

### 5.4 헬스 체크

```bash
curl http://localhost:6002/health
```
응답 예:
```json
{
  "status": "ok",
  "embedding": {
    "model_loaded": true,
    "model_path": ".../embedding_qwen3_0_6b",
    "device": "cuda:0",
    "gpu_available": true,
    "gpu_count": 1
  },
  "reranker": {
    "model_loaded": true,
    "model_path": ".../reranker_qwen3_0_6b",
    "device": "cuda:0",
    "gpu_available": true,
    "gpu_count": 1
  },
  "server_time_kst": "2026-05-22 17:45:00.123"
}
```

---

## 6. Docker 실행

### 6.1 단독 실행

```bash
cd /home/pps-nipa/jenkins/dev/embedding
docker build -t pps/embed:v0.0.1 -f Dockerfile .

docker rm -f embed_v1 2>/dev/null || true

docker run -d --restart always \
  --gpus all \
  -e APP_PORT=5000 \
  -p 5000:5000 \
  -v "$(pwd)/src/resources/model:/app/src/resources/model:ro" \
  --name embed_v1 \
  pps/embed:v0.0.1

docker logs -f embed_v1
```

마운트 디렉토리 안에는 두 폴더가 모두 있어야 합니다.
```
src/resources/model/
├── embedding_qwen3_0_6b/
└── reranker_qwen3_0_6b/
```

### 6.2 Jenkins 통합 배포

- 이미지 태그: `.env.<env>` 의 `IMG_EMBEDDING=<registry>/embedding:<env>`.
- compose 서비스명: `embedding`.
- 모델 디렉토리: 호스트 → 컨테이너 read-only 마운트.

```bash
sudo /home/pps-nipa/jenkins/dev/docker.sh dev up embedding
```

---

## 7. API 사용법

### 7.1 임베딩 — `POST /embed`

```bash
curl -X POST "http://localhost:6002/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["안녕하세요", "텍스트 임베딩 테스트입니다"],
    "max_length": 512,
    "normalize": true,
    "batch_size": 32
  }'
```

응답:
```json
{
  "embeddings": [[0.012, -0.034, ...], [0.045, 0.023, ...]],
  "dimension": 1024,
  "count": 2,
  "usage": {
    "total_texts": 2,
    "inference_time": 0.0234,
    "batch_size": 32,
    "num_batches": 1
  },
  "timestamp": {
    "request_kst": "2026-05-22 17:45:00.000",
    "response_kst": "2026-05-22 17:45:00.023",
    "total_elapsed_sec": 0.0234
  }
}
```

요청 필드(`EmbedRequest`):

| 필드 | 기본 | 범위 | 설명 |
|------|------|------|------|
| `texts` | 필수 | 최소 1개 | 임베딩 대상 텍스트 |
| `max_length` | 512 | 1~8192 | 토크나이저 truncation 길이 |
| `normalize` | true | bool | L2 정규화 (cosine 유사도 사용 시 권장) |
| `batch_size` | 32 | 1~256 | GPU 배치 |

### 7.2 리랭킹 — `POST /rerank`

```bash
curl -X POST "http://localhost:6002/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Qwen 리랭커 모델은 무엇인가요?",
    "documents": [
      "Qwen3-Reranker-0.6B는 Alibaba에서 공개한 리랭커 모델입니다.",
      "오늘 점심으로 김밥을 먹었습니다.",
      "리랭커는 검색 결과의 순서를 재정렬하는 데 사용됩니다."
    ],
    "instruction": "Given a web search query, retrieve relevant passages that answer the query",
    "top_n": 2,
    "batch_size": 8,
    "max_length": 8192
  }'
```

응답:
```json
{
  "results": [
    {"index": 0, "score": 0.9821, "document": "Qwen3-Reranker-0.6B는 ..."},
    {"index": 2, "score": 0.7613, "document": "리랭커는 검색 결과의 ..."}
  ],
  "count": 2,
  "usage": {
    "total_documents": 3,
    "inference_time": 0.1421,
    "batch_size": 8,
    "num_batches": 1
  },
  "timestamp": {
    "request_kst": "2026-05-22 17:45:01.000",
    "response_kst": "2026-05-22 17:45:01.142",
    "total_elapsed_sec": 0.1421
  }
}
```

요청 필드(`RerankRequest`):

| 필드 | 기본 | 범위 | 설명 |
|------|------|------|------|
| `query` | 필수 | min_length 1 | 리랭킹 기준 쿼리 |
| `documents` | 필수 | min_length 1 | 후보 문서 리스트 |
| `instruction` | (기본 지시문 자동 사용) | - | "Given a web search query, retrieve relevant passages that answer the query" |
| `top_n` | null | ≥ 1 | 상위 N개만 반환 |
| `batch_size` | 8 | 1~64 | 배치 |
| `max_length` | 8192 | 1~32768 | 토크나이저 max_length |

> 결과의 `index` 는 **요청 시 documents 배열에서의 원래 인덱스**. 정렬은 `score` 내림차순.

### 7.3 Swagger / ReDoc
- `http://localhost:6002/docs`
- `http://localhost:6002/redoc`

---

## 8. 백엔드 연동 가이드 (RAG 파이프라인 예시)

표준 RAG 흐름에서 본 서비스의 위치:

```
[문서 인덱싱]                              [검색 시점]
 PDF/DOCX/HWP → ocr_api                   사용자 질의
        │                                     │
        ▼                                     ▼
   markdown/text 청크                   embedding /embed
        │                                     │
        ▼                                     ▼
 embedding /embed  ──► Milvus(벡터 DB)   Milvus 검색 (top-K, 예: 50)
                                              │
                                              ▼
                                       embedding /rerank
                                              │
                                              ▼
                                        상위 N개 컨텍스트
                                              │
                                              ▼
                                          llm_api /generate
```

- 내부 베이스 URL: 백엔드 환경변수로 `EMBEDDING_BASE=http://embedding:${EMBEDDING_PORT_INTERNAL}` 통일.
- 임베딩 벡터 차원: 모델에 따라 다름(0.6B = 1024 차원). 응답의 `dimension` 으로 확인.
- 리랭커 점수는 `[0, 1]` 범위(softmax(`yes`) 확률). 임계값은 백엔드가 정책 결정.

---

## 9. 트러블슈팅

| 증상 | 원인 | 조치 |
|------|------|------|
| `503 임베딩 모델이 아직 로드되지 않았습니다.` | lifespan 로딩 미완료 또는 실패 | `/health` 의 `embedding.model_loaded` 확인 → 로그 점검 |
| `503 리랭커 모델이 아직 로드되지 않았습니다.` | 동상 | `/health` 의 `reranker.model_loaded` 확인. **임베딩은 정상이어도 리랭커만 실패할 수 있음** |
| `리랭커 모델 파일이 누락되었습니다: [...]` | tokenizer.json 또는 config.json 부재 | `python test/download_model.py --reranker` 재실행 |
| OOM (CUDA out of memory) | 임베딩 + 리랭커 동시 로딩으로 VRAM 부족 | 1) `batch_size` 축소  2) `max_length` 축소  3) **두 서비스를 별도 컨테이너로 분리** (섹션 1.1 참조) |
| 정규화된 cosine 유사도가 비정상 | `normalize: false` 로 받고 cosine 계산 | `normalize: true` 로 받거나 클라이언트에서 L2 정규화 |
| 리랭커 스코어가 모두 0 | `_compute_scores` 의 yes/no 토큰 ID가 모델 vocab 에 없음 | 모델 폴더 재다운로드. 다른 변형 모델 사용 시 `token_true_id`/`token_false_id` 조정 |
| `trust_remote_code` 관련 경고 | Qwen3 정상 동작에 필요 | 그대로 사용. 변경 금지 |
| 첫 호출 latency 매우 높음 | 모델 첫 호출 시 CUDA 커널 컴파일 | 부팅 직후 warm-up 호출 1회 추가 |

---

## 10. 관련 문서
- 개발 표준 → [`./development-standards.md`](./development-standards.md)
- LLM 서비스 → [`/home/pps-nipa/jenkins/dev/LLM/docs/`](../../LLM/docs/)
- OCR 서비스 (인덱싱 전처리) → [`/home/pps-nipa/jenkins/dev/ocr/docs/`](../../ocr/docs/)
- Jenkins 통합 배포 → [`/home/pps-nipa/jenkins/docs/development-environment.md`](../../../docs/development-environment.md)
