# Qwen3 Embedding & Reranker API Server

Qwen3 Embedding / Reranker 모델 기반 텍스트 임베딩 및 리랭킹 API 서버입니다.

## 프로젝트 구조

```
embedding/
├── Dockerfile
├── pyproject.toml
├── README.md
├── src/
│   ├── api.py                    # FastAPI 앱 & 라우트 (/embed, /rerank, /health)
│   ├── v1/
│   │   ├── service.py            # 임베딩 서비스 레이어
│   │   ├── reranker.py           # 리랭커 서비스 레이어
│   │   └── utils/
│   │       └── model_loader.py   # 임베딩/리랭커 모델 로드 유틸리티
│   └── resources/
│       └── model/                # 다운로드된 모델 저장 위치
│           ├── embedding_qwen3_0_6b/
│           └── reranker_qwen3_0_6b/
└── test/
    └── download_model.py         # HF 임베딩/리랭커 모델 다운로드 스크립트
```

## 환경 변수

| 변수명 | 설명 | 기본값 |
|--------|------|--------|
| `EMBEDDING_MODEL_PATH` | 임베딩 모델 로컬 경로 | `src/resources/model/embedding_qwen3_0_6b` |
| `RERANKER_MODEL_PATH` | 리랭커 모델 로컬 경로 | `src/resources/model/reranker_qwen3_0_6b` |
| `EMBEDDING_PORT` | 서버 포트 | `6002` |

## 실행

### 로컬 실행

```bash
# 모델 다운로드 (최초 1회)
python test/download_model.py              # 임베딩 모델만 (기본)
python test/download_model.py --reranker   # 리랭커 모델만
python test/download_model.py --all        # 임베딩 + 리랭커 모두

# 서버 실행
PYTHONPATH=. uvicorn src.api:app --host 0.0.0.0 --port 6002
```

### Docker

```bash
docker build -t embedding-server .
docker run --gpus all -p 6002:6002 \
  -v /path/to/models:/app/src/resources/model \
  -e EMBEDDING_PORT=6002 \
  embedding-server
```

## API

### `GET /health`

서버 상태와 임베딩/리랭커 모델 로드 여부를 반환합니다.

### `POST /embed`

텍스트 리스트를 임베딩 벡터로 변환합니다.

**Request:**
```json
{
  "texts": ["안녕하세요", "텍스트 임베딩 테스트입니다"],
  "max_length": 512,
  "normalize": true,
  "batch_size": 32
}
```

**Response:**
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
    "request_kst": "2025-01-01 12:00:00.000",
    "response_kst": "2025-01-01 12:00:00.023",
    "total_elapsed_sec": 0.0234
  }
}
```

### `POST /rerank`

`query`와 `documents` 쌍의 적합도를 Qwen3 리랭커로 평가하고 점수 내림차순으로 정렬한 결과를 반환합니다.

**Request:**
```json
{
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
}
```

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `query` | string | (필수) | 리랭킹 기준 쿼리 |
| `documents` | string[] | (필수) | 후보 문서 리스트 |
| `instruction` | string \| null | 기본 지시문 | 리랭커에게 전달할 추가 지시문 |
| `top_n` | int \| null | null (전체) | 상위 N개만 반환 |
| `batch_size` | int | 8 | 배치 크기 |
| `max_length` | int | 8192 | 토크나이저 최대 길이 |

**Response:**
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
    "request_kst": "2025-01-01 12:00:00.000",
    "response_kst": "2025-01-01 12:00:00.142",
    "total_elapsed_sec": 0.1421
  }
}
```

> `index`는 요청 시 전달한 `documents` 배열에서의 원래 인덱스입니다. 정렬은 `score` 내림차순.
