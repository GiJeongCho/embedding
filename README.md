# Qwen3 Embedding API Server

Qwen3 Embedding 모델 기반 텍스트 임베딩 API 서버입니다.

## 프로젝트 구조

```
embeded/
├── Dockerfile
├── pyproject.toml
├── README.md
├── src/
│   ├── api.py                    # FastAPI 앱 & 라우트
│   ├── v1/
│   │   ├── service.py            # 임베딩 서비스 레이어
│   │   └── utils/
│   │       └── model_loader.py   # 모델 로드 유틸리티
│   └── resources/
│       └── model/                # 다운로드된 모델 저장 위치
└── test/
    └── download_model.py         # HF 모델 다운로드 스크립트
```

## 환경 변수

| 변수명 | 설명 | 기본값 |
|--------|------|--------|
| `EMBEDDING_MODEL_PATH` | 로컬 모델 경로 | `src/resources/model/embedding_qwen3_0_6b` |
| `EMBEDDING_PORT` | 서버 포트 | `6002` |

## 실행

### 로컬 실행

```bash
# 모델 다운로드 (최초 1회)
python test/download_model.py

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

서버 상태를 반환합니다.

### `POST /embed`

텍스트 리스트를 임베딩 벡터로 변환합니다.

**Request:**
```json
{
  "texts": ["안녕하세요", "텍스트 임베딩 테스트입니다"],
  "max_length": 512
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
    "inference_time": 0.0234
  },
  "timestamp": {
    "request_kst": "2025-01-01 12:00:00.000",
    "response_kst": "2025-01-01 12:00:00.023",
    "total_elapsed_sec": 0.0234
  }
}
```
