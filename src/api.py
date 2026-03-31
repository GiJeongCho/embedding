import time
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.v1.service import EmbeddingService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))

service = EmbeddingService()


def kst_now() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Qwen3 임베딩 모델을 로드합니다...")
    try:
        service.load()
        logger.info("임베딩 모델 로드 완료. API 서비스 준비됨.")
    except Exception as e:
        logger.error("임베딩 모델 로드 실패: %s", e)
    yield


app = FastAPI(
    title="Qwen3 Embedding API",
    description="Qwen3 Embedding 모델 기반 텍스트 임베딩 API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, description="임베딩할 텍스트 리스트")
    max_length: int = Field(512, ge=1, le=8192, description="토크나이저 최대 길이")


@app.get("/health")
def health():
    return {"status": "ok", "model": service.get_status(), "server_time_kst": kst_now()}


@app.post("/embed")
def embed(req: EmbedRequest):
    """텍스트 리스트를 임베딩 벡터로 변환합니다."""
    if service.model is None:
        raise HTTPException(status_code=503, detail="임베딩 모델이 아직 로드되지 않았습니다.")

    request_time = kst_now()
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"[{request_time}] POST /embed")
    print(f"  texts count: {len(req.texts)}")
    print(f"  max_length: {req.max_length}")
    for i, text in enumerate(req.texts[:3]):
        print(f"  text[{i}]: {text[:100]}{'...' if len(text) > 100 else ''}")
    if len(req.texts) > 3:
        print(f"  ... and {len(req.texts) - 3} more")

    try:
        result = service.embed(
            texts=req.texts,
            max_length=req.max_length,
        )
    except Exception as e:
        logger.error("임베딩 실패: %s", e, exc_info=True)
        print(f"[{kst_now()}] ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    response_time = kst_now()
    total_elapsed = round(time.time() - t0, 4)

    response = {
        **result,
        "timestamp": {
            "request_kst": request_time,
            "response_kst": response_time,
            "total_elapsed_sec": total_elapsed,
        },
    }

    print(f"[{response_time}] 응답 완료 ({total_elapsed}s)")
    print(f"  dimension: {result['dimension']}, count: {result['count']}")
    print(f"{'='*60}\n")

    return response


if __name__ == "__main__":
    import os
    port = int(os.getenv("EMBEDDING_PORT", "5000"))
    uvicorn.run("src.api:app", host="0.0.0.0", port=port, reload=False)
