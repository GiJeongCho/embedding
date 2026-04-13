"""Qwen3 Embedding 서비스 레이어.

모델 로딩과 임베딩 추론 로직을 캡슐화합니다.
"""

import os
import time
import logging
from typing import List

import numpy as np
import torch

from src.v1.utils.model_loader import load_model, DEFAULT_MODEL_PATH

logger = logging.getLogger(__name__)


def _mean_pooling(outputs, attention_mask):
    """last_hidden_state에 attention_mask 기반 평균 풀링을 적용합니다."""
    token_embeddings = outputs.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask_expanded, dim=1)
    counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return summed / counts


class EmbeddingService:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = None
        self.model_path = os.getenv("EMBEDDING_MODEL_PATH", DEFAULT_MODEL_PATH)

    def load(self):
        """로컬 임베딩 모델을 GPU/CPU에 로드합니다."""
        self.tokenizer, self.model, self.device = load_model(self.model_path)

    def _check_ready(self):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("임베딩 모델이 로드되지 않았습니다.")

    def _embed_batch(
        self,
        texts: List[str],
        max_length: int,
        normalize: bool,
    ) -> np.ndarray:
        """단일 배치를 임베딩합니다."""
        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding="longest",
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = _mean_pooling(outputs, inputs["attention_mask"])

        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy().astype("float32")

    def embed(
        self,
        texts: List[str],
        max_length: int = 512,
        normalize: bool = True,
        batch_size: int = 32,
    ) -> dict:
        """텍스트 리스트를 임베딩 벡터로 변환합니다."""
        self._check_ready()

        t0 = time.time()
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_emb = self._embed_batch(batch, max_length, normalize)
            all_embeddings.append(batch_emb)

        embeddings = np.concatenate(all_embeddings, axis=0)
        elapsed = time.time() - t0

        return {
            "embeddings": embeddings.tolist(),
            "dimension": embeddings.shape[1],
            "count": len(texts),
            "usage": {
                "total_texts": len(texts),
                "inference_time": round(elapsed, 4),
                "batch_size": batch_size,
                "num_batches": len(all_embeddings),
            },
        }

    def get_status(self) -> dict:
        return {
            "model_loaded": self.model is not None,
            "model_path": self.model_path,
            "device": str(self.device) if self.device else None,
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
