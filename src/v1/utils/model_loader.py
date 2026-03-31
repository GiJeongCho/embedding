"""Qwen3 Embedding 로컬 모델 로드 유틸리티.

서비스에서 import하여 사용:
    from src.v1.utils.model_loader import load_model, DEFAULT_MODEL_PATH
"""

import os
import time
import logging
import threading
from typing import Tuple

import torch
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.abspath(os.path.join(
    _UTILS_DIR, "..", "..", "resources", "model", "embedding_qwen3_0_6b",
))

_load_lock = threading.Lock()


def load_model(
    model_path: str = DEFAULT_MODEL_PATH,
) -> Tuple[AutoTokenizer, AutoModel, torch.device]:
    """로컬에 저장된 임베딩 모델과 토크나이저를 GPU/CPU에 로드합니다."""
    with _load_lock:
        logger.info("임베딩 모델 로딩 중: %s", model_path)

        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"모델 디렉토리가 존재하지 않습니다: {model_path}")

        t0 = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        ).to(device)
        model.eval()

        elapsed = time.time() - t0
        logger.info("임베딩 모델 로딩 완료 (%.1fs) | device: %s", elapsed, device)
        return tokenizer, model, device
