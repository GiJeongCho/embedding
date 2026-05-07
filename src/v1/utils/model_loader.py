"""Qwen3 Embedding / Reranker 로컬 모델 로드 유틸리티.

서비스에서 import하여 사용:
    from src.v1.utils.model_loader import load_model, DEFAULT_MODEL_PATH
    from src.v1.utils.model_loader import load_reranker, DEFAULT_RERANKER_PATH
"""

import os
import time
import logging
import threading
from typing import Tuple

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

logger = logging.getLogger(__name__)

_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.abspath(os.path.join(
    _UTILS_DIR, "..", "..", "resources", "model", "embedding_qwen3_0_6b",
))
DEFAULT_RERANKER_PATH = os.path.abspath(os.path.join(
    _UTILS_DIR, "..", "..", "resources", "model", "reranker_qwen3_0_6b",
))

_load_lock = threading.Lock()
_reranker_load_lock = threading.Lock()


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


def load_reranker(
    model_path: str = DEFAULT_RERANKER_PATH,
) -> Tuple[AutoTokenizer, AutoModelForCausalLM, torch.device, int, int]:
    """로컬에 저장된 Qwen3 리랭커 모델을 GPU/CPU에 로드합니다.

    Returns:
        (tokenizer, model, device, token_true_id, token_false_id)
    """
    with _reranker_load_lock:
        logger.info("리랭커 모델 로딩 중: %s", model_path)

        need_files = [
            os.path.join(model_path, "config.json"),
            os.path.join(model_path, "tokenizer.json"),
        ]
        missing = [p for p in need_files if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"리랭커 모델 파일이 누락되었습니다: {missing}")

        t0 = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            padding_side="left",
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        ).to(device)
        model.eval()

        token_true_id = tokenizer.convert_tokens_to_ids("yes")
        token_false_id = tokenizer.convert_tokens_to_ids("no")

        elapsed = time.time() - t0
        logger.info("리랭커 모델 로딩 완료 (%.1fs) | device: %s", elapsed, device)
        return tokenizer, model, device, token_true_id, token_false_id
