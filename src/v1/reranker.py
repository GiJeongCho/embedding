"""Qwen3 Reranker 서비스 레이어.

리랭커 모델 로딩과 (query, document) 쌍에 대한 적합도 점수 계산을 캡슐화합니다.
"""

import os
import time
import logging
from typing import List, Optional, Tuple

import torch

from src.v1.utils.model_loader import load_reranker, DEFAULT_RERANKER_PATH

logger = logging.getLogger(__name__)

_DEFAULT_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    'Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _format_pair(instruction: str, query: str, doc: str) -> str:
    return (
        f"<Instruct>: {instruction}\n"
        f"<Query>: {query}\n"
        f"<Document>: {doc}"
    )


class RerankerService:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device: Optional[torch.device] = None
        self.token_true_id: Optional[int] = None
        self.token_false_id: Optional[int] = None
        self.model_path = os.getenv("RERANKER_MODEL_PATH", DEFAULT_RERANKER_PATH)

    def load(self):
        """로컬 리랭커 모델을 로드합니다."""
        (
            self.tokenizer,
            self.model,
            self.device,
            self.token_true_id,
            self.token_false_id,
        ) = load_reranker(self.model_path)

    def _check_ready(self):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("리랭커 모델이 로드되지 않았습니다.")

    @torch.no_grad()
    def _compute_scores(
        self,
        pairs: List[str],
        max_length: int,
    ) -> List[float]:
        """리랭크 점수 계산 (yes/no 로짓 기반)."""
        if not pairs:
            return []

        prefix_tokens = self.tokenizer.encode(_PREFIX, add_special_tokens=False)
        suffix_tokens = self.tokenizer.encode(_SUFFIX, add_special_tokens=False)

        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=max_length - len(prefix_tokens) - len(suffix_tokens),
        )

        for i, ele in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = prefix_tokens + ele + suffix_tokens

        inputs = self.tokenizer.pad(
            inputs, padding=True, return_tensors="pt", max_length=max_length
        )
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)

        batch_logits = self.model(**inputs).logits[:, -1, :]
        true_vec = batch_logits[:, self.token_true_id]
        false_vec = batch_logits[:, self.token_false_id]
        stacked = torch.stack([false_vec, true_vec], dim=1)
        log_probs = torch.nn.functional.log_softmax(stacked, dim=1)
        scores = log_probs[:, 1].exp().tolist()
        return scores

    def rerank(
        self,
        query: str,
        documents: List[str],
        instruction: Optional[str] = None,
        top_n: Optional[int] = None,
        batch_size: int = 8,
        max_length: int = 8192,
    ) -> dict:
        """단일 query와 다중 documents 사이의 적합도 점수를 계산하고 정렬합니다."""
        self._check_ready()

        instr = instruction or _DEFAULT_INSTRUCTION
        pairs = [_format_pair(instr, query, doc) for doc in documents]

        t0 = time.time()
        all_scores: List[float] = []
        num_batches = 0
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i : i + batch_size]
            try:
                all_scores.extend(self._compute_scores(chunk, max_length=max_length))
            except Exception as exc:
                logger.warning("리랭크 배치 실패 (idx=%d): %s", i, exc)
                all_scores.extend([0.0] * len(chunk))
            num_batches += 1

        elapsed = time.time() - t0

        ranked: List[Tuple[int, str, float]] = [
            (idx, doc, float(score))
            for idx, (doc, score) in enumerate(zip(documents, all_scores))
        ]
        ranked.sort(key=lambda x: x[2], reverse=True)

        if top_n is not None and top_n > 0:
            ranked = ranked[:top_n]

        results = [
            {"index": idx, "score": score, "document": doc}
            for idx, doc, score in ranked
        ]

        return {
            "results": results,
            "count": len(results),
            "usage": {
                "total_documents": len(documents),
                "inference_time": round(elapsed, 4),
                "batch_size": batch_size,
                "num_batches": num_batches,
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
