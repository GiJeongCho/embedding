"""Qwen3 Embedding / Reranker 모델 다운로드 스크립트.

사용법:
    python test/download_model.py                  # 기본 임베딩 모델(0.6B)만 다운로드
    python test/download_model.py --reranker       # 리랭커 모델만 다운로드
    python test/download_model.py --all            # 임베딩 + 리랭커 모두 다운로드
    python test/download_model.py --hf_token <TOKEN>
"""

import os
from typing import Iterable, Optional

from huggingface_hub import snapshot_download, login

BASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src", "resources", "model"
)

EMBEDDING_MODELS = {
    "Qwen/Qwen3-Embedding-0.6B": "embedding_qwen3_0_6b",
    # "Qwen/Qwen3-Embedding-4B": "embedding_qwen3_4b",
    # "Qwen/Qwen3-Embedding-8B": "embedding_qwen3_8b",
}

RERANKER_MODELS = {
    "Qwen/Qwen3-Reranker-0.6B": {
        "folder": "reranker_qwen3_0_6b",
        "include": [
            "*.safetensors",
            "config.json",
            "generation_config.json",
            "preprocessor_config.json",
            "tokenizer.*",
            "vocab.json",
            "vocab.txt",
            "merges.txt",
            "special_tokens_map.json",
        ],
    },
}


def _hf_download(
    model_id: str,
    local_dir: str,
    *,
    token: Optional[str] = None,
    include: Optional[Iterable[str]] = None,
) -> str:
    if token:
        login(token=token, add_to_git_credential=False)
    os.makedirs(local_dir, exist_ok=True)
    return snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        allow_patterns=list(include) if include else None,
        resume_download=True,
    )


def download_embedding(
    model_ids: Optional[list[str]] = None,
    token: Optional[str] = None,
):
    """임베딩 모델을 다운로드합니다."""
    targets: dict[str, str] = {}
    if model_ids:
        for mid in model_ids:
            if mid in EMBEDDING_MODELS:
                targets[mid] = EMBEDDING_MODELS[mid]
            else:
                print(f"[경고] 알 수 없는 임베딩 모델 ID: {mid}")
    else:
        default_id = "Qwen/Qwen3-Embedding-0.6B"
        targets[default_id] = EMBEDDING_MODELS[default_id]

    os.makedirs(BASE_DIR, exist_ok=True)
    for model_id, folder_name in targets.items():
        save_path = os.path.join(BASE_DIR, folder_name)
        print(f"[Embedding] Downloading {model_id} → {save_path}")
        _hf_download(model_id=model_id, local_dir=save_path, token=token)
        print(f"✅ Completed: {model_id}\n")


def download_reranker(
    model_ids: Optional[list[str]] = None,
    token: Optional[str] = None,
):
    """리랭커 모델을 다운로드합니다."""
    targets: dict[str, dict] = {}
    if model_ids:
        for mid in model_ids:
            if mid in RERANKER_MODELS:
                targets[mid] = RERANKER_MODELS[mid]
            else:
                print(f"[경고] 알 수 없는 리랭커 모델 ID: {mid}")
    else:
        default_id = "Qwen/Qwen3-Reranker-0.6B"
        targets[default_id] = RERANKER_MODELS[default_id]

    os.makedirs(BASE_DIR, exist_ok=True)
    for model_id, plan in targets.items():
        save_path = os.path.join(BASE_DIR, plan["folder"])
        print(f"[Reranker] Downloading {model_id} → {save_path}")
        _hf_download(
            model_id=model_id,
            local_dir=save_path,
            token=token,
            include=plan.get("include"),
        )
        print(f"✅ Completed: {model_id}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="임베딩 + 리랭커 모두 다운로드")
    parser.add_argument("--reranker", action="store_true", help="리랭커 모델만 다운로드")
    parser.add_argument("--embedding", action="store_true", help="임베딩 모델만 다운로드")
    parser.add_argument("--hf_token", default=None, help="Hugging Face 토큰(필요 시)")
    args = parser.parse_args()

    if args.all:
        download_embedding(token=args.hf_token)
        download_reranker(token=args.hf_token)
    elif args.reranker:
        download_reranker(token=args.hf_token)
    elif args.embedding:
        download_embedding(token=args.hf_token)
    else:
        download_embedding(token=args.hf_token)
        print("리랭커도 받으려면: python test/download_model.py --reranker")
        print("전체를 받으려면: python test/download_model.py --all")
