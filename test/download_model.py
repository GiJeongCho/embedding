"""Qwen3 Embedding 모델 다운로드 스크립트.

사용법:
    python test/download_model.py
"""

import os
from huggingface_hub import snapshot_download

BASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src", "resources", "model"
)

MODELS = {
    "Qwen/Qwen3-Embedding-0.6B": "embedding_qwen3_0_6b",
    # "Qwen/Qwen3-Embedding-4B": "embedding_qwen3_4b",
    # "Qwen/Qwen3-Embedding-8B": "embedding_qwen3_8b",
}


def download_models(model_ids: list[str] | None = None):
    """
    지정된 모델을 다운로드합니다.
    model_ids가 None이면 기본 모델(0.6B)만 다운로드합니다.
    """
    targets = {}
    if model_ids:
        for mid in model_ids:
            if mid in MODELS:
                targets[mid] = MODELS[mid]
            else:
                print(f"[경고] 알 수 없는 모델 ID: {mid}")
    else:
        default_id = "Qwen/Qwen3-Embedding-0.6B"
        targets[default_id] = MODELS[default_id]

    os.makedirs(BASE_DIR, exist_ok=True)

    for model_id, folder_name in targets.items():
        save_path = os.path.join(BASE_DIR, folder_name)
        print(f"Downloading {model_id} → {save_path}")
        snapshot_download(
            repo_id=model_id,
            local_dir=save_path,
            local_dir_use_symlinks=False,
        )
        print(f"✅ Completed: {model_id}\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        download_models(list(MODELS.keys()))
    else:
        download_models()
        print("전체 모델을 다운로드하려면: python test/download_model.py --all")
