from __future__ import annotations

import os
from pathlib import Path

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


ROOT = Path(__file__).resolve().parents[2]

RETRIEVAL_BACKEND = os.getenv("HYMY_RETRIEVAL_BACKEND", "vector")
VECTOR_DB_PATH = ROOT / "data" / "qdrant_db"
EMBEDDING_MODEL = os.getenv("HYMY_EMBEDDING_MODEL", "BAAI/bge-m3")
RERANKER_MODEL = os.getenv("HYMY_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RECALL_TOP_K = int(os.getenv("HYMY_RECALL_TOP_K", "30"))
FINAL_TOP_K = int(os.getenv("HYMY_FINAL_TOP_K", "8"))
DENSE_WEIGHT = float(os.getenv("HYMY_DENSE_WEIGHT", "0.7"))
SPARSE_WEIGHT = float(os.getenv("HYMY_SPARSE_WEIGHT", "0.3"))
# 问答检索：原问题 7，原问题+回答 3；无原问题的原创帖在召回/精排阶段额外加权。
TRIGGER_FIELD_WEIGHT = float(os.getenv("HYMY_TRIGGER_FIELD_WEIGHT", "0.7"))
TRIGGER_CONTENT_FIELD_WEIGHT = float(os.getenv("HYMY_TRIGGER_CONTENT_FIELD_WEIGHT", "0.3"))
ORIGINAL_POST_BOOST = float(os.getenv("HYMY_ORIGINAL_POST_BOOST", "1.25"))
COLLECTION_NAME = "hymy_quotes"
HF_CACHE_DIR = Path.home() / ".cache" / "huggingface"
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
LOCAL_MODEL_ROOT = ROOT / "models"


def _detect_device() -> str:
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _default_batch_size(device: str) -> int:
    if device == "cuda":
        return 32
    if device == "mps":
        return 8
    return 8


DEVICE = _detect_device()
ENCODE_BATCH_SIZE = int(os.getenv("HYMY_ENCODE_BATCH_SIZE", str(_default_batch_size(DEVICE))))
