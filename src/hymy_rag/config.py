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
ENCODE_BATCH_SIZE = int(os.getenv("HYMY_ENCODE_BATCH_SIZE", "32"))
COLLECTION_NAME = "hymy_quotes"
HF_CACHE_DIR = Path.home() / ".cache" / "huggingface"
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
LOCAL_MODEL_ROOT = ROOT / "models"

if torch is not None and torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
