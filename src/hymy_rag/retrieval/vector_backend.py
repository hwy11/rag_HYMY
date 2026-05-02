from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm
from transformers import modeling_utils as transformers_modeling_utils
from transformers.utils import import_utils as transformers_import_utils

from ..config import (
    COLLECTION_NAME,
    DENSE_WEIGHT,
    DEVICE,
    EMBEDDING_MODEL,
    ENCODE_BATCH_SIZE,
    FINAL_TOP_K,
    HF_CACHE_DIR,
    HF_ENDPOINT,
    LOCAL_MODEL_ROOT,
    RECALL_TOP_K,
    RERANKER_MODEL,
    SPARSE_WEIGHT,
    VECTOR_DB_PATH,
)


@dataclass(slots=True)
class SearchFilters:
    domains: list[str] = field(default_factory=list)
    quote_types: list[str] = field(default_factory=list)
    date_from: str | None = None
    time_sensitivities: list[str] = field(default_factory=list)


@dataclass(slots=True)
class VectorBuildReport:
    quote_count: int
    point_count: int
    encode_seconds: float
    total_seconds: float
    points_per_second: float
    disk_usage_bytes: int
    peak_vram_bytes: int
    device: str
    collection_name: str


class VectorBackend:
    def __init__(
        self,
        db_path: Path = VECTOR_DB_PATH,
        embedding_model: str = EMBEDDING_MODEL,
        reranker_model: str = RERANKER_MODEL,
        device: str = DEVICE,
        batch_size: int = ENCODE_BATCH_SIZE,
        recall_top_k: int = RECALL_TOP_K,
        final_top_k: int = FINAL_TOP_K,
        dense_weight: float = DENSE_WEIGHT,
        sparse_weight: float = SPARSE_WEIGHT,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self.db_path = db_path
        self.embedding_model_name = embedding_model
        self.reranker_model_name = reranker_model
        self.device = device
        self.batch_size = batch_size
        self.recall_top_k = recall_top_k
        self.final_top_k = final_top_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.collection_name = collection_name
        self.hf_cache_dir = HF_CACHE_DIR
        self._client: QdrantClient | None = None
        self._embedder: BGEM3FlagModel | None = None
        self._reranker: FlagReranker | None = None

    def rebuild_index(self, rows: list[dict[str, Any]]) -> VectorBuildReport:
        started = time.perf_counter()
        points_to_encode = self._prepare_points(rows)
        quote_count = len(rows)
        point_count = len(points_to_encode)
        self.db_path.mkdir(parents=True, exist_ok=True)
        client = self._client_or_create()
        if client.collection_exists(self.collection_name):
            client.delete_collection(self.collection_name)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        print(f"[vector] backend=device={self.device} quotes={quote_count} points={point_count}")
        encode_started = time.perf_counter()
        encoded_points = self._encode_points(points_to_encode)
        encode_seconds = time.perf_counter() - encode_started
        vector_size = len(encoded_points[0]["dense"])
        client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )
        self._upsert_points(encoded_points)
        total_seconds = time.perf_counter() - started
        disk_usage_bytes = _directory_size(self.db_path)
        peak_vram_bytes = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        points_per_second = point_count / encode_seconds if encode_seconds else 0.0
        print(
            "[vector] build summary "
            f"collection={self.collection_name} points={point_count} "
            f"encode={encode_seconds:.2f}s total={total_seconds:.2f}s "
            f"speed={points_per_second:.2f} points/s peak_accel_mem={_format_bytes(peak_vram_bytes)} "
            f"disk={_format_bytes(disk_usage_bytes)}"
        )
        return VectorBuildReport(
            quote_count=quote_count,
            point_count=point_count,
            encode_seconds=encode_seconds,
            total_seconds=total_seconds,
            points_per_second=points_per_second,
            disk_usage_bytes=disk_usage_bytes,
            peak_vram_bytes=peak_vram_bytes,
            device=self.device,
            collection_name=self.collection_name,
        )

    def search(
        self,
        query: str,
        top_k: int = FINAL_TOP_K,
        filters: SearchFilters | None = None,
    ) -> list[dict[str, Any]]:
        filters = filters or SearchFilters()
        dense_query, sparse_query = self._encode_query(query)
        client = self._client_or_create()
        dense_hits = client.query_points(
            collection_name=self.collection_name,
            query=dense_query,
            using="dense",
            limit=max(50, self.recall_top_k),
            with_payload=True,
        ).points
        sparse_hits = []
        if sparse_query.indices:
            sparse_hits = client.query_points(
                collection_name=self.collection_name,
                query=sparse_query,
                using="sparse",
                limit=max(50, self.recall_top_k),
                with_payload=True,
            ).points

        fused = self._fuse_hits(dense_hits, sparse_hits)
        deduped = self._dedupe_by_quote(fused)[: max(self.recall_top_k, top_k)]
        reranked = self._rerank(query, deduped)
        filtered = [row for row in reranked if _matches_filters(row, filters)]
        return filtered[:top_k]

    def _prepare_points(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prepared: list[dict[str, Any]] = []
        next_point_id = 1
        for row in rows:
            content = str(row.get("content") or "").strip()
            if not content:
                continue
            quote_id = str(row.get("id") or row.get("source_id") or "")
            trigger = str(row.get("trigger") or row.get("source_question") or "").strip()
            payload = dict(row)
            payload["quote_id"] = quote_id
            payload["trigger"] = trigger
            payload["source_question"] = str(row.get("source_question") or trigger)
            prepared.append(
                {
                    "id": next_point_id,
                    "text": content,
                    "payload": {**payload, "field": "content", "point_key": f"{quote_id}::content"},
                }
            )
            next_point_id += 1
            if trigger:
                prepared.append(
                    {
                        "id": next_point_id,
                        "text": f"{trigger} [SEP] {content}",
                        "payload": {
                            **payload,
                            "field": "trigger_content",
                            "point_key": f"{quote_id}::trigger_content",
                        },
                    }
                )
                next_point_id += 1
        return prepared

    def _encode_points(self, points_to_encode: list[dict[str, Any]]) -> list[dict[str, Any]]:
        embedder = self._embedder_or_create()
        encoded_points: list[dict[str, Any]] = []
        progress = tqdm(
            range(0, len(points_to_encode), self.batch_size),
            desc="Encoding BGE-M3",
            unit="batch",
        )
        for start in progress:
            batch = points_to_encode[start : start + self.batch_size]
            result = embedder.encode(
                [item["text"] for item in batch],
                batch_size=self.batch_size,
                return_dense=True,
                return_sparse=True,
            )
            dense_vecs = result["dense_vecs"]
            sparse_vecs = result["lexical_weights"]
            for item, dense, sparse in zip(batch, dense_vecs, sparse_vecs):
                encoded_points.append(
                    {
                        "id": item["id"],
                        "payload": item["payload"],
                        "dense": dense.tolist(),
                        "sparse": _lexical_weights_to_sparse_vector(sparse),
                    }
                )
        return encoded_points

    def _upsert_points(self, encoded_points: list[dict[str, Any]]) -> None:
        client = self._client_or_create()
        progress = tqdm(
            range(0, len(encoded_points), self.batch_size),
            desc="Upserting Qdrant",
            unit="batch",
        )
        for start in progress:
            batch = encoded_points[start : start + self.batch_size]
            client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=item["id"],
                        vector={"dense": item["dense"], "sparse": item["sparse"]},
                        payload=item["payload"],
                    )
                    for item in batch
                ],
            )

    def _encode_query(self, query: str) -> tuple[list[float], models.SparseVector]:
        embedder = self._embedder_or_create()
        result = embedder.encode(
            [query],
            batch_size=1,
            return_dense=True,
            return_sparse=True,
        )
        dense_query = result["dense_vecs"][0].tolist()
        sparse_query = _lexical_weights_to_sparse_vector(result["lexical_weights"][0])
        return dense_query, sparse_query

    def _fuse_hits(
        self,
        dense_hits: list[models.ScoredPoint],
        sparse_hits: list[models.ScoredPoint],
    ) -> list[dict[str, Any]]:
        dense_scores = _normalize_scores(dense_hits)
        sparse_scores = _normalize_scores(sparse_hits)
        merged: dict[str, dict[str, Any]] = {}
        for point in dense_hits:
            merged[str(point.id)] = {
                "point": point,
                "dense_score": dense_scores.get(str(point.id), 0.0),
                "sparse_score": 0.0,
            }
        for point in sparse_hits:
            entry = merged.setdefault(
                str(point.id),
                {"point": point, "dense_score": 0.0, "sparse_score": 0.0},
            )
            entry["point"] = point
            entry["sparse_score"] = sparse_scores.get(str(point.id), 0.0)

        fused_rows: list[dict[str, Any]] = []
        for point_id, entry in merged.items():
            point = entry["point"]
            recall_score = (
                self.dense_weight * entry["dense_score"] + self.sparse_weight * entry["sparse_score"]
            )
            payload = dict(point.payload or {})
            payload["point_id"] = point_id
            payload["recall_score"] = round(recall_score, 4)
            payload["dense_score"] = round(entry["dense_score"], 4)
            payload["sparse_score"] = round(entry["sparse_score"], 4)
            fused_rows.append(payload)
        fused_rows.sort(key=lambda row: row["recall_score"], reverse=True)
        return fused_rows[: max(self.recall_top_k, self.final_top_k)]

    def _dedupe_by_quote(self, fused_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: dict[str, dict[str, Any]] = {}
        for row in fused_rows:
            quote_id = str(row.get("quote_id") or row.get("id") or row.get("source_id") or "")
            current = deduped.get(quote_id)
            if current is None or row["recall_score"] > current["recall_score"]:
                deduped[quote_id] = row
        rows = list(deduped.values())
        rows.sort(key=lambda row: row["recall_score"], reverse=True)
        return rows

    def _rerank(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not candidates:
            return []
        reranker = self._reranker_or_create()
        pairs = [(query, str(row.get("content") or "")) for row in candidates]
        rerank_scores = reranker.compute_score(pairs)
        if not isinstance(rerank_scores, list):
            rerank_scores = [rerank_scores]
        reranked: list[dict[str, Any]] = []
        for row, score in zip(candidates, rerank_scores):
            reranked.append(
                {
                    **row,
                    "score": round(float(row["recall_score"]), 4),
                    "rerank_score": round(float(score), 4),
                }
            )
        reranked.sort(key=lambda row: row["rerank_score"], reverse=True)
        return reranked

    def _client_or_create(self) -> QdrantClient:
        if self._client is None:
            self.db_path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(self.db_path))
        return self._client

    def _embedder_or_create(self) -> BGEM3FlagModel:
        if self._embedder is None:
            self._prepare_hf_environment()
            model_source = self._resolve_model_source(self.embedding_model_name)
            print(f"[vector] loading embedder {model_source} on {self.device}")
            self._embedder = BGEM3FlagModel(
                str(model_source),
                use_fp16=self.device == "cuda",
                devices=self.device,
                cache_dir=str(self.hf_cache_dir),
                batch_size=self.batch_size,
                return_dense=True,
                return_sparse=True,
            )
            print(
                f"[vector] embedder ready cache={self.hf_cache_dir} "
                f"model_size={_format_bytes(_model_cache_size(model_source, self.hf_cache_dir))} "
                f"accel_mem={_format_bytes(_accelerator_memory_allocated())}"
            )
        return self._embedder

    def _reranker_or_create(self) -> FlagReranker:
        if self._reranker is None:
            self._prepare_hf_environment()
            model_source = self._resolve_model_source(self.reranker_model_name)
            print(f"[vector] loading reranker {model_source} on {self.device}")
            self._reranker = FlagReranker(
                str(model_source),
                use_fp16=self.device == "cuda",
                devices=self.device,
                cache_dir=str(self.hf_cache_dir),
                batch_size=max(self.batch_size, 64),
            )
            print(
                f"[vector] reranker ready cache={self.hf_cache_dir} "
                f"model_size={_format_bytes(_model_cache_size(model_source, self.hf_cache_dir))} "
                f"accel_mem={_format_bytes(_accelerator_memory_allocated())}"
            )
        return self._reranker

    def _prepare_hf_environment(self) -> None:
        os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        self.hf_cache_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(transformers_import_utils, "check_torch_load_is_safe"):
            # We only load trusted local Hugging Face checkpoints here.
            transformers_import_utils.check_torch_load_is_safe = lambda: None
        if hasattr(transformers_modeling_utils, "check_torch_load_is_safe"):
            transformers_modeling_utils.check_torch_load_is_safe = lambda: None

    def _resolve_model_source(self, model_name: str) -> Path | str:
        local_path = LOCAL_MODEL_ROOT / model_name.split("/")[-1]
        if local_path.exists():
            return local_path
        return model_name


def _normalize_scores(points: list[models.ScoredPoint]) -> dict[str, float]:
    if not points:
        return {}
    max_score = max(point.score for point in points) or 1.0
    return {str(point.id): point.score / max_score for point in points}


def _lexical_weights_to_sparse_vector(weights: dict[str, float]) -> models.SparseVector:
    items: list[tuple[int, float]] = []
    for key, value in (weights or {}).items():
        if value == 0:
            continue
        items.append((_sparse_index_from_key(key), float(value)))
    items.sort(key=lambda item: item[0])
    return models.SparseVector(
        indices=[index for index, _ in items],
        values=[score for _, score in items],
    )


def _sparse_index_from_key(key: str | int) -> int:
    if isinstance(key, int):
        return key
    text = str(key).strip()
    try:
        return int(text)
    except ValueError:
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "little", signed=False)


def _model_cache_size(model_name: Path | str, cache_dir: Path) -> int:
    model_path = Path(model_name)
    if model_path.exists():
        return _directory_size(model_path)
    hub_dir = cache_dir / "hub" / f"models--{model_name.replace('/', '--')}"
    return _directory_size(hub_dir)


def _accelerator_memory_allocated() -> int:
    if torch.cuda.is_available():
        return int(torch.cuda.memory_allocated())
    mps = getattr(torch, "mps", None)
    if mps is not None and hasattr(mps, "current_allocated_memory"):
        try:
            return int(mps.current_allocated_memory())
        except RuntimeError:
            return 0
    return 0


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def _format_bytes(size: int) -> str:
    if size <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{size} B"


def _matches_filters(row: dict[str, Any], filters: SearchFilters) -> bool:
    if filters.domains and not set(filters.domains).intersection(row.get("domains", [])):
        return False
    if filters.quote_types and row.get("type") not in set(filters.quote_types):
        return False
    if filters.date_from and not _date_at_least(str(row.get("date", "unknown")), filters.date_from):
        return False
    if filters.time_sensitivities and row.get("time_sensitivity") not in set(filters.time_sensitivities):
        return False
    return True


def _date_at_least(value: str, threshold: str) -> bool:
    if value == "unknown":
        return False
    return value[:10] >= threshold[:10]
