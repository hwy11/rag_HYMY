from __future__ import annotations

import json
from pathlib import Path

from .io import read_jsonl


def build_status_report(
    raw_dir: Path,
    processed_path: Path,
    tagged_path: Path,
    index_path: Path,
    persona_dir: Path,
    distill_dir: Path,
) -> str:
    raw_files = sorted(path for path in raw_dir.rglob("*.json") if path.is_file()) if raw_dir.exists() else []
    clean_count = len(read_jsonl(processed_path))
    tagged_count = len(read_jsonl(tagged_path))
    persona_files = sorted(path for path in persona_dir.glob("*.md") if path.is_file()) if persona_dir.exists() else []
    distill_files = sorted(path for path in distill_dir.glob("*.md") if path.is_file()) if distill_dir.exists() else []
    index_count = _index_doc_count(index_path)

    lines = [
        f"原始 JSON 文件：{len(raw_files)}",
        f"清洗后语录：{clean_count}",
        f"已打标语录：{tagged_count}",
        f"本地索引条目：{index_count}",
        f"领域蒸馏语料：{len(distill_files)}",
        f"Persona 文档：{len(persona_files)}",
        "",
        "下一步建议：",
        _next_step(raw_files, clean_count, tagged_count, index_count, distill_files, persona_files),
    ]
    return "\n".join(lines)


def _index_doc_count(index_path: Path) -> int:
    if not index_path.exists():
        return 0
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("docs", "records", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return len(value)
    if isinstance(payload, list):
        return len(payload)
    return 0


def _next_step(
    raw_files: list[Path],
    clean_count: int,
    tagged_count: int,
    index_count: int,
    distill_files: list[Path],
    persona_files: list[Path],
) -> str:
    if not raw_files:
        return "先把原始 JSON 放进 data/raw/，然后运行 ingest。"
    if clean_count == 0:
        return "先运行 ingest，把可用回答清洗进 data/processed/quotes_clean.jsonl。"
    if tagged_count == 0:
        return "运行 make-tag-batches，把批次发给模型打标，再用 import-tagged 导回。"
    if index_count == 0:
        return "运行 build-index，先把检索链路建起来。"
    if not distill_files:
        return "运行 prepare-persona，把长期资产语录整理成领域蒸馏语料。"
    if not persona_files:
        return "请先手动蒸馏 persona/meta_thinking.md，参考 prompts/distill_master_prompt.md。"
    return "现在可以直接 ask，或者继续迭代 persona 文档。"
