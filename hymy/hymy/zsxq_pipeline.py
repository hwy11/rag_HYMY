from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .io_utils import write_json
from .paths import OUTPUT_DIR, ensure_output_dir
from .pipeline import generate_enrichment
from .storage import RAW_TOPICS_PATH


BATCH_SIZE = 1000
ZSXQ_BATCH_START = 11


def strip_html(value: str | None) -> str:
    if not value:
        return ""
    text = re.sub(r"<[^>]+>", "", value)
    return text.replace("\r\n", "\n").strip()


def format_publish_time(iso_value: str | None) -> str:
    if not iso_value:
        return "Unknown"
    try:
        normalized = iso_value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        return dt.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return iso_value[:16].replace("T", " ")


def topic_to_processed_entry(topic: dict[str, Any]) -> dict[str, Any] | None:
    content = topic.get("talk") or topic.get("question") or topic.get("solution") or topic.get("task") or {}
    text = strip_html(content.get("text"))
    answer = strip_html((topic.get("answer") or {}).get("text"))
    if not text:
        return None
    return {
        "id": str(topic.get("topic_id") or topic.get("id") or ""),
        "publish_time": format_publish_time(topic.get("create_time")),
        "question": text,
        "answer": answer or None,
    }


def load_processed_entries_from_raw() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if not RAW_TOPICS_PATH.exists():
        return entries

    seen: set[str] = set()
    with RAW_TOPICS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            topic = json.loads(line)
            entry = topic_to_processed_entry(topic)
            if not entry or entry["id"] in seen:
                continue
            seen.add(entry["id"])
            entries.append(entry)

    entries.sort(key=lambda item: item["publish_time"], reverse=True)
    return entries


def _chunk_entries(entries: list[dict[str, Any]], size: int = BATCH_SIZE) -> list[list[dict[str, Any]]]:
    return [entries[index : index + size] for index in range(0, len(entries), size)]


def _clear_zsxq_batches() -> None:
    for path in OUTPUT_DIR.glob("processed_data_*.json"):
        stem = path.stem
        if stem.endswith("_enriched"):
            stem = stem[: -len("_enriched")]
        try:
            batch_num = int(stem.split("_")[-1])
        except ValueError:
            continue
        if batch_num >= ZSXQ_BATCH_START:
            path.unlink(missing_ok=True)


def export_zsxq_outputs() -> dict[str, Any]:
    ensure_output_dir()
    entries = load_processed_entries_from_raw()
    chunks = _chunk_entries(entries)
    generated_files: list[str] = []
    _clear_zsxq_batches()

    for offset, chunk in enumerate(chunks):
        batch_index = ZSXQ_BATCH_START + offset
        processed_path = OUTPUT_DIR / f"processed_data_{batch_index}.json"
        enriched_path = OUTPUT_DIR / f"processed_data_{batch_index}_enriched.json"

        write_json(processed_path, chunk, indent=2)
        generated_files.append(str(processed_path))

        enriched_entries = []
        for entry in chunk:
            summary, keywords = generate_enrichment(entry.get("question"))
            enriched_entries.append(
                {
                    **entry,
                    "summary": summary,
                    "keywords": keywords,
                }
            )

        write_json(enriched_path, enriched_entries, indent=4)
        generated_files.append(str(enriched_path))

    return {
        "total_entries": len(entries),
        "batch_count": len(chunks),
        "generated_files": generated_files,
    }


def append_raw_topics(topics: list[dict[str, Any]]) -> None:
    RAW_TOPICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RAW_TOPICS_PATH.open("a", encoding="utf-8") as file:
        for topic in topics:
            file.write(json.dumps(topic, ensure_ascii=False))
            file.write("\n")
