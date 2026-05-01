import json
from pathlib import Path
from typing import Any

from .paths import ROOT_DIR


DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CONFIG_PATH = DATA_DIR / "config.json"
STATE_PATH = DATA_DIR / "state.json"
RAW_TOPICS_PATH = RAW_DIR / "topics.jsonl"
SEEN_TOPIC_IDS_PATH = RAW_DIR / "seen_topic_ids.json"


DEFAULT_CONFIG = {
    "authorization": "",
    "user_agent": "",
    "group_id": "",
    "topics_url": "",
    "scope": "",
    "crawl_mode": "after_baseline",
    "window_start_time": "",
    "window_end_time": "",
    "max_new_topics_per_run": 50,
    "auto_export": True,
}

DEFAULT_STATE = {
    "baseline_time": "",
    "progress_time": "",
    "current_page_range": "",
    "run_mode": "",
    "run_limit": 0,
    "run_got": 0,
    "phase": "idle",
    "phase_detail": "",
    "_internal": {
        "history_complete": False,
        "last_run_at": "",
        "last_success_at": "",
        "latest_topic_id": "",
        "latest_topic_time": "",
        "export_summary": {},
        "pages_scanned": 0,
        "last_page_new_topics": 0,
        "last_error": "",
    },
}


def ensure_data_dirs() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    RAW_DIR.mkdir(exist_ok=True)


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def load_config() -> dict[str, Any]:
    ensure_data_dirs()
    config = DEFAULT_CONFIG.copy()
    config.update(_load_json(CONFIG_PATH, {}))
    return config


def save_config(config: dict[str, Any]) -> dict[str, Any]:
    ensure_data_dirs()
    merged = DEFAULT_CONFIG.copy()
    merged.update(config)
    _write_json(CONFIG_PATH, merged)
    return merged


def load_state() -> dict[str, Any]:
    ensure_data_dirs()
    state = DEFAULT_STATE.copy()
    state.update(_load_json(STATE_PATH, {}))
    internal = DEFAULT_STATE["_internal"].copy()
    internal.update(state.get("_internal", {}))
    state["_internal"] = internal
    return state


def save_state(state: dict[str, Any]) -> dict[str, Any]:
    ensure_data_dirs()
    merged = DEFAULT_STATE.copy()
    merged.update(state)
    internal = DEFAULT_STATE["_internal"].copy()
    internal.update(merged.get("_internal", {}))
    merged["_internal"] = internal
    _write_json(STATE_PATH, merged)
    return merged


def load_seen_topic_ids() -> set[str]:
    ensure_data_dirs()
    return set(_load_json(SEEN_TOPIC_IDS_PATH, []))


def save_seen_topic_ids(topic_ids: set[str]) -> None:
    ensure_data_dirs()
    _write_json(SEEN_TOPIC_IDS_PATH, sorted(topic_ids))


def infer_existing_latest_publish_time() -> str:
    ensure_data_dirs()
    patterns = [
        "processed_data_*.json",
        "output/processed_data_*.json",
        "output/zsxq_processed_data*.json",
    ]
    latest = ""
    for pattern in patterns:
        for path in ROOT_DIR.glob(pattern):
            try:
                data = _load_json(path, [])
            except Exception:
                continue
            if not isinstance(data, list):
                continue
            for item in data:
                if not isinstance(item, dict):
                    continue
                publish_time = item.get("publish_time")
                if not publish_time or publish_time == "Unknown":
                    continue
                if publish_time > latest:
                    latest = publish_time
    return latest
