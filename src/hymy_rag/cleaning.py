from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .io import read_json
from .models import CleanQuote


DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})")
PUNCT_RE = re.compile(r"[\W_]+", re.UNICODE)
SYMBOL_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)
PURE_NUMBER_RE = re.compile(r"^\d+$")
SHORT_FILLER_BLACKLIST = {
    "插",
    "幸福",
    "真离谱",
    "怎么会事？",
    "🐂回，速归！",
    "[呲牙]",
}
FILTER_LEVELS = {"strict", "loose", "none"}


@dataclass
class IngestReport:
    total_files: int = 0
    successful_files: int = 0
    raw_rows: int = 0
    skipped_files: list[tuple[Path, str]] = field(default_factory=list)
    quotes: list[CleanQuote] = field(default_factory=list)
    filter_counts: Counter[str] = field(default_factory=Counter)
    filtered_examples: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    dedup_counts: Counter[str] = field(default_factory=Counter)


def load_clean_quotes(
    paths: list[Path],
    keep_empty_answer: bool = False,
    dedup_threshold: float = 0.92,
    filter_level: str = "strict",
) -> list[CleanQuote]:
    report = load_clean_quotes_report(
        paths,
        keep_empty_answer=keep_empty_answer,
        dedup_threshold=dedup_threshold,
        filter_level=filter_level,
    )
    return report.quotes


def load_clean_quotes_report(
    paths: list[Path],
    keep_empty_answer: bool = False,
    dedup_threshold: float = 0.92,
    filter_level: str = "strict",
) -> IngestReport:
    if filter_level not in FILTER_LEVELS:
        raise ValueError(f"unsupported filter_level: {filter_level}")
    resolved_paths = _expand_json_paths(paths)
    quotes: list[CleanQuote] = []
    normalized_quotes: list[str] = []
    seen_exact: dict[str, int] = {}
    candidate_buckets: dict[str, list[int]] = {}
    report = IngestReport(total_files=len(resolved_paths))
    for path in resolved_paths:
        try:
            data = read_json(path)
        except Exception as exc:
            report.skipped_files.append((path, _friendly_reason(exc)))
            continue
        if not isinstance(data, list):
            report.skipped_files.append((path, "文件内容不是 JSON 数组"))
            continue
        report.successful_files += 1
        report.raw_rows += len(data)
        for item in data:
            decision = _clean_item(item, path, keep_empty_answer, filter_level)
            if isinstance(decision, FilteredOut):
                report.filter_counts[decision.reason] += 1
                samples = report.filtered_examples.setdefault(decision.reason, [])
                if len(samples) < 5:
                    samples.append(decision.sample)
                continue
            if decision is None:
                continue
            quote = decision
            fingerprint = _fingerprint(quote.content)
            if fingerprint in seen_exact:
                report.dedup_counts["exact_duplicate"] += 1
                continue
            normalized = _normalized_text(quote.content)
            duplicate_index = _find_near_duplicate(
                existing=normalized_quotes,
                candidate=normalized,
                threshold=dedup_threshold,
                buckets=candidate_buckets,
            )
            if duplicate_index is not None:
                existing = quotes[duplicate_index]
                if len(quote.content) > len(existing.content):
                    quotes[duplicate_index] = quote
                    normalized_quotes[duplicate_index] = normalized
                    del seen_exact[_fingerprint(existing.content)]
                    seen_exact[fingerprint] = duplicate_index
                    _register_candidate_bucket(candidate_buckets, normalized, duplicate_index)
                report.dedup_counts["near_duplicate"] += 1
                continue
            seen_exact[fingerprint] = len(quotes)
            index = len(quotes)
            quotes.append(quote)
            normalized_quotes.append(normalized)
            _register_candidate_bucket(candidate_buckets, normalized, index)
    report.quotes = quotes
    return report


@dataclass
class FilteredOut:
    reason: str
    sample: dict[str, str]


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_item(item: Any, path: Path, keep_empty_answer: bool, filter_level: str) -> CleanQuote | FilteredOut | None:
    if not isinstance(item, dict):
        return None
    source_id = str(item.get("id") or "")
    raw_time = str(item.get("publish_time") or item.get("time") or "")
    date = _extract_date(raw_time)
    raw_question = normalize_text(str(item.get("question") or item.get("content") or ""))
    has_answer = item.get("answer") is not None and normalize_text(str(item.get("answer") or "")) != ""
    raw_answer = normalize_text(str(item.get("answer") or ""))
    if has_answer:
        content = raw_answer
        trigger = raw_question or None
        type_origin = "reply"
    else:
        if not keep_empty_answer and not raw_question:
            return None
        content = raw_question
        trigger = None
        type_origin = "original_post"
    if not content:
        return None
    reason = _filter_reason(content=content, trigger=trigger, filter_level=filter_level)
    if reason:
        return FilteredOut(
            reason=reason,
            sample={
                "id": source_id,
                "type_origin": type_origin,
                "content": content[:120],
                "trigger": (trigger or "")[:120],
            },
        )
    quote_id = f"{path.stem}-{source_id or len(content)}"
    return CleanQuote(
        id=quote_id,
        source_id=source_id,
        date=date,
        content=content,
        trigger=trigger,
        type_origin=type_origin,
        source_question=trigger or "",
        raw_time=raw_time,
    )


def _filter_reason(content: str, trigger: str | None, filter_level: str) -> str | None:
    if filter_level == "none":
        return None
    normalized_content = normalize_text(content)
    normalized_trigger = normalize_text(trigger or "")
    if filter_level == "strict":
        if len(normalized_content) < 8 and len(normalized_trigger) < 8:
            return "too_short_both"
    if _is_pure_digits(normalized_content):
        return "pure_digits"
    if _is_pure_symbol_or_emoji(normalized_content):
        return "pure_symbol_or_emoji"
    if normalized_content in SHORT_FILLER_BLACKLIST:
        return "blacklist_short_filler"
    return None


def _extract_date(raw_time: str) -> str:
    match = DATE_RE.match(raw_time.strip())
    return match.group(1) if match else "unknown"


def _fingerprint(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _expand_json_paths(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path.is_dir():
            candidates = sorted(item for item in path.rglob("*.json") if item.is_file())
        else:
            candidates = [path]
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                expanded.append(candidate)
    return expanded


def _normalized_text(text: str) -> str:
    compact = normalize_text(text)
    compact = compact.replace("，", ",").replace("。", ".").replace("；", ";").replace("：", ":")
    compact = compact.replace("！", "!").replace("？", "?")
    compact = PUNCT_RE.sub("", compact.lower())
    return compact


def _find_near_duplicate(
    existing: list[str],
    candidate: str,
    threshold: float,
    buckets: dict[str, list[int]],
) -> int | None:
    candidate_indices = _candidate_indices(buckets, candidate)
    for index in candidate_indices:
        text = existing[index]
        if not text or not candidate:
            continue
        if not _roughly_same_length(text, candidate):
            continue
        if SequenceMatcher(None, text, candidate).ratio() > threshold:
            return index
    return None


def _candidate_indices(buckets: dict[str, list[int]], candidate: str) -> list[int]:
    keys = _bucket_keys(candidate)
    if not keys:
        return list(range(sum(len(v) for v in buckets.values())))
    indices: set[int] = set()
    for key in keys:
        indices.update(buckets.get(key, []))
    return sorted(indices)


def _register_candidate_bucket(buckets: dict[str, list[int]], text: str, index: int) -> None:
    for key in _bucket_keys(text):
        buckets.setdefault(key, []).append(index)


def _bucket_keys(text: str) -> list[str]:
    if not text:
        return []
    compact = text[:10]
    tail = text[-10:]
    middle_start = max(0, len(text) // 2 - 3)
    middle = text[middle_start : middle_start + 6]
    keys = {
        f"p:{compact}",
        f"p6:{text[:6]}",
        f"s:{tail}",
        f"s6:{text[-6:]}",
        f"m:{middle}",
        f"ps:{text[:4]}:{tail[-4:]}",
    }
    return sorted(keys)


def _roughly_same_length(left: str, right: str) -> bool:
    shorter = max(1, min(len(left), len(right)))
    longer = max(len(left), len(right))
    return longer / shorter <= 1.35


def _is_pure_symbol_or_emoji(text: str) -> bool:
    if not text:
        return False
    if re.search(r"[\u4e00-\u9fffA-Za-z0-9]", text):
        return False
    return bool(SYMBOL_ONLY_RE.fullmatch(text))


def _is_pure_digits(text: str) -> bool:
    return bool(text) and bool(PURE_NUMBER_RE.fullmatch(text))


def _friendly_reason(exc: Exception) -> str:
    if isinstance(exc, UnicodeDecodeError):
        return "文件编码不是 UTF-8"
    if exc.__class__.__name__ == "JSONDecodeError":
        return f"JSON 格式错误：{exc}"
    return str(exc)
