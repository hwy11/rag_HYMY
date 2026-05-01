from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path

from .io import read_jsonl


SPECIAL_TOKEN_RE = re.compile(
    r"(?:\[[^\]\n]{1,8}\]|[A-Za-z0-9_]*[🐂🐶🐔📕➕🆗][A-Za-z0-9_🐂🐶🐔📕➕🆗]*|top\d+|[A-Za-z]+TV)",
    re.IGNORECASE,
)
SEED_TERMS = [
    "两个字",
    "星球",
    "收米",
    "带专",
    "老登",
    "牛回",
    "赛道一",
    "韩信",
    "顶真",
    "遥遥领先",
    "魔",
    "ip",
    "top2",
    "情感TV",
    "情感tv",
    "带专🐶",
    "二本🐶",
    "🐂回",
    "老油子",
    "388888",
    "985",
    "211",
    "996",
]


def write_slang_candidates(clean_path: Path, output_path: Path, min_count: int = 5) -> int:
    rows = read_jsonl(clean_path)
    texts = [str(row.get("content") or "") for row in rows if str(row.get("content") or "").strip()]
    counter: Counter[str] = Counter()
    contexts: dict[str, list[str]] = defaultdict(list)
    for text in texts:
        seen_in_row: set[str] = set()
        for token in SPECIAL_TOKEN_RE.findall(text):
            token = token.strip()
            if not token:
                continue
            seen_in_row.add(token)
        for term in SEED_TERMS:
            if term.lower() == "ip":
                found = re.search(r"\bip\b", text, re.IGNORECASE)
                if found:
                    seen_in_row.add("ip")
            elif term in text:
                seen_in_row.add(term)
        for token in seen_in_row:
            if _should_skip_token(token):
                continue
            counter[token] += 1
            if len(contexts[token]) < 5:
                contexts[token].append(_context_snippet(text, token))
    candidates = sorted((token, count) for token, count in counter.items() if count > min_count)
    lines = [
        "| 词 | 出现次数 | 5个上下文示例 | 我（codex）的猜测含义 |",
        "| --- | ---: | --- | --- |",
    ]
    for token, count in candidates:
        examples = "<br>".join(contexts[token][:5])
        lines.append(f"| {token} | {count} | {examples} | {_guess_meaning(token)} |")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(candidates)


def _context_snippet(text: str, token: str) -> str:
    flattened = text.replace("\n", " / ")
    idx = flattened.lower().find(token.lower())
    if idx < 0:
        return flattened[:80]
    start = max(0, idx - 30)
    end = min(len(flattened), idx + len(token) + 30)
    return flattened[start:end]


def _guess_meaning(token: str) -> str:
    if token in {"星球", "➕", "没有➕星球?[呲牙]"}:
        return "可能和付费社群/加入圈层有关"
    if token in {"收米", "388888"}:
        return "可能和收费、报价、金额表达有关"
    if any(mark in token for mark in ("🐶", "🐂", "🐔")):
        return "可能是自嘲、讽刺或圈内标签"
    if token.lower() in {"ip", "top2", "tv"} or "top" in token.lower():
        return "可能是平台术语、等级缩写或圈内简称"
    if token in {"两个字", "韩信", "赛道一"}:
        return "可能是固定话头、暗号式表达或内部梗"
    return "可能是口头禅、圈内梗或半公开暗语"


def _should_skip_token(token: str) -> bool:
    if token.isdigit():
        return True
    if len(token) == 1 and re.fullmatch(r"[一-龥]", token):
        return True
    return False
