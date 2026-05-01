from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .io import read_jsonl
from .models import TaggedQuote


TOKEN_RE = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]")


def build_index(tagged_path: Path, index_path: Path) -> int:
    rows = read_jsonl(tagged_path)
    docs = [TaggedQuote.from_dict(row) for row in rows]
    doc_tokens = [_tokenize(_search_text(doc)) for doc in docs]
    doc_freq: Counter[str] = Counter()
    for tokens in doc_tokens:
        doc_freq.update(set(tokens))
    total = max(len(docs), 1)
    idf = {token: math.log((1 + total) / (1 + freq)) + 1 for token, freq in doc_freq.items()}
    vectors = [_vectorize(tokens, idf) for tokens in doc_tokens]
    payload = {
        "docs": [doc.to_dict() for doc in docs],
        "idf": idf,
        "vectors": vectors,
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return len(docs)


def search_index(
    index_path: Path,
    query: str,
    top_k: int = 12,
    domain: str | None = None,
    domains: list[str] | None = None,
    quote_type: str | None = None,
    quote_types: list[str] | None = None,
    date_from: str | None = None,
    time_sensitivities: list[str] | None = None,
    preferred_time_sensitivity: str | None = None,
) -> list[dict[str, Any]]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    idf: dict[str, float] = payload["idf"]
    query_vector = _vectorize(_tokenize(query), idf)
    allowed_domains = set(domains or ([] if not domain else [domain]))
    allowed_types = set(quote_types or ([] if not quote_type else [quote_type]))
    allowed_time_sensitivities = set(time_sensitivities or [])
    scored: list[tuple[float, dict[str, Any]]] = []
    for doc, vector in zip(payload["docs"], payload["vectors"]):
        if allowed_domains and not allowed_domains.intersection(doc.get("domains", [])):
            continue
        if allowed_types and doc.get("type") not in allowed_types:
            continue
        if date_from and not _date_at_least(doc.get("date", "unknown"), date_from):
            continue
        if allowed_time_sensitivities and doc.get("time_sensitivity") not in allowed_time_sensitivities:
            continue
        score = _cosine(query_vector, vector)
        score += _time_sensitivity_bonus(doc, preferred_time_sensitivity)
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [{**doc, "score": round(score, 4)} for score, doc in scored[:top_k]]


def _search_text(doc: TaggedQuote) -> str:
    parts = [
        doc.content,
        " ".join(doc.domains),
        doc.source_question,
    ]
    return "\n".join(part for part in parts if part)


def _tokenize(text: str) -> list[str]:
    basic = TOKEN_RE.findall(text.lower())
    compact = "".join(ch for ch in text if "\u4e00" <= ch <= "\u9fff")
    grams: list[str] = []
    for size in (2, 3):
        grams.extend(compact[index : index + size] for index in range(max(len(compact) - size + 1, 0)))
    return basic + grams


def _vectorize(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(token for token in tokens if token in idf)
    if not counts:
        return {}
    total = sum(counts.values())
    vector = {token: (count / total) * idf[token] for token, count in counts.items()}
    norm = math.sqrt(sum(value * value for value in vector.values())) or 1
    return {token: value / norm for token, value in vector.items()}


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(token, 0.0) for token, value in left.items())


def _date_at_least(value: str, threshold: str) -> bool:
    if value == "unknown":
        return False
    normalized = value[:10]
    target = threshold[:10]
    return normalized >= target


def _time_sensitivity_bonus(doc: dict[str, Any], preferred: str | None) -> float:
    if not preferred:
        return 0.0
    return 0.05 if doc.get("time_sensitivity") == preferred else 0.0
