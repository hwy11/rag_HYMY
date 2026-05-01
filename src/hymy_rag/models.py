from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CleanQuote:
    id: str
    source_id: str
    date: str
    content: str
    trigger: str | None = None
    type_origin: str = "reply"
    source_question: str = ""
    raw_time: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "date": self.date,
            "content": self.content,
            "trigger": self.trigger,
            "type_origin": self.type_origin,
            "source_question": self.source_question,
            "raw_time": self.raw_time,
        }


@dataclass
class TaggedQuote:
    id: str
    source_id: str
    date: str = "unknown"
    type_origin: str = "reply"
    source_question: str = ""
    content: str = ""
    domains: list[str] = field(default_factory=list)
    type: str = "unknown"
    time_sensitivity: str = "unknown"
    confidence: str = "unknown"
    key_concepts: list[str] = field(default_factory=list)
    one_line_summary: str = ""
    context_hint: str = ""

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "TaggedQuote":
        content = str(row.get("content") or "").strip()
        if not content:
            raise ValueError("tagged quote content is empty")
        return cls(
            id=str(row.get("id") or row.get("source_id") or ""),
            source_id=str(row.get("source_id") or row.get("id") or ""),
            date=str(row.get("date") or "unknown"),
            type_origin=str(row.get("type_origin") or "reply"),
            source_question=str(row.get("source_question") or ""),
            content=content,
            domains=_as_str_list(row.get("domains")),
            type=str(row.get("type") or "unknown"),
            time_sensitivity=str(row.get("time_sensitivity") or "unknown"),
            confidence=str(row.get("confidence") or "unknown"),
            key_concepts=_as_str_list(row.get("key_concepts")),
            one_line_summary=str(row.get("one_line_summary") or ""),
            context_hint=str(row.get("context_hint") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "source_id": self.source_id,
            "date": self.date,
            "type_origin": self.type_origin,
            "source_question": self.source_question,
            "content": self.content,
            "domains": self.domains,
            "type": self.type,
        }
        if self.time_sensitivity and self.time_sensitivity != "unknown":
            payload["time_sensitivity"] = self.time_sensitivity
        if self.confidence and self.confidence != "unknown":
            payload["confidence"] = self.confidence
        if self.key_concepts:
            payload["key_concepts"] = self.key_concepts
        if self.one_line_summary:
            payload["one_line_summary"] = self.one_line_summary
        if self.context_hint:
            payload["context_hint"] = self.context_hint
        return payload


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []
