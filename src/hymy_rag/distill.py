from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .io import read_jsonl
from .models import TaggedQuote


CORE_TYPES = {"思维方式", "价值观", "方法论"}


def export_domain_corpora(tagged_path: Path, output_dir: Path) -> int:
    rows = read_jsonl(tagged_path)
    quotes = [TaggedQuote.from_dict(row) for row in rows]
    grouped: dict[str, list[TaggedQuote]] = defaultdict(list)
    for quote in quotes:
        if quote.type not in CORE_TYPES:
            continue
        domains = quote.domains or ["未分类"]
        for domain in domains:
            grouped[domain].append(quote)

    output_dir.mkdir(parents=True, exist_ok=True)
    for domain, items in grouped.items():
        target = output_dir / f"{_slugify(domain)}.md"
        target.write_text(_render_domain_markdown(domain, items), encoding="utf-8")
    return len(grouped)


def export_master_corpus(tagged_path: Path, output_path: Path) -> int:
    rows = read_jsonl(tagged_path)
    quotes = [TaggedQuote.from_dict(row) for row in rows]
    core_quotes = [quote for quote in quotes if quote.type in CORE_TYPES]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render_master_markdown(core_quotes), encoding="utf-8")
    return len(core_quotes)


def _render_domain_markdown(domain: str, quotes: list[TaggedQuote]) -> str:
    lines = [
        f"# {domain} 领域语料",
        "",
        "以下内容已经筛选为更适合蒸馏思维 DNA 的长期资产语录。",
        "",
    ]
    for index, quote in enumerate(_sort_quotes(quotes), 1):
        meta = " | ".join(
            [
                quote.date or "unknown",
                quote.type or "unknown",
                f"置信:{quote.confidence or 'unknown'}",
            ]
        )
        lines.extend(
            [
                f"## {index}. {meta}",
                quote.content,
            ]
        )
        if quote.one_line_summary:
            lines.append(f"摘要：{quote.one_line_summary}")
        if quote.key_concepts:
            lines.append(f"关键词：{'、'.join(quote.key_concepts)}")
        if quote.context_hint:
            lines.append(f"上下文：{quote.context_hint}")
        if quote.source_question:
            lines.append(f"原问题：{quote.source_question}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _render_master_markdown(quotes: list[TaggedQuote]) -> str:
    lines = [
        "# 元思维总语料",
        "",
        "以下内容来自所有领域中被判定为思维方式、价值观、方法论的语录，可用于总纲蒸馏。",
        "",
    ]
    for index, quote in enumerate(_sort_quotes(quotes), 1):
        domains = "、".join(quote.domains or ["未分类"])
        meta = " | ".join(
            [
                quote.date or "unknown",
                domains,
                quote.type or "unknown",
                f"置信:{quote.confidence or 'unknown'}",
            ]
        )
        lines.extend([f"## {index}. {meta}", quote.content])
        if quote.one_line_summary:
            lines.append(f"摘要：{quote.one_line_summary}")
        if quote.key_concepts:
            lines.append(f"关键词：{'、'.join(quote.key_concepts)}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _sort_quotes(quotes: list[TaggedQuote]) -> list[TaggedQuote]:
    return sorted(
        quotes,
        key=lambda quote: (
            quote.date == "unknown",
            quote.date,
            quote.id,
        ),
    )


def _slugify(text: str) -> str:
    slug = "".join(ch if ch.isalnum() else "_" for ch in text.strip())
    slug = slug.strip("_")
    return slug or "untitled"
