from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any


class PersonaPromptMissingError(RuntimeError):
    pass


def build_prompt_package(
    question: str,
    results: list[dict[str, Any]],
    template_path: Path,
    persona_dir: Path,
    output_path: Path,
    current_context: str = "",
    persona_name: str = "meta_thinking.md",
    max_tokens: int | None = None,
) -> str:
    template = template_path.read_text(encoding="utf-8")
    persona = _load_persona(persona_dir, persona_name=persona_name)
    retrieved = _format_results(results)
    package = template.format(
        persona=persona,
        retrieved_quotes=retrieved,
        question=question,
        current_date=date.today().isoformat(),
        current_context=current_context.strip() or "暂无补充上下文。",
    )
    package = _trim_package(package, results, template, persona, question, current_context, max_tokens)
    output_path.write_text(package, encoding="utf-8")
    return package


def _load_persona(persona_dir: Path, persona_name: str) -> str:
    persona_path = (persona_dir / persona_name).resolve()
    expected_root = persona_dir.resolve()
    if expected_root not in persona_path.parents and persona_path != expected_root:
        raise PersonaPromptMissingError("persona 文件路径越界，请检查 --persona 参数")
    if not persona_path.exists() or not persona_path.is_file():
        raise PersonaPromptMissingError("请先手动蒸馏 meta_thinking.md，参考 prompts/distill_master_prompt.md")
    return persona_path.read_text(encoding="utf-8").strip()


def _format_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return "没有检索到相关语录。"
    blocks: list[str] = []
    for index, row in enumerate(results, 1):
        domains = "、".join(row.get("domains") or [])
        meta_parts = [row.get("date", "unknown")]
        if domains:
            meta_parts.append(f"领域:{domains}")
        meta_parts.append(f"类型:{row.get('type', 'unknown')}")
        if row.get("rerank_score") is not None:
            meta_parts.append(f"rerank:{row.get('rerank_score', 0)}")
        meta_parts.append(f"recall:{row.get('score', 0)}")
        meta = " | ".join(meta_parts)
        block = [
            f"## {index}. {meta}",
            row.get("content", ""),
        ]
        if row.get("source_question"):
            block.append(f"原问题：{row['source_question']}")
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


def _trim_package(
    package: str,
    results: list[dict[str, Any]],
    template: str,
    persona: str,
    question: str,
    current_context: str,
    max_tokens: int | None,
) -> str:
    if not max_tokens or max_tokens <= 0:
        return package
    if _estimate_tokens(package) <= max_tokens:
        return package
    trimmed_results = list(results)
    trimmed_persona = persona
    while trimmed_results:
        trimmed_results.pop()
        candidate = template.format(
            persona=trimmed_persona,
            retrieved_quotes=_format_results(trimmed_results),
            question=question,
            current_date=date.today().isoformat(),
            current_context=current_context.strip() or "暂无补充上下文。",
        )
        if _estimate_tokens(candidate) <= max_tokens:
            return candidate
    while _estimate_tokens(trimmed_persona) > max_tokens // 2 and len(trimmed_persona) > 120:
        trimmed_persona = trimmed_persona[: max(120, int(len(trimmed_persona) * 0.7))].rstrip() + "\n\n[Persona 已按 token 限额截断]"
        candidate = template.format(
            persona=trimmed_persona,
            retrieved_quotes="检索结果过长，已全部截断。请缩小问题范围或调低 top-k。",
            question=question,
            current_date=date.today().isoformat(),
            current_context=current_context.strip() or "暂无补充上下文。",
        )
        if _estimate_tokens(candidate) <= max_tokens:
            return candidate
    fallback = template.format(
        persona=trimmed_persona,
        retrieved_quotes="检索结果过长，已全部截断。请缩小问题范围或调低 top-k。",
        question=question,
        current_date=date.today().isoformat(),
        current_context=current_context.strip() or "暂无补充上下文。",
    )
    if _estimate_tokens(fallback) <= max_tokens:
        return fallback
    minimal_persona = "System prompt 过长，已在本地截断。请精简 persona/meta_thinking.md。"
    return template.format(
        persona=minimal_persona,
        retrieved_quotes="检索结果过长，已全部截断。请缩小问题范围或调低 top-k。",
        question=question,
        current_date=date.today().isoformat(),
        current_context=current_context.strip() or "暂无补充上下文。",
    )


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 2)
