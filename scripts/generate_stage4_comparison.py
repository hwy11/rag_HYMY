from __future__ import annotations

from pathlib import Path

from hymy_rag.search import search_index


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data" / "index" / "quotes_index.json"
OUTPUT = ROOT / "stage4_comparison.md"
QUERIES = [
    "失恋怎么走出来呢",
    "30岁还在体制内要不要跳出来",
    "怎么判断一个人值不值得交往",
    "现在该不该买房",
    "怎么走出习得性无助",
]


def main() -> None:
    lines: list[str] = ["# Stage 4 Comparison", ""]
    for index, query in enumerate(QUERIES, 1):
        sparse = search_index(INDEX, query, top_k=5, backend="sparse")
        vector = search_index(INDEX, query, top_k=5, backend="vector")
        lines.append(f"## Query {index}")
        lines.append(query)
        lines.append("")
        lines.append("### Sparse")
        lines.extend(_format_rows(sparse))
        lines.append("")
        lines.append("### Vector")
        lines.extend(_format_rows(vector))
        lines.append("")
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUTPUT)


def _format_rows(rows: list[dict[str, object]]) -> list[str]:
    if not rows:
        return ["(无结果)"]
    lines: list[str] = []
    for row in rows:
        trigger = str(row.get("trigger") or row.get("source_question") or "").replace("\n", " ").strip()
        content = str(row.get("content") or "").replace("\n", " ").strip()
        domains = "、".join(row.get("domains") or []) if isinstance(row.get("domains"), list) else "-"
        rerank = row.get("rerank_score")
        score_text = f"{row.get('score', 0)}"
        if rerank is not None:
            score_text = f"recall={row.get('score', 0)}, rerank={rerank}"
        lines.append(
            f"- {row.get('source_id', row.get('id', ''))} | {row.get('date', 'unknown')} | "
            f"{domains} | {row.get('type', 'unknown')} | {score_text}"
        )
        lines.append(f"  content: {content[:80]}")
        lines.append(f"  trigger: {trigger[:80] or '-'}")
    return lines


if __name__ == "__main__":
    main()
