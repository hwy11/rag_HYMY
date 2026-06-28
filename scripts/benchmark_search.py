from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hymy_rag.search import search_index

INDEX = ROOT / "data" / "index" / "quotes_index.json"
QUERIES = [
    "失恋怎么走出来呢",
    "30岁还在体制内要不要跳出来",
    "怎么判断一个人值不值得交往",
    "现在该不该买房",
    "怎么走出习得性无助",
]


def main() -> None:
    for index, query in enumerate(QUERIES, 1):
        print(f"\n=== Query {index}: {query} ===")
        results = search_index(INDEX, query, top_k=5, backend="vector")
        for row in results:
            trigger = (row.get("trigger") or row.get("source_question") or "-")[:80].replace("\n", " ")
            content = (row.get("content") or "")[:80].replace("\n", " ")
            print(
                f"{row.get('source_id')} | {row.get('date')} | "
                f"rerank={row.get('rerank_score')} recall={row.get('score')} | field={row.get('field')}"
            )
            print(f"  trigger: {trigger}")
            print(f"  content: {content}")


if __name__ == "__main__":
    main()
