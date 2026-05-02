from __future__ import annotations

import argparse
from pathlib import Path

from .config import FINAL_TOP_K, RETRIEVAL_BACKEND
from .cleaning import load_clean_quotes_report
from .distill import export_domain_corpora, export_master_corpus
from .io import write_jsonl
from .package import PersonaPromptMissingError, build_prompt_package
from .search import build_index, search_index
from .slang import write_slang_candidates
from .status import build_status_report
from .tagging import (
    TaggedImportError,
    TaggingConfigError,
    estimate_tagging_cost,
    import_tagged,
    make_batches,
    run_tagging,
)


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed" / "quotes_clean.jsonl"
TAGGED = ROOT / "data" / "processed" / "quotes_tagged.jsonl"
INDEX = ROOT / "data" / "index" / "quotes_index.json"
DISTILL_DIR = ROOT / "data" / "distill"


def main() -> None:
    parser = argparse.ArgumentParser(prog="hymy-rag")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init")

    ingest = sub.add_parser("ingest")
    ingest.add_argument("paths", nargs="+", type=Path)
    ingest.add_argument("--keep-empty-answer", action="store_true")
    ingest.add_argument("--dedup-threshold", type=float, default=0.92)
    ingest.add_argument("--filter-level", choices=["strict", "loose", "none"], default="strict")

    batches = sub.add_parser("make-tag-batches")
    batches.add_argument("--batch-size", type=int, default=20)
    batches.add_argument("--dry-run", action="store_true")
    batches.add_argument("--skip-existing", action="store_true")

    tag = sub.add_parser("tag")
    tag.add_argument("--batch-limit", type=int)
    tag.add_argument("--batch-names")
    tag.add_argument("--all", action="store_true")
    tag.add_argument("--retries", type=int, default=3)

    tagged = sub.add_parser("import-tagged")
    tagged.add_argument("paths", nargs="+", type=Path)
    tagged.add_argument("--force", action="store_true")

    index = sub.add_parser("build-index")
    index.add_argument("--backend", choices=["sparse", "vector"], default=RETRIEVAL_BACKEND)

    persona = sub.add_parser("prepare-persona")
    persona.add_argument("--output-dir", type=Path, default=DISTILL_DIR)

    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=FINAL_TOP_K)
    ask.add_argument("--domains")
    ask.add_argument("--type")
    ask.add_argument("--date-from")
    ask.add_argument("--prefer-time-sensitivity")
    ask.add_argument("--context", default="")
    ask.add_argument("--output", type=Path, default=ROOT / "clipboard.md")
    ask.add_argument("--persona", default="meta_thinking.md")
    ask.add_argument("--max-tokens", type=int, default=0)
    ask.add_argument("--backend", choices=["sparse", "vector"], default=RETRIEVAL_BACKEND)

    sub.add_parser("status")

    slang = sub.add_parser("scan-slang")
    slang.add_argument("--input", type=Path, default=PROCESSED)
    slang.add_argument("--output", type=Path, default=ROOT / "data" / "slang_candidates.md")
    slang.add_argument("--min-count", type=int, default=5)

    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)

    args = parser.parse_args()
    if args.command == "init":
        _init()
    elif args.command == "ingest":
        report = load_clean_quotes_report(
            args.paths,
            keep_empty_answer=args.keep_empty_answer,
            dedup_threshold=args.dedup_threshold,
            filter_level=args.filter_level,
        )
        for path, reason in report.skipped_files:
            print(f"⚠️ 跳过 {path.name}：原因 {reason}")
        if report.filter_counts:
            print("过滤统计：")
            for reason, count in sorted(report.filter_counts.items()):
                print(f"- {reason}: {count}")
        if report.dedup_counts:
            print("去重统计：")
            for reason, count in sorted(report.dedup_counts.items()):
                print(f"- {reason}: {count}")
        write_jsonl(PROCESSED, [quote.to_dict() for quote in report.quotes])
        print(
            f"共处理 {report.total_files} 个文件，成功 {report.successful_files}，跳过 {len(report.skipped_files)}\n"
            f"原始记录 {report.raw_rows} 条\n"
            f"已清洗 {len(report.quotes)} 条，写入 {PROCESSED}"
        )
    elif args.command == "make-tag-batches":
        summary = make_batches(
            clean_path=PROCESSED,
            output_dir=ROOT / "data" / "tagging_batches",
            prompt_path=ROOT / "prompts" / "tagging_prompt.md",
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            skip_existing=args.skip_existing,
            tagged_dir=ROOT / "data" / "tagged",
        )
        cost = estimate_tagging_cost(summary)
        prefix = "dry-run 预估" if args.dry_run else "已生成"
        print(
            f"{prefix} {summary['batch_count']} 个打标批次，覆盖 {summary['record_count']} 条待打标语录\n"
            f"已跳过已打标去重后语录：{summary['existing_tagged_unique']} 条\n"
            f"清洗后语料总量：{summary['total_row_count']} 条\n"
            f"输入 token 预估：{summary['input_tokens_estimate']}\n"
            f"输出 token 预估：{summary['output_tokens_estimate']}\n"
            f"GPT-4o-mini 预估成本：{cost['gpt_4o_mini_rmb']} 元\n"
            f"Claude Haiku 预估成本：{cost['claude_haiku_rmb']} 元"
        )
    elif args.command == "tag":
        try:
            summary = run_tagging(
                batches_dir=ROOT / "data" / "tagging_batches",
                output_dir=ROOT / "data" / "tagged",
                prompt_path=ROOT / "prompts" / "tagging_prompt.md",
                batch_limit=args.batch_limit or (None if args.all else 10),
                batch_names=[name.strip() for name in args.batch_names.split(",") if name.strip()] if args.batch_names else None,
                run_all=args.all,
                retries=args.retries,
            )
        except TaggingConfigError as exc:
            raise SystemExit(str(exc))
        print(
            f"打标结束：本次计划 {summary.selected_batch_count} 批，成功新增 {summary.completed_count} 批，失败 {summary.failed_count} 批\n"
            f"累计已打标 {summary.cumulative_tagged_count} 条 / 总 {summary.total_available_count} 条\n"
            f"实际 token：prompt {summary.prompt_tokens} / completion {summary.completion_tokens} / total {summary.total_tokens}\n"
            f"总耗时：{round(summary.elapsed_seconds / 60, 1)} 分钟\n"
            f"平均每批耗时：{round(summary.average_batch_seconds, 1)} 秒\n"
            f"总成本：{summary.actual_cost_rmb} 元\n"
            f"结果目录：{summary.output_dir}"
        )
        if summary.failed_batches:
            print("失败批次：" + ", ".join(summary.failed_batches))
    elif args.command == "import-tagged":
        try:
            result = import_tagged(args.paths, TAGGED, force=args.force)
        except TaggedImportError as exc:
            raise SystemExit(str(exc))
        print(f"已导入 {result.count} 条打标语录，扫描 {result.files_found} 个文件，写入 {TAGGED}")
    elif args.command == "build-index":
        report = build_index(TAGGED, INDEX, backend=args.backend)
        if report.backend == "vector":
            print(
                f"已构建 vector 索引\n"
                f"- device: {report.device}\n"
                f"- collection: {report.collection_name}\n"
                f"- quotes: {report.quote_count}\n"
                f"- points: {report.point_count}\n"
                f"- total_seconds: {round(report.elapsed_seconds, 2)}\n"
                f"- points_per_second: {round(report.points_per_second, 2)}\n"
                f"- peak_vram_bytes: {report.peak_vram_bytes}\n"
                f"- disk_usage_bytes: {report.disk_usage_bytes}"
            )
        else:
            print(f"已构建 {report.indexed_count} 条语录的 sparse 本地索引")
    elif args.command == "prepare-persona":
        domain_count = export_domain_corpora(TAGGED, args.output_dir)
        master_count = export_master_corpus(TAGGED, args.output_dir / "_master.md")
        print(f"已导出 {domain_count} 个领域语料文件，并汇总 {master_count} 条长期资产语录")
    elif args.command == "ask":
        domain_filters = [item.strip() for item in (args.domains or "").split(",") if item.strip()]
        results = search_index(
            INDEX,
            args.question,
            top_k=args.top_k,
            domains=domain_filters,
            quote_type=args.type,
            date_from=args.date_from,
            preferred_time_sensitivity=args.prefer_time_sensitivity,
            backend=args.backend,
        )
        _print_results(results)
        try:
            package = build_prompt_package(
                question=args.question,
                results=results,
                template_path=ROOT / "prompts" / "package_template.md",
                persona_dir=ROOT / "persona",
                output_path=args.output,
                current_context=args.context,
                persona_name=args.persona,
                max_tokens=args.max_tokens,
            )
        except PersonaPromptMissingError as exc:
            raise SystemExit(str(exc))
        token_estimate = max(1, len(package) // 2)
        print(f"已写入 {args.output}，共 {len(package)} 字，预估 {token_estimate} tokens")
    elif args.command == "status":
        report = build_status_report(
            raw_dir=ROOT / "data" / "raw",
            processed_path=PROCESSED,
            tagged_path=TAGGED,
            index_path=INDEX,
            persona_dir=ROOT / "persona",
            distill_dir=DISTILL_DIR,
        )
        print(report)
    elif args.command == "scan-slang":
        count = write_slang_candidates(args.input, args.output, min_count=args.min_count)
        print(f"已输出 {count} 个黑话候选，写入 {args.output}")
    elif args.command == "serve":
        from .webapp import create_app

        import uvicorn

        uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")


def _init() -> None:
    for path in [
        ROOT / "data" / "raw",
        ROOT / "data" / "processed",
        ROOT / "data" / "tagged",
        ROOT / "data" / "tagging_batches",
        ROOT / "data" / "index",
        ROOT / "data" / "distill",
        ROOT / "persona",
        ROOT / "prompts",
    ]:
        path.mkdir(parents=True, exist_ok=True)
    print("项目目录已就绪")


def _print_results(results: list[dict[str, object]]) -> None:
    if not results:
        print("未召回到结果。")
        return
    print("召回结果：")
    for index, row in enumerate(results, 1):
        domains = "、".join(row.get("domains") or []) if isinstance(row.get("domains"), list) else ""
        rerank_score = row.get("rerank_score")
        trigger = str(row.get("trigger") or row.get("source_question") or "").replace("\n", " ").strip()
        content = str(row.get("content") or "").replace("\n", " ").strip()
        print(
            f"{index}. {row.get('source_id', row.get('id', ''))} | {row.get('date', 'unknown')} | "
            f"{domains or '-'} | {row.get('type', 'unknown')} | "
            f"recall={row.get('score', 0)} | "
            f"rerank={rerank_score if rerank_score is not None else '-'}"
        )
        print(f"   content: {content[:80]}")
        if trigger:
            print(f"   trigger: {trigger[:80]}")


if __name__ == "__main__":
    main()
