from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib import error, request

from .io import merge_jsonl_by_id, read_json, read_jsonl, write_json, write_jsonl
from .models import TaggedQuote

ACTIVE_BATCH_MANIFEST = "active_batches.json"


PRICING_PER_MILLION_TOKENS_RMB = {
    "gpt-4o-mini_input": 1.1,
    "gpt-4o-mini_output": 4.4,
    "claude-haiku_input": 6.0,
    "claude-haiku_output": 30.0,
}


class TaggedImportError(RuntimeError):
    pass


class TaggingConfigError(RuntimeError):
    pass


class RemoteTaggingError(RuntimeError):
    pass


@dataclass
class TaggedImportResult:
    count: int
    files_found: int
    added_count: int = 0
    updated_count: int = 0


@dataclass
class BatchTagRunResult:
    batch_name: str
    output_path: Path
    record_count: int
    elapsed_seconds: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    skipped_existing: bool = False
    recovered_records: int = 0


@dataclass
class TagRunSummary:
    selected_batch_count: int
    total_record_count: int
    completed_count: int
    failed_count: int
    failed_batches: list[str]
    output_dir: Path
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    elapsed_seconds: float = 0.0
    actual_cost_rmb: float = 0.0
    cumulative_tagged_count: int = 0
    total_available_count: int = 0
    average_batch_seconds: float = 0.0


def make_batches(
    clean_path: Path,
    output_dir: Path,
    prompt_path: Path,
    batch_size: int,
    dry_run: bool = False,
    skip_existing: bool = False,
    tagged_dir: Path | None = None,
) -> dict[str, int]:
    rows = read_jsonl(clean_path)
    system_prompt, user_prefix = load_prompt_parts(prompt_path)
    prompt_overhead = system_prompt + "\n\n" + user_prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_prefix = f"pending_{batch_size:02d}_batch_"
    tagged_ids = _collect_tagged_ids(tagged_dir) if skip_existing else set()
    pending_rows = [row for row in rows if str(row["id"]) not in tagged_ids]
    _clear_pending_batches(output_dir)
    if tagged_dir is not None and tagged_dir.exists():
        _clear_pending_tagged_outputs(tagged_dir, batch_prefix)
    count = 0
    slim_rows = [_slim_prompt_row(row) for row in pending_rows]
    batch_files: list[str] = []
    for start in range(0, len(pending_rows), batch_size):
        count += 1
        batch = pending_rows[start : start + batch_size]
        slim_batch = slim_rows[start : start + batch_size]
        batch_json = output_dir / f"{batch_prefix}{count:03d}.json"
        batch_prompt = output_dir / f"{batch_prefix}{count:03d}_prompt.md"
        batch_files.append(batch_json.name)
        if not dry_run:
            batch_json.write_text(json.dumps(batch, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            batch_prompt.write_text(
                f"[system]\n{system_prompt}\n\n[user]\n{user_prefix}\n" + json.dumps(slim_batch, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    manifest = {
        "batch_prefix": batch_prefix,
        "batch_size": batch_size,
        "total_rows": len(rows),
        "existing_tagged_unique": len(tagged_ids),
        "pending_rows": len(pending_rows),
        "batch_files": batch_files,
    }
    if not dry_run:
        write_json(output_dir / ACTIVE_BATCH_MANIFEST, manifest)
    return {
        "batch_count": count,
        "record_count": len(pending_rows),
        "total_row_count": len(rows),
        "existing_tagged_unique": len(tagged_ids),
        "input_tokens_estimate": _estimate_batch_input_tokens(slim_rows, prompt_overhead),
        "output_tokens_estimate": _estimate_batch_output_tokens(pending_rows),
    }


def import_tagged(
    paths: list[Path],
    output_path: Path,
    force: bool = False,
    merge: bool = False,
) -> TaggedImportResult:
    resolved_paths = _expand_json_paths(paths)
    if not resolved_paths:
        raise TaggedImportError("未找到任何打标文件，请检查路径")
    incoming: list[TaggedQuote] = []
    seen: set[str] = set()
    for path in resolved_paths:
        data = read_json(path)
        if not isinstance(data, list):
            raise ValueError(f"{path} must be a JSON array")
        for item in data:
            quote = TaggedQuote.from_dict(item)
            key = quote.id or quote.content
            if key in seen:
                continue
            seen.add(key)
            incoming.append(quote)
    if not incoming and not force:
        raise TaggedImportError("导入结果为 0 条，已拒绝覆盖已有 quotes_tagged.jsonl；如需强制覆盖，请加 --force")

    existing_ids = {str(row.get("id")) for row in read_jsonl(output_path)} if merge else set()
    incoming_rows = [quote.to_dict() for quote in incoming]
    if merge:
        merged_rows = merge_jsonl_by_id(output_path, incoming_rows)
        added_count = sum(1 for row in incoming_rows if str(row.get("id")) not in existing_ids)
        updated_count = len(incoming_rows) - added_count
        write_jsonl(output_path, merged_rows)
        return TaggedImportResult(
            count=len(merged_rows),
            files_found=len(resolved_paths),
            added_count=added_count,
            updated_count=updated_count,
        )

    write_jsonl(output_path, incoming_rows)
    return TaggedImportResult(count=len(incoming_rows), files_found=len(resolved_paths))


def run_tagging(
    batches_dir: Path,
    output_dir: Path,
    prompt_path: Path,
    batch_limit: int | None = None,
    batch_names: list[str] | None = None,
    run_all: bool = False,
    retries: int = 3,
    failed_log_path: Path | None = None,
) -> TagRunSummary:
    if not run_all and batch_limit is None and not batch_names:
        raise TaggingConfigError("请使用 --batch-limit N、--batch-names 或 --all")
    config = load_tagging_env()
    batch_paths = _load_active_batch_paths(batches_dir)
    if not batch_paths:
        raise TaggingConfigError("未找到任何打标批次，请先运行 make-tag-batches")
    if batch_names:
        wanted = set(batch_names)
        selected = [path for path in batch_paths if path.stem in wanted]
        missing = [name for name in batch_names if name not in {path.stem for path in selected}]
        if missing:
            raise TaggingConfigError("这些批次不存在：" + ", ".join(missing))
    else:
        selected = batch_paths if run_all else batch_paths[: batch_limit or 0]
    if not selected:
        raise TaggingConfigError("没有可执行的打标批次")
    total_records = sum(len(read_json(path)) for path in selected)
    print(f"即将打标 {total_records} 条，{len(selected)} 批")

    output_dir.mkdir(parents=True, exist_ok=True)
    failed_log = failed_log_path or output_dir / "failed_batches.log"
    completed = 0
    failed: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    started_at = time.time()
    for index, batch_path in enumerate(selected, start=1):
        batch_name = batch_path.stem
        output_path = output_dir / f"{batch_name}_tagged.json"
        try:
            result = tag_single_batch(
                batch_path=batch_path,
                prompt_path=prompt_path,
                output_path=output_path,
                api_base=config["api_base"],
                api_key=config["api_key"],
                model=config["model"],
                retries=retries,
            )
            if not result.skipped_existing:
                completed += 1
                prompt_tokens += result.prompt_tokens
                completion_tokens += result.completion_tokens
                total_tokens += result.total_tokens
        except Exception as exc:
            failed.append(batch_name)
            _append_failed_log(failed_log, batch_name, str(exc))
        elapsed = max(1.0, time.time() - started_at)
        avg_seconds = elapsed / index
        remaining = len(selected) - index
        eta_minutes = int(round((avg_seconds * remaining) / 60))
        print(f"已完成 {index}/{len(selected)} 批，预计剩余时间 {eta_minutes} 分钟")
    return TagRunSummary(
        selected_batch_count=len(selected),
        total_record_count=total_records,
        completed_count=completed,
        failed_count=len(failed),
        failed_batches=failed,
        output_dir=output_dir,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        elapsed_seconds=time.time() - started_at,
        actual_cost_rmb=_actual_cost_rmb(config["model"], prompt_tokens, completion_tokens),
        cumulative_tagged_count=len(_collect_tagged_ids(output_dir)),
        total_available_count=len(read_jsonl(Path(__file__).resolve().parents[2] / "data" / "processed" / "quotes_clean.jsonl")),
        average_batch_seconds=(time.time() - started_at) / max(len(selected), 1),
    )


def tag_single_batch(
    batch_path: Path,
    prompt_path: Path,
    output_path: Path,
    api_base: str,
    api_key: str,
    model: str,
    retries: int = 3,
) -> BatchTagRunResult:
    batch_rows = read_json(batch_path)
    if output_path.exists():
        return BatchTagRunResult(
            batch_name=batch_path.stem,
            output_path=output_path,
            record_count=len(batch_rows),
            elapsed_seconds=0.0,
            skipped_existing=True,
        )
    system_prompt, user_prefix = load_prompt_parts(prompt_path)
    started = time.time()
    merged, usage = _tag_rows_with_recovery(
        rows=batch_rows,
        batch_name=batch_path.stem,
        system_prompt=system_prompt,
        user_prefix=user_prefix,
        api_base=api_base,
        api_key=api_key,
        model=model,
        retries=retries,
        preferred_chunk_size=min(20, max(1, len(batch_rows))),
    )
    write_json(output_path, merged)
    return BatchTagRunResult(
        batch_name=batch_path.stem,
        output_path=output_path,
        record_count=len(merged),
        elapsed_seconds=time.time() - started,
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        total_tokens=usage.get("total_tokens", 0),
        recovered_records=max(0, len(batch_rows) - min(len(batch_rows), usage.get("direct_match_count", len(batch_rows)))),
    )


def load_tagging_env(env_path: Path | None = None) -> dict[str, str]:
    resolved_env_path = env_path or Path(".env")
    env_values: dict[str, str] = {}
    if resolved_env_path.exists():
        env_values.update(_parse_env_file(resolved_env_path))
    file_base = env_values.get("OPENROUTER_API_BASE", "").strip()
    file_key = env_values.get("OPENROUTER_API_KEY", "").strip()
    file_model = env_values.get("OPENROUTER_MODEL", "").strip()
    api_base = os.getenv("OPENROUTER_API_BASE", file_base).strip()
    api_key = os.getenv("OPENROUTER_API_KEY", file_key).strip()
    model = os.getenv("OPENROUTER_MODEL", file_model).strip()
    if not resolved_env_path.exists() and not api_key:
        raise TaggingConfigError("未找到 .env，且 OPENROUTER_API_KEY 也未配置")
    if not api_key:
        raise TaggingConfigError("OPENROUTER_API_KEY 缺失，请检查 .env 或环境变量")
    if not api_base:
        raise TaggingConfigError("OPENROUTER_API_BASE 缺失，请检查 .env 或环境变量")
    if not model:
        raise TaggingConfigError("OPENROUTER_MODEL 缺失，请检查 .env 或环境变量")
    return {"api_base": api_base, "api_key": api_key, "model": model}


def load_prompt_parts(prompt_path: Path) -> tuple[str, str]:
    text = prompt_path.read_text(encoding="utf-8")
    match = re.search(r"\[system\]\s*(.*?)\s*\[user\]\s*(.*)", text, re.DOTALL)
    if not match:
        raise TaggingConfigError("tagging_prompt.md 缺少 [system]/[user] 分段")
    return match.group(1).strip(), match.group(2).strip()


def _load_active_batch_paths(batches_dir: Path) -> list[Path]:
    manifest_path = batches_dir / ACTIVE_BATCH_MANIFEST
    if manifest_path.exists():
        payload = read_json(manifest_path)
        files = payload.get("batch_files") or []
        return [batches_dir / str(name) for name in files if (batches_dir / str(name)).exists()]
    return sorted(
        path
        for path in batches_dir.glob("pending_batch_*.json")
        if path.is_file() and not path.name.endswith("_prompt.md")
    )


def _expand_json_paths(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if not path.exists():
            continue
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


def _collect_tagged_ids(tagged_dir: Path | None) -> set[str]:
    if tagged_dir is None or not tagged_dir.exists():
        return set()
    ids: set[str] = set()
    for path in sorted(tagged_dir.glob("*_tagged.json")):
        try:
            data = read_json(path)
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for row in data:
            rid = str(row.get("id") or "").strip()
            if rid:
                ids.add(rid)
    return ids


def _clear_pending_batches(output_dir: Path) -> None:
    for path in output_dir.glob("pending_batch_*"):
        if path.is_file():
            path.unlink()
    for path in output_dir.glob("pending_*_batch_*"):
        if path.is_file():
            path.unlink()
    manifest_path = output_dir / ACTIVE_BATCH_MANIFEST
    if manifest_path.exists():
        manifest_path.unlink()


def _clear_pending_tagged_outputs(tagged_dir: Path, batch_prefix: str) -> None:
    for path in tagged_dir.glob(f"{batch_prefix}*_tagged.json"):
        if path.is_file():
            path.unlink()


def estimate_tagging_cost(summary: dict[str, int]) -> dict[str, float]:
    input_tokens = summary["input_tokens_estimate"]
    output_tokens = summary["output_tokens_estimate"]
    return {
        "gpt_4o_mini_rmb": round(
            input_tokens / 1_000_000 * PRICING_PER_MILLION_TOKENS_RMB["gpt-4o-mini_input"]
            + output_tokens / 1_000_000 * PRICING_PER_MILLION_TOKENS_RMB["gpt-4o-mini_output"],
            2,
        ),
        "claude_haiku_rmb": round(
            input_tokens / 1_000_000 * PRICING_PER_MILLION_TOKENS_RMB["claude-haiku_input"]
            + output_tokens / 1_000_000 * PRICING_PER_MILLION_TOKENS_RMB["claude-haiku_output"],
            2,
        ),
    }


def _estimate_batch_input_tokens(rows: list[dict[str, object]], prompt_text: str) -> int:
    rows_text = json.dumps(rows, ensure_ascii=False)
    return _estimate_tokens(prompt_text) + _estimate_tokens(rows_text)


def _estimate_batch_output_tokens(rows: list[dict[str, object]]) -> int:
    per_row = 60
    return len(rows) * per_row


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 2)


def _slim_prompt_row(row: dict[str, object]) -> dict[str, object]:
    prompt_row: dict[str, object] = {
        "id": row["id"],
        "content": row["content"],
    }
    trigger = str(row.get("trigger") or "").strip()
    if trigger:
        prompt_row["trigger"] = trigger
    return prompt_row


def _tag_rows_with_recovery(
    rows: list[dict[str, object]],
    batch_name: str,
    system_prompt: str,
    user_prefix: str,
    api_base: str,
    api_key: str,
    model: str,
    retries: int,
    preferred_chunk_size: int = 20,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    slim_batch = [_slim_prompt_row(row) for row in rows]
    user_prompt = user_prefix + "\n" + json.dumps(slim_batch, ensure_ascii=False, indent=2)
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            tagged_rows, usage = _call_remote_tagger(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                api_base=api_base,
                api_key=api_key,
                model=model,
            )
            tagged_map = _index_tagged_rows(rows, tagged_rows)
            if len(tagged_map) == len(rows):
                merged = _merge_tagged_rows_from_map(rows, tagged_map)
                usage["direct_match_count"] = len(rows)
                return merged, usage
            if tagged_map:
                missing_rows = [row for row in rows if str(row["id"]) not in tagged_map]
                recovered_rows, recovered_usage = _recover_missing_rows(
                    missing_rows=missing_rows,
                    batch_name=batch_name,
                    system_prompt=system_prompt,
                    user_prefix=user_prefix,
                    api_base=api_base,
                    api_key=api_key,
                    model=model,
                    retries=retries,
                    preferred_chunk_size=preferred_chunk_size,
                )
                recovered_map = {str(row["id"]): row for row in recovered_rows}
                merged_map = {**tagged_map, **recovered_map}
                merged = _merge_tagged_rows_from_map(rows, merged_map)
                usage = _merge_usage(usage, recovered_usage)
                usage["direct_match_count"] = len(tagged_map)
                return merged, usage
            raise RemoteTaggingError(f"返回条数不匹配，期望 {len(rows)} 条，实际 0 条")
        except Exception as exc:
            last_error = str(exc)
            if attempt >= retries:
                break
            time.sleep(_retry_wait_seconds(exc, attempt))
    if len(rows) == 1:
        raise RemoteTaggingError(f"{batch_name} 失败：{last_error}")
    recovered_rows, recovered_usage = _recover_missing_rows(
        missing_rows=rows,
        batch_name=batch_name,
        system_prompt=system_prompt,
        user_prefix=user_prefix,
        api_base=api_base,
        api_key=api_key,
        model=model,
        retries=retries,
        preferred_chunk_size=preferred_chunk_size,
    )
    recovered_map = {str(row["id"]): row for row in recovered_rows}
    merged = _merge_tagged_rows_from_map(rows, recovered_map)
    recovered_usage["direct_match_count"] = 0
    return merged, recovered_usage


def _recover_missing_rows(
    missing_rows: list[dict[str, object]],
    batch_name: str,
    system_prompt: str,
    user_prefix: str,
    api_base: str,
    api_key: str,
    model: str,
    retries: int,
    preferred_chunk_size: int,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    if not missing_rows:
        return [], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if len(missing_rows) == 1:
        return _tag_rows_with_recovery(
            rows=missing_rows,
            batch_name=f"{batch_name}#single",
            system_prompt=system_prompt,
            user_prefix=user_prefix,
            api_base=api_base,
            api_key=api_key,
            model=model,
            retries=retries,
            preferred_chunk_size=1,
        )
    chunk_size = min(preferred_chunk_size, max(1, len(missing_rows) // 2))
    if chunk_size >= len(missing_rows):
        chunk_size = max(1, len(missing_rows) // 2)
    merged_rows: list[dict[str, object]] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for index, chunk in enumerate(_chunk_rows(missing_rows, chunk_size), start=1):
        chunk_rows, chunk_usage = _tag_rows_with_recovery(
            rows=chunk,
            batch_name=f"{batch_name}#split{index}",
            system_prompt=system_prompt,
            user_prefix=user_prefix,
            api_base=api_base,
            api_key=api_key,
            model=model,
            retries=retries,
            preferred_chunk_size=max(1, min(preferred_chunk_size, len(chunk) // 2 or 1)),
        )
        merged_rows.extend(chunk_rows)
        usage = _merge_usage(usage, chunk_usage)
    return merged_rows, usage


def _chunk_rows(rows: list[dict[str, object]], chunk_size: int) -> list[list[dict[str, object]]]:
    return [rows[start : start + chunk_size] for start in range(0, len(rows), chunk_size)]


def _merge_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {
        "prompt_tokens": int(left.get("prompt_tokens", 0)) + int(right.get("prompt_tokens", 0)),
        "completion_tokens": int(left.get("completion_tokens", 0)) + int(right.get("completion_tokens", 0)),
        "total_tokens": int(left.get("total_tokens", 0)) + int(right.get("total_tokens", 0)),
        "direct_match_count": int(left.get("direct_match_count", 0)) + int(right.get("direct_match_count", 0)),
    }


def _merge_tagged_rows(batch_rows: list[dict[str, object]], tagged_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    tagged_map = _index_tagged_rows(batch_rows, tagged_rows)
    if len(tagged_map) != len(batch_rows):
        raise RemoteTaggingError(f"返回条数不匹配，期望 {len(batch_rows)} 条，实际 {len(tagged_map)} 条")
    return _merge_tagged_rows_from_map(batch_rows, tagged_map)


def _index_tagged_rows(
    batch_rows: list[dict[str, object]],
    tagged_rows: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    source_map = {str(row["id"]): row for row in batch_rows}
    tagged_map: dict[str, dict[str, object]] = {}
    for tagged in tagged_rows:
        tagged_id = str(tagged.get("id") or "").strip()
        if tagged_id and tagged_id in source_map and tagged_id not in tagged_map:
            tagged_map[tagged_id] = tagged
    return tagged_map


def _merge_tagged_rows_from_map(
    batch_rows: list[dict[str, object]],
    tagged_map: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    source_map = {str(row["id"]): row for row in batch_rows}
    merged: list[dict[str, object]] = []
    for source in batch_rows:
        tagged_id = str(source["id"])
        if tagged_id not in tagged_map:
            raise RemoteTaggingError(f"缺少 id={tagged_id} 的打标结果")
        tagged = tagged_map[tagged_id]
        source = source_map[tagged_id]
        merged.append(
            TaggedQuote(
                id=str(source["id"]),
                source_id=str(source["source_id"]),
                date=str(source.get("date") or "unknown")[:10],
                type_origin=str(source.get("type_origin") or "reply"),
                source_question=str(source.get("source_question") or ""),
                content=str(source["content"]),
                domains=_normalize_domains(tagged.get("domains"), source_text=_tag_source_text(source)),
                type=_normalize_type(tagged.get("type"), source_text=_tag_source_text(source)),
            ).to_dict()
        )
    return merged


def _normalize_type(value: object, source_text: str) -> str:
    allowed = {"思维方式", "方法论", "价值观", "事实陈述/推测预测", "故事案例"}
    text = str(value or "").strip()
    if text not in allowed:
        text = "事实陈述/推测预测"
    override = _infer_type(source_text)
    return override or text


def _normalize_domains(value: object, source_text: str) -> list[str]:
    allowed = {"投资", "健康", "认知", "搞钱", "价值观", "人际", "学习", "决策", "职业", "教育", "情绪", "社会观察", "内容创作", "两性", "消费"}
    aliases = {
        "社会": "社会观察",
        "社会议题": "社会观察",
        "趋势": "社会观察",
        "内容": "内容创作",
        "创作": "内容创作",
        "自媒体": "内容创作",
        "感情": "两性",
        "感情关系": "两性",
        "男女": "两性",
        "消费观": "消费",
    }
    values = []
    for item in _as_list(value):
        mapped = aliases.get(item, item)
        if mapped in allowed and mapped not in values:
            values.append(mapped)
    inferred = _infer_domains(source_text)
    if not values:
        values = inferred
    else:
        values = _merge_domains(values, inferred)
    if not values:
        raise RemoteTaggingError("domains 为空或非法")
    return values[:3]


def _as_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _tag_source_text(source: dict[str, object]) -> str:
    content = str(source.get("content") or "")
    trigger = str(source.get("trigger") or "")
    return content + "\n" + trigger


def _infer_domains(text: str) -> list[str]:
    rules = [
        ("投资", ["a股", "股票", "基金", "券商", "房价", "贷款", "利率", "征信", "银行", "比特币", "买房"]),
        ("健康", ["睡眠", "锻炼", "心率", "脂肪", "蛋白粉", "缺钾", "身体", "减肥", "桑拿"]),
        ("职业", ["工作", "打工", "上班", "岗位", "入职", "求职", "offer", "跳槽"]),
        ("教育", ["高考", "学校", "师范", "专业", "学生", "老师", "法考", "读研", "大学"]),
        ("消费", ["买车", "电单车", "酒店", "房租", "消费", "一晚", "块钱", "价格", "租"]),
        ("内容创作", ["直播", "自媒体", "写文章", "mcn", "流量", "星球", "视频"]),
        ("两性", ["女粉", "女助理", "师生恋", "出轨", "结婚", "女人", "男人", "艹比"]),
        ("情绪", ["开心", "难受", "焦虑", "麻了", "气死", "崩溃", "流泪"]),
        ("搞钱", ["赚钱", "收米", "挣钱", "搞钱", "财富", "财商", "守财"]),
        ("认知", ["认知", "本质", "反思", "判断", "看法"]),
        ("价值观", ["没必要", "愿意", "自由自在", "不忘初心", "该涨价", "一步到位"]),
        ("决策", ["怎么选", "选择", "重点看", "注意", "建议", "判断"]),
        ("人际", ["父母", "朋友", "粉丝", "家人"]),
        ("学习", ["学习", "ai", "考试", "复习"]),
        ("社会观察", ["领导", "网友", "舆论", "法律", "社会", "账号", "注销", "白👻"]),
    ]
    lowered = text.lower()
    hits: list[str] = []
    for domain, keywords in rules:
        if any(keyword in lowered for keyword in keywords):
            hits.append(domain)
    return hits[:3]


def _infer_type(text: str) -> str | None:
    lowered = text.lower()
    if any(mark in lowered for mark in ["第一", "第二", "第三", "一是", "二是", "三是", "重点看", "步骤", "先", "再", "建议", "注意"]):
        return "方法论"
    if any(mark in lowered for mark in ["我宁可", "没必要", "一步到位", "自由自在", "不忘初心", "守财奴"]):
        return "价值观"
    if any(mark in lowered for mark in ["这世界就是", "好人不一定", "坏人不一定", "本质", "说白了", "还是那句话"]):
        return "思维方式"
    if any(mark in lowered for mark in ["有一次", "有个", "之前有个", "某个人", "我认识一个"]):
        return "故事案例"
    return None


def _merge_domains(model_domains: list[str], inferred_domains: list[str]) -> list[str]:
    generic = {"社会观察", "情绪", "认知", "价值观"}
    merged: list[str] = []
    for domain in model_domains + inferred_domains:
        if domain not in merged:
            merged.append(domain)
    non_generic = [domain for domain in merged if domain not in generic]
    if non_generic:
        merged = non_generic + [domain for domain in merged if domain in generic and domain != "社会观察"]
    deduped: list[str] = []
    for domain in merged:
        if domain not in deduped:
            deduped.append(domain)
    if len(deduped) > 1 and "社会观察" in deduped:
        deduped = [domain for domain in deduped if domain != "社会观察"]
    return deduped[:3]


def _call_remote_tagger(
    system_prompt: str,
    user_prompt: str,
    api_base: str,
    api_key: str,
    model: str,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    endpoint = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    req = request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://local.hymy-rag",
            "X-Title": "hymy-rag",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=180) as response:
            body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RemoteTaggingError(f"HTTP {exc.code}: {detail or exc.reason}") from exc
    except error.URLError as exc:
        raise RemoteTaggingError(f"网络错误: {exc.reason}") from exc
    payload = json.loads(body)
    content = payload["choices"][0]["message"]["content"]
    usage_payload = payload.get("usage") or {}
    usage = {
        "prompt_tokens": int(usage_payload.get("prompt_tokens") or 0),
        "completion_tokens": int(usage_payload.get("completion_tokens") or 0),
        "total_tokens": int(usage_payload.get("total_tokens") or 0),
    }
    return _parse_json_array(content), usage


def _parse_json_array(text: str) -> list[dict[str, object]]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"(\[\s*{.*}\s*\])", stripped, re.DOTALL)
        if not match:
            raise RemoteTaggingError("模型返回内容不是合法 JSON 数组")
        parsed = json.loads(match.group(1))
    if not isinstance(parsed, list):
        raise RemoteTaggingError("模型返回内容不是 JSON 数组")
    return parsed


def _retry_wait_seconds(exc: Exception, attempt: int) -> int:
    message = str(exc)
    if "HTTP 429" in message:
        return min(24, 5 * attempt)
    return min(12, 2 * attempt)


def _append_failed_log(path: Path, batch_name: str, reason: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", encoding="utf-8") as file:
        file.write(f"{timestamp}\t{batch_name}\t{reason}\n")


def _parse_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip("\"'")
    return env


def _actual_cost_rmb(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    if model.endswith(":free"):
        return 0.0
    return round(
        prompt_tokens / 1_000_000 * PRICING_PER_MILLION_TOKENS_RMB["gpt-4o-mini_input"]
        + completion_tokens / 1_000_000 * PRICING_PER_MILLION_TOKENS_RMB["gpt-4o-mini_output"],
        2,
    )
