from __future__ import annotations

import random
import time
from datetime import datetime, timedelta
from threading import Lock, Thread
from typing import Any
from urllib.parse import parse_qs, urlparse

from .storage import (
    infer_existing_latest_publish_time,
    load_config,
    load_seen_topic_ids,
    load_state,
    save_seen_topic_ids,
    save_state,
)
from .zsxq_client import DEFAULT_COUNT, ZsxqClient, ZsxqConfig
from .zsxq_pipeline import append_raw_topics, export_zsxq_outputs, format_publish_time

# 抓取模式语义，注释是契约，下面实现必须逐行服从它。
#
# after_baseline
# - 输入：baseline_time，加上 progress_time 作为动态游标；如果 progress_time 为空，就从 baseline_time 开始。
# - 从哪开始：从 baseline_time 之后紧邻的时间窗口开始，而不是从 2026 年最新页倒着扫。
# - 到哪结束：每次只扫 [progress_time, progress_time + 7 天] 这段窗口；窗口内按 end_time 从新到旧翻页。
# - 终止条件：拿满 run_limit，或者当前窗口扫完后再推进到下一窗口，直到推进到现在时间为止。
#
# time_window
# - 输入：window_start_time 和 window_end_time。
# - 从哪开始：直接从这个时间段内部开始。
# - 到哪结束：只允许落在 [window_start_time, window_end_time] 的内容。
# - 终止条件：拿满 run_limit，或者这个时间段已经翻完。
#
# full_history
# - 输入：内部 older_history_end_time 游标；为空时表示从最新页开始。
# - 从哪开始：从 older_history_end_time 对应的旧内容页开始；为空时就是最新页。
# - 到哪结束：一路往更早的内容翻。
# - 终止条件：拿满 run_limit，或者接口已经没有更旧的内容。


def _utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _random_pause_seconds() -> int:
    return random.randint(20, 90)


def _normalize_local_time(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    if len(value) == 10:
        return f"{value} 00:00"
    return value[:16]


def _parse_local_time(value: str) -> datetime:
    return datetime.strptime(_normalize_local_time(value), "%Y-%m-%d %H:%M")


def _format_local_time(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M")


def _current_local_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _local_to_api_time(value: str, is_end: bool = False) -> str:
    normalized = _normalize_local_time(value)
    if not normalized:
        return ""
    seconds = "59.999" if is_end else "00.000"
    return normalized.replace(" ", "T") + f":{seconds}+0800"


def _previous_end_time(api_time: str) -> str:
    dt = datetime.strptime(api_time, "%Y-%m-%dT%H:%M:%S.%f%z")
    previous = dt - timedelta(seconds=1)
    return previous.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + previous.strftime("%z")


def _shift_api_time(api_time: str, seconds: int) -> str:
    dt = datetime.strptime(api_time, "%Y-%m-%dT%H:%M:%S.%f%z")
    shifted = dt + timedelta(seconds=seconds)
    return shifted.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + shifted.strftime("%z")


def _parse_group_id(topics_url: str) -> str:
    if not topics_url:
        return ""
    path_parts = [part for part in urlparse(topics_url).path.split("/") if part]
    if "groups" in path_parts:
        idx = path_parts.index("groups")
        if idx + 1 < len(path_parts):
            return path_parts[idx + 1]
    query = parse_qs(urlparse(topics_url).query)
    if "group_id" in query:
        return query["group_id"][0]
    return ""


def _page_range(topics: list[dict[str, Any]]) -> str:
    if not topics:
        return ""
    newest = format_publish_time(topics[0].get("create_time"))
    oldest = format_publish_time(topics[-1].get("create_time"))
    if not newest or newest == "Unknown" or not oldest or oldest == "Unknown":
        return ""
    return f"{oldest} 到 {newest}"


def _public_state(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline_time": state.get("baseline_time", ""),
        "progress_time": state.get("progress_time", ""),
        "current_page_range": state.get("current_page_range", ""),
        "run_mode": state.get("run_mode") or config.get("crawl_mode", "after_baseline"),
        "run_limit": state.get("run_limit") or int(config.get("max_new_topics_per_run", 50) or 50),
        "run_got": state.get("run_got", 0),
        "phase": state.get("phase", "idle"),
        "phase_detail": state.get("phase_detail", ""),
    }


class CrawlManager:
    def __init__(self) -> None:
        self._lock = Lock()
        self._thread: Thread | None = None

    def get_runtime_status(self) -> dict[str, Any]:
        config = load_config()
        state = load_state()
        baseline_time = state.get("baseline_time") or infer_existing_latest_publish_time()
        if baseline_time and not state.get("progress_time"):
            state["progress_time"] = baseline_time
        state["baseline_time"] = baseline_time
        save_state(state)
        return _public_state(state, config)

    def start_crawl(self) -> dict[str, Any]:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return {"ok": False, "message": "已有抓取任务在运行"}

            config = load_config()
            state = load_state()
            baseline_time = state.get("baseline_time") or infer_existing_latest_publish_time()
            run_mode = config.get("crawl_mode", "after_baseline")
            run_limit = max(int(config.get("max_new_topics_per_run", 50) or 50), 1)

            save_state(
                {
                    **state,
                    "baseline_time": baseline_time,
                    "progress_time": state.get("progress_time") or baseline_time,
                    "current_page_range": "",
                    "run_mode": run_mode,
                    "run_limit": run_limit,
                    "run_got": 0,
                    "phase": "fetching",
                    "phase_detail": "正在准备抓取",
                    "_internal": {
                        **state.get("_internal", {}),
                        "last_run_at": _utc_now(),
                        "last_error": "",
                        "pages_scanned": 0,
                        "last_page_new_topics": 0,
                    },
                }
            )

            self._thread = Thread(target=self._run_crawl, daemon=True)
            self._thread.start()
            return {"ok": True, "message": "抓取任务已启动"}

    def test_connection(self) -> dict[str, Any]:
        return ZsxqClient(self._load_runtime_config()).test_connection()

    def _load_runtime_config(self) -> ZsxqConfig:
        raw = load_config()
        group_id = raw.get("group_id") or _parse_group_id(raw.get("topics_url", ""))
        return ZsxqConfig(
            authorization=raw.get("authorization", ""),
            user_agent=raw.get("user_agent", ""),
            group_id=group_id,
            topics_url=raw.get("topics_url", ""),
            scope=raw.get("scope", "all") or "all",
        )

    def _save_public_state(
        self,
        *,
        phase: str,
        phase_detail: str,
        run_mode: str | None = None,
        run_limit: int | None = None,
        run_got: int | None = None,
        progress_time: str | None = None,
        current_page_range: str | None = None,
        internal_updates: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = load_state()
        internal = state.get("_internal", {}).copy()
        if internal_updates:
            internal.update(internal_updates)
        if run_mode is not None:
            state["run_mode"] = run_mode
        if run_limit is not None:
            state["run_limit"] = run_limit
        if run_got is not None:
            state["run_got"] = run_got
        if progress_time is not None:
            state["progress_time"] = progress_time
        if current_page_range is not None:
            state["current_page_range"] = current_page_range
        state["phase"] = phase
        state["phase_detail"] = phase_detail
        state["_internal"] = internal
        return save_state(state)

    def _fetch_topics_with_fallback(
        self,
        client: ZsxqClient,
        count: int,
        begin_time: str | None = None,
        end_time: str | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        fallback_steps = [0, -1, -2, -5, -10, -30, -60]
        last_error: Exception | None = None

        for step in fallback_steps:
            try_end_time = _shift_api_time(end_time, step) if end_time and step else end_time
            try:
                topics = client.fetch_topics(
                    count=count,
                    begin_time=begin_time,
                    end_time=try_end_time,
                )
                return topics, try_end_time
            except RuntimeError as exc:
                last_error = exc
                if "1059" not in str(exc) or not end_time:
                    raise
                self._save_public_state(
                    phase="fetching",
                    phase_detail="接口时间点不稳定，正在自动回退到更早的结束时间重试",
                )

        if last_error:
            raise last_error
        return [], end_time

    def _fetch_latest_topic(self, client: ZsxqClient) -> tuple[str, str]:
        topics, _ = self._fetch_topics_with_fallback(client=client, count=1)
        if not topics:
            return "", ""
        return str(topics[0].get("topic_id") or ""), topics[0].get("create_time") or ""

    def _run_crawl(self) -> None:
        try:
            config_raw = load_config()
            config = self._load_runtime_config()
            if not config.authorization or not config.user_agent or not config.group_id:
                raise ValueError("缺少 Authorization、User-Agent 或 group_id")

            client = ZsxqClient(config)
            seen_ids = load_seen_topic_ids()
            state = load_state()
            baseline_time = state.get("baseline_time") or infer_existing_latest_publish_time()
            crawl_mode = config_raw.get("crawl_mode", "after_baseline")
            max_new_topics = max(int(config_raw.get("max_new_topics_per_run", 50) or 50), 1)
            auto_export = bool(config_raw.get("auto_export", True))

            self._save_public_state(
                phase="fetching",
                phase_detail="开始抓取正文",
                run_mode=crawl_mode,
                run_limit=max_new_topics,
                run_got=0,
                progress_time=state.get("progress_time") or baseline_time,
                current_page_range="",
                internal_updates={"last_error": "", "pages_scanned": 0, "last_page_new_topics": 0},
            )

            if crawl_mode == "time_window":
                start_time = _normalize_local_time(config_raw.get("window_start_time", ""))
                end_time = _normalize_local_time(config_raw.get("window_end_time", ""))
                if not start_time or not end_time:
                    raise ValueError("时间窗口模式需要开始时间和结束时间")
                total_new, progress_time, history_complete = self._crawl_time_window(
                    client=client,
                    seen_ids=seen_ids,
                    start_time=start_time,
                    end_time=end_time,
                    max_new_topics=max_new_topics,
                )
            elif crawl_mode == "full_history":
                total_new, progress_time, history_complete = self._crawl_older_history(
                    client=client,
                    seen_ids=seen_ids,
                    max_new_topics=max_new_topics,
                    end_time=state.get("_internal", {}).get("older_history_end_time"),
                )
            else:
                total_new, progress_time, history_complete = self._crawl_forward_from_cutoff(
                    client=client,
                    seen_ids=seen_ids,
                    cutoff_time=baseline_time,
                    max_new_topics=max_new_topics,
                    cursor_time=state.get("progress_time") or baseline_time,
                )

            save_seen_topic_ids(seen_ids)
            export_summary = export_zsxq_outputs() if auto_export else {}
            try:
                latest_topic_id, latest_topic_time = self._fetch_latest_topic(client)
            except Exception:
                latest_topic_id, latest_topic_time = "", ""

            self._save_public_state(
                phase="done",
                phase_detail=(
                    "这次没有发现符合条件且未抓过的新正文"
                    if total_new == 0
                    else f"本次新增 {total_new} 条正文"
                ),
                run_mode=crawl_mode,
                run_limit=max_new_topics,
                run_got=total_new,
                progress_time=progress_time,
                internal_updates={
                    "history_complete": history_complete,
                    "last_success_at": _utc_now(),
                    "latest_topic_id": latest_topic_id,
                    "latest_topic_time": latest_topic_time,
                    "export_summary": export_summary,
                },
            )
        except Exception as exc:
            self._save_public_state(
                phase="error",
                phase_detail=f"抓取失败：{exc}",
                internal_updates={"last_error": str(exc)},
            )

    def _crawl_forward_from_cutoff(
        self,
        client: ZsxqClient,
        seen_ids: set[str],
        cutoff_time: str,
        max_new_topics: int,
        cursor_time: str,
    ) -> tuple[int, str, bool]:
        total_new = 0
        pages_scanned = 0
        current_start = _parse_local_time(cursor_time or cutoff_time)
        now_time = _parse_local_time(_current_local_time())

        while total_new < max_new_topics and current_start < now_time:
            current_end = min(current_start + timedelta(days=7), now_time)
            begin_local = _format_local_time(current_start)
            end_local = _format_local_time(current_end)
            begin_api = _local_to_api_time(begin_local, is_end=False)
            current_end_api = _local_to_api_time(end_local, is_end=True)

            while total_new < max_new_topics:
                self._save_public_state(
                    phase="fetching",
                    phase_detail=f"正在补 {begin_local} 之后的新内容，第 {pages_scanned + 1} 页",
                    progress_time=begin_local,
                )
                topics, effective_end_time = self._fetch_topics_with_fallback(
                    client=client,
                    count=DEFAULT_COUNT,
                    begin_time=begin_api,
                    end_time=current_end_api,
                )
                if not topics:
                    break

                pages_scanned += 1
                page_range = _page_range(topics)
                oldest_api_time = topics[-1].get("create_time") or ""
                oldest_time = format_publish_time(oldest_api_time)
                page_new_topics = []

                for topic in topics:
                    publish_time = format_publish_time(topic.get("create_time"))
                    if publish_time <= cutoff_time:
                        continue
                    topic_id = str(topic.get("topic_id") or "")
                    if not topic_id or topic_id in seen_ids:
                        continue
                    seen_ids.add(topic_id)
                    page_new_topics.append(topic)

                page_new_topics = page_new_topics[: max_new_topics - total_new]
                projected_total = total_new + len(page_new_topics)
                self._save_public_state(
                    phase="fetching",
                    phase_detail=f"正在补 {begin_local} 之后的新内容，第 {pages_scanned} 页",
                    run_got=projected_total,
                    progress_time=begin_local,
                    current_page_range=page_range,
                    internal_updates={"pages_scanned": pages_scanned, "last_page_new_topics": len(page_new_topics)},
                )
                if page_new_topics:
                    append_raw_topics(page_new_topics)
                    total_new = projected_total
                    if total_new >= max_new_topics:
                        return total_new, begin_local, current_end >= now_time

                if len(topics) < DEFAULT_COUNT or not oldest_time or oldest_time <= begin_local:
                    break
                current_end_api = _previous_end_time(oldest_api_time) if oldest_api_time else effective_end_time
                self._pause_between_pages(total_new)

            current_start = current_end
            next_progress = _format_local_time(current_start)
            self._save_public_state(
                phase="fetching",
                phase_detail=f"正在推进到下一段时间窗口，当前已经补到 {next_progress}",
                progress_time=next_progress,
            )

        return total_new, _format_local_time(current_start), current_start >= now_time

    def _crawl_older_history(
        self,
        client: ZsxqClient,
        seen_ids: set[str],
        max_new_topics: int,
        end_time: str | None,
    ) -> tuple[int, str, bool]:
        total_new = 0
        pages_scanned = 0
        cursor_end_time = end_time

        while total_new < max_new_topics:
            self._save_public_state(
                phase="fetching",
                phase_detail=f"正在回填更早的旧内容，第 {pages_scanned + 1} 页",
            )
            topics, effective_end_time = self._fetch_topics_with_fallback(
                client=client,
                count=DEFAULT_COUNT,
                end_time=cursor_end_time,
            )
            if not topics:
                return total_new, load_state().get("progress_time", ""), True

            pages_scanned += 1
            page_range = _page_range(topics)
            oldest_api_time = topics[-1].get("create_time") or ""
            oldest_time = format_publish_time(oldest_api_time)
            page_new_topics = []

            for topic in topics:
                topic_id = str(topic.get("topic_id") or "")
                if not topic_id or topic_id in seen_ids:
                    continue
                seen_ids.add(topic_id)
                page_new_topics.append(topic)

            page_new_topics = page_new_topics[: max_new_topics - total_new]
            projected_total = total_new + len(page_new_topics)
            self._save_public_state(
                phase="fetching",
                phase_detail=f"正在回填更早的旧内容，第 {pages_scanned} 页",
                run_got=projected_total,
                progress_time=oldest_time or load_state().get("progress_time", ""),
                current_page_range=page_range,
                internal_updates={"pages_scanned": pages_scanned, "last_page_new_topics": len(page_new_topics)},
            )
            if page_new_topics:
                append_raw_topics(page_new_topics)
                total_new = projected_total
                if total_new >= max_new_topics:
                    return total_new, oldest_time or load_state().get("progress_time", ""), False

            if len(topics) < DEFAULT_COUNT:
                return total_new, oldest_time or load_state().get("progress_time", ""), True

            cursor_end_time = _previous_end_time(oldest_api_time) if oldest_api_time else effective_end_time
            state = load_state()
            internal = state.get("_internal", {}).copy()
            internal["older_history_end_time"] = cursor_end_time
            save_state({**state, "_internal": internal})
            self._pause_between_pages(total_new)

        return total_new, load_state().get("progress_time", ""), False

    def _crawl_time_window(
        self,
        client: ZsxqClient,
        seen_ids: set[str],
        start_time: str,
        end_time: str,
        max_new_topics: int,
    ) -> tuple[int, str, bool]:
        total_new = 0
        pages_scanned = 0
        begin_api_time = _local_to_api_time(start_time, is_end=False)
        current_end_time = _local_to_api_time(end_time, is_end=True)

        while total_new < max_new_topics:
            self._save_public_state(
                phase="fetching",
                phase_detail=f"正在抓 {start_time} 到 {end_time} 这段时间，第 {pages_scanned + 1} 页",
                progress_time=start_time,
            )
            topics, effective_end_time = self._fetch_topics_with_fallback(
                client=client,
                count=DEFAULT_COUNT,
                begin_time=begin_api_time,
                end_time=current_end_time,
            )
            if not topics:
                return total_new, start_time, True

            pages_scanned += 1
            page_range = _page_range(topics)
            oldest_api_time = topics[-1].get("create_time") or ""
            oldest_time = format_publish_time(oldest_api_time)
            page_new_topics = []

            for topic in topics:
                topic_id = str(topic.get("topic_id") or "")
                if not topic_id or topic_id in seen_ids:
                    continue
                seen_ids.add(topic_id)
                page_new_topics.append(topic)

            page_new_topics = page_new_topics[: max_new_topics - total_new]
            projected_total = total_new + len(page_new_topics)
            self._save_public_state(
                phase="fetching",
                phase_detail=f"正在抓 {start_time} 到 {end_time} 这段时间，第 {pages_scanned} 页",
                run_got=projected_total,
                progress_time=oldest_time or start_time,
                current_page_range=page_range,
                internal_updates={"pages_scanned": pages_scanned, "last_page_new_topics": len(page_new_topics)},
            )
            if page_new_topics:
                append_raw_topics(page_new_topics)
                total_new = projected_total
                if total_new >= max_new_topics:
                    return total_new, oldest_time or start_time, False

            if len(topics) < DEFAULT_COUNT or not oldest_time or oldest_time <= start_time:
                return total_new, oldest_time or start_time, True

            current_end_time = _previous_end_time(oldest_api_time) if oldest_api_time else effective_end_time
            self._pause_between_pages(total_new)

        return total_new, load_state().get("progress_time", start_time), False

    def _pause_between_pages(self, total_new: int) -> None:
        wait_seconds = _random_pause_seconds()
        self._save_public_state(
            phase="sleeping",
            phase_detail=f"按限流休息 {wait_seconds} 秒",
            run_got=total_new,
        )
        time.sleep(wait_seconds)
