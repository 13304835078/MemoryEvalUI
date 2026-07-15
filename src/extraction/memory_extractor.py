from __future__ import annotations

import math
import re
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any, Callable

import pandas as pd
import yaml

from src.extraction.document_parser import (
    DocumentParseResult,
    extract_long_memory,
    extract_memory_document,
    extract_user_md,
    normalize_memory_document_body,
    normalize_user_md_body,
    parse_memory_document,
)
from src.extraction.contracts import (
    CallStatus,
    CaseStatus,
    InheritanceSource,
    ParseStatus,
    get_extraction_task_profile,
)
from src.extraction.client import (
    MemoryExtractionClient,
    MemoryExtractionConfig,
    MockMemoryExtractionClient,
    extract_answer_from_response,
)
from src.persistence import append_jsonl_rows, atomic_write_bytes, atomic_write_jsonl
from src.runtime_paths import APP_HOME, DATA_DIR
from src.schema import TaskType
from src.ui.global_rate_limiter import api_rate_scope, wait_for_global_rate_slot
from src.llm_api import make_prompt_cache_id

PROJECT_ROOT = APP_HOME
EXTRACTION_OUTPUT_DIR = DATA_DIR / "extractions"


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def sanitize_filename(name: str) -> str:
    if not name:
        return "ALL"
    return re.sub(r'[\\/:*?"<>|，,\s]+', "_", str(name).strip())


def parse_generation_prompt_templates(text: str, suffix: str = "") -> dict[str, str]:
    """Return create/update templates; flat prompts are used for both modes."""
    loaded: Any = text
    if suffix.lower() in {".yaml", ".yml"}:
        loaded = yaml.safe_load(text)

    if isinstance(loaded, str):
        prompt = loaded.strip()
        return {"create": prompt, "update": prompt}
    if not isinstance(loaded, dict):
        raise ValueError("无法从提示词内容中提取模板")

    prompt_config = loaded.get("memory_extraction", loaded)
    if not isinstance(prompt_config, dict):
        raise ValueError("提示词中的 memory_extraction 不是 object")

    fallback = ""
    for key in ("prompt", "system", "user"):
        value = prompt_config.get(key)
        if isinstance(value, str) and value.strip():
            fallback = value.strip()
            break
    create = str(prompt_config.get("create_template") or fallback).strip()
    update = str(prompt_config.get("update_template") or fallback or create).strip()
    create = create or update
    if not create or not update:
        raise ValueError("提示词中未找到 create_template/update_template 或通用 prompt")
    return {"create": create, "update": update}


def load_generation_prompt_templates(prompt_file: str | Path) -> dict[str, str]:
    path = Path(prompt_file)
    return parse_generation_prompt_templates(
        path.read_text(encoding="utf-8"),
        path.suffix,
    )


def load_generation_prompt(prompt_file: str | Path) -> str:
    return load_generation_prompt_templates(prompt_file)["update"]


def _round_to_int(value: Any) -> int:
    text = clean_cell(value)
    if not text:
        return 0
    try:
        return int(float(text))
    except ValueError:
        return 0


def split_sessions(df: pd.DataFrame) -> list[list[dict[str, Any]]]:
    sessions: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        row_dict = row.to_dict()
        round_no = _round_to_int(row_dict.get("轮次", 0))
        if round_no == 1 and current:
            sessions.append(current)
            current = []
        current.append(row_dict)

    if current:
        sessions.append(current)

    return sessions


def build_dialogue_history(chunk: list[dict[str, Any]]) -> str:
    turns: list[str] = []
    for row in chunk:
        query = clean_cell(row.get("query", ""))
        answer = clean_cell(row.get("answer", ""))
        if not query and not answer:
            continue

        lines: list[str] = []
        if query:
            lines.append(f"- user: {query}")
        if answer:
            lines.append(f"- assistant: {answer}")
        turns.append("\n".join(lines))

    return "\n\n".join(turns).strip()


def build_user_prompt(current_user_md: str, formatted_history: str) -> str:
    profile = get_extraction_task_profile(TaskType.USER_MD)
    return profile.build_user_message(clean_cell(current_user_md), formatted_history)


def build_long_memory_prompt(current_memory: str, formatted_history: str) -> str:
    profile = get_extraction_task_profile(TaskType.LONG_MEMORY)
    return profile.build_user_message(clean_cell(current_memory), formatted_history)


class MemoryExtractionRunner:
    def __init__(
        self,
        config: MemoryExtractionConfig,
        prompt_text: str,
        client: MemoryExtractionClient | None = None,
        *,
        task_type: TaskType = TaskType.USER_MD,
        create_prompt_text: str = "",
        update_prompt_text: str = "",
    ):
        self.config = config
        self.task_type = TaskType(task_type)
        self.task_profile = get_extraction_task_profile(self.task_type)
        self.document_name = self.task_profile.document_name
        self.prompt_text = update_prompt_text or prompt_text
        self.create_prompt_text = create_prompt_text or self.prompt_text
        self.update_prompt_text = update_prompt_text or prompt_text or self.create_prompt_text
        if self.config.prompt_cache_location != "none" and not self.config.prompt_cache_id:
            self.config.prompt_cache_id = make_prompt_cache_id(
                f"memory_extract_{self.task_type.value}",
                self.config.model,
                self.create_prompt_text,
                self.update_prompt_text,
            )
        self.client = client or (MockMemoryExtractionClient(config) if config.mock else MemoryExtractionClient(config))

    @staticmethod
    def _find_reviewer(rows: list[dict[str, Any]]) -> str:
        for row in rows:
            reviewer = clean_cell(row.get("评测人", ""))
            if reviewer:
                return reviewer
        return "未知"

    @staticmethod
    def _filter_reviewers(df: pd.DataFrame, reviewer_filter: str | None) -> pd.DataFrame:
        if not reviewer_filter:
            return df
        names = [name.strip() for name in re.split(r"[,，]", reviewer_filter) if name.strip()]
        if not names:
            return df
        filtered = df[df["评测人"].isin(names)].copy()
        if filtered.empty:
            available = sorted([x for x in df["评测人"].dropna().unique() if x])
            raise ValueError(f"未找到评测人: {names}。Excel 中可选评测人包括: {available}")
        return filtered

    @staticmethod
    def _count_chunks(sessions: list[list[dict[str, Any]]], chunk_size: int) -> int:
        return sum(math.ceil(len(session) / chunk_size) for session in sessions if session)

    def _process_sessions(
        self,
        sessions: list[list[dict[str, Any]]],
        chunk_size: int,
        total_chunks_for_progress: int,
        wait_for_rate_slot: Callable[[], None],
        progress_callback: Callable[[int, int, str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        chunk_rows_callback: Callable[[list[dict[str, Any]]], None] | None = None,
    ) -> dict[str, Any]:
        completed_chunks = 0
        api_call_count = 0
        final_rows: list[dict[str, Any]] = []
        memory_by_reviewer: dict[str, str] = {}
        last_reviewer = ""
        last_source_reviewer_segment: str | None = None
        status_counts: dict[str, int] = {}
        call_status_counts: dict[str, int] = {}
        parse_status_counts: dict[str, int] = {}
        case_status_counts: dict[str, int] = {}

        for session_index, session_rows in enumerate(sessions, start=1):
            source_session_index = int(session_rows[0].get("__session_index", session_index)) if session_rows else session_index
            reviewer = self._find_reviewer(session_rows)
            source_reviewer_segment = str(session_rows[0].get("__source_reviewer_segment", "")) if session_rows else ""
            if self.task_type == TaskType.LONG_MEMORY:
                reviewer_changed = bool(last_reviewer and reviewer != last_reviewer)
                source_segment_changed = bool(
                    source_reviewer_segment
                    and last_source_reviewer_segment is not None
                    and source_reviewer_segment != last_source_reviewer_segment
                )
                if reviewer_changed or source_segment_changed:
                    memory_by_reviewer.clear()
            current_memory = memory_by_reviewer.get(reviewer, "")
            last_reviewer = reviewer
            last_source_reviewer_segment = source_reviewer_segment or last_source_reviewer_segment

            for chunk_start in range(0, len(session_rows), chunk_size):
                if should_stop is not None and should_stop():
                    break

                chunk = session_rows[chunk_start: chunk_start + chunk_size]
                chunk_id = chunk_start // chunk_size + 1
                history = build_dialogue_history(chunk)
                llm_result = ""
                llm_reasoning = ""
                status = "UNKNOWN"
                error = ""
                old_memory = current_memory
                candidate_document = ""
                parsed_document = ""
                parse_result = DocumentParseResult(None, "not_attempted", 0.0)
                call_status = CallStatus.NOT_ATTEMPTED
                parse_status = ParseStatus.NOT_ATTEMPTED
                case_status = CaseStatus.SKIP
                inheritance_source = InheritanceSource.NONE
                propagation_status = "not_applicable"
                mode_label = "create" if not current_memory else "update"

                if not history:
                    status = "SKIPPED_EMPTY"
                    call_status = CallStatus.SKIPPED
                    parse_result = DocumentParseResult(
                        None,
                        "skipped_empty",
                        0.0,
                        ("当前分块没有可提取的对话正文",),
                    )
                else:
                    if progress_callback:
                        progress_callback(
                            completed_chunks,
                            total_chunks_for_progress,
                            f"请求排队中：Session {source_session_index} Chunk {chunk_id}",
                        )
                    wait_for_rate_slot()
                    if should_stop is not None and should_stop():
                        break

                    api_call_count += 1
                    if progress_callback:
                        progress_callback(
                            completed_chunks,
                            total_chunks_for_progress,
                            f"正在调用模型：Session {source_session_index} Chunk {chunk_id}",
                        )
                    messages = [
                        {
                            "role": "system",
                            "content": self.create_prompt_text if mode_label == "create" else self.update_prompt_text,
                        },
                        {
                            "role": "user",
                            "content": self.task_profile.build_user_message(current_memory, history),
                        },
                    ]
                    success, result, reasoning, error = self.client.chat_with_retry(messages)
                    llm_result = result or ""
                    llm_reasoning = reasoning or ""
                    if success:
                        call_status = CallStatus.SUCCESS
                        parse_result = parse_memory_document(llm_result, self.document_name)
                        parsed = parse_result.document
                        if parsed is None:
                            raw_fallback = (
                                normalize_memory_document_body(llm_result, self.document_name)
                                if self.task_profile.parser_fallback_policy == "raw_output_requires_review"
                                else ""
                            )
                            if raw_fallback:
                                status = "SUCCESS_UNSTRUCTURED"
                                candidate_document = raw_fallback
                                parse_status = ParseStatus.RAW_FALLBACK
                                case_status = CaseStatus.REVIEW_REQUIRED
                                inheritance_source = InheritanceSource.RAW_OUTPUT
                                propagation_status = "blocked_low_confidence"
                                parse_result = DocumentParseResult(
                                    raw_fallback,
                                    f"raw_fallback:{parse_result.method}",
                                    0.25,
                                    parse_result.warnings
                                    + (
                                        "未可靠识别正文边界：该输出保留为待复核候选，但不会继承到后续分块；"
                                        "后续继续使用最近一次可靠正文",
                                    ),
                                )
                            else:
                                status = "OUTPUT_EMPTY"
                                parse_status = ParseStatus.EMPTY
                                error = error or f"模型未返回可用于评测的 {self.document_name} 内容"
                        else:
                            status = "SUCCESS"
                            parsed_document = parsed
                            parse_status = ParseStatus.STRUCTURED if parsed else ParseStatus.EMPTY
                            case_status = CaseStatus.READY
                            if self.task_profile.preserve_previous_on_empty and not parsed:
                                candidate_document = current_memory
                                inheritance_source = InheritanceSource.PREVIOUS_DOCUMENT
                                propagation_status = "kept_previous_document"
                                parse_result = DocumentParseResult(
                                    parsed,
                                    parse_result.method,
                                    parse_result.confidence,
                                    parse_result.warnings + ("正文为空，按长期记忆无变化处理",),
                                )
                            else:
                                candidate_document = parsed
                                inheritance_source = InheritanceSource.PARSED_DOCUMENT
                                current_memory = parsed
                                memory_by_reviewer[reviewer] = current_memory
                                propagation_status = "accepted"
                    else:
                        stop_after_retry = bool(should_stop is not None and should_stop())
                        status = "STOPPED" if stop_after_retry else "API_FAILED"
                        call_status = CallStatus.STOPPED if stop_after_retry else CallStatus.FAILED
                        error = error or ("记忆提取收到终止请求" if stop_after_retry else llm_result)
                        parse_result = DocumentParseResult(
                            None,
                            "stopped" if stop_after_retry else "api_failed",
                            0.0,
                            (error[:500],) if error else (),
                        )

                status_counts[status] = status_counts.get(status, 0) + 1
                call_status_counts[call_status.value] = call_status_counts.get(call_status.value, 0) + 1
                parse_status_counts[parse_status.value] = parse_status_counts.get(parse_status.value, 0) + 1
                case_status_counts[case_status.value] = case_status_counts.get(case_status.value, 0) + 1
                completed_chunks += 1

                chunk_output_rows: list[dict[str, Any]] = []
                for row_index, row in enumerate(chunk):
                    is_last = row_index == len(chunk) - 1
                    output_row = dict(row)
                    common_output = {
                        "session_id": source_session_index,
                        "chunk_id": chunk_id,
                        "评测人": reviewer,
                        "status": status if is_last else "",
                        "task_profile_id": self.task_profile.profile_id if is_last else "",
                        "call_status": call_status.value if is_last else "",
                        "parse_status": parse_status.value if is_last else "",
                        "case_status": case_status.value if is_last else "",
                        "error": error if is_last else "",
                        "reasoning": llm_reasoning if is_last else "",
                        "old_effective_document": old_memory if is_last else "",
                        "raw_output": llm_result if is_last else "",
                        "parsed_document": parsed_document if is_last else "",
                        "effective_document": candidate_document if is_last else "",
                        "inheritance_source": inheritance_source.value if is_last else "",
                        "propagation_status": propagation_status if is_last else "",
                        "parse_method": parse_result.method if is_last else "",
                        "parse_confidence": parse_result.confidence if is_last else "",
                        "parse_warnings": "；".join(parse_result.warnings) if is_last else "",
                    }
                    if self.task_type == TaskType.LONG_MEMORY:
                        common_output.update({
                            "当前使用的模板": mode_label if is_last else "",
                            "旧MEMORY.md": old_memory if is_last else "",
                            "MEMORY.md": candidate_document if is_last else "",
                            "模型原始返回": llm_result if is_last else "",
                        })
                    else:
                        common_output.update({
                            "result": llm_result if is_last else "",
                            "user.md": candidate_document if is_last else "",
                        })
                    output_row.update(common_output)
                    chunk_output_rows.append(output_row)

                final_rows.extend(chunk_output_rows)
                if chunk_rows_callback is not None:
                    chunk_rows_callback(chunk_output_rows)

                if progress_callback:
                    progress_callback(
                        completed_chunks,
                        total_chunks_for_progress,
                        f"已完成 Session {source_session_index} Chunk {chunk_id}：{status}",
                    )

            if should_stop is not None and should_stop():
                break

        return {
            "final_rows": final_rows,
            "completed_chunks": completed_chunks,
            "api_calls": api_call_count,
            "status_counts": status_counts,
            "call_status_counts": call_status_counts,
            "parse_status_counts": parse_status_counts,
            "case_status_counts": case_status_counts,
            "sessions": len(sessions),
            "stopped": bool(should_stop is not None and should_stop()),
        }

    def process_excel(
        self,
        file_path: str | Path,
        output_path: str | Path,
        sheet_name: str | int | None = 0,
        reviewer_filter: str | None = None,
        chunk_size: int = 10,
        progress_callback: Callable[[int, int, str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        emit_parallel_chunk_progress: bool = False,
        priority_provider: Callable[[], int] | None = None,
        concurrency_provider: Callable[[], int] | None = None,
    ) -> dict[str, Any]:
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if not self.create_prompt_text.strip() or not self.update_prompt_text.strip():
            raise ValueError("提取提示词为空")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        journal_path = output_path.with_suffix(".journal.jsonl")
        atomic_write_jsonl(journal_path, [])

        def append_chunk_rows(rows: list[dict[str, Any]]) -> None:
            append_jsonl_rows(journal_path, rows)

        df = pd.read_excel(file_path, sheet_name=sheet_name if sheet_name not in ("", None) else 0)
        required_cols = {"轮次", "query", "answer", "评测人"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Excel 缺少必要列: {sorted(missing)}")

        df = df.copy()
        df["评测人"] = df["评测人"].apply(clean_cell)
        df["query"] = df["query"].apply(clean_cell)
        df["answer"] = df["answer"].apply(clean_cell)
        df["轮次"] = pd.to_numeric(df["轮次"], errors="coerce").fillna(0).astype(int)
        df = self._filter_reviewers(df, reviewer_filter)
        df = df.reset_index(drop=True)
        df["__source_order"] = range(len(df))

        concurrency = min(100, max(1, int(getattr(self.config, "concurrency", 1) or 1)))
        request_interval = float(self.config.request_interval or 0)
        rate_scope = api_rate_scope(self.config.api_base, self.config.api_token)

        def current_priority() -> int:
            if priority_provider is not None:
                return max(1, min(10, int(priority_provider() or 5)))
            return max(1, min(10, int(getattr(self.config, "priority", 5) or 5)))

        def current_concurrency(limit: int) -> int:
            if concurrency_provider is not None:
                return min(limit, 100, max(1, int(concurrency_provider() or concurrency)))
            return min(limit, concurrency)

        def wait_for_rate_slot() -> None:
            wait_for_global_rate_slot(
                rate_scope,
                request_interval,
                disabled=bool(self.config.mock),
                should_stop=should_stop,
                priority=current_priority(),
            )

        if hasattr(self.client, "rate_limit_wait_callback"):
            self.client.rate_limit_wait_callback = wait_for_rate_slot
        if hasattr(self.client, "should_stop_callback"):
            self.client.should_stop_callback = should_stop

        global_sessions = split_sessions(df)
        for session_index, session_rows in enumerate(global_sessions, start=1):
            for row in session_rows:
                row["__session_index"] = session_index
                source_order = int(row.get("__source_order", 0))
                df.loc[source_order, "__session_index"] = session_index
        total_chunks = self._count_chunks(global_sessions, chunk_size)
        final_rows: list[dict[str, Any]] = []
        completed_chunks = 0
        live_completed_chunks = 0
        api_call_count = 0
        session_count = len(global_sessions)
        status_counts: dict[str, int] = {}
        call_status_counts: dict[str, int] = {}
        parse_status_counts: dict[str, int] = {}
        case_status_counts: dict[str, int] = {}
        progress_lock = Lock()

        def merge_result(result: dict[str, Any]) -> None:
            nonlocal completed_chunks, api_call_count
            final_rows.extend(result["final_rows"])
            completed_chunks += int(result.get("completed_chunks", 0) or 0)
            api_call_count += int(result.get("api_calls", 0) or 0)
            for key, value in (result.get("status_counts") or {}).items():
                status_counts[key] = status_counts.get(key, 0) + int(value)
            for target, source_key in (
                (call_status_counts, "call_status_counts"),
                (parse_status_counts, "parse_status_counts"),
                (case_status_counts, "case_status_counts"),
            ):
                for key, value in (result.get(source_key) or {}).items():
                    target[key] = target.get(key, 0) + int(value)

        def make_parallel_progress_callback(reviewer: str) -> Callable[[int, int, str], None] | None:
            if not progress_callback or not emit_parallel_chunk_progress:
                return None
            reviewer_done = {"value": 0}

            def on_parallel_progress(done: int, _total: int, message: str) -> None:
                nonlocal live_completed_chunks
                with progress_lock:
                    delta = max(0, int(done) - reviewer_done["value"])
                    if delta:
                        live_completed_chunks += delta
                        reviewer_done["value"] = int(done)
                    current = live_completed_chunks
                progress_callback(current, total_chunks, f"评测人 {reviewer}：{message}")

            return on_parallel_progress

        if concurrency <= 1 and concurrency_provider is None:
            result = self._process_sessions(
                global_sessions,
                chunk_size,
                total_chunks,
                wait_for_rate_slot,
                progress_callback=progress_callback,
                should_stop=should_stop,
                chunk_rows_callback=append_chunk_rows,
            )
            merge_result(result)
        else:
            if self.task_type == TaskType.LONG_MEMORY:
                reviewer_segments = (
                    df["__source_reviewer_segment"]
                    if "__source_reviewer_segment" in df.columns
                    else df["评测人"].ne(df["评测人"].shift()).cumsum()
                )
                groups = [
                    (f"{str(group.iloc[0]['评测人'] or '未知')}（区段 {segment_id}）", group.copy())
                    for segment_id, group in df.groupby(reviewer_segments, sort=False)
                ]
            else:
                groups = [
                    (str(reviewer or "未知"), group.copy())
                    for reviewer, group in df.groupby("评测人", sort=False, dropna=False)
                ]
            if len(groups) <= 1:
                result = self._process_sessions(
                    global_sessions,
                    chunk_size,
                    total_chunks,
                    wait_for_rate_slot,
                    progress_callback=progress_callback,
                    should_stop=should_stop,
                    chunk_rows_callback=append_chunk_rows,
                )
                merge_result(result)
            else:
                group_iter = iter(groups)
                future_map = {}

                def submit_next(executor: ThreadPoolExecutor) -> bool:
                    if should_stop is not None and should_stop():
                        return False
                    try:
                        reviewer, group_df = next(group_iter)
                    except StopIteration:
                        return False
                    future_map[executor.submit(
                        self._process_sessions,
                        split_sessions(group_df),
                        chunk_size,
                        total_chunks,
                        wait_for_rate_slot,
                        make_parallel_progress_callback(reviewer),
                        should_stop,
                        append_chunk_rows,
                    )] = reviewer
                    return True

                with ThreadPoolExecutor(max_workers=min(100, len(groups))) as executor:
                    for _ in range(current_concurrency(len(groups))):
                        if not submit_next(executor):
                            break
                    while future_map:
                        done, _pending = wait(set(future_map), return_when=FIRST_COMPLETED)
                        for future in done:
                            reviewer = future_map.pop(future)
                            result = future.result()
                            merge_result(result)
                            if progress_callback and not emit_parallel_chunk_progress:
                                progress_callback(
                                    completed_chunks,
                                    total_chunks,
                                    f"已完成评测人 {reviewer}：{completed_chunks}/{total_chunks}",
                                )
                        while len(future_map) < current_concurrency(len(groups)) and submit_next(executor):
                            pass

        final_rows = sorted(final_rows, key=lambda row: int(row.get("__source_order", 0)))
        for row in final_rows:
            row.pop("__source_order", None)
            row.pop("__session_index", None)
            row.pop("__source_reviewer_segment", None)
        excel_buffer = BytesIO()
        pd.DataFrame(final_rows).to_excel(excel_buffer, index=False)
        atomic_write_bytes(output_path, excel_buffer.getvalue())

        return {
            "output_path": str(output_path),
            "journal_path": str(journal_path),
            "rows": len(final_rows),
            "sessions": session_count,
            "chunks": total_chunks,
            "api_calls": api_call_count,
            "status_counts": status_counts,
            "call_status_counts": call_status_counts,
            "parse_status_counts": parse_status_counts,
            "case_status_counts": case_status_counts,
            "concurrency": concurrency,
            "task_type": self.task_type.value,
            "task_profile_id": self.task_profile.profile_id,
            "document_name": self.document_name,
            "stopped": bool(should_stop is not None and should_stop()),
        }
