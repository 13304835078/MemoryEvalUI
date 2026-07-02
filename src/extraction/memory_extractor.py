from __future__ import annotations

import json
import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Callable

import pandas as pd
import requests
import yaml

from src.eval.judge_client import RealJudgeClient
from src.schema import EvalConfig


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTRACTION_OUTPUT_DIR = PROJECT_ROOT / "data" / "extractions"


@dataclass
class MemoryExtractionConfig:
    api_base: str = ""
    api_token: str = ""
    model: str = ""
    max_tokens: int = 50000
    timeout: int = 100
    request_interval: float = 10.0
    max_retries: int = 2
    retry_sleep: float = 15.0
    enable_thinking: bool = True
    send_enable_thinking: bool = True
    skip_special_tokens: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    concurrency: int = 1
    mock: bool = False

    @classmethod
    def from_eval_config(
        cls,
        config: EvalConfig,
        model: str = "",
        max_tokens: int = 50000,
        request_interval: float | None = None,
        max_retries: int | None = None,
        retry_sleep: float | None = None,
        enable_thinking: bool = True,
        timeout: int | None = None,
    ) -> "MemoryExtractionConfig":
        return cls(
            api_base=config.judge_api_base_url,
            api_token=config.judge_api_bearer_token,
            model=model or config.judge_model,
            max_tokens=max_tokens,
            timeout=int(timeout if timeout is not None else (config.judge_timeout or 100)),
            request_interval=float(
                request_interval if request_interval is not None else getattr(config, "judge_request_interval", 10.0)
            ),
            max_retries=int(max_retries if max_retries is not None else max(0, (config.judge_max_retries or 1) - 1)),
            retry_sleep=float(retry_sleep if retry_sleep is not None else getattr(config, "judge_qps_backoff", 15.0)),
            enable_thinking=enable_thinking,
            send_enable_thinking=True,
            skip_special_tokens=False,
            temperature=getattr(config, "judge_temperature", None),
            top_p=getattr(config, "judge_top_p", None),
            top_k=getattr(config, "judge_top_k", None),
            concurrency=min(100, max(1, int(getattr(config, "judge_concurrency", 1) or 1))),
            mock=bool(getattr(config, "mock", False)),
        )


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


def normalize_user_md_body(body: str | None) -> str:
    if body is None:
        return ""

    text = str(body).strip()
    if not text:
        return ""

    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^---+\s*USER\.md\s*---+$", line, flags=re.IGNORECASE):
            continue
        if re.match(r"^#{1,6}\s*(Think|Reasoning|Output|USER\.md)\s*$", line, flags=re.IGNORECASE):
            continue

        line = re.sub(r"^[*•]\s+", "- ", line)
        line = re.sub(r"^\d+[.、]\s+", "- ", line)
        lines.append(line)

    return "\n".join(lines).strip()


def extract_user_md(text: str | None) -> str | None:
    if text is None:
        return None

    raw = str(text).strip()
    if not raw:
        return None

    raw = re.sub(r"^\s*```(?:markdown|md|text)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```\s*$", "", raw)

    has_think = re.search(r"(?im)^\s*#{1,6}\s*(Think|Reasoning)\b", raw)
    has_output_or_user_md = re.search(
        r"(?im)^\s*#{1,6}\s*Output\s*$|---+\s*USER\.md\s*---+",
        raw,
    )
    if has_think and not has_output_or_user_md:
        return None

    markers = list(re.finditer(
        r"(?im)^\s*#{1,6}\s*Output\s*$|---+\s*USER\.md\s*---+",
        raw,
    ))
    if markers:
        content = raw[markers[-1].end():].strip()
        return normalize_user_md_body(content)

    if re.search(r"(?m)^\s*[-*•]\s*\S+[:：]", raw):
        return normalize_user_md_body(raw)

    return None


def extract_answer_from_response(response_data: dict) -> tuple[str | None, str]:
    if not isinstance(response_data, dict):
        return None, ""

    choices = response_data.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            message = choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
                if content is not None:
                    return str(content), str(reasoning or "")
                if reasoning:
                    return str(reasoning), str(reasoning)
            if "text" in choice:
                return str(choice.get("text") or ""), ""

    for key in ("result", "content", "message"):
        if key in response_data:
            return str(response_data.get(key) or ""), ""

    data = response_data.get("data")
    if isinstance(data, dict):
        for key in ("content", "answer", "response"):
            if key in data:
                return str(data.get(key) or ""), ""
    elif data is not None:
        return str(data), ""

    return None, ""


def load_generation_prompt(prompt_file: str | Path) -> str:
    path = Path(prompt_file)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        loaded = yaml.safe_load(text)
        if isinstance(loaded, dict):
            for key in ("user", "prompt", "system"):
                if key in loaded:
                    return str(loaded[key])
        if isinstance(loaded, str):
            return loaded
        raise ValueError(f"无法从 YAML 提取 prompt：{path}")
    return text


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
    current_user_md = clean_cell(current_user_md)
    existing_block = f"--- USER.md ---\n{current_user_md}" if current_user_md else "--- USER.md ---"
    return (
        f"## [现有文件内容]\n"
        f"{existing_block}\n\n"
        f"## [最新对话内容]\n"
        f"{formatted_history}"
    )


class MemoryExtractionClient:
    def __init__(self, config: MemoryExtractionConfig):
        self.config = config

    def _build_payload(self, messages: list[dict[str, str]]) -> dict:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": int(self.config.max_tokens),
            "stream": False,
            "messages": messages,
            "extra_body": {
                "skip_special_tokens": bool(self.config.skip_special_tokens),
            },
        }
        if self.config.send_enable_thinking:
            payload["extra_body"]["enable_thinking"] = bool(self.config.enable_thinking)
        if self.config.temperature is not None:
            payload["temperature"] = float(self.config.temperature)
        if self.config.top_p is not None:
            payload["top_p"] = float(self.config.top_p)
        if self.config.top_k not in (None, ""):
            payload["top_k"] = int(self.config.top_k)
        return payload

    def _request_once(self, messages: list[dict[str, str]]) -> tuple[bool, str, str, str]:
        url = RealJudgeClient._normalize_chat_completions_url(self.config.api_base)
        headers = {
            "Content-Type": "application/json",
            "Authorization": RealJudgeClient._build_auth_header(self.config.api_token),
        }
        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(self._build_payload(messages), ensure_ascii=False).encode("utf-8"),
            timeout=int(self.config.timeout),
        )
        raw_text = response.text
        response.raise_for_status()
        data = response.json()
        answer, reasoning = extract_answer_from_response(data)
        if answer is None:
            return False, f"无法从响应中提取答案。完整响应: {raw_text[:1000]}", "", raw_text
        return True, answer, reasoning or "", raw_text

    def chat_with_retry(self, messages: list[dict[str, str]]) -> tuple[bool, str, str, str]:
        last_error = ""
        attempts = int(self.config.max_retries) + 1
        for attempt in range(1, attempts + 1):
            try:
                success, answer, reasoning, raw = self._request_once(messages)
                if success:
                    return True, answer or "", reasoning or "", ""
                last_error = answer or raw or "API 返回失败"
            except requests.exceptions.RequestException as exc:
                last_error = f"请求错误: {exc}"
            except json.JSONDecodeError as exc:
                last_error = f"JSON解析错误: {exc}"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"

            logger.warning("记忆提取 API 调用失败，第 %s/%s 次：%s", attempt, attempts, last_error)
            if attempt < attempts and self.config.retry_sleep > 0:
                sleep_seconds = float(self.config.retry_sleep)
                if RealJudgeClient._is_rate_limit_error(last_error):
                    sleep_seconds = RealJudgeClient(self.config_to_eval_config())._get_rate_limit_backoff(last_error)
                time.sleep(sleep_seconds)

        return False, f"API CALL FAILED: {last_error}", "", last_error

    def config_to_eval_config(self) -> EvalConfig:
        return EvalConfig(
            judge_qps_backoff=float(self.config.retry_sleep or 12.0),
        )


class MockMemoryExtractionClient(MemoryExtractionClient):
    def __init__(self, config: MemoryExtractionConfig):
        super().__init__(config)
        self.counter = 0

    def chat_with_retry(self, messages: list[dict[str, str]]) -> tuple[bool, str, str, str]:
        self.counter += 1
        user_message = messages[-1]["content"] if messages else ""
        existing = ""
        marker = "## [现有文件内容]"
        dialogue_marker = "## [最新对话内容]"
        if marker in user_message and dialogue_marker in user_message:
            existing = user_message.split(marker, 1)[1].split(dialogue_marker, 1)[0].strip()
            existing = existing.replace("--- USER.md ---", "").strip()
        lines = [line.strip() for line in existing.splitlines() if line.strip()]
        lines.append(f"- [MOCK] 已处理第 {self.counter} 个记忆提取 chunk")
        content = "# Output\n--- USER.md ---\n" + "\n".join(lines)
        return True, content, "[MOCK] 未调用真实接口。", ""


class MemoryExtractionRunner:
    def __init__(
        self,
        config: MemoryExtractionConfig,
        prompt_text: str,
        client: MemoryExtractionClient | None = None,
    ):
        self.config = config
        self.prompt_text = prompt_text
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
    ) -> dict[str, Any]:
        completed_chunks = 0
        api_call_count = 0
        final_rows: list[dict[str, Any]] = []
        user_md_by_reviewer: dict[str, str] = {}
        status_counts: dict[str, int] = {}

        for session_index, session_rows in enumerate(sessions, start=1):
            source_session_index = int(session_rows[0].get("__session_index", session_index)) if session_rows else session_index
            reviewer = self._find_reviewer(session_rows)
            current_user_md = user_md_by_reviewer.get(reviewer, "")

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

                if not history:
                    status = "SKIPPED_EMPTY"
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
                        {"role": "system", "content": self.prompt_text},
                        {"role": "user", "content": build_user_prompt(current_user_md, history)},
                    ]
                    success, result, reasoning, error = self.client.chat_with_retry(messages)
                    llm_result = result or ""
                    llm_reasoning = reasoning or ""
                    if success:
                        parsed = extract_user_md(llm_result)
                        if parsed is None:
                            status = "PARSE_FAILED"
                            error = error or "无法从模型输出中解析 USER.md"
                        else:
                            status = "SUCCESS"
                            current_user_md = parsed
                            user_md_by_reviewer[reviewer] = current_user_md
                    else:
                        status = "API_FAILED"
                        error = error or llm_result

                status_counts[status] = status_counts.get(status, 0) + 1
                completed_chunks += 1

                for row_index, row in enumerate(chunk):
                    is_last = row_index == len(chunk) - 1
                    output_row = dict(row)
                    output_row.update({
                        "session_id": source_session_index,
                        "chunk_id": chunk_id,
                        "评测人": reviewer,
                        "status": status if is_last else "",
                        "error": error if is_last else "",
                        "result": llm_result if is_last else "",
                        "reasoning": llm_reasoning if is_last else "",
                        "user.md": current_user_md if is_last else "",
                    })
                    final_rows.append(output_row)

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
    ) -> dict[str, Any]:
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if not self.prompt_text.strip():
            raise ValueError("提取提示词为空")

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
        rate_lock = Lock()
        next_request_at = {"value": time.monotonic()}

        def wait_for_rate_slot() -> None:
            if request_interval <= 0:
                return
            with rate_lock:
                now = time.monotonic()
                wait_seconds = max(0.0, next_request_at["value"] - now)
                next_request_at["value"] = max(now, next_request_at["value"]) + request_interval

            while wait_seconds > 0:
                if should_stop is not None and should_stop():
                    return
                sleep_seconds = min(1.0, wait_seconds)
                time.sleep(sleep_seconds)
                wait_seconds -= sleep_seconds

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
        progress_lock = Lock()

        def merge_result(result: dict[str, Any]) -> None:
            nonlocal completed_chunks, api_call_count
            final_rows.extend(result["final_rows"])
            completed_chunks += int(result.get("completed_chunks", 0) or 0)
            api_call_count += int(result.get("api_calls", 0) or 0)
            for key, value in (result.get("status_counts") or {}).items():
                status_counts[key] = status_counts.get(key, 0) + int(value)

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

        if concurrency <= 1:
            result = self._process_sessions(
                global_sessions,
                chunk_size,
                total_chunks,
                wait_for_rate_slot,
                progress_callback=progress_callback,
                should_stop=should_stop,
            )
            merge_result(result)
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
                )
                merge_result(result)
            else:
                with ThreadPoolExecutor(max_workers=min(concurrency, len(groups))) as executor:
                    future_map = {
                        executor.submit(
                            self._process_sessions,
                            split_sessions(group_df),
                            chunk_size,
                            total_chunks,
                            wait_for_rate_slot,
                            make_parallel_progress_callback(reviewer),
                            should_stop,
                        ): reviewer
                        for reviewer, group_df in groups
                    }
                    for future in as_completed(future_map):
                        reviewer = future_map[future]
                        result = future.result()
                        merge_result(result)
                        if progress_callback and not emit_parallel_chunk_progress:
                            progress_callback(
                                completed_chunks,
                                total_chunks,
                                f"已完成评测人 {reviewer}：{completed_chunks}/{total_chunks}",
                            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_rows = sorted(final_rows, key=lambda row: int(row.get("__source_order", 0)))
        for row in final_rows:
            row.pop("__source_order", None)
            row.pop("__session_index", None)
        pd.DataFrame(final_rows).to_excel(output_path, index=False)

        return {
            "output_path": str(output_path),
            "rows": len(final_rows),
            "sessions": session_count,
            "chunks": total_chunks,
            "api_calls": api_call_count,
            "status_counts": status_counts,
            "concurrency": concurrency,
            "stopped": bool(should_stop is not None and should_stop()),
        }
