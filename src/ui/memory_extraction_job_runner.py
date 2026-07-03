from __future__ import annotations

import json
import math
import re
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.extraction.memory_extractor import (
    EXTRACTION_OUTPUT_DIR,
    MemoryExtractionConfig,
    MemoryExtractionRunner,
    clean_cell,
    sanitize_filename,
    split_sessions,
)
from src.ui.data_service import prepare_cases_from_run_output, save_cases
from src.ui.state_io import atomic_write_json, state_file_lock


MEMORY_EXTRACTION_JOBS_DIR = EXTRACTION_OUTPUT_DIR / "jobs"


@dataclass
class MemoryExtractionJobConfig:
    job_id: str
    input_path: str
    output_path: str
    prompt_text: str
    prompt_version: str
    sheet_name: str | int | None = 0
    reviewer_filter: str = ""
    chunk_size: int = 10
    auto_make_cases: bool = True
    case_model_name: str = "unknown"
    case_prompt_version: str = "unknown"
    extraction_config: MemoryExtractionConfig = field(default_factory=MemoryExtractionConfig)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_time(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def job_dir(job_id: str) -> Path:
    return MEMORY_EXTRACTION_JOBS_DIR / sanitize_filename(job_id)


def state_path(job_id: str) -> Path:
    return job_dir(job_id) / "state.json"


def stop_path(job_id: str) -> Path:
    return job_dir(job_id) / "STOP"


def read_memory_extraction_job_state(job_id: str) -> dict[str, Any]:
    path = state_path(job_id)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_memory_extraction_job_state(job_id: str, state: dict[str, Any]) -> None:
    path = state_path(job_id)
    with state_file_lock(path):
        state["heartbeat_at"] = utc_now()
        atomic_write_json(path, state)


def list_memory_extraction_job_ids() -> list[str]:
    if not MEMORY_EXTRACTION_JOBS_DIR.exists():
        return []
    paths = [path for path in MEMORY_EXTRACTION_JOBS_DIR.iterdir() if path.is_dir()]
    paths.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return [path.name for path in paths]


def request_memory_extraction_stop(job_id: str) -> None:
    path = stop_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(utc_now(), encoding="utf-8")


def memory_extraction_stop_requested(job_id: str) -> bool:
    return stop_path(job_id).exists()


def memory_extraction_job_is_running(job_id: str) -> bool:
    state = read_memory_extraction_job_state(job_id)
    if memory_extraction_job_is_stale(state):
        mark_memory_extraction_job_interrupted(job_id)
        return False
    return state.get("status") == "running"


def memory_extraction_job_stale_after_seconds(state: dict[str, Any]) -> float:
    config = state.get("config") or {}
    extraction_config = config.get("extraction_config") if isinstance(config.get("extraction_config"), dict) else {}
    timeout = float(extraction_config.get("timeout") or 120)
    retries = float(extraction_config.get("max_retries") or 2)
    backoff = float(extraction_config.get("retry_sleep") or 15)
    interval = float(extraction_config.get("request_interval") or 0)
    return max(300.0, timeout * 2 + (retries + 1) * max(backoff, 5.0) + interval + 120.0)


def memory_extraction_job_is_stale(state: dict[str, Any]) -> bool:
    if state.get("status") != "running":
        return False
    heartbeat = _parse_time(str(state.get("heartbeat_at") or state.get("updated_at") or ""))
    if heartbeat is None:
        return False
    if heartbeat.tzinfo is None:
        heartbeat = heartbeat.replace(tzinfo=timezone.utc)
    elapsed = (datetime.now(timezone.utc) - heartbeat).total_seconds()
    return elapsed > memory_extraction_job_stale_after_seconds(state)


def mark_memory_extraction_job_interrupted(job_id: str) -> dict[str, Any]:
    state = read_memory_extraction_job_state(job_id)
    if not state or state.get("status") != "running":
        return state
    state["status"] = "interrupted"
    state["stage"] = "已中断"
    state["message"] = "后台记忆提取任务可能已中断：长时间没有心跳。可以重新启动任务，或使用已生成的中间文件继续后续流程。"
    state["finished_at"] = utc_now()
    state["updated_at"] = utc_now()
    write_memory_extraction_job_state(job_id, state)
    return state


def _safe_config(config: MemoryExtractionJobConfig) -> dict[str, Any]:
    value = asdict(config)
    value.pop("prompt_text", None)
    extraction_config = value.get("extraction_config")
    if isinstance(extraction_config, dict):
        extraction_config.pop("api_token", None)
    return value


def _write_state(
    config: MemoryExtractionJobConfig,
    *,
    status: str = "running",
    stage: str,
    done: int,
    total: int,
    message: str,
    started_at: str,
    extra: dict[str, Any] | None = None,
) -> None:
    state = {
        "job_id": config.job_id,
        "status": status,
        "stage": stage,
        "done": int(done),
        "total": int(total),
        "message": message,
        "input_path": config.input_path,
        "output_path": config.output_path,
        "journal_path": str(Path(config.output_path).with_suffix(".journal.jsonl")),
        "started_at": started_at,
        "updated_at": utc_now(),
        "config": _safe_config(config),
    }
    if extra:
        state.update(extra)
    write_memory_extraction_job_state(config.job_id, state)


def estimate_total_chunks(config: MemoryExtractionJobConfig) -> tuple[int, int]:
    df = pd.read_excel(config.input_path, sheet_name=config.sheet_name if config.sheet_name not in ("", None) else 0)
    required_cols = {"轮次", "query", "answer", "评测人"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Excel 缺少必要列: {sorted(missing)}")

    df = df.copy()
    df["评测人"] = df["评测人"].apply(clean_cell)
    df["轮次"] = pd.to_numeric(df["轮次"], errors="coerce").fillna(0).astype(int)
    reviewer_filter = str(config.reviewer_filter or "").strip()
    if reviewer_filter:
        names = [name.strip() for name in re.split(r"[,，]", reviewer_filter) if name.strip()]
        if names:
            df = df[df["评测人"].isin(names)]

    sessions = split_sessions(df)
    chunk_size = max(1, int(config.chunk_size or 1))
    total_chunks = sum(math.ceil(len(session) / chunk_size) for session in sessions)
    return len(sessions), total_chunks


def run_memory_extraction_job(config: MemoryExtractionJobConfig) -> None:
    started_at = utc_now()
    if stop_path(config.job_id).exists():
        stop_path(config.job_id).unlink()

    _write_state(
        config,
        stage="准备",
        done=0,
        total=0,
        message="后台记忆提取任务已启动，正在读取输入 Excel。",
        started_at=started_at,
    )

    try:
        session_count, estimated_chunks = estimate_total_chunks(config)
        _write_state(
            config,
            stage="准备",
            done=0,
            total=estimated_chunks,
            message=f"已读取输入 Excel：{session_count} 个 session，预计 {estimated_chunks} 个提取 chunk，准备调用模型。",
            started_at=started_at,
            extra={"estimated_sessions": session_count, "estimated_chunks": estimated_chunks},
        )

        runner = MemoryExtractionRunner(
            config=config.extraction_config,
            prompt_text=config.prompt_text,
        )

        def on_progress(done: int, total: int, message: str) -> None:
            _write_state(
                config,
                stage="记忆提取",
                done=done,
                total=total,
                message=message,
                started_at=started_at,
            )

        stats = runner.process_excel(
            config.input_path,
            config.output_path,
            sheet_name=config.sheet_name,
            reviewer_filter=config.reviewer_filter or None,
            chunk_size=int(config.chunk_size),
            progress_callback=on_progress,
            should_stop=lambda: memory_extraction_stop_requested(config.job_id),
            emit_parallel_chunk_progress=True,
        )

        if stats.get("stopped"):
            _write_state(
                config,
                status="stopped",
                stage="已终止",
                done=int(stats.get("chunks", 0) or 0),
                total=int(stats.get("chunks", 0) or 0),
                message="记忆提取已按终止请求停止。",
                started_at=started_at,
                extra={"stats": stats, "finished_at": utc_now()},
            )
            return

        extra: dict[str, Any] = {"stats": stats}
        done = int(stats.get("chunks", 0) or 0)
        total = int(stats.get("chunks", 0) or 0)
        message = f"记忆提取完成：{config.output_path}"

        if config.auto_make_cases:
            _write_state(
                config,
                stage="生成 case",
                done=done,
                total=total,
                message="记忆提取完成，正在生成评测 case。",
                started_at=started_at,
                extra=extra,
            )
            cases, missed_cases, convert_stats = prepare_cases_from_run_output(
                config.output_path,
                model=config.case_model_name,
                prompt_version=config.case_prompt_version,
                chunk_size=int(config.chunk_size),
                return_missed=True,
            )
            case_filename = (
                f"{sanitize_filename(config.case_model_name)}_"
                f"{sanitize_filename(config.case_prompt_version)}_user_md_cases_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            )
            cases_path = save_cases(cases, case_filename)
            missed_path = ""
            if missed_cases:
                missed_filename = (
                    f"{sanitize_filename(config.case_model_name)}_"
                    f"{sanitize_filename(config.case_prompt_version)}_user_md_missed_cases_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
                )
                missed_path = save_cases(missed_cases, missed_filename)
            extra.update({
                "case_stats": convert_stats,
                "cases_path": cases_path,
                "missed_cases_path": missed_path,
            })
            message = f"记忆提取和 case 生成完成：完整 case {len(cases)} 条，漏抽 case {len(missed_cases)} 条。"

        preview_rows = []
        output_path = Path(config.output_path)
        if output_path.exists():
            preview_rows = pd.read_excel(output_path).fillna("").head(50).to_dict("records")

        _write_state(
            config,
            status="completed",
            stage="完成",
            done=total,
            total=total,
            message=message,
            started_at=started_at,
            extra={**extra, "preview_rows": preview_rows, "finished_at": utc_now()},
        )

    except Exception as exc:
        state = read_memory_extraction_job_state(config.job_id)
        _write_state(
            config,
            status="failed",
            stage="失败",
            done=int(state.get("done", 0) or 0),
            total=int(state.get("total", 0) or 0),
            message=f"记忆提取失败：{type(exc).__name__}: {exc}",
            started_at=started_at,
            extra={
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "finished_at": utc_now(),
            },
        )
