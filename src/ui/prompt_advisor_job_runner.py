from __future__ import annotations

import json
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.runtime_paths import DATA_DIR
from src.schema import EvalConfig
from src.persistence import atomic_write_text
from src.ui.background_tasks import (
    list_task_job_ids,
    parse_time as _parse_time,
    read_json_state,
    request_stop_file,
    stop_file_exists,
    task_job_dir,
    task_state_path,
    task_stop_path,
    utc_datetime,
    utc_now,
)
from src.ui.prompt_advisor import call_prompt_advisor
from src.ui.state_io import atomic_write_json


PROMPT_ADVISOR_JOBS_DIR = DATA_DIR / "prompt_advisor_jobs"


class PromptAdvisorJobStopped(Exception):
    pass


@dataclass
class PromptAdvisorJobConfig:
    job_id: str
    task_type: str
    evidence: list[dict[str, Any]]
    current_judge_prompt: str
    extraction_prompt: str = ""
    target: str = "judge_prompt"
    advisor_mode: str = "absolute_eval"
    min_evidence: int = 3
    source_name: str = ""
    eval_config: EvalConfig = field(default_factory=EvalConfig)


def job_dir(job_id: str) -> Path:
    return task_job_dir(PROMPT_ADVISOR_JOBS_DIR, job_id)


def state_path(job_id: str) -> Path:
    return task_state_path(PROMPT_ADVISOR_JOBS_DIR, job_id)


def stop_path(job_id: str) -> Path:
    return task_stop_path(PROMPT_ADVISOR_JOBS_DIR, job_id)


def result_path(job_id: str) -> Path:
    return job_dir(job_id) / "result.json"


def raw_path(job_id: str) -> Path:
    return job_dir(job_id) / "raw.txt"


def read_prompt_advisor_job_state(job_id: str) -> dict[str, Any]:
    return read_json_state(state_path(job_id))


def write_prompt_advisor_job_state(job_id: str, state: dict[str, Any]) -> None:
    state["heartbeat_at"] = utc_now()
    atomic_write_json(state_path(job_id), state)


def list_prompt_advisor_job_ids() -> list[str]:
    return list_task_job_ids(PROMPT_ADVISOR_JOBS_DIR)


def request_prompt_advisor_stop(job_id: str) -> None:
    request_stop_file(stop_path(job_id))


def prompt_advisor_stop_requested(job_id: str) -> bool:
    return stop_file_exists(stop_path(job_id))


def prompt_advisor_job_stale_after_seconds(state: dict[str, Any]) -> float:
    config = state.get("config") or {}
    eval_config = config.get("eval_config") if isinstance(config.get("eval_config"), dict) else {}
    timeout = float(eval_config.get("judge_timeout") or 120)
    retries = float(eval_config.get("judge_max_retries") or 3)
    backoff = float(eval_config.get("judge_qps_backoff") or 12)
    return max(300.0, timeout * 2 + retries * max(backoff, 5.0) + 180.0)


def prompt_advisor_job_is_stale(state: dict[str, Any]) -> bool:
    if state.get("status") != "running":
        return False
    heartbeat = _parse_time(str(state.get("heartbeat_at") or state.get("updated_at") or ""))
    if heartbeat is None:
        return False
    if heartbeat.tzinfo is None:
        heartbeat = heartbeat.replace(tzinfo=utc_datetime().tzinfo)
    return (utc_datetime() - heartbeat).total_seconds() > prompt_advisor_job_stale_after_seconds(state)


def mark_prompt_advisor_job_interrupted(job_id: str) -> dict[str, Any]:
    state = read_prompt_advisor_job_state(job_id)
    if not state or state.get("status") != "running":
        return state
    state["status"] = "interrupted"
    state["stage"] = "已中断"
    state["message"] = "后台提示词建议任务可能已中断：长时间没有心跳。可以重新启动任务。"
    state["finished_at"] = utc_now()
    state["updated_at"] = utc_now()
    write_prompt_advisor_job_state(job_id, state)
    return state


def prompt_advisor_job_is_running(job_id: str) -> bool:
    state = read_prompt_advisor_job_state(job_id)
    if prompt_advisor_job_is_stale(state):
        mark_prompt_advisor_job_interrupted(job_id)
        return False
    return state.get("status") == "running"


def _safe_config(config: PromptAdvisorJobConfig) -> dict[str, Any]:
    value = asdict(config)
    value.pop("current_judge_prompt", None)
    value.pop("extraction_prompt", None)
    value["evidence"] = f"{len(config.evidence)} 条证据"
    eval_config = value.get("eval_config")
    if isinstance(eval_config, dict):
        eval_config.pop("judge_api_bearer_token", None)
        eval_config["judge_max_attempts"] = int(eval_config.get("judge_max_retries") or 1)
    return value


def _write_state(
    config: PromptAdvisorJobConfig,
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
        "source_name": config.source_name,
        "started_at": started_at,
        "updated_at": utc_now(),
        "config": _safe_config(config),
    }
    if extra:
        state.update(extra)
    write_prompt_advisor_job_state(config.job_id, state)


def _result_summary(result: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    patch_result = result.get("extraction_prompt_patch_result")
    if not isinstance(patch_result, dict):
        patch_result = {}
    return {
        "can_suggest": result.get("can_suggest"),
        "error": result.get("error", ""),
        "candidate_judge_prompt_chars": len(str(result.get("candidate_judge_prompt") or "")),
        "candidate_extraction_prompt_chars": len(str(result.get("candidate_extraction_prompt") or "")),
        "advisor_flow": result.get("advisor_flow", ""),
        "applied_patch_edits": len(patch_result.get("applied_edits") or []),
        "skipped_patch_edits": len(patch_result.get("skipped_edits") or []),
        "risk_count": len(result.get("risks") or []),
    }


def _write_result_files(job_id: str, result: dict[str, Any] | None, raw: str | None) -> tuple[Path, Path]:
    job_dir(job_id).mkdir(parents=True, exist_ok=True)
    result_file = result_path(job_id)
    raw_file = raw_path(job_id)
    atomic_write_json(result_file, result or {})
    atomic_write_text(raw_file, raw or "")
    return result_file, raw_file


def load_prompt_advisor_job_result(job_id: str, state: dict[str, Any] | None = None) -> tuple[dict[str, Any] | None, str]:
    state = state or read_prompt_advisor_job_state(job_id)
    result_file = Path(str(state.get("result_path") or result_path(job_id)))
    raw_file = Path(str(state.get("raw_path") or raw_path(job_id)))

    result: dict[str, Any] | None = None
    raw = ""
    if result_file.exists():
        loaded = read_json_state(result_file)
        result = loaded if isinstance(loaded, dict) and loaded.get("status") != "corrupt" else None
    elif isinstance(state.get("result"), dict):
        result = state.get("result")

    if raw_file.exists():
        raw = raw_file.read_text(encoding="utf-8")
    else:
        raw = str(state.get("raw") or "")
    return result, raw


def run_prompt_advisor_job(config: PromptAdvisorJobConfig) -> None:
    started_at = utc_now()
    if stop_path(config.job_id).exists():
        stop_path(config.job_id).unlink()

    _write_state(
        config,
        stage="准备",
        done=0,
        total=1,
        message=f"已收集 {len(config.evidence)} 条证据，准备生成提示词建议。",
        started_at=started_at,
    )

    try:
        if prompt_advisor_stop_requested(config.job_id):
            _write_state(
                config,
                status="stopped",
                stage="已终止",
                done=0,
                total=1,
                message="提示词建议任务已在调用前终止。",
                started_at=started_at,
                extra={"finished_at": utc_now()},
            )
            return

        _write_state(
            config,
            stage="调用模型",
            done=0,
            total=3,
            message="正在调用提示词建议模型。切换页面不会中断任务。",
            started_at=started_at,
        )

        def on_progress(done: int, total: int, stage: str, message: str) -> None:
            if prompt_advisor_stop_requested(config.job_id):
                raise PromptAdvisorJobStopped()
            _write_state(
                config,
                stage=stage,
                done=done,
                total=total,
                message=message,
                started_at=started_at,
            )

        result, raw = call_prompt_advisor(
            config.eval_config,
            evidence=config.evidence,
            current_judge_prompt=config.current_judge_prompt,
            extraction_prompt=config.extraction_prompt,
            target=config.target,
            advisor_mode=config.advisor_mode,
            min_evidence=config.min_evidence,
            progress_callback=on_progress,
        )
        status = "completed" if result and result.get("can_suggest") is not False else "completed"
        saved_result_path, saved_raw_path = _write_result_files(config.job_id, result or {}, raw or "")
        state = read_prompt_advisor_job_state(config.job_id)
        _write_state(
            config,
            status=status,
            stage="完成",
            done=int(state.get("total", 1) or 1),
            total=int(state.get("total", 1) or 1),
            message="提示词建议生成完成。",
            started_at=started_at,
            extra={
                "result_path": str(saved_result_path),
                "raw_path": str(saved_raw_path),
                "summary": _result_summary(result or {}),
                "finished_at": utc_now(),
            },
        )
    except PromptAdvisorJobStopped:
        state = read_prompt_advisor_job_state(config.job_id)
        _write_state(
            config,
            status="stopped",
            stage="已终止",
            done=int(state.get("done", 0) or 0),
            total=int(state.get("total", 1) or 1),
            message="提示词建议任务已按请求终止。",
            started_at=started_at,
            extra={"finished_at": utc_now()},
        )
    except Exception:
        state = read_prompt_advisor_job_state(config.job_id)
        _write_state(
            config,
            status="failed",
            stage="失败",
            done=int(state.get("done", 0) or 0),
            total=int(state.get("total", 1) or 1),
            message="提示词建议任务失败。",
            started_at=started_at,
            extra={"traceback": traceback.format_exc(), "finished_at": utc_now()},
        )
