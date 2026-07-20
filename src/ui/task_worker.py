from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import traceback
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from src.persistence import atomic_write_text
from src.runtime_paths import APP_HOME, DATA_DIR, IS_FROZEN, SOURCE_ROOT, active_workspace_id, activate_workspace
from src.schema import Case, EvalConfig, EvalResult, results_from_jsonl
from src.ui.state_io import atomic_write_json, state_file_lock


TASK_REQUESTS_DIR = DATA_DIR / "task_requests"
_SECRET_ENV = "MEMORY_EVAL_TASK_SECRETS_B64"
_SECRET_KEYS = {
    "api_token",
    "extraction_api_token",
    "advisor_api_token",
    "judge_api_bearer_token",
    "judge_hmac_secret_key",
}


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _extract_secrets(value: Any, path: tuple[str, ...] = ()) -> tuple[Any, dict[str, str]]:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        secrets: dict[str, str] = {}
        for key, item in value.items():
            key_text = str(key)
            item_path = (*path, key_text)
            if key_text in _SECRET_KEYS and str(item or ""):
                cleaned[key_text] = ""
                secrets[".".join(item_path)] = str(item)
            else:
                cleaned_item, nested = _extract_secrets(item, item_path)
                cleaned[key_text] = cleaned_item
                secrets.update(nested)
        return cleaned, secrets
    if isinstance(value, list):
        cleaned_list = []
        secrets: dict[str, str] = {}
        for index, item in enumerate(value):
            cleaned_item, nested = _extract_secrets(item, (*path, str(index)))
            cleaned_list.append(cleaned_item)
            secrets.update(nested)
        return cleaned_list, secrets
    return value, {}


def _restore_secrets(payload: dict[str, Any], secrets: dict[str, str]) -> None:
    for dotted_path, secret in secrets.items():
        parts = dotted_path.split(".")
        target: Any = payload
        for part in parts[:-1]:
            target = target[int(part)] if isinstance(target, list) else target[part]
        if isinstance(target, list):
            target[int(parts[-1])] = secret
        else:
            target[parts[-1]] = secret


def _request_path(job_id: str) -> Path:
    safe = "".join(char if char.isalnum() or char in "_-" else "_" for char in str(job_id))
    return Path(TASK_REQUESTS_DIR) / safe / "request.json"


def _write_request(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with state_file_lock(path):
        atomic_write_json(path, payload)


def launch_background_task(
    kind: str,
    config: Any,
    *,
    cases: list[Case] | None = None,
    existing_results: list[EvalResult] | None = None,
) -> int:
    """Launch a task in a detached process, independent of Streamlit reruns."""
    job_id = str(getattr(config, "job_id", None) or getattr(config, "run_id", None) or "task")
    request_path = _request_path(job_id)
    payload = {
        "version": 1,
        "kind": str(kind),
        "job_id": job_id,
        "workspace_id": active_workspace_id(),
        "status": "queued",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": _json_value(config),
        "cases": [case.to_dict() for case in (cases or [])],
        "existing_results": [result.to_dict() for result in (existing_results or [])],
    }
    public_payload, secrets = _extract_secrets(payload)
    _write_request(request_path, public_payload)

    env = os.environ.copy()
    env["MEMORY_EVAL_WORKSPACE_ID"] = active_workspace_id()
    if secrets:
        encoded = base64.b64encode(json.dumps(secrets, ensure_ascii=False).encode("utf-8")).decode("ascii")
        env[_SECRET_ENV] = encoded

    if IS_FROZEN:
        command = [sys.executable, "--background-worker", str(request_path)]
    else:
        command = [sys.executable, "-m", "src.ui.task_worker", "--request", str(request_path)]

    log_path = request_path.parent / "worker.log"
    log_handle = open(log_path, "ab", buffering=0)
    kwargs: dict[str, Any] = {
        "cwd": str(APP_HOME if IS_FROZEN else SOURCE_ROOT),
        "env": env,
        "stdin": subprocess.DEVNULL,
        "stdout": log_handle,
        "stderr": log_handle,
        "close_fds": True,
    }
    if os.name == "nt":
        kwargs["creationflags"] = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        )
    else:
        kwargs["start_new_session"] = True
    try:
        process = subprocess.Popen(command, **kwargs)
    finally:
        log_handle.close()

    with state_file_lock(request_path):
        try:
            latest = json.loads(request_path.read_text(encoding="utf-8"))
        except Exception:
            latest = dict(public_payload)
        if latest.get("status") == "queued":
            latest["status"] = "launched"
        latest["pid"] = int(process.pid)
        latest["launched_at"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(request_path, latest)
    return int(process.pid)


def _eval_config(value: dict[str, Any]) -> EvalConfig:
    return EvalConfig(**{
        key: item for key, item in value.items() if key in EvalConfig.__dataclass_fields__
    })


def _run_request(payload: dict[str, Any]) -> None:
    kind = str(payload.get("kind") or "")
    config_data = dict(payload.get("config") or {})
    if kind == "eval":
        from src.ui.eval_job_runner import EvalJobConfig, run_eval_job

        config_data["eval_config"] = _eval_config(dict(config_data.get("eval_config") or {}))
        config = EvalJobConfig(**config_data)
        cases = [Case.from_dict(item) for item in payload.get("cases") or []]
        existing = [EvalResult.from_dict(item) for item in payload.get("existing_results") or []]
        output_path = Path(config.output_path)
        if output_path.exists():
            existing = results_from_jsonl(str(output_path))
        run_eval_job(config, cases, existing)
        return

    if kind == "memory_extraction":
        from src.extraction.client import MemoryExtractionConfig
        from src.ui.memory_extraction_job_runner import MemoryExtractionJobConfig, run_memory_extraction_job

        config_data["extraction_config"] = MemoryExtractionConfig(**dict(config_data.get("extraction_config") or {}))
        run_memory_extraction_job(MemoryExtractionJobConfig(**config_data))
        return

    if kind == "prompt_advisor":
        from src.ui.prompt_advisor_job_runner import PromptAdvisorJobConfig, run_prompt_advisor_job

        config_data["eval_config"] = _eval_config(dict(config_data.get("eval_config") or {}))
        run_prompt_advisor_job(PromptAdvisorJobConfig(**config_data))
        return

    if kind == "judge_ab":
        from src.ui.judge_ab_job_runner import JudgeAbJobConfig, run_judge_ab_job

        config_data["eval_config"] = _eval_config(dict(config_data.get("eval_config") or {}))
        cases = [Case.from_dict(item) for item in payload.get("cases") or []]
        run_judge_ab_job(JudgeAbJobConfig(**config_data), cases)
        return

    if kind == "extraction_prompt_ab":
        from src.extraction.client import MemoryExtractionConfig
        from src.loop.validation_gate import ValidationGateConfig
        from src.ui.extraction_prompt_ab_job_runner import (
            ExtractionPromptAbJobConfig,
            run_extraction_prompt_ab_job,
        )

        config_data["extraction_config"] = MemoryExtractionConfig(
            **dict(config_data.get("extraction_config") or {})
        )
        for side_key in ("extraction_config_a", "extraction_config_b"):
            if config_data.get(side_key):
                config_data[side_key] = MemoryExtractionConfig(
                    **dict(config_data.get(side_key) or {})
                )
        config_data["eval_config"] = _eval_config(dict(config_data.get("eval_config") or {}))
        if config_data.get("comparison_config"):
            config_data["comparison_config"] = _eval_config(
                dict(config_data.get("comparison_config") or {})
            )
        else:
            config_data["comparison_config"] = EvalConfig(
                **asdict(config_data["eval_config"])
            )
        config_data["validation_config"] = ValidationGateConfig(
            **dict(config_data.get("validation_config") or {})
        )
        run_extraction_prompt_ab_job(ExtractionPromptAbJobConfig(**config_data))
        return

    if kind == "closed_loop":
        from src.loop.closed_loop import ClosedLoopConfig, run_closed_loop

        config_data["eval_config"] = _eval_config(dict(config_data.get("eval_config") or {}))
        run_closed_loop(ClosedLoopConfig(**config_data))
        return

    raise ValueError(f"未知后台任务类型：{kind}")


def run_worker(request_path: str | Path) -> int:
    path = Path(request_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    workspace_id = str(payload.get("workspace_id") or os.environ.get("MEMORY_EVAL_WORKSPACE_ID", ""))
    activate_workspace(workspace_id)
    secret_text = os.environ.pop(_SECRET_ENV, "")
    if secret_text:
        secrets = json.loads(base64.b64decode(secret_text).decode("utf-8"))
        _restore_secrets(payload, secrets)

    public_payload, _secrets = _extract_secrets(payload)
    public_payload["status"] = "running"
    public_payload["worker_pid"] = os.getpid()
    public_payload["started_at"] = datetime.now(timezone.utc).isoformat()
    _write_request(path, public_payload)
    try:
        _run_request(payload)
        public_payload["status"] = "finished"
        public_payload["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_request(path, public_payload)
        return 0
    except BaseException as exc:
        public_payload["status"] = "failed"
        public_payload["finished_at"] = datetime.now(timezone.utc).isoformat()
        public_payload["error"] = f"{type(exc).__name__}: {exc}"
        public_payload["traceback"] = traceback.format_exc()
        _write_request(path, public_payload)
        atomic_write_text(path.parent / "worker_error.txt", public_payload["traceback"])
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args(argv)
    return run_worker(args.request)


if __name__ == "__main__":
    raise SystemExit(main())
