from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def utc_now() -> str:
    return utc_datetime().isoformat()


def utc_datetime() -> datetime:
    return datetime.now(timezone.utc)


def parse_time(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def task_job_dir(base_dir: str | Path, job_id: str, *, sanitize: Callable[[str], str] | None = None) -> Path:
    safe_job_id = sanitize(job_id) if sanitize is not None else job_id
    return Path(base_dir) / safe_job_id


def task_state_path(base_dir: str | Path, job_id: str, *, sanitize: Callable[[str], str] | None = None) -> Path:
    return task_job_dir(base_dir, job_id, sanitize=sanitize) / "state.json"


def task_stop_path(base_dir: str | Path, job_id: str, *, sanitize: Callable[[str], str] | None = None) -> Path:
    return task_job_dir(base_dir, job_id, sanitize=sanitize) / "STOP"


def read_json_state(path: str | Path) -> dict[str, Any]:
    state_path = Path(path)
    if not state_path.exists():
        return {}
    try:
        value = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def list_task_job_ids(base_dir: str | Path) -> list[str]:
    root = Path(base_dir)
    if not root.exists():
        return []
    paths = [path for path in root.iterdir() if path.is_dir()]
    paths.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return [path.name for path in paths]


def request_stop_file(path: str | Path) -> None:
    stop_file = Path(path)
    stop_file.parent.mkdir(parents=True, exist_ok=True)
    stop_file.write_text(utc_now(), encoding="utf-8")


def stop_file_exists(path: str | Path) -> bool:
    return Path(path).exists()
