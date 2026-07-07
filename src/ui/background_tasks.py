from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.persistence import atomic_write_text, backup_corrupt_file


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
        return _mark_corrupt_state(state_path, "状态文件不是合法 JSON")
    if not isinstance(value, dict):
        return _mark_corrupt_state(state_path, "状态文件 JSON 顶层不是 object")
    return value


def _mark_corrupt_state(path: Path, message: str) -> dict[str, Any]:
    try:
        backup = backup_corrupt_file(path)
    except Exception as exc:
        return {
            "job_id": path.parent.name,
            "status": "corrupt",
            "stage": "状态文件损坏",
            "message": f"{message}；备份失败：{type(exc).__name__}: {exc}",
            "_state_error": message,
            "_state_path": str(path),
        }
    state = {
        "job_id": path.parent.name,
        "status": "corrupt",
        "stage": "状态文件损坏",
        "message": f"{message}；已备份到 {backup}",
        "updated_at": utc_now(),
        "heartbeat_at": utc_now(),
        "_state_error": message,
        "_state_path": str(path),
        "_state_corrupt_path": str(backup) if backup else "",
    }
    try:
        atomic_write_text(path, json.dumps(state, ensure_ascii=False, indent=2))
    except Exception:
        pass
    return state


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
    atomic_write_text(stop_file, utc_now())


def stop_file_exists(path: str | Path) -> bool:
    return Path(path).exists()
