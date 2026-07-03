from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from src import runtime_paths


def _read_json(path: Path) -> dict[str, Any]:
    try:
        if path.exists() and path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        return {}
    return {}


def _git_value(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(runtime_paths.SOURCE_ROOT), *args],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def get_build_info() -> dict[str, Any]:
    for path in (
        runtime_paths.APP_HOME / "build_info.json",
        runtime_paths.BUNDLED_ROOT / "build_info.json",
        runtime_paths.SOURCE_ROOT / ".tmp" / "build_info.json",
    ):
        data = _read_json(path)
        if data:
            return data

    status = _git_value("status", "--porcelain")
    return {
        "app_name": "MemoryEvalUI",
        "version": "dev",
        "git_commit": _git_value("rev-parse", "--short", "HEAD"),
        "git_branch": _git_value("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(status),
        "build_mode": "development",
    }


def format_build_label(info: dict[str, Any] | None = None) -> str:
    info = info or get_build_info()
    version = str(info.get("version") or "dev")
    commit = str(info.get("git_commit") or "unknown")
    dirty = " dirty" if info.get("git_dirty") else ""
    return f"{version} / {commit}{dirty}"
