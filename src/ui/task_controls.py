from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.ui.state_io import atomic_write_json, state_file_lock


DEFAULT_PRIORITY = 5


def normalize_int(value: Any, default: int, *, min_value: int, max_value: int) -> int:
    try:
        current = int(value)
    except (TypeError, ValueError):
        current = int(default)
    return min(max_value, max(min_value, current))


def normalize_float(value: Any, default: float, *, min_value: float, max_value: float) -> float:
    try:
        current = float(value)
    except (TypeError, ValueError):
        current = float(default)
    return min(max_value, max(min_value, current))


def normalize_priority(value: Any, default: int = DEFAULT_PRIORITY) -> int:
    return normalize_int(value, default, min_value=1, max_value=10)


def read_task_controls(path: str | Path) -> dict[str, Any]:
    control_path = Path(path)
    if not control_path.exists():
        return {}
    try:
        value = json.loads(control_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def write_task_controls(path: str | Path, controls: dict[str, Any]) -> dict[str, Any]:
    control_path = Path(path)
    control_path.parent.mkdir(parents=True, exist_ok=True)
    with state_file_lock(control_path):
        normalized = dict(controls)
        if "priority" in normalized:
            normalized["priority"] = normalize_priority(normalized.get("priority"))
        atomic_write_json(control_path, normalized)
        return normalized


def merge_task_controls(path: str | Path, updates: dict[str, Any]) -> dict[str, Any]:
    control_path = Path(path)
    control_path.parent.mkdir(parents=True, exist_ok=True)
    with state_file_lock(control_path):
        current = read_task_controls(control_path)
        for key, value in updates.items():
            if value is None:
                current.pop(key, None)
            else:
                current[key] = value
        return write_task_controls(control_path, current)


def init_task_controls(path: str | Path, defaults: dict[str, Any]) -> dict[str, Any]:
    return write_task_controls(path, defaults)


def control_int(
    controls: dict[str, Any],
    key: str,
    default: int,
    *,
    min_value: int = 1,
    max_value: int = 100,
) -> int:
    return normalize_int(controls.get(key), default, min_value=min_value, max_value=max_value)


def control_float(
    controls: dict[str, Any],
    key: str,
    default: float,
    *,
    min_value: float = 0.0,
    max_value: float = 300.0,
) -> float:
    return normalize_float(controls.get(key), default, min_value=min_value, max_value=max_value)


def control_priority(controls: dict[str, Any], default: int = DEFAULT_PRIORITY) -> int:
    return normalize_priority(controls.get("priority"), default)
