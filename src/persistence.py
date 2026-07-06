from __future__ import annotations

import json
import os
import threading
import time
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def file_lock(path: str | Path) -> threading.RLock:
    key = str(Path(path).resolve())
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _LOCKS[key] = lock
        return lock


def atomic_write_text(
    path: str | Path,
    text: str,
    *,
    encoding: str = "utf-8",
    retries: int = 8,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = file_lock(target)

    with lock:
        tmp = target.with_name(f"{target.name}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp")
        try:
            with open(tmp, "w", encoding=encoding, newline="") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())

            last_error: PermissionError | None = None
            for attempt in range(max(1, retries)):
                try:
                    tmp.replace(target)
                    return
                except PermissionError as exc:
                    last_error = exc
                    time.sleep(min(0.5, 0.05 * (attempt + 1)))
            if last_error is not None:
                raise last_error
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass


def atomic_write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    payload = "".join(_json_dumps(row) + "\n" for row in rows)
    atomic_write_text(path, payload)


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    append_jsonl_rows(path, [row])


def append_jsonl_rows(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = file_lock(target)
    with lock:
        with open(target, "a", encoding="utf-8", newline="") as handle:
            for row in rows:
                handle.write(_json_dumps(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def read_jsonl(path: str | Path, *, tolerate_trailing_partial: bool = True) -> list[dict[str, Any]]:
    target = Path(path)
    lines = target.read_text(encoding="utf-8-sig").splitlines()
    rows: list[dict[str, Any]] = []
    nonempty_indexes = [index for index, line in enumerate(lines) if line.strip()]
    last_nonempty = nonempty_indexes[-1] if nonempty_indexes else -1

    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if tolerate_trailing_partial and index == last_nonempty:
                break
            raise
        if not isinstance(value, dict):
            raise ValueError(f"JSONL 第 {index + 1} 行必须是 object")
        rows.append(value)
    return rows


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "item") and callable(value.item):
        try:
            item_value = value.item()
            if item_value is not value:
                return item_value
        except Exception:
            pass
    if hasattr(value, "isoformat") and callable(value.isoformat):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)
