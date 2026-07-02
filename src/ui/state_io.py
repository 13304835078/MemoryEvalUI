from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def state_file_lock(path: str | Path) -> threading.RLock:
    key = str(Path(path).resolve())
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _LOCKS[key] = lock
        return lock


def atomic_write_json(path: str | Path, data: dict[str, Any], *, retries: int = 8) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    lock = state_file_lock(target)

    with lock:
        tmp = target.with_name(f"{target.name}.{threading.get_ident()}.{time.time_ns()}.tmp")
        try:
            tmp.write_text(payload, encoding="utf-8")
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
