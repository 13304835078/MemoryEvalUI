from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.runtime_paths import APP_HOME


RATE_LIMIT_DB = APP_HOME / "system" / "global_rate_limit.sqlite3"
RATE_SLOT_SAFETY_SECONDS = 0.002
_THREAD_CONTEXT = threading.local()
_SCHEMA_LOCK = threading.Lock()
_SCHEMA_READY = False


@dataclass(frozen=True)
class _RateWaiter:
    waiter_id: str
    priority: int
    interval: float
    enqueued_at: float


def normalize_priority(priority: int | None) -> int:
    try:
        value = int(priority if priority is not None else 5)
    except (TypeError, ValueError):
        value = 5
    return min(10, max(1, value))


def set_current_task_priority(priority: int | None) -> None:
    _THREAD_CONTEXT.priority = normalize_priority(priority)


def current_task_priority(default: int = 5) -> int:
    return normalize_priority(getattr(_THREAD_CONTEXT, "priority", default))


def api_rate_scope(api_base: str = "", token: str = "") -> str:
    """Return a cross-process scope without exposing the credential."""
    base = str(api_base or "").strip().rstrip("/") or "default-api"
    token_text = str(token or "").strip()
    token_hash = hashlib.sha256(token_text.encode("utf-8")).hexdigest()[:16] if token_text else "no-token"
    return f"{base}|{token_hash}"


def _connect() -> sqlite3.Connection:
    global _SCHEMA_READY
    Path(RATE_LIMIT_DB).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(RATE_LIMIT_DB), timeout=30.0, isolation_level=None)
    connection.execute("PRAGMA busy_timeout=30000")
    connection.execute("PRAGMA journal_mode=WAL")
    if not _SCHEMA_READY:
        with _SCHEMA_LOCK:
            if not _SCHEMA_READY:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS rate_scopes ("
                    "scope TEXT PRIMARY KEY, next_request_at REAL NOT NULL)"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS rate_waiters ("
                    "waiter_id TEXT PRIMARY KEY, scope TEXT NOT NULL, priority INTEGER NOT NULL, "
                    "interval_seconds REAL NOT NULL, enqueued_at REAL NOT NULL)"
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_rate_waiters_scope ON rate_waiters(scope, enqueued_at)"
                )
                _SCHEMA_READY = True
    return connection


def _effective_priority(waiter: _RateWaiter, now: float) -> int:
    aging_window = max(1.0, waiter.interval * 3.0)
    age_bonus = int(max(0.0, now - waiter.enqueued_at) / aging_window)
    return min(10, waiter.priority + age_bonus)


def _select_waiter(waiters: list[_RateWaiter], now: float) -> _RateWaiter | None:
    if not waiters:
        return None
    return min(
        waiters,
        key=lambda waiter: (-_effective_priority(waiter, now), waiter.enqueued_at, waiter.waiter_id),
    )


def _remove_waiter(waiter_id: str) -> None:
    try:
        with _connect() as connection:
            connection.execute("DELETE FROM rate_waiters WHERE waiter_id = ?", (waiter_id,))
    except sqlite3.Error:
        pass


def wait_for_global_rate_slot(
    scope: str,
    interval_seconds: float,
    *,
    disabled: bool = False,
    should_stop: Callable[[], bool] | None = None,
    priority: int | None = None,
) -> float:
    """Reserve a request slot shared by all users and detached task processes."""
    interval = max(0.0, float(interval_seconds or 0.0))
    if disabled or interval <= 0:
        return 0.0

    monotonic_started = time.monotonic()
    enqueued_at = time.time()
    effective_priority = normalize_priority(priority if priority is not None else current_task_priority())
    waiter_id = f"{os.getpid()}-{threading.get_ident()}-{uuid.uuid4().hex}"
    with _connect() as connection:
        connection.execute(
            "INSERT INTO rate_waiters(waiter_id, scope, priority, interval_seconds, enqueued_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (waiter_id, scope, effective_priority, interval, enqueued_at),
        )

    reserved = False
    has_waited = False
    try:
        while True:
            if should_stop is not None and should_stop():
                return max(0.0, time.monotonic() - monotonic_started) if has_waited else 0.0

            timeout = 0.1
            try:
                connection = _connect()
                try:
                    connection.execute("BEGIN IMMEDIATE")
                    now = time.time()
                    connection.execute("DELETE FROM rate_waiters WHERE enqueued_at < ?", (now - 3600.0,))
                    rows = connection.execute(
                        "SELECT waiter_id, priority, interval_seconds, enqueued_at "
                        "FROM rate_waiters WHERE scope = ?",
                        (scope,),
                    ).fetchall()
                    waiters = [
                        _RateWaiter(str(row[0]), int(row[1]), float(row[2]), float(row[3]))
                        for row in rows
                    ]
                    selected = _select_waiter(waiters, now)
                    row = connection.execute(
                        "SELECT next_request_at FROM rate_scopes WHERE scope = ?",
                        (scope,),
                    ).fetchone()
                    next_request_at = float(row[0]) if row else now
                    if selected is not None and selected.waiter_id == waiter_id and now >= next_request_at:
                        strictest_interval = max((item.interval for item in waiters), default=interval)
                        connection.execute(
                            "INSERT INTO rate_scopes(scope, next_request_at) VALUES (?, ?) "
                            "ON CONFLICT(scope) DO UPDATE SET next_request_at = excluded.next_request_at",
                            (scope, now + strictest_interval + RATE_SLOT_SAFETY_SECONDS),
                        )
                        connection.execute("DELETE FROM rate_waiters WHERE waiter_id = ?", (waiter_id,))
                        connection.commit()
                        reserved = True
                        return max(0.0, time.monotonic() - monotonic_started) if has_waited else 0.0
                    connection.commit()
                    if selected is not None and selected.waiter_id == waiter_id:
                        timeout = min(0.25, max(0.01, next_request_at - now))
                    else:
                        timeout = 0.1
                finally:
                    connection.close()
            except sqlite3.OperationalError:
                timeout = 0.1
            has_waited = True
            time.sleep(timeout)
    finally:
        if not reserved:
            _remove_waiter(waiter_id)


def reset_global_rate_limits() -> None:
    try:
        with _connect() as connection:
            connection.execute("DELETE FROM rate_waiters")
            connection.execute("DELETE FROM rate_scopes")
    finally:
        set_current_task_priority(5)
