from __future__ import annotations

import hashlib
import threading
import time
from typing import Callable


_LOCK = threading.Lock()
_NEXT_REQUEST_AT: dict[str, float] = {}


def api_rate_scope(api_base: str = "", token: str = "") -> str:
    """Return a process-local rate limit scope without exposing tokens."""
    base = str(api_base or "").strip().rstrip("/") or "default-api"
    token_text = str(token or "").strip()
    token_hash = hashlib.sha1(token_text.encode("utf-8")).hexdigest()[:12] if token_text else "no-token"
    return f"{base}|{token_hash}"


def wait_for_global_rate_slot(
    scope: str,
    interval_seconds: float,
    *,
    disabled: bool = False,
    should_stop: Callable[[], bool] | None = None,
) -> float:
    """Wait until a request slot can be reserved for a shared API scope.

    This coordinates concurrent Streamlit background threads in the same process.
    It deliberately does not persist across process restarts.
    """
    interval = max(0.0, float(interval_seconds or 0.0))
    if disabled or interval <= 0:
        return 0.0

    total_waited = 0.0
    while True:
        if should_stop is not None and should_stop():
            return total_waited

        with _LOCK:
            now = time.monotonic()
            next_at = _NEXT_REQUEST_AT.get(scope, now)
            wait_seconds = max(0.0, next_at - now)
            if wait_seconds <= 0:
                _NEXT_REQUEST_AT[scope] = now + interval
                return total_waited

        sleep_seconds = min(1.0, wait_seconds)
        time.sleep(sleep_seconds)
        total_waited += sleep_seconds


def reset_global_rate_limits() -> None:
    with _LOCK:
        _NEXT_REQUEST_AT.clear()
