import time

from src.ui.global_rate_limiter import api_rate_scope, reset_global_rate_limits, wait_for_global_rate_slot


def test_api_rate_scope_hides_token():
    scope = api_rate_scope("http://example/api", "Bearer secret-token")

    assert "secret-token" not in scope
    assert scope.startswith("http://example/api|")


def test_global_rate_limiter_queues_same_scope():
    reset_global_rate_limits()
    scope = api_rate_scope("http://example/api", "token")

    first = wait_for_global_rate_slot(scope, 0.02)
    start = time.monotonic()
    second = wait_for_global_rate_slot(scope, 0.02)
    elapsed = time.monotonic() - start

    assert first == 0
    assert second > 0
    assert elapsed >= 0.015


def test_global_rate_limiter_does_not_queue_different_scope():
    reset_global_rate_limits()
    wait_for_global_rate_slot(api_rate_scope("http://example/a", "token"), 0.05)
    start = time.monotonic()
    wait_for_global_rate_slot(api_rate_scope("http://example/b", "token"), 0.05)

    assert time.monotonic() - start < 0.03


def test_global_rate_limiter_stop_does_not_reserve_future_slot():
    reset_global_rate_limits()
    scope = api_rate_scope("http://example/api", "token")

    wait_for_global_rate_slot(scope, 0.05)
    wait_for_global_rate_slot(scope, 0.05, should_stop=lambda: True)
    start = time.monotonic()
    wait_for_global_rate_slot(scope, 0.05)
    elapsed = time.monotonic() - start

    assert elapsed < 0.08
