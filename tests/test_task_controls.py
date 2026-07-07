from src.ui.global_rate_limiter import current_task_priority, set_current_task_priority
from src.ui.task_controls import (
    control_int,
    control_priority,
    merge_task_controls,
    read_task_controls,
    write_task_controls,
)


def test_task_controls_roundtrip_and_normalize_priority(tmp_path):
    path = tmp_path / "job" / "controls.json"

    written = write_task_controls(path, {"priority": 99, "judge_concurrency": 3})
    assert written["priority"] == 10

    merged = merge_task_controls(path, {"priority": 0, "judge_concurrency": 8})
    assert merged["priority"] == 1
    assert read_task_controls(path)["judge_concurrency"] == 8

    controls = read_task_controls(path)
    assert control_priority(controls) == 1
    assert control_int(controls, "judge_concurrency", 1, min_value=1, max_value=100) == 8


def test_thread_priority_context_is_normalized():
    set_current_task_priority(12)
    assert current_task_priority() == 10

    set_current_task_priority(-2)
    assert current_task_priority() == 1

    set_current_task_priority(5)
