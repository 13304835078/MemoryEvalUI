import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.loop import closed_loop
from src.ui.background_tasks import read_json_state
from src.ui import (
    eval_job_runner,
    extraction_prompt_ab_job_runner,
    judge_ab_job_runner,
    memory_extraction_job_runner,
    prompt_advisor_job_runner,
)


def _stale_time() -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=1000)).isoformat()


def test_eval_job_is_running_marks_stale_job_interrupted(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_job_runner, "EVAL_JOBS_DIR", tmp_path)
    path = tmp_path / "job-1" / "state.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({
            "status": "running",
            "heartbeat_at": _stale_time(),
            "done": 2,
            "total": 5,
            "config": {
                "eval_config": {
                    "judge_timeout": 1,
                    "judge_max_retries": 1,
                    "judge_qps_backoff": 1,
                    "judge_request_interval": 0,
                }
            },
        }),
        encoding="utf-8",
    )

    assert eval_job_runner.eval_job_is_running("job-1") is False

    updated = eval_job_runner.read_eval_job_state("job-1")
    assert updated["status"] == "interrupted"
    assert updated["done"] == 2
    assert updated["total"] == 5
    assert updated["finished_at"]
    assert updated["heartbeat_at"]


def test_read_json_state_marks_corrupt_file_and_keeps_backup(tmp_path):
    path = tmp_path / "job-1" / "state.json"
    path.parent.mkdir(parents=True)
    path.write_text("{bad json", encoding="utf-8")

    state = read_json_state(path)

    assert state["status"] == "corrupt"
    assert state["_state_error"]
    backup_path = state["_state_corrupt_path"]
    assert backup_path
    assert Path(backup_path).read_text(encoding="utf-8") == "{bad json"

    restored = read_json_state(path)
    assert restored["status"] == "corrupt"
    assert restored["_state_corrupt_path"] == backup_path


def test_closed_loop_reader_marks_corrupt_state_and_keeps_backup(tmp_path, monkeypatch):
    monkeypatch.setattr(closed_loop, "CLOSED_LOOP_DIR", tmp_path)
    path = tmp_path / "loop-corrupt" / "state.json"
    path.parent.mkdir(parents=True)
    path.write_text("{bad json", encoding="utf-8")

    state = closed_loop.read_loop_state("loop-corrupt")

    assert state["status"] == "corrupt"
    assert Path(state["_state_corrupt_path"]).read_text(encoding="utf-8") == "{bad json"


def test_loop_is_running_marks_stale_loop_interrupted(tmp_path, monkeypatch):
    monkeypatch.setattr(closed_loop, "CLOSED_LOOP_DIR", tmp_path)
    path = tmp_path / "loop-1" / "state.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({
            "status": "running",
            "heartbeat_at": _stale_time(),
            "events": [],
            "config": {
                "extraction_timeout": 1,
                "extraction_max_retries": 1,
                "extraction_retry_sleep": 1,
                "eval_config": {
                    "judge_timeout": 1,
                    "judge_max_retries": 1,
                    "judge_qps_backoff": 1,
                },
            },
        }),
        encoding="utf-8",
    )

    assert closed_loop.loop_is_running("loop-1") is False

    updated = closed_loop.read_loop_state("loop-1")
    assert updated["status"] == "interrupted"
    assert updated["finished_at"]
    assert updated["heartbeat_at"]
    assert updated["events"]
    assert updated["events"][-1]["level"] == "warning"


def test_memory_extraction_job_is_running_marks_stale_job_interrupted(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_extraction_job_runner, "MEMORY_EXTRACTION_JOBS_DIR", tmp_path)
    path = tmp_path / "memory-1" / "state.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({
            "status": "running",
            "heartbeat_at": _stale_time(),
            "done": 3,
            "total": 8,
            "config": {
                "extraction_config": {
                    "timeout": 1,
                    "max_retries": 1,
                    "retry_sleep": 1,
                    "request_interval": 0,
                }
            },
        }),
        encoding="utf-8",
    )

    assert memory_extraction_job_runner.memory_extraction_job_is_running("memory-1") is False

    updated = memory_extraction_job_runner.read_memory_extraction_job_state("memory-1")
    assert updated["status"] == "interrupted"
    assert updated["stage"] == "已中断"
    assert updated["done"] == 3
    assert updated["total"] == 8
    assert updated["finished_at"]


def test_memory_extraction_job_auto_generates_cases_after_extraction(tmp_path, monkeypatch):
    jobs_dir = tmp_path / "jobs"
    output_path = tmp_path / "result.xlsx"
    saved_filenames: list[str] = []

    class FakeRunner:
        def __init__(self, **kwargs):
            pass

        def process_excel(self, *args, **kwargs):
            kwargs["progress_callback"](1, 1, "done")
            return {"chunks": 1, "stopped": False}

    monkeypatch.setattr(memory_extraction_job_runner, "MEMORY_EXTRACTION_JOBS_DIR", jobs_dir)
    monkeypatch.setattr(memory_extraction_job_runner, "estimate_total_chunks", lambda config: (1, 1))
    monkeypatch.setattr(memory_extraction_job_runner, "MemoryExtractionRunner", FakeRunner)
    monkeypatch.setattr(
        memory_extraction_job_runner,
        "prepare_cases_from_run_output",
        lambda *args, **kwargs: ([], [], {"generated_cases": 0}),
    )
    monkeypatch.setattr(
        memory_extraction_job_runner,
        "save_cases",
        lambda cases, filename: saved_filenames.append(filename) or str(tmp_path / filename),
    )

    config = memory_extraction_job_runner.MemoryExtractionJobConfig(
        job_id="memory-auto-cases",
        input_path=str(tmp_path / "input.xlsx"),
        output_path=str(output_path),
        prompt_text="prompt",
        prompt_version="v1",
        auto_make_cases=True,
        case_model_name="model",
        case_prompt_version="prompt-v1",
    )

    memory_extraction_job_runner.run_memory_extraction_job(config)

    state = memory_extraction_job_runner.read_memory_extraction_job_state(config.job_id)
    assert state["status"] == "completed"
    assert state["cases_path"].endswith(saved_filenames[0])
    assert saved_filenames[0].startswith("model_prompt-v1_user_md_cases_")
    assert saved_filenames[0].endswith(".jsonl")


def test_prompt_advisor_job_is_running_marks_stale_job_interrupted(tmp_path, monkeypatch):
    monkeypatch.setattr(prompt_advisor_job_runner, "PROMPT_ADVISOR_JOBS_DIR", tmp_path)
    path = tmp_path / "advisor-1" / "state.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({
            "status": "running",
            "heartbeat_at": _stale_time(),
            "done": 0,
            "total": 1,
            "config": {
                "eval_config": {
                    "judge_timeout": 1,
                    "judge_max_retries": 1,
                    "judge_qps_backoff": 1,
                }
            },
        }),
        encoding="utf-8",
    )

    assert prompt_advisor_job_runner.prompt_advisor_job_is_running("advisor-1") is False

    updated = prompt_advisor_job_runner.read_prompt_advisor_job_state("advisor-1")
    assert updated["status"] == "interrupted"
    assert updated["stage"] == "已中断"
    assert updated["finished_at"]


def test_judge_ab_job_is_running_marks_stale_job_interrupted(tmp_path, monkeypatch):
    monkeypatch.setattr(judge_ab_job_runner, "JUDGE_AB_JOBS_DIR", tmp_path)
    path = tmp_path / "ab-1" / "state.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({
            "status": "running",
            "heartbeat_at": _stale_time(),
            "done": 1,
            "total": 4,
            "config": {
                "eval_config": {
                    "judge_timeout": 1,
                    "judge_max_retries": 1,
                    "judge_qps_backoff": 1,
                    "judge_request_interval": 0,
                }
            },
        }),
        encoding="utf-8",
    )

    assert judge_ab_job_runner.judge_ab_job_is_running("ab-1") is False

    updated = judge_ab_job_runner.read_judge_ab_job_state("ab-1")
    assert updated["status"] == "interrupted"
    assert updated["stage"] == "已中断"
    assert updated["done"] == 1
    assert updated["total"] == 4
    assert updated["finished_at"]


def test_extraction_prompt_ab_job_marks_stale_job_interrupted(tmp_path, monkeypatch):
    monkeypatch.setattr(extraction_prompt_ab_job_runner, "EXTRACTION_PROMPT_AB_JOBS_DIR", tmp_path)
    path = tmp_path / "extract-ab-1" / "state.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({
            "status": "running",
            "heartbeat_at": _stale_time(),
            "done": 400,
            "total": 1000,
            "config": {
                "extraction_config": {"timeout": 1, "max_attempts": 1, "retry_sleep": 1},
                "eval_config": {"judge_timeout": 1, "judge_max_attempts": 1, "judge_qps_backoff": 1},
            },
        }),
        encoding="utf-8",
    )

    assert extraction_prompt_ab_job_runner.extraction_prompt_ab_job_is_running("extract-ab-1") is False

    updated = extraction_prompt_ab_job_runner.read_extraction_prompt_ab_job_state("extract-ab-1")
    assert updated["status"] == "interrupted"
    assert updated["done"] == 400
    assert updated["total"] == 1000
    assert updated["finished_at"]


def test_closed_loop_state_updates_are_serialized_under_parallel_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(closed_loop, "CLOSED_LOOP_DIR", tmp_path)
    closed_loop.write_loop_state("loop-1", {
        "status": "running",
        "events": [],
        "rounds": [],
        "config": {},
    })

    def worker(index: int) -> None:
        closed_loop.update_state("loop-1", lambda state: (
            state.update({"stage": f"progress-{index}"}),
            closed_loop._round_record(state, 1).update({"latest_message": f"message-{index}"}),
            closed_loop.append_event(state, f"event-{index}"),
        ))

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(30)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    state = closed_loop.read_loop_state("loop-1")
    assert len(state["events"]) == 30
    assert {event["message"] for event in state["events"]} == {f"event-{index}" for index in range(30)}
    assert state["rounds"][0]["latest_message"].startswith("message-")
