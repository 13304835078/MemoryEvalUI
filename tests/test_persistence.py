from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path

import pytest

from src.persistence import append_jsonl_rows, atomic_write_jsonl, atomic_write_text, read_jsonl
from src.schema import EvalResult, append_result_to_jsonl, results_from_jsonl, results_to_jsonl


def _result(score: float, comment: str) -> EvalResult:
    return EvalResult(
        case_id="case-1",
        task_type="user_md_update",
        score_total=score,
        scores={"correctness": score},
        comment=comment,
        model_name="model",
        prompt_version="prompt",
        judge_model="judge",
        judge_prompt_version="judge-prompt",
    )


def test_result_journal_keeps_latest_duplicate_and_tolerates_partial_tail(tmp_path: Path):
    path = tmp_path / "results.jsonl"
    results_to_jsonl([_result(4.0, "first")], str(path))
    append_result_to_jsonl(_result(5.0, "latest"), str(path))
    with open(path, "ab") as handle:
        handle.write(b'{"case_id":"partial')

    restored = results_from_jsonl(str(path))

    assert len(restored) == 1
    assert restored[0].score_total == 5.0
    assert restored[0].comment == "latest"


def test_atomic_write_preserves_old_file_when_replace_fails(tmp_path: Path, monkeypatch):
    path = tmp_path / "state.txt"
    path.write_text("old", encoding="utf-8")

    def fail_replace(self, target):
        raise PermissionError("locked")

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(PermissionError):
        atomic_write_text(path, "new", retries=1)

    assert path.read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob("*.tmp"))


def test_jsonl_writers_serialize_excel_date_and_time_values(tmp_path: Path):
    path = tmp_path / "journal.jsonl"
    atomic_write_jsonl(path, [{
        "日期": date(2026, 7, 6),
        "时间": time(9, 30, 5),
        "时间戳": datetime(2026, 7, 6, 9, 30, 5),
    }])
    append_jsonl_rows(path, [{"普通列": "ok", "打卡时间": time(18, 45)}])

    rows = read_jsonl(path)

    assert rows[0]["日期"] == "2026-07-06"
    assert rows[0]["时间"] == "09:30:05"
    assert rows[0]["时间戳"] == "2026-07-06T09:30:05"
    assert rows[1]["打卡时间"] == "18:45:00"
