from __future__ import annotations

from pathlib import Path

import pytest

from src.persistence import atomic_write_text
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
