import json

from src.schema import EvalResult
from src.ui.rule_ref_validation import (
    summarize_rule_ref_validation,
    validate_result_rule_refs,
)


def make_result(**kwargs) -> EvalResult:
    defaults = {
        "case_id": "c1",
        "task_type": "user_md_update",
        "score_total": 5.0,
        "scores": {},
        "comment": "ok",
        "error_tags": [],
        "fatal_error": False,
        "extraction_prompt_version": "extract_v1",
        "extraction_prompt_hash": "",
    }
    defaults.update(kwargs)
    return EvalResult(**defaults)


def test_rule_ref_validation_accepts_real_prompt_snippet():
    result = make_result(
        rule_refs=["## Rule A"],
        diagnostics=[{"rule_refs": ["R2: do not store temporary tasks"]}],
        comment="Follows ## Rule A.",
    )
    report = validate_result_rule_refs(
        result,
        extraction_prompt_text="## Rule A\nR2: do not store temporary tasks\n",
    )

    assert report["status"] == "ok"
    assert report["invalid_refs"] == []
    assert report["total_refs"] == 2


def test_rule_ref_validation_marks_missing_refs():
    result = make_result(rule_refs=[], diagnostics=[])
    report = validate_result_rule_refs(result, extraction_prompt_text="## Rule A\n")

    assert report["status"] == "missing"
    assert report["missing_required"] is True


def test_rule_ref_validation_detects_raw_response_hallucinated_ref():
    raw = {
        "score_total": 5,
        "scores": {},
        "comment": "mentions R3",
        "error_tags": [],
        "fatal_error": False,
        "rule_refs": ["R3"],
        "diagnostics": [{"rule_refs": ["R2"]}],
    }
    result = make_result(
        rule_refs=["R2"],
        comment="mentions R3",
        raw_response=json.dumps(raw, ensure_ascii=False),
    )
    report = validate_result_rule_refs(result, extraction_prompt_text="R2: allowed rule\n")

    assert report["status"] == "invalid"
    assert report["raw_invalid_refs"] == ["R3"]
    assert report["comment_invalid_rule_ids"] == ["R3"]


def test_rule_ref_validation_summary_counts_statuses():
    reports = [
        {"status": "ok", "checked": True},
        {"status": "invalid", "checked": True},
        {"status": "missing", "checked": True},
        {"status": "not_applicable", "checked": False},
    ]

    summary = summarize_rule_ref_validation(reports)

    assert summary["ok"] == 1
    assert summary["invalid"] == 1
    assert summary["missing"] == 1
    assert summary["not_checked"] == 1
