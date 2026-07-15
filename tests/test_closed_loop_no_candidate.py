from src.loop.closed_loop import _classify_no_candidate_reason
from src.schema import EvalResult


def _result(case_id: str, *, score: float = 5.0, fatal: bool = False, tags: list[str] | None = None) -> EvalResult:
    return EvalResult(
        case_id=case_id,
        task_type="user_md_update",
        score_total=score,
        error_tags=list(tags or []),
        fatal_error=fatal,
    )


def test_classify_no_candidate_keeps_quality_fatal_as_valid_evidence():
    reason = _classify_no_candidate_reason(
        results=[_result("c1", fatal=True), _result("c2", fatal=True)],
        evidence=[
            {"case_id": "c1", "fatal_error": True, "evidence_mode": "issue_or_low_score"},
            {"case_id": "c2", "fatal_error": True, "evidence_mode": "issue_or_low_score"},
        ],
        advisor_result={"candidate_prompt_source": "no_valid_incremental_patch"},
    )

    assert reason["category"] == "no_safe_patch"
    assert reason["status"] == "paused_no_safe_patch"


def test_classify_no_candidate_as_eval_chain_failed_for_judge_runtime_failures():
    failed = [
        EvalResult.from_parse_failure(
            case_id=f"c{index}",
            task_type="user_md_update",
            raw="API error: QPS limit exceeded",
        )
        for index in (1, 2)
    ]
    reason = _classify_no_candidate_reason(
        results=failed,
        evidence=[],
        advisor_result={"candidate_prompt_source": "no_valid_incremental_patch"},
    )

    assert reason["category"] == "eval_chain_failed"
    assert reason["status"] == "paused_eval_failed"


def test_classify_no_candidate_as_no_change_needed_when_only_weak_context_exists():
    reason = _classify_no_candidate_reason(
        results=[_result("c1"), _result("c2")],
        evidence=[
            {"case_id": "c1", "fatal_error": False, "evidence_mode": "weak_context_from_result"},
            {"case_id": "c2", "fatal_error": False, "evidence_mode": "weak_context_from_result"},
        ],
        advisor_result={"can_suggest": True, "candidate_prompt_source": "no_valid_incremental_patch"},
    )

    assert reason["category"] == "no_change_needed"
    assert reason["status"] == "completed_no_change"


def test_classify_no_candidate_as_no_safe_patch_when_valid_issues_exist():
    reason = _classify_no_candidate_reason(
        results=[_result("c1", score=4.0, tags=["missing_key_info"])],
        evidence=[
            {"case_id": "c1", "fatal_error": False, "evidence_mode": "issue_or_low_score"},
        ],
        advisor_result={"can_suggest": True, "candidate_prompt_source": "no_valid_incremental_patch"},
    )

    assert reason["category"] == "no_safe_patch"
    assert reason["status"] == "paused_no_safe_patch"
