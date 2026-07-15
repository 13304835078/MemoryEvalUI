from src.eval.run_quality import compute_run_quality
from src.schema import Case, EvalResult, TaskType


def _case(case_id: str, *, call: str = "success", parse: str = "structured") -> Case:
    return Case(
        case_id=case_id,
        task_type=TaskType.USER_MD,
        session_id="s1",
        candidate_output="- item" if parse != "empty" else None,
        metadata={"call_status": call, "parse_status": parse},
    )


def test_run_quality_separates_execution_and_quality_failures():
    scored = EvalResult(case_id="ok", task_type="user_md_update", score_total=4.0)
    judge_failed = EvalResult.from_parse_failure(
        case_id="judge_fail", task_type="user_md_update", raw="API error: QPS limit exceeded"
    )
    summary = compute_run_quality(
        [scored, judge_failed],
        cases=[_case("ok"), _case("judge_fail")],
        missed_cases=[
            _case("empty", call="success", parse="empty"),
            _case("api", call="failed", parse="not_attempted"),
            _case("unknown", call="not_attempted", parse="not_attempted"),
        ],
    )

    assert summary["conditional_avg_score"] == 4.0
    assert summary["judge_failures"] == 1
    assert summary["extraction_quality_failures"] == 1
    assert summary["extraction_infrastructure_failures"] == 1
    assert summary["end_to_end_score"] == 2.0
    assert summary["run_complete"] is False
    assert summary["replacement_eligible"] is False
