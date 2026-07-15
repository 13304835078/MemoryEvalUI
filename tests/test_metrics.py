import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.schema import EvalResult
from src.eval.metrics import compute_aggregations, group_by


def make_sample_results() -> list[EvalResult]:
    return [
        EvalResult(case_id="c1", task_type="user_md_update", score_total=4.5,
                   scores={"correctness": 5, "coverage": 4, "update_logic": 4,
                           "memory_boundary": 5, "conciseness": 4, "format": 5},
                   error_tags=["verbose_or_noisy"], judge_model="m1"),
        EvalResult(case_id="c2", task_type="user_md_update", score_total=3.0,
                   scores={"correctness": 3, "coverage": 3, "update_logic": 2,
                           "memory_boundary": 3, "conciseness": 4, "format": 4},
                   error_tags=["over_memory", "short_term_pollution"], judge_model="m1"),
        EvalResult(case_id="c3", task_type="user_md_update", score_total=1.0,
                   scores={"correctness": 1, "coverage": 1, "update_logic": 1,
                           "memory_boundary": 1, "conciseness": 1, "format": 1},
                   error_tags=["hallucination", "wrong_fact"], fatal_error=True, judge_model="m2"),
    ]


def test_compute_aggregations():
    results = make_sample_results()
    stats = compute_aggregations(results)
    assert stats["total_cases"] == 3
    assert stats["fatal_errors"] == 1
    assert abs(stats["fatal_rate"] - (1.0 / 3.0)) < 0.001
    assert 2.5 < stats["avg_score_total"] < 3.0  # (4.5 + 3.0 + 1.0) / 3 = 2.833

    dims = stats["avg_dimension_scores"]
    assert "correctness" in dims
    assert round(dims["correctness"], 2) == 3.0  # (5+3+1)/3

    tags = dict(stats["error_tags"])
    assert tags.get("over_memory") == 1
    assert tags.get("hallucination") == 1
    assert tags.get("verbose_or_noisy") == 1


def test_group_by():
    results = make_sample_results()
    groups = group_by(results, "judge_model")
    assert "m1" in groups
    assert "m2" in groups
    assert len(groups["m1"]) == 2
    assert len(groups["m2"]) == 1


def test_empty_results():
    stats = compute_aggregations([])
    assert stats["total_cases"] == 0


def test_runtime_failure_is_not_counted_as_zero_score():
    success = EvalResult(case_id="ok", task_type="user_md_update", score_total=4.0)
    failure = EvalResult.from_parse_failure(
        case_id="failed",
        task_type="user_md_update",
        raw="API error: QPS limit exceeded",
    )

    stats = compute_aggregations([success, failure])

    assert stats["total_cases"] == 2
    assert stats["scored_cases"] == 1
    assert stats["judge_failures"] == 1
    assert stats["avg_score_total"] == 4.0
    assert stats["run_complete"] is False
