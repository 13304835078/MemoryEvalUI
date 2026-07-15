from __future__ import annotations

from typing import Any, Iterable

from src.eval.result_status import result_is_score_eligible


def _meta(case: Any) -> dict[str, Any]:
    value = getattr(case, "metadata", {})
    return value if isinstance(value, dict) else {}


def _classify_missed_case(case: Any) -> str:
    metadata = _meta(case)
    call_status = str(metadata.get("call_status") or "").strip().lower()
    parse_status = str(metadata.get("parse_status") or "").strip().lower()

    if call_status in {"failed", "stopped"}:
        return "extraction_infrastructure_failure"
    if call_status == "success" and parse_status in {"empty", "unknown", "not_attempted", ""}:
        return "extraction_quality_failure"
    return "excluded_or_unknown"


def compute_run_quality(
    results: Iterable[Any],
    *,
    cases: Iterable[Any] | None = None,
    missed_cases: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Combine extraction and Judge outcomes without turning execution failures into zero scores."""
    result_list = list(results or [])
    case_list = list(cases or [])
    missed_list = list(missed_cases or [])

    scored = [item for item in result_list if result_is_score_eligible(item)]
    judge_failures = [item for item in result_list if not result_is_score_eligible(item)]
    conditional_avg = (
        round(sum(float(getattr(item, "score_total", 0.0) or 0.0) for item in scored) / len(scored), 4)
        if scored
        else 0.0
    )

    missed_counts = {
        "extraction_infrastructure_failure": 0,
        "extraction_quality_failure": 0,
        "excluded_or_unknown": 0,
    }
    for case in missed_list:
        missed_counts[_classify_missed_case(case)] += 1

    ready_count = len(case_list) if case_list else len(result_list)
    extraction_quality_failures = missed_counts["extraction_quality_failure"]
    extraction_infrastructure_failures = missed_counts["extraction_infrastructure_failure"]
    excluded_count = missed_counts["excluded_or_unknown"]

    extraction_known = ready_count + extraction_quality_failures
    extraction_coverage = ready_count / extraction_known if extraction_known else 0.0
    infrastructure_denominator = extraction_known + extraction_infrastructure_failures
    infrastructure_success_rate = (
        extraction_known / infrastructure_denominator if infrastructure_denominator else 0.0
    )

    # Successful extraction with no usable body is a quality miss and contributes zero.
    # Judge execution failures remain outside the denominator and make the run provisional.
    end_to_end_denominator = len(scored) + extraction_quality_failures
    end_to_end_score = (
        sum(float(getattr(item, "score_total", 0.0) or 0.0) for item in scored)
        / end_to_end_denominator
        if end_to_end_denominator
        else 0.0
    )

    unresolved_execution_failures = len(judge_failures) + extraction_infrastructure_failures
    run_complete = unresolved_execution_failures == 0 and excluded_count == 0
    replacement_eligible = run_complete and bool(scored or extraction_quality_failures)

    return {
        "total_result_rows": len(result_list),
        "scored_cases": len(scored),
        "judge_failures": len(judge_failures),
        "conditional_avg_score": round(conditional_avg, 4),
        "ready_cases": ready_count,
        "extraction_quality_failures": extraction_quality_failures,
        "extraction_infrastructure_failures": extraction_infrastructure_failures,
        "excluded_or_unknown_chunks": excluded_count,
        "extraction_coverage": round(extraction_coverage, 4),
        "infrastructure_success_rate": round(infrastructure_success_rate, 4),
        "end_to_end_score": round(end_to_end_score, 4),
        "end_to_end_denominator": end_to_end_denominator,
        "unresolved_execution_failures": unresolved_execution_failures,
        "run_complete": run_complete,
        "replacement_eligible": replacement_eligible,
        "status": "complete" if run_complete else "provisional",
    }
