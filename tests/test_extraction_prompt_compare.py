from src.eval.extraction_prompt_compare import (
    build_extraction_prompt_diff,
    build_extraction_pairs,
    compare_extraction_prompt_pairs,
    compare_extraction_prompt_runs,
    deterministic_pairwise_result,
    source_case_key,
)
from src.loop.validation_gate import ValidationGateConfig
from src.schema import Case, EvalResult, TaskType


def _case(
    case_id: str,
    *,
    reviewer: str = "reviewer-1",
    missed: bool = False,
    output: str = "# USER.md\n- value",
) -> Case:
    return Case(
        case_id=case_id,
        task_type=TaskType.USER_MD,
        session_id="session-1",
        candidate_output=None if missed else output,
        model_name="model",
        prompt_version="prompt-a" if case_id.endswith("a") else "prompt-b",
        metadata={
            "reviewer": reviewer,
            "source_session_id": "session-1",
            "chunk_index_in_session": 0,
            "row_start": 1,
            "row_end": 10,
            "call_status": "success",
            "parse_status": "empty" if missed else "parsed",
        },
    )


def _result(case_id: str, score: float, *, eligible: bool = True) -> EvalResult:
    return EvalResult(
        case_id=case_id,
        task_type=TaskType.USER_MD.value,
        score_total=score,
        scores={"correctness": score, "coverage": score},
        comment="comment",
        error_tags=[] if score >= 4.0 else ["missing_key_info"],
        evaluation_status="success" if eligible else "judge_api_failed",
        score_eligible=eligible,
        failure_type="" if eligible else "judge_api_error",
    )


def _lenient_gate() -> ValidationGateConfig:
    return ValidationGateConfig(
        min_score_delta=0.1,
        min_paired_cases=1,
        min_paired_clusters=1,
        require_statistical_confidence=False,
        max_prompt_growth_ratio=1.0,
    )


def test_source_case_key_ignores_prompt_specific_case_id() -> None:
    assert source_case_key(_case("generated-a")) == source_case_key(_case("different-b"))


def test_compare_pairs_same_source_and_recommends_improved_candidate() -> None:
    case_a = _case("case-a")
    case_b = _case("case-b", output="# USER.md\n- improved value")

    report = compare_extraction_prompt_runs(
        cases_a=[case_a],
        cases_b=[case_b],
        missed_cases_a=[],
        missed_cases_b=[],
        results_a=[_result(case_a.case_id, 4.0)],
        results_b=[_result(case_b.case_id, 4.5)],
        prompt_a="baseline prompt",
        prompt_b="candidate prompt",
        validation_config=_lenient_gate(),
    )

    assert report["recommendation"] == "建议选择 B"
    assert report["validation_gate"]["paired_case_count"] == 1
    assert report["rows"][0]["comparison"] == "B较优"
    assert report["rows"][0]["score_delta_b_minus_a"] == 0.5
    assert report["rows"][0]["chunk_id"] == 1
    assert report["rows"][0]["scores_b"]["correctness"] == 4.5


def test_compare_does_not_turn_judge_failure_into_zero_score() -> None:
    case_a = _case("case-a")
    case_b = _case("case-b")

    report = compare_extraction_prompt_runs(
        cases_a=[case_a],
        cases_b=[case_b],
        missed_cases_a=[],
        missed_cases_b=[],
        results_a=[_result(case_a.case_id, 4.0)],
        results_b=[_result(case_b.case_id, 0.0, eligible=False)],
        prompt_a="baseline",
        prompt_b="candidate",
        validation_config=_lenient_gate(),
    )

    row = report["rows"][0]
    assert report["recommendation"] == "暂不定版"
    assert row["comparison"] == "不可比较"
    assert row["score_b"] is None
    assert report["quality_b"]["conditional_avg_score"] == 0.0
    assert report["quality_b"]["judge_failures"] == 1


def test_identical_outputs_are_ties_even_when_judge_scores_differ() -> None:
    case_a = _case("case-a")
    case_b = _case("case-b")

    report = compare_extraction_prompt_runs(
        cases_a=[case_a],
        cases_b=[case_b],
        missed_cases_a=[],
        missed_cases_b=[],
        results_a=[_result(case_a.case_id, 4.0)],
        results_b=[_result(case_b.case_id, 4.8)],
        prompt_a="baseline",
        prompt_b="candidate",
        validation_config=_lenient_gate(),
    )

    assert report["recommendation"] == "证据不足，暂时保留 A"
    assert report["validation_gate"]["paired_score_delta"] == 0.0
    assert report["rows"][0]["comparison"] == "输出相同"
    assert report["judge_disagreement_on_identical_output_count"] == 1


def test_compare_treats_candidate_missed_chunk_as_coverage_regression() -> None:
    case_a = _case("case-a")
    missed_b = _case("missed-b", missed=True)

    report = compare_extraction_prompt_runs(
        cases_a=[case_a],
        cases_b=[],
        missed_cases_a=[],
        missed_cases_b=[missed_b],
        results_a=[_result(case_a.case_id, 5.0)],
        results_b=[],
        prompt_a="baseline",
        prompt_b="candidate",
        validation_config=_lenient_gate(),
    )

    assert report["recommendation"] == "建议保留 A"
    assert report["quality_b"]["extraction_coverage"] == 0.0
    assert report["rows"][0]["comparison"] == "A独有"


def test_compare_rejects_ambiguous_duplicate_source_keys() -> None:
    duplicate_a_1 = _case("case-a-1")
    duplicate_a_2 = _case("case-a-2")
    case_b = _case("case-b")

    report = compare_extraction_prompt_runs(
        cases_a=[duplicate_a_1, duplicate_a_2],
        cases_b=[case_b],
        missed_cases_a=[],
        missed_cases_b=[],
        results_a=[],
        results_b=[],
        prompt_a="baseline",
        prompt_b="candidate",
        validation_config=_lenient_gate(),
    )

    assert report["recommendation"] == "暂不定版"
    assert report["duplicate_source_keys"]
    assert report["rows"] == []


def test_build_diff_keeps_dialogue_rows_and_writes_chunk_result_at_boundary() -> None:
    rows_a = [
        {
            "session_id": 1,
            "chunk_id": 1,
            "query": "q1",
            "answer": "a1",
            "评测人": "reviewer-1",
        },
        {
            "session_id": 1,
            "chunk_id": 1,
            "query": "q2",
            "answer": "a2",
            "评测人": "reviewer-1",
            "effective_document": "A output",
            "reasoning": "A reasoning",
        },
    ]
    rows_b = [
        {**rows_a[0]},
        {
            **rows_a[1],
            "effective_document": "B output",
            "reasoning": "B reasoning",
        },
    ]
    comparison_rows = [
        {
            "reviewer": "reviewer-1",
            "session_id": "1",
            "chunk_id": 1,
            "extraction_a": "可评测",
            "extraction_b": "可评测",
            "score_a": 4.0,
            "score_b": 4.5,
            "score_delta_b_minus_a": 0.5,
            "comparison": "B较优",
            "comparison_note": "B 改善。",
        }
    ]

    row_diff, chunk_diff, include_reasoning = build_extraction_prompt_diff(
        rows_a,
        rows_b,
        comparison_rows,
    )

    assert include_reasoning is True
    assert len(row_diff) == 2
    assert row_diff[0]["query"] == "q1"
    assert row_diff[0]["A提取结果"] == ""
    assert row_diff[1]["A提取结果"] == "A output"
    assert row_diff[1]["B提取结果"] == "B output"
    assert row_diff[1]["A_reasoning"] == "A reasoning"
    assert row_diff[1]["B-A"] == 0.5
    assert row_diff[1]["对比结论"] == "B较优"
    assert chunk_diff[0]["query"] == "q1\n\nq2"


def test_build_diff_omits_reasoning_columns_when_both_versions_have_none() -> None:
    rows = [
        {
            "session_id": 1,
            "chunk_id": 1,
            "query": "q1",
            "answer": "a1",
            "评测人": "reviewer-1",
            "effective_document": "same output",
        }
    ]

    row_diff, chunk_diff, include_reasoning = build_extraction_prompt_diff(rows, rows, [])

    assert include_reasoning is False
    assert "A_reasoning" not in row_diff[0]
    assert "B_reasoning" not in chunk_diff[0]


def test_direct_pairwise_skips_model_for_identical_outputs() -> None:
    case_a = _case("case-a")
    case_b = _case("case-b")
    pairs, duplicates = build_extraction_pairs(
        cases_a=[case_a],
        cases_b=[case_b],
        missed_cases_a=[],
        missed_cases_b=[],
    )

    result = deterministic_pairwise_result(pairs[0])

    assert duplicates == []
    assert result is not None
    assert result["winner"] == "TIE"
    assert result["comparison_kind"] == "identical_output"


def test_direct_pairwise_does_not_rejudge_unchanged_historical_differences() -> None:
    case_a = _case("case-a", output="# USER.md\n- historical A")
    case_b = _case("case-b", output="# USER.md\n- historical B")
    case_a.old_memory = case_a.candidate_output
    case_b.old_memory = case_b.candidate_output
    pairs, _ = build_extraction_pairs(
        cases_a=[case_a],
        cases_b=[case_b],
        missed_cases_a=[],
        missed_cases_b=[],
    )

    result = deterministic_pairwise_result(pairs[0])

    assert result is not None
    assert result["winner"] == "HISTORICAL_DIFFERENCE"
    assert result["issues_a"] == []
    assert result["issues_b"] == []


def test_direct_pairwise_leaves_single_quality_miss_for_neutral_judge() -> None:
    case_a = _case("case-a")
    missed_b = _case("missed-b", missed=True)
    missed_b.metadata["call_status"] = "success"
    missed_b.metadata["parse_status"] = "empty"
    pairs, _ = build_extraction_pairs(
        cases_a=[case_a],
        cases_b=[],
        missed_cases_a=[],
        missed_cases_b=[missed_b],
    )

    result = deterministic_pairwise_result(pairs[0])

    assert result is None


def test_direct_pairwise_excludes_extraction_api_failure_from_winner() -> None:
    case_a = _case("case-a")
    missed_b = _case("missed-b", missed=True)
    missed_b.metadata["call_status"] = "failed"
    pairs, _ = build_extraction_pairs(
        cases_a=[case_a],
        cases_b=[],
        missed_cases_a=[],
        missed_cases_b=[missed_b],
    )

    result = deterministic_pairwise_result(pairs[0])

    assert result is not None
    assert result["winner"] == "INSUFFICIENT"
    assert result["status"] == "infrastructure_failure"


def test_direct_pairwise_report_uses_wins_without_absolute_scores() -> None:
    case_a = _case("case-a")
    case_b = _case("case-b", output="# USER.md\n- improved value")
    source_key = source_case_key(case_a)
    report = compare_extraction_prompt_pairs(
        cases_a=[case_a],
        cases_b=[case_b],
        missed_cases_a=[],
        missed_cases_b=[],
        pairwise_results=[
            {
                "source_key": source_key,
                "status": "success",
                "model": "pairwise-model",
                "winner": "B",
                "confidence": "high",
                "reason": "B 更完整。",
                "rule_refs": ["覆盖规则"],
                "evidence_refs": ["用户事实"],
                "issues_a": ["遗漏"],
                "issues_b": [],
                "error_tags_a": ["missing_key_info"],
                "error_tags_b": [],
                "strengths_a": [],
                "strengths_b": ["完整"],
            }
        ],
        prompt_a="prompt A",
        prompt_b="prompt B",
        validation_config=_lenient_gate(),
    )

    assert report["comparison_mode"] == "candidate_neutral_pairwise_v2"
    assert report["recommendation"] == "建议选择 B"
    assert report["validation_gate"]["paired_preference_delta"] == 1.0
    assert report["rows"][0]["comparison"] == "B较优"
    assert "score_a" not in report["rows"][0]


def test_prompt_design_quality_is_only_a_tiebreaker() -> None:
    case_a = _case("case-a")
    case_b = _case("case-b", output="# USER.md\n- equivalent wording")
    source_key = source_case_key(case_a)
    report = compare_extraction_prompt_pairs(
        cases_a=[case_a],
        cases_b=[case_b],
        missed_cases_a=[],
        missed_cases_b=[],
        pairwise_results=[
            {
                "source_key": source_key,
                "status": "success",
                "winner": "TIE",
                "decision_basis": "equivalent",
                "confidence": "high",
                "reason": "共同质量相当。",
            }
        ],
        prompt_a="prompt A",
        prompt_b="prompt B",
        validation_config=_lenient_gate(),
        evaluation_protocol={
            "prompt_quality_a": {"overall": 3.0},
            "prompt_quality_b": {"overall": 4.0},
        },
    )

    assert report["recommendation"] == "建议选择 B"
    assert "次级依据" in report["recommendation_reason"]


def test_policy_difference_is_excluded_from_pairwise_wins() -> None:
    case_a = _case("case-a")
    case_b = _case("case-b", output="# USER.md\n- policy-specific value")
    source_key = source_case_key(case_a)
    report = compare_extraction_prompt_pairs(
        cases_a=[case_a],
        cases_b=[case_b],
        missed_cases_a=[],
        missed_cases_b=[],
        pairwise_results=[
            {
                "source_key": source_key,
                "status": "success",
                "winner": "POLICY_DIFFERENCE",
                "decision_basis": "policy_difference",
                "policy_differences": ["准入范围不同"],
                "reason": "属于策略差异。",
            }
        ],
        prompt_a="prompt A",
        prompt_b="prompt B",
        validation_config=_lenient_gate(),
    )

    assert report["winner_counts"]["策略差异"] == 1
    assert report["validation_gate"]["paired_case_count"] == 0
    assert report["recommendation"] == "暂不定版"


def test_historical_baseline_difference_is_excluded_and_history_is_visible() -> None:
    case_a = _case("case-a")
    case_b = _case("case-b", output="# USER.md\n- inherited B value")
    case_a.old_memory = "# USER.md\n- inherited A value"
    case_b.old_memory = "# USER.md\n- inherited B value"
    source_key = source_case_key(case_a)
    report = compare_extraction_prompt_pairs(
        cases_a=[case_a],
        cases_b=[case_b],
        missed_cases_a=[],
        missed_cases_b=[],
        pairwise_results=[
            {
                "source_key": source_key,
                "status": "success",
                "winner": "HISTORICAL_DIFFERENCE",
                "decision_basis": "historical_baseline_difference",
                "reason": "差异来自历史基线。",
            }
        ],
        prompt_a="prompt A",
        prompt_b="prompt B",
        validation_config=_lenient_gate(),
    )

    assert report["winner_counts"]["历史基线差异"] == 1
    assert report["validation_gate"]["paired_case_count"] == 0
    assert report["rows"][0]["history_input_a"] == "已输入"
    assert report["rows"][0]["history_input_b"] == "已输入"
    assert report["rows"][0]["history_baseline_relation"] == "不同"
