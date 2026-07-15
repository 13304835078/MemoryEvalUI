from pathlib import Path

import pandas as pd
import pytest

from src.loop.dataset_split import split_excel_by_reviewer_session
from src.loop.validation_gate import ValidationGateConfig, evaluate_candidate_gate
from src.loop import closed_loop
from src.schema import EvalConfig
from src.schema import Case, EvalResult, TaskType


def _result(case_id: str, score: float) -> EvalResult:
    return EvalResult(
        case_id=case_id,
        task_type=TaskType.USER_MD.value,
        score_total=score,
        scores={"correctness": score},
    )


def _case(case_id: str, reviewer: str = "") -> Case:
    return Case(
        case_id=case_id,
        task_type=TaskType.USER_MD,
        session_id=case_id,
        candidate_output="- x",
        metadata={"reviewer": reviewer} if reviewer else {},
    )


def test_dataset_split_is_deterministic_and_keeps_sessions_whole(tmp_path: Path):
    rows = []
    for session_index in range(12):
        reviewer_index = session_index % 6
        rows.extend([
            {"轮次": 1, "query": f"q{session_index}-1", "answer": "a", "评测人": f"r{reviewer_index}"},
            {"轮次": 2, "query": f"q{session_index}-2", "answer": "a", "评测人": f"r{reviewer_index}"},
        ])
    source = tmp_path / "input.xlsx"
    pd.DataFrame(rows).to_excel(source, index=False)

    first = split_excel_by_reviewer_session(source, tmp_path / "split1")
    second = split_excel_by_reviewer_session(source, tmp_path / "split2")

    assert first["partition_group_counts"] == {"discovery": 4, "validation": 1, "locked_test": 1}
    assert [(g["group_id"], g["partition"]) for g in first["groups"]] == [
        (g["group_id"], g["partition"]) for g in second["groups"]
    ]
    partition_queries = {}
    reviewer_partitions: dict[str, set[str]] = {}
    for partition, path in first["partition_paths"].items():
        frame = pd.read_excel(path)
        assert len(frame) % 2 == 0
        partition_queries[partition] = set(frame["query"].astype(str))
        source_positions = [int(value.split("-")[0][1:]) for value in frame["query"].astype(str)]
        assert source_positions == sorted(source_positions)
        for reviewer in set(frame["评测人"].astype(str)):
            reviewer_partitions.setdefault(reviewer, set()).add(partition)
        if "r0" in set(frame["评测人"].astype(str)):
            assert frame.loc[frame["评测人"].astype(str) == "r0", "__source_reviewer_segment"].nunique() == 2
    assert not (partition_queries["discovery"] & partition_queries["validation"])
    assert not (partition_queries["discovery"] & partition_queries["locked_test"])
    assert all(len(partitions) == 1 for partitions in reviewer_partitions.values())


def test_dataset_split_rejects_fewer_than_three_reviewer_histories(tmp_path: Path):
    source = tmp_path / "too_small.xlsx"
    pd.DataFrame([
        {"轮次": 1, "query": "q1", "answer": "a", "评测人": "r1"},
        {"轮次": 1, "query": "q2", "answer": "a", "评测人": "r2"},
    ]).to_excel(source, index=False)

    with pytest.raises(ValueError, match="3 位不同评测人"):
        split_excel_by_reviewer_session(source, tmp_path / "split")


def test_dataset_split_reserves_validation_reviewers_for_statistical_gate(tmp_path: Path):
    source = tmp_path / "four_reviewers.xlsx"
    pd.DataFrame([
        {"轮次": 1, "query": f"q{index}", "answer": "a", "评测人": f"r{index}"}
        for index in range(4)
    ]).to_excel(source, index=False)

    manifest = split_excel_by_reviewer_session(
        source,
        tmp_path / "split_with_gate",
        min_validation_reviewers=2,
    )

    assert manifest["partition_group_counts"] == {
        "discovery": 1,
        "validation": 2,
        "locked_test": 1,
    }
    assert manifest["minimum_partition_reviewers"]["validation"] == 2


def test_dataset_split_rejects_gate_minimum_larger_than_reviewer_pool(tmp_path: Path):
    source = tmp_path / "three_reviewers.xlsx"
    pd.DataFrame([
        {"轮次": 1, "query": f"q{index}", "answer": "a", "评测人": f"r{index}"}
        for index in range(3)
    ]).to_excel(source, index=False)

    with pytest.raises(ValueError, match="至少需要 4 位不同评测人"):
        split_excel_by_reviewer_session(
            source,
            tmp_path / "split_with_gate",
            min_validation_reviewers=2,
        )


def test_validation_gate_rejects_incomplete_run_and_accepts_real_improvement():
    cases = [_case("c1", "r1"), _case("c2", "r2")]
    champion = [_result("c1", 4.0), _result("c2", 4.0)]
    candidate = [_result("c1", 4.2), _result("c2", 4.2)]
    accepted = evaluate_candidate_gate(
        champion,
        candidate,
        champion_cases=cases,
        candidate_cases=cases,
        champion_prompt="abc",
        candidate_prompt="abd",
        config=ValidationGateConfig(
            min_score_delta=0.1,
            max_prompt_growth_ratio=1.0,
            min_paired_cases=2,
            min_paired_clusters=2,
            bootstrap_samples=200,
        ),
    )
    assert accepted["accepted"] is True

    candidate[1] = EvalResult.from_parse_failure(
        case_id="c2", task_type=TaskType.USER_MD.value, raw="API error: timeout"
    )
    rejected = evaluate_candidate_gate(
        champion,
        candidate,
        champion_cases=cases,
        candidate_cases=cases,
        champion_prompt="abc",
        candidate_prompt="abd",
        config=ValidationGateConfig(
            min_score_delta=0.0,
            max_prompt_growth_ratio=1.0,
            min_paired_cases=2,
            min_paired_clusters=2,
            bootstrap_samples=200,
        ),
    )
    assert rejected["accepted"] is False
    assert any("比较不完整" in reason for reason in rejected["reasons"])


def test_validation_gate_rejects_improvement_without_enough_independent_evidence():
    cases = [_case("c1", "same_reviewer"), _case("c2", "same_reviewer")]
    result = evaluate_candidate_gate(
        [_result("c1", 4.0), _result("c2", 4.0)],
        [_result("c1", 4.5), _result("c2", 4.5)],
        champion_cases=cases,
        candidate_cases=cases,
        champion_prompt="abc",
        candidate_prompt="abd",
        config=ValidationGateConfig(
            min_score_delta=0.1,
            max_prompt_growth_ratio=1.0,
            min_paired_cases=2,
            min_paired_clusters=2,
            bootstrap_samples=200,
        ),
    )

    assert result["accepted"] is False
    assert result["paired_cluster_count"] == 1
    assert any("统计证据不足" in reason for reason in result["reasons"])


def test_closed_loop_evaluation_uses_frozen_initial_rule_contract(tmp_path: Path, monkeypatch):
    captured = {}

    class CapturingRunner:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(closed_loop, "EvalRunner", CapturingRunner)
    config = closed_loop.ClosedLoopConfig(
        run_id="frozen_contract",
        input_excel_path="unused.xlsx",
        extraction_prompt_text="# 初始规则",
        extraction_prompt_version="initial_v1",
        evaluation_rule_prompt_text="# 冻结规则",
        evaluation_rule_prompt_version="contract_v1",
        eval_config=EvalConfig(mock=True),
    )

    closed_loop._evaluate_cases(
        config,
        [],
        2,
        "# 候选自行修改后的规则",
        "candidate_v2",
        tmp_path / "results.jsonl",
    )

    assert captured["extraction_prompt_text"] == "# 冻结规则"
    assert captured["extraction_prompt_version"] == "contract_v1"


def test_trusted_closed_loop_runs_holdout_protocol_in_mock_mode(tmp_path: Path, monkeypatch):
    source = tmp_path / "trusted_input.xlsx"
    pd.DataFrame([
        {"轮次": 1, "query": "q1", "answer": "a1", "评测人": "r1"},
        {"轮次": 1, "query": "q2", "answer": "a2", "评测人": "r2"},
        {"轮次": 1, "query": "q3", "answer": "a3", "评测人": "r3"},
        {"轮次": 1, "query": "q4", "answer": "a4", "评测人": "r4"},
    ]).to_excel(source, index=False)
    monkeypatch.setattr(closed_loop, "CLOSED_LOOP_DIR", tmp_path / "closed_loop")

    config = closed_loop.ClosedLoopConfig(
        run_id="trusted_mock",
        input_excel_path=str(source),
        protocol_version="v2_holdout",
        rounds=1,
        chunk_size=1,
        extraction_model="mock-extractor",
        extraction_prompt_text="# 提取规则\n- 只记录稳定信息。",
        extraction_prompt_version="extract_v1",
        extraction_request_interval=0,
        judge_prompt_text="请输出评分 JSON。",
        judge_prompt_version="judge_v1",
        advisor_model="mock-advisor",
        eval_config=EvalConfig(mock=True, judge_model="mock-judge"),
    )

    closed_loop.run_closed_loop(config)

    state = closed_loop.read_loop_state(config.run_id)
    assert state["protocol"]["version"] == "v2_holdout"
    assert state["split_manifest"]["partition_group_counts"] == {
        "discovery": 1, "validation": 2, "locked_test": 1,
    }
    assert state["rounds"][0]["discovery"]["run_quality"]["run_complete"] is True
    assert state["rounds"][0]["validation_gate"]["accepted"] is False
    assert state["status"] == "validation_rejected"
    assert state["locked_test"]["advisor_visible"] is False
