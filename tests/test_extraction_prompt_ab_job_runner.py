from pathlib import Path

import pandas as pd

from src.extraction.client import MemoryExtractionConfig
from src.loop.validation_gate import ValidationGateConfig
from src.schema import EvalConfig, TaskType
from src.ui import extraction_prompt_ab_job_runner as runner


def _write_extraction(path: Path, output: str) -> None:
    pd.DataFrame(
        [
            {
                "轮次": 1,
                "query": "我长期住在杭州",
                "answer": "收到",
                "评测人": "reviewer-1",
                "session_id": 1,
                "chunk_id": 1,
                "call_status": "success",
                "parse_status": "structured",
                "case_status": "ready",
                "effective_document": output,
                "user.md": output,
            }
        ]
    ).to_excel(path, index=False)


def test_job_can_compare_two_existing_extraction_files_without_absolute_scoring(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(runner, "EXTRACTION_PROMPT_AB_JOBS_DIR", tmp_path / "jobs")
    extraction_a = tmp_path / "a.xlsx"
    extraction_b = tmp_path / "b.xlsx"
    _write_extraction(extraction_a, "# USER.md\n- 常住杭州")
    _write_extraction(extraction_b, "# USER.md\n- 用户长期居住在杭州")

    extraction_config_a = MemoryExtractionConfig(mock=True, model="extract-model-a")
    extraction_config_b = MemoryExtractionConfig(mock=True, model="extract-model-b")
    comparison_config = EvalConfig(mock=True, judge_model="pairwise-model")
    config = runner.ExtractionPromptAbJobConfig(
        job_id="existing-ab",
        task_type=TaskType.USER_MD.value,
        input_path="",
        prompt_a_text="prompt A",
        prompt_a_version="A-v1",
        prompt_b_text="prompt B",
        prompt_b_version="B-v1",
        judge_prompt_text="直接比较",
        judge_prompt_version="pairwise-v1",
        evaluation_rule_prompt_text="只记录长期事实",
        evaluation_rule_prompt_version="rule-v1",
        side_a_mode="existing",
        side_b_mode="existing",
        existing_extraction_a_path=str(extraction_a),
        existing_extraction_b_path=str(extraction_b),
        extraction_config=extraction_config_a,
        extraction_config_a=extraction_config_a,
        extraction_config_b=extraction_config_b,
        eval_config=comparison_config,
        comparison_config=comparison_config,
        validation_config=ValidationGateConfig(require_statistical_confidence=False),
    )

    runner.run_extraction_prompt_ab_job(config)

    state = runner.read_extraction_prompt_ab_job_state(config.job_id)
    report = runner.load_extraction_prompt_ab_report(config.job_id)
    assert state["status"] == "completed"
    assert report["comparison_mode"] == "candidate_neutral_pairwise_v2"
    assert report["evaluation_protocol"]["protocol_version"] == "candidate_neutral_common_core_v2"
    assert report["input_modes"] == {"A": "existing", "B": "existing"}
    assert report["model_roles"]["extraction_model_a"] == "extract-model-a"
    assert report["model_roles"]["extraction_model_b"] == "extract-model-b"
    assert runner.pairwise_results_path(config.job_id).exists()
    assert runner.diff_excel_path(config.job_id).exists()
    diff = pd.read_excel(runner.diff_excel_path(config.job_id), sheet_name="逐Chunk对比")
    assert "A相对问题" in diff.columns
    assert "B相对优点" in diff.columns
    assert "对比调用状态" not in diff.columns
    assert "A总分" not in diff.columns
    assert "B总分" not in diff.columns


def test_job_can_reuse_one_existing_file_as_source_for_the_other_extraction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(runner, "EXTRACTION_PROMPT_AB_JOBS_DIR", tmp_path / "jobs")
    extraction_a = tmp_path / "a.xlsx"
    _write_extraction(extraction_a, "# USER.md\n- 常住杭州")
    extraction_config_a = MemoryExtractionConfig(mock=True, model="existing-model-a")
    extraction_config_b = MemoryExtractionConfig(mock=True, model="new-model-b")
    comparison_config = EvalConfig(mock=True, judge_model="pairwise-model")
    config = runner.ExtractionPromptAbJobConfig(
        job_id="one-existing-ab",
        task_type=TaskType.USER_MD.value,
        input_path=str(extraction_a),
        prompt_a_text="prompt A",
        prompt_a_version="A-v1",
        prompt_b_text="prompt B",
        prompt_b_create_text="prompt B create",
        prompt_b_version="B-v1",
        judge_prompt_text="直接比较",
        judge_prompt_version="pairwise-v1",
        evaluation_rule_prompt_text="只记录长期事实",
        evaluation_rule_prompt_version="rule-v1",
        side_a_mode="existing",
        side_b_mode="extract",
        existing_extraction_a_path=str(extraction_a),
        extraction_config=extraction_config_a,
        extraction_config_a=extraction_config_a,
        extraction_config_b=extraction_config_b,
        eval_config=comparison_config,
        comparison_config=comparison_config,
        validation_config=ValidationGateConfig(require_statistical_confidence=False),
    )

    runner.run_extraction_prompt_ab_job(config)

    state = runner.read_extraction_prompt_ab_job_state(config.job_id)
    assert state["status"] == "completed"
    assert state["stats_a"]["extraction"]["api_calls"] == 0
    assert state["stats_b"]["extraction"]["api_calls"] == 1
    assert runner.extraction_path(config.job_id, "B").exists()
