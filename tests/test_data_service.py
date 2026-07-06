import pandas as pd

from src.eval.metrics import flatten_results
from src.schema import Case, EvalResult, TaskType
from src.ui.data_service import dataframe_to_excel_bytes, find_case_for_result, list_files, load_results_bytes


def test_list_files_matches_suffix_case_insensitively(tmp_path):
    lower = tmp_path / "result_a.jsonl"
    upper = tmp_path / "result_b.JSONL"
    ignored = tmp_path / "result.xlsx"
    lower.write_text("{}\n", encoding="utf-8")
    upper.write_text("{}\n", encoding="utf-8")
    ignored.write_text("not-jsonl", encoding="utf-8")

    files = list_files(tmp_path, ".jsonl")

    assert files == [str(lower), str(upper)]


def test_exported_csv_and_excel_can_be_loaded_back_as_results():
    original = EvalResult(
        case_id="case_1",
        task_type="user_md_update",
        score_total=4.25,
        scores={"correctness": 4.0, "coverage": 4.5},
        comment="遗漏一项",
        error_tags=["missing_key_info"],
        fatal_error=False,
        model_name="model-a",
        prompt_version="prompt-v1",
        judge_model="judge-a",
        judge_prompt_version="judge-v1",
        extraction_prompt_version="extract-v1",
        extraction_prompt_hash="abc123",
        diagnostics=[{"dimension": "coverage", "reason": "漏记"}],
        rule_refs=["A4 兴趣爱好"],
        evidence_refs=["用户表达长期偏好"],
        output_refs=["新 USER.md 未记录"],
        timestamp="2026-07-02T00:00:00+00:00",
    )
    frame = pd.DataFrame(flatten_results([original]))
    csv_bytes = frame.to_csv(index=False).encode("utf-8-sig")
    excel_bytes = dataframe_to_excel_bytes(frame)

    csv_result = load_results_bytes(csv_bytes, "results.csv")[0]
    excel_result = load_results_bytes(excel_bytes, "results.xlsx")[0]

    for restored in (csv_result, excel_result):
        assert restored.case_id == original.case_id
        assert restored.score_total == original.score_total
        assert restored.scores == original.scores
        assert restored.error_tags == original.error_tags
        assert restored.diagnostics == original.diagnostics
        assert restored.rule_refs == original.rule_refs
        assert restored.evidence_refs == original.evidence_refs
        assert restored.output_refs == original.output_refs


def test_find_case_for_result_requires_matching_task_type():
    user_case = Case(
        case_id="same_id",
        task_type=TaskType.USER_MD,
        session_id="s1",
        old_memory="旧用户画像",
        candidate_output="候选用户画像",
        model_name="model",
        prompt_version="prompt",
    )
    memory_case = Case(
        case_id="same_id",
        task_type=TaskType.LONG_MEMORY,
        session_id="s1",
        old_memory="旧长期记忆",
        candidate_output="候选长期记忆",
        model_name="model",
        prompt_version="prompt",
    )
    result = EvalResult(
        case_id="same_id",
        task_type=TaskType.LONG_MEMORY.value,
        score_total=4.5,
        model_name="model",
        prompt_version="prompt",
    )

    assert find_case_for_result([user_case, memory_case], result) is memory_case
    assert find_case_for_result([user_case], result) is None
