import pandas as pd

from src.eval.metrics import flatten_results
from src.schema import EvalResult
from src.ui.data_service import dataframe_to_excel_bytes, list_files, load_results_bytes


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
