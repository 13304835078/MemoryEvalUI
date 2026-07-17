import pandas as pd
from pathlib import Path

from src.eval.metrics import compute_aggregations, flatten_results
from src.schema import Case, EvalResult, TaskType
from src.ui import data_service
from src.ui.data_service import (
    dataframe_to_excel_bytes,
    eval_result_resume_key,
    eval_result_row_key,
    find_case_for_result,
    list_files,
    load_results_bytes,
    results_to_dataframe,
)


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


def test_overview_row_keys_preserve_scored_and_failed_results():
    scored = EvalResult(
        case_id="scored",
        task_type=TaskType.USER_MD.value,
        score_total=4.5,
        scores={"correctness": 4.5},
        model_name="model",
        prompt_version="prompt",
        judge_model="judge",
        judge_prompt_version="judge-prompt",
        extraction_prompt_hash="extract-hash",
        evaluation_fingerprint="fingerprint-scored",
    )
    failed = EvalResult.from_parse_failure(
        case_id="failed",
        task_type=TaskType.USER_MD.value,
        raw="Judge 输出不是可解析 JSON",
        model_name="model",
        prompt_version="prompt",
        judge_model="judge",
        judge_prompt_version="judge-prompt",
        extraction_prompt_hash="extract-hash",
        evaluation_fingerprint="fingerprint-failed",
    )
    results = [scored, failed]
    frame = results_to_dataframe(results)
    filtered_keys = {eval_result_row_key(row) for _, row in frame.iterrows()}
    filtered_results = [result for result in results if eval_result_resume_key(result) in filtered_keys]

    stats = compute_aggregations(filtered_results)

    assert filtered_results == results
    assert stats["total_cases"] == 2
    assert stats["scored_cases"] == 1
    assert stats["judge_failures"] == 1
    assert stats["avg_score_total"] == 4.5
    assert failed.failure_message == "Judge 输出不是可解析 JSON"


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


def test_saved_case_and_upload_names_cannot_escape_storage_directory(tmp_path, monkeypatch):
    cases_dir = tmp_path / "cases"
    uploads_dir = tmp_path / "uploads"
    monkeypatch.setattr(data_service, "CASES_DIR", cases_dir)
    monkeypatch.setattr(data_service, "UPLOAD_DIR", uploads_dir)

    case = Case(case_id="c1", task_type=TaskType.USER_MD, session_id="s1")
    cases_path = Path(data_service.save_cases([case], "../unsafe:name"))

    class Uploaded:
        name = "../report?.xlsx"

        @staticmethod
        def getvalue():
            return b"excel"

    upload_path = Path(data_service.save_uploaded_file(Uploaded(), suffix=".xlsx"))

    assert cases_path.parent.resolve() == cases_dir.resolve()
    assert cases_path.name == "unsafe_name.jsonl"
    assert upload_path.parent.resolve() == uploads_dir.resolve()
    assert "?" not in upload_path.name
    assert upload_path.read_bytes() == b"excel"


def test_case_generation_uses_explicit_status_contract_before_legacy_status(tmp_path):
    input_path = tmp_path / "contract.xlsx"
    pd.DataFrame({
        "轮次": [1, 1],
        "query": ["第一条", "第二条"],
        "answer": ["好的", "好的"],
        "评测人": ["alice", "alice"],
        "status": ["API_FAILED", ""],
        "call_status": ["success", "failed"],
        "parse_status": ["raw_fallback", "not_attempted"],
        "case_status": ["review_required", "skip"],
        "raw_output": ["未带固定结构的候选正文", "API error"],
        "parsed_document": ["", ""],
        "effective_document": ["未带固定结构的候选正文", ""],
        "inheritance_source": ["raw_output", "none"],
    }).to_excel(input_path, index=False)

    cases, missed, stats = data_service.prepare_cases_from_run_output(
        input_path,
        model="model",
        prompt_version="prompt",
        chunk_size=1,
        return_missed=True,
    )

    assert len(cases) == 1
    assert len(missed) == 1
    assert cases[0].candidate_output == "未带固定结构的候选正文"
    assert cases[0].metadata["case_status"] == "review_required"
    assert cases[0].metadata["extraction_status"] == "needs_parse_review"
    assert missed[0].metadata["call_status"] == "failed"
    assert stats["case_status_counts"] == {"review_required": 1, "skip": 1}
