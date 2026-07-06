import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.schema import Case, EvalResult, TaskType
from src.loaders.excel_loader import ExcelLoader
from src.ui.data_service import (
    append_result,
    case_resume_key,
    eval_result_resume_key,
    load_results,
    prepare_long_memory_cases_from_run_output,
    prepare_cases_from_run_output,
)


def test_excel_loader_basic():
    df = pd.DataFrame({
        "case_id": ["case_1", "case_2"],
        "session_id": ["s1", "s2"],
        "旧USER.md": ["old content 1", "old content 2"],
        "query": ["user msg 1", "user msg 2"],
        "answer": ["assistant msg 1", "assistant msg 2"],
        "新USER.md": ["new content 1", "new content 2"],
    })
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        tmp = f.name
    try:
        df.to_excel(tmp, index=False)
        loader = ExcelLoader(TaskType.USER_MD)
        cases = loader.load(tmp)
        assert len(cases) == 2
        assert cases[0].old_memory == "old content 1"
        assert cases[0].dialogue[0].role == "user"
        assert cases[0].dialogue[0].content == "user msg 1"
        assert cases[0].dialogue[1].role == "assistant"
        assert cases[0].dialogue[1].content == "assistant msg 1"
        assert cases[0].candidate_output == "new content 1"
        assert cases[1].old_memory == "old content 2"
    finally:
        os.unlink(tmp)


def test_excel_loader_column_aliases():
    df = pd.DataFrame({
        "Case_ID": ["c1"],
        "Session": ["s1"],
        "旧画像": ["old"],
        "query": ["q"],
        "answer": ["a"],
        "模型输出": ["new"],
        "模型": ["test-model"],
        "版本": ["v2"],
    })
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        tmp = f.name
    try:
        df.to_excel(tmp, index=False)
        loader = ExcelLoader(TaskType.USER_MD)
        cases = loader.load(tmp)
        assert len(cases) == 1
        case = cases[0]
        assert case.case_id == "c1"
        assert case.old_memory == "old"
        assert case.candidate_output == "new"
        assert case.model_name == "test-model"
        assert case.prompt_version == "v2"
    finally:
        os.unlink(tmp)


def test_prepare_run_output_chunks_by_session_turn_and_chunk_size():
    df = pd.DataFrame({
        "session_id": ["s1", "s1", "s1", "s2", "s2", "s3", "s4", "s4"],
        "轮次": [1, 2, 3, 1, 2, 1, 1, 2],
        "query": ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8"],
        "answer": ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8"],
        "评测人": ["alice", "alice", "alice", "alice", "alice", "bob", "alice", "alice"],
        "result": ["", "raw1", "raw2", "", "raw3", "raw4", "", ""],
        "reasoning": ["", "r1", "r2", "", "r3", "r4", "", ""],
        "user.md": ["", "- 偏好: A", "- 偏好: A2", "", "- 偏好: A3", "- 偏好: B", "", ""],
    })
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        tmp = f.name
    try:
        df.to_excel(tmp, index=False)
        cases, stats = prepare_cases_from_run_output(
            tmp,
            model="m",
            prompt_version="p",
            chunk_size=2,
            return_stats=True,
        )
        assert len(cases) == 4
        assert stats["total_chunks"] == 5
        assert stats["generated_cases"] == 4
        assert stats["skipped_chunks"] == 1
        assert stats["skipped_chunk_details"][0]["row_start"] == 7
        assert stats["skipped_chunk_details"][0]["row_end"] == 8

        assert cases[0].old_memory is None
        assert cases[0].candidate_output == "- 偏好: A"
        assert cases[0].metadata["reasoning"] == "r1"
        assert cases[0].metadata["source_session_id"] == "s1"
        assert cases[0].metadata["row_start"] == 1
        assert cases[0].metadata["row_end"] == 2
        assert len(cases[0].dialogue) == 4

        assert cases[1].old_memory == "- 偏好: A"
        assert cases[1].candidate_output == "- 偏好: A2"
        assert cases[1].metadata["reasoning"] == "r2"
        assert cases[1].metadata["row_start"] == 3
        assert cases[1].metadata["row_end"] == 3

        assert cases[2].old_memory == "- 偏好: A2"
        assert cases[2].candidate_output == "- 偏好: A3"
        assert cases[2].metadata["source_session_id"] == "s2"
        assert cases[2].metadata["reasoning"] == "r3"

        assert cases[3].old_memory is None
        assert cases[3].candidate_output == "- 偏好: B"
        assert cases[3].metadata["reviewer"] == "bob"

        skipped_ranges = {(c.metadata["row_start"], c.metadata["row_end"]) for c in cases}
        assert (7, 8) not in skipped_ranges
    finally:
        os.unlink(tmp)


def test_prepare_run_output_can_return_missed_cases():
    df = pd.DataFrame({
        "session_id": ["s1", "s1", "s2"],
        "轮次": [1, 2, 1],
        "query": ["q1", "q2", "q3"],
        "answer": ["a1", "a2", "a3"],
        "评测人": ["alice", "alice", "alice"],
        "result": ["", "", ""],
        "reasoning": ["", "", ""],
        "user.md": ["", "", "- 记忆: ok"],
    })
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        tmp = f.name
    try:
        df.to_excel(tmp, index=False)
        cases, missed_cases, stats = prepare_cases_from_run_output(
            tmp,
            model="m",
            prompt_version="p",
            chunk_size=2,
            return_missed=True,
        )

        assert len(cases) == 1
        assert len(missed_cases) == 1
        assert stats["generated_cases"] == 1
        assert stats["missed_cases"] == 1
        missed = missed_cases[0]
        assert missed.case_id.startswith("missed_")
        assert missed.candidate_output is None
        assert missed.metadata["extraction_status"] == "missed_extraction"
        assert missed.metadata["row_start"] == 1
        assert missed.metadata["row_end"] == 2
        assert missed.metadata["boundary_row"] == 2
        assert missed.metadata["skip_reason"] == "chunk_last_row_missing_user_md_result_reasoning"
    finally:
        os.unlink(tmp)


def test_prepare_run_output_all_missed_does_not_raise_with_return_missed():
    df = pd.DataFrame({
        "session_id": ["s1", "s1"],
        "轮次": [1, 2],
        "query": ["q1", "q2"],
        "answer": ["a1", "a2"],
        "评测人": ["alice", "alice"],
        "result": ["", ""],
        "reasoning": ["", ""],
        "user.md": ["", ""],
    })
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        tmp = f.name
    try:
        df.to_excel(tmp, index=False)
        cases, missed_cases, stats = prepare_cases_from_run_output(
            tmp,
            model="m",
            prompt_version="p",
            chunk_size=2,
            return_missed=True,
        )

        assert cases == []
        assert len(missed_cases) == 1
        assert stats["generated_cases"] == 0
        assert stats["missed_cases"] == 1
    finally:
        os.unlink(tmp)


def test_prepare_long_memory_cases_uses_memory_columns_and_resets_when_reviewer_changes():
    df = pd.DataFrame({
        "session_id": ["s1", "s1", "s2", "s3", "s4"],
        "轮次": [1, 2, 1, 1, 1],
        "query": ["q1", "q2", "q3", "q4", "q5"],
        "answer": ["a1", "a2", "a3", "a4", "a5"],
        "评测人": ["alice", "alice", "alice", "bob", "alice"],
        "MEMORY.md": ["", "- 计划：A1", "- 计划：A2", "- 计划：B1", "- 计划：A3"],
        "模型原始返回": ["", "raw1", "raw2", "raw3", "raw4"],
        "reasoning": ["", "r1", "r2", "r3", "r4"],
    })
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        tmp = f.name
    try:
        df.to_excel(tmp, index=False)
        cases, missed, stats = prepare_long_memory_cases_from_run_output(
            tmp,
            model="memory-model",
            prompt_version="memory-v1",
            chunk_size=2,
            return_missed=True,
        )

        assert not missed
        assert stats["generated_cases"] == 4
        assert [case.task_type for case in cases] == [TaskType.LONG_MEMORY] * 4
        assert cases[0].old_memory is None
        assert cases[0].candidate_output == "- 计划：A1"
        assert cases[1].old_memory == "- 计划：A1"
        assert cases[1].candidate_output == "- 计划：A2"
        assert cases[2].old_memory is None
        assert cases[2].candidate_output == "- 计划：B1"
        assert cases[3].old_memory is None
        assert cases[3].candidate_output == "- 计划：A3"
        assert cases[0].metadata["candidate_source_column"] == "MEMORY.md"
        assert cases[0].metadata["document_name"] == "MEMORY.md"
    finally:
        os.unlink(tmp)


def test_prepare_long_memory_cases_accepts_existing_program_column_name():
    df = pd.DataFrame({
        "轮次": [1],
        "query": ["我准备考研"],
        "answer": ["好的"],
        "评测人": ["alice"],
        "生成的MEMORY.md正文": ["- 长期计划：用户正在准备考研"],
        "模型原始返回": ["raw"],
        "reasoning": ["reason"],
    })
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        tmp = f.name
    try:
        df.to_excel(tmp, index=False)
        cases = prepare_long_memory_cases_from_run_output(tmp)

        assert len(cases) == 1
        assert cases[0].candidate_output == "- 长期计划：用户正在准备考研"
        assert cases[0].metadata["candidate_source_column"] == "生成的MEMORY.md正文"
    finally:
        os.unlink(tmp)


def test_resume_key_and_append_result():
    result = EvalResult(
        case_id="c1",
        task_type="user_md_update",
        score_total=5.0,
        model_name="m",
        prompt_version="p",
        judge_model="judge",
        judge_prompt_version="judge_v1",
    )
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        tmp = f.name
    try:
        append_result(tmp, result)
        restored = load_results(tmp)
        assert len(restored) == 1
        assert eval_result_resume_key(restored[0]) == ("c1", "m", "p", "judge", "judge_v1", "")

        case = Case(
            case_id="c1",
            task_type=TaskType.USER_MD,
            session_id="s1",
            model_name="m",
            prompt_version="p",
        )
        assert case_resume_key(case, "judge", "judge_v1") == eval_result_resume_key(restored[0])
        assert case_resume_key(case, "judge", "judge_v1", "hash_a") != case_resume_key(
            case,
            "judge",
            "judge_v1",
            "hash_b",
        )
    finally:
        os.unlink(tmp)
