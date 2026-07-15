import os
import sys
import tempfile
import threading
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extraction.memory_extractor import (
    MemoryExtractionConfig,
    MemoryExtractionClient,
    MemoryExtractionRunner,
    MockMemoryExtractionClient,
    build_dialogue_history,
    build_long_memory_prompt,
    build_user_prompt,
    extract_answer_from_response,
    extract_long_memory,
    extract_user_md,
    load_generation_prompt_templates,
    parse_memory_document,
    split_sessions,
)
from src.persistence import read_jsonl
from src.schema import TaskType
from src.ui.data_service import (
    prepare_cases_from_run_output,
    prepare_long_memory_cases_from_run_output,
)
from src.ui.memory_extraction_job_runner import MemoryExtractionJobConfig, estimate_total_chunks


class FakeExtractionClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.messages = []

    def chat_with_retry(self, messages):
        self.messages.append(messages)
        if not self.outputs:
            return False, "no output", "", "no output"
        return self.outputs.pop(0)


class PromptAwareFakeExtractionClient:
    def __init__(self):
        self.messages = []
        self.lock = threading.Lock()

    def chat_with_retry(self, messages):
        user_message = messages[-1]["content"]
        with self.lock:
            self.messages.append(user_message)

        if "Alice 1" in user_message:
            return True, "# Output\n--- USER.md ---\n- Alice memory 1", "", ""
        if "Alice 2" in user_message:
            assert "- Alice memory 1" in user_message
            return True, "# Output\n--- USER.md ---\n- Alice memory 2", "", ""
        if "Alice 3" in user_message:
            assert "- Alice memory 2" in user_message
            return True, "# Output\n--- USER.md ---\n- Alice memory 3", "", ""
        if "Bob 1" in user_message:
            assert "- Alice memory" not in user_message
            return True, "# Output\n--- USER.md ---\n- Bob memory 1", "", ""
        return False, "unexpected prompt", "", "unexpected prompt"


class LongMemoryReviewerSwitchFakeClient:
    def chat_with_retry(self, messages):
        user_message = messages[-1]["content"]
        if "Alice first" in user_message:
            return True, "- 计划：Alice first", "", ""
        if "Bob" in user_message:
            assert "Alice first" not in user_message
            return True, "- 计划：Bob", "", ""
        if "Alice returns" in user_message:
            assert "Alice first" not in user_message
            return True, "- 计划：Alice returns", "", ""
        return False, "unexpected prompt", "", "unexpected prompt"


class RetryingExtractionClient(MemoryExtractionClient):
    def __init__(self, config):
        super().__init__(config)
        self.attempts = 0

    def _request_once(self, messages):
        self.attempts += 1
        if self.attempts == 1:
            return False, "temporary failure", "", "raw"
        return True, "# Output\n--- USER.md ---\n- 城市：杭州", "", "raw"


def test_extract_user_md_from_output_marker():
    text = """
# Think
分析过程

# Output
--- USER.md ---
1. 喜欢粤菜
* 常住上海
"""
    assert extract_user_md(text) == "1. 喜欢粤菜\n* 常住上海"


def test_extract_user_md_rejects_reasoning_only():
    assert extract_user_md("# Reasoning\n用户说了一个临时请求") is None
    assert extract_user_md("# Reasoning: 用户只表达了临时需求\n## 分析\n- 不应记录") is None


def test_extract_user_md_supports_chinese_output_marker_and_preserves_sections():
    text = """
*输出*
## 第一分区：用户基础静态信息台账
- 常驻地：杭州

## 第二分区：长期稳定行为存档
1. 日常饮食行为
   - 无记录
"""

    parsed = parse_memory_document(text, "USER.md")

    assert parsed.method == "output_marker"
    assert parsed.confidence > 0.9
    assert parsed.document == (
        "## 第一分区：用户基础静态信息台账\n"
        "- 常驻地：杭州\n\n"
        "## 第二分区：长期稳定行为存档\n"
        "1. 日常饮食行为\n"
        "   - 无记录"
    )


def test_extract_user_md_supports_json_and_structured_markdown_fallback():
    json_result = parse_memory_document(
        '{"document_type":"USER.md","document":"## 基本信息\\n- 城市：杭州"}',
        "USER.md",
    )
    fallback_result = parse_memory_document(
        "## 第一分区：用户基础静态信息台账\n- 城市：杭州",
        "USER.md",
    )

    assert json_result.document == "## 基本信息\n- 城市：杭州"
    assert json_result.method == "json:document"
    assert fallback_result.document == "## 第一分区：用户基础静态信息台账\n- 城市：杭州"
    assert fallback_result.method == "structured_markdown_fallback"
    assert fallback_result.warnings

    mismatch = parse_memory_document(
        '{"document_type":"MEMORY.md","document":"- 不应作为 USER.md"}',
        "USER.md",
    )
    assert mismatch.document is None
    assert mismatch.method == "json_document_type_mismatch"


def test_extract_long_memory_and_build_update_prompt():
    assert extract_long_memory(
        "# Output\n--- MEMORY.md ---\n- 长期计划：用户正在准备考研"
    ) == "- 长期计划：用户正在准备考研"
    prompt = build_long_memory_prompt("- 长期计划：准备考研", "- user: 目标改为明年")
    assert "*现有长期记忆*\n- 长期计划：准备考研" in prompt
    assert "*新增对话记录*\n- user: 目标改为明年" in prompt


def test_load_generation_prompt_templates_reads_create_and_update_yaml(tmp_path):
    path = tmp_path / "memory.yaml"
    path.write_text(
        "memory_extraction:\n"
        "  create_template: |\n"
        "    create rules\n"
        "  update_template: |\n"
        "    update rules\n",
        encoding="utf-8",
    )

    templates = load_generation_prompt_templates(path)

    assert templates == {"create": "create rules", "update": "update rules"}


def test_extract_answer_from_response_supports_reasoning():
    content, reasoning = extract_answer_from_response({
        "choices": [
            {
                "message": {
                    "content": "answer",
                    "reasoning_content": "reason",
                }
            }
        ]
    })

    assert content == "answer"
    assert reasoning == "reason"


def test_memory_extraction_retry_reserves_global_rate_slot():
    client = RetryingExtractionClient(MemoryExtractionConfig(max_retries=1, retry_sleep=0))
    waits = []
    client.rate_limit_wait_callback = lambda: waits.append("waited")

    success, answer, _reasoning, _error = client.chat_with_retry([
        {"role": "user", "content": "test"},
    ])

    assert success is True
    assert "城市：杭州" in answer
    assert waits == ["waited"]


def test_memory_extraction_retry_stops_before_next_request():
    client = RetryingExtractionClient(MemoryExtractionConfig(max_retries=2, retry_sleep=0))
    client.should_stop_callback = lambda: True

    success, answer, _reasoning, error = client.chat_with_retry([
        {"role": "user", "content": "test"},
    ])

    assert success is False
    assert answer == "STOP REQUESTED"
    assert error == "STOP_REQUESTED"
    assert client.attempts == 1


def test_split_sessions_and_prompt_building():
    df = pd.DataFrame({
        "轮次": [1, 2, 1],
        "query": ["q1", "q2", "q3"],
        "answer": ["a1", "a2", "a3"],
        "评测人": ["r", "r", "r"],
    })
    sessions = split_sessions(df)

    assert len(sessions) == 2
    assert len(sessions[0]) == 2
    history = build_dialogue_history(sessions[0])
    assert "- user: q1" in history
    assert "- assistant: a2" in history
    prompt = build_user_prompt("- 旧画像", history)
    assert "## [现有文件内容]" in prompt
    assert "--- USER.md ---\n- 旧画像" in prompt


def test_memory_extraction_runner_outputs_run_user_compatible_excel():
    df = pd.DataFrame({
        "轮次": [1, 2, 1],
        "query": ["我喜欢粤菜", "我常住上海", "帮我查天气"],
        "answer": ["好的", "记下了", "今天晴"],
        "评测人": ["张三", "张三", "张三"],
    })
    fake_client = FakeExtractionClient([
        (True, "# Output\n--- USER.md ---\n- 喜欢粤菜\n- 常住上海", "r1", ""),
        (True, "# Output\n--- USER.md ---\n", "r2", ""),
    ])

    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp) / "input.xlsx"
        output_path = Path(tmp) / "output.xlsx"
        df.to_excel(input_path, index=False)

        runner = MemoryExtractionRunner(
            MemoryExtractionConfig(model="mock", request_interval=0),
            prompt_text="提取 USER.md",
            client=fake_client,
        )
        stats = runner.process_excel(input_path, output_path, chunk_size=2)

        assert stats["chunks"] == 2
        assert stats["api_calls"] == 2
        assert stats["status_counts"] == {"SUCCESS": 2}
        assert stats["call_status_counts"] == {"success": 2}
        assert stats["parse_status_counts"] == {"structured": 1, "empty": 1}
        assert stats["case_status_counts"] == {"ready": 2}
        assert stats["task_profile_id"] == "user_md_default_v1"
        assert Path(stats["journal_path"]).exists()
        journal_rows = read_jsonl(stats["journal_path"])
        assert len(journal_rows) == 3
        assert [row.get("status") for row in journal_rows] == ["", "SUCCESS", "SUCCESS"]

        out = pd.read_excel(output_path).fillna("")
        assert out.loc[0, "user.md"] == ""
        assert out.loc[1, "user.md"] == "- 喜欢粤菜\n- 常住上海"
        assert out.loc[2, "user.md"] == ""
        assert out.loc[1, "reasoning"] == "r1"
        assert out.loc[1, "parse_method"] == "document_separator"
        assert out.loc[1, "parse_confidence"] > 0.9
        assert out.loc[1, "call_status"] == "success"
        assert out.loc[1, "parse_status"] == "structured"
        assert out.loc[1, "case_status"] == "ready"
        assert out.loc[1, "raw_output"].startswith("# Output")
        assert out.loc[1, "parsed_document"] == "- 喜欢粤菜\n- 常住上海"
        assert out.loc[1, "effective_document"] == "- 喜欢粤菜\n- 常住上海"
        assert out.loc[1, "inheritance_source"] == "parsed_document"
        assert out.loc[2, "old_effective_document"] == "- 喜欢粤菜\n- 常住上海"

        cases, missed, convert_stats = prepare_cases_from_run_output(
            output_path,
            model="mock",
            prompt_version="extract_v1",
            chunk_size=2,
            return_missed=True,
        )
        assert len(cases) == 2
        assert len(missed) == 0
        assert convert_stats["generated_cases"] == 2
        assert cases[0].candidate_output == "- 喜欢粤菜\n- 常住上海"
        assert cases[1].old_memory == "- 喜欢粤菜\n- 常住上海"
        assert cases[0].metadata["task_profile_id"] == "user_md_default_v1"
        assert cases[0].metadata["call_status"] == "success"


def test_unstructured_extraction_uses_raw_output_as_low_confidence_candidate():
    df = pd.DataFrame({
        "轮次": [1, 1],
        "query": ["first", "second"],
        "answer": ["ok", "ok"],
        "评测人": ["alice", "alice"],
    })
    fake_client = FakeExtractionClient([
        (True, "# Output\n--- USER.md ---\n- 城市：杭州", "", ""),
        (True, "模型没有给出可识别的正文", "reason", ""),
    ])

    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp) / "input.xlsx"
        output_path = Path(tmp) / "output.xlsx"
        df.to_excel(input_path, index=False)

        runner = MemoryExtractionRunner(
            MemoryExtractionConfig(model="mock", request_interval=0),
            prompt_text="提取 USER.md",
            client=fake_client,
        )
        runner.process_excel(input_path, output_path, chunk_size=1)

        out = pd.read_excel(output_path).fillna("")
        assert out.loc[1, "status"] == "SUCCESS_UNSTRUCTURED"
        assert out.loc[1, "user.md"] == "模型没有给出可识别的正文"
        assert out.loc[1, "parse_method"] == "raw_fallback:unrecognized"
        assert out.loc[1, "parse_confidence"] == 0.25
        assert out.loc[1, "call_status"] == "success"
        assert out.loc[1, "parse_status"] == "raw_fallback"
        assert out.loc[1, "case_status"] == "review_required"
        assert out.loc[1, "parsed_document"] == ""
        assert out.loc[1, "effective_document"] == "模型没有给出可识别的正文"
        assert out.loc[1, "inheritance_source"] == "raw_output"

        cases, missed, stats = prepare_cases_from_run_output(
            output_path,
            model="mock",
            prompt_version="v1",
            chunk_size=1,
            return_missed=True,
        )

        assert len(cases) == 2
        assert len(missed) == 0
        assert cases[1].candidate_output == "模型没有给出可识别的正文"
        assert cases[1].old_memory == "- 城市：杭州"
        assert cases[1].metadata["parse_method"] == "raw_fallback:unrecognized"
        assert cases[1].metadata["case_status"] == "review_required"
        assert cases[1].metadata["extraction_status"] == "needs_parse_review"
        assert stats["missed_reason_counts"] == {}


def test_low_confidence_raw_fallback_does_not_pollute_next_chunk_inheritance():
    df = pd.DataFrame({
        "轮次": [1, 1, 1],
        "query": ["first", "second", "third"],
        "answer": ["ok", "ok", "ok"],
        "评测人": ["alice", "alice", "alice"],
    })
    fake_client = FakeExtractionClient([
        (True, "# Output\n--- USER.md ---\n- 城市：杭州", "", ""),
        (True, "模型没有给出可识别的正文", "reason", ""),
        (True, "# Output\n--- USER.md ---\n- 城市：杭州\n- 职业：工程师", "", ""),
    ])

    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp) / "input.xlsx"
        output_path = Path(tmp) / "output.xlsx"
        df.to_excel(input_path, index=False)
        runner = MemoryExtractionRunner(
            MemoryExtractionConfig(model="mock", request_interval=0),
            prompt_text="提取 USER.md",
            client=fake_client,
        )
        runner.process_excel(input_path, output_path, chunk_size=1)

        output = pd.read_excel(output_path).fillna("")
        assert output.loc[1, "propagation_status"] == "blocked_low_confidence"
        assert "不会继承到后续分块" in output.loc[1, "parse_warnings"]
        assert output.loc[2, "old_effective_document"] == "- 城市：杭州"
        assert "模型没有给出可识别的正文" not in fake_client.messages[2][1]["content"]
        cases = prepare_cases_from_run_output(
            output_path,
            model="mock",
            prompt_version="v1",
            chunk_size=1,
        )
        assert cases[2].old_memory == "- 城市：杭州"


def test_legacy_parse_failed_row_is_recovered_from_raw_result():
    df = pd.DataFrame({
        "轮次": [1],
        "query": ["我常住杭州"],
        "answer": ["好的"],
        "评测人": ["alice"],
        "status": ["PARSE_FAILED"],
        "result": ["用户常住杭州"],
        "user.md": [""],
        "parse_method": ["unrecognized"],
    })

    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp) / "legacy.xlsx"
        df.to_excel(input_path, index=False)

        cases, missed, stats = prepare_cases_from_run_output(
            input_path,
            model="mock",
            prompt_version="v1",
            chunk_size=1,
            return_missed=True,
        )

        assert len(cases) == 1
        assert not missed
        assert cases[0].candidate_output == "用户常住杭州"
        assert "原始输出生成候选 case" in cases[0].metadata["parse_warnings"]
        assert stats["generated_cases"] == 1


def test_api_failure_is_separate_from_parse_state_and_skips_case():
    df = pd.DataFrame({
        "轮次": [1],
        "query": ["我常住杭州"],
        "answer": ["好的"],
        "评测人": ["alice"],
    })
    fake_client = FakeExtractionClient([
        (False, "API CALL FAILED", "", "timeout"),
    ])

    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp) / "input.xlsx"
        output_path = Path(tmp) / "output.xlsx"
        df.to_excel(input_path, index=False)

        runner = MemoryExtractionRunner(
            MemoryExtractionConfig(model="mock", request_interval=0),
            prompt_text="提取 USER.md",
            client=fake_client,
        )
        runner.process_excel(input_path, output_path, chunk_size=1)

        output = pd.read_excel(output_path).fillna("")
        assert output.loc[0, "status"] == "API_FAILED"
        assert output.loc[0, "call_status"] == "failed"
        assert output.loc[0, "parse_status"] == "not_attempted"
        assert output.loc[0, "case_status"] == "skip"
        assert output.loc[0, "effective_document"] == ""

        cases, missed, stats = prepare_cases_from_run_output(
            output_path,
            model="mock",
            prompt_version="v1",
            chunk_size=1,
            return_missed=True,
        )
        assert not cases
        assert len(missed) == 1
        assert missed[0].metadata["call_status"] == "failed"
        assert stats["case_status_counts"] == {"skip": 1}


def test_long_memory_extraction_uses_create_then_update_and_outputs_cases():
    df = pd.DataFrame({
        "轮次": [1, 2],
        "query": ["我准备考研", "目标改为明年"],
        "answer": ["好的", "已更新"],
        "评测人": ["张三", "张三"],
    })
    fake_client = FakeExtractionClient([
        (True, "- 长期计划：用户正在准备考研", "r1", ""),
        (True, "- 长期计划：用户计划明年考研", "r2", ""),
    ])

    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp) / "input.xlsx"
        output_path = Path(tmp) / "output.xlsx"
        df.to_excel(input_path, index=False)

        runner = MemoryExtractionRunner(
            MemoryExtractionConfig(model="mock", request_interval=0),
            prompt_text="update rules",
            create_prompt_text="create rules",
            update_prompt_text="update rules",
            task_type=TaskType.LONG_MEMORY,
            client=fake_client,
        )
        stats = runner.process_excel(input_path, output_path, chunk_size=1)

        assert stats["task_type"] == TaskType.LONG_MEMORY.value
        assert fake_client.messages[0][0]["content"] == "create rules"
        assert fake_client.messages[1][0]["content"] == "update rules"
        assert "*现有长期记忆*" not in fake_client.messages[0][1]["content"]
        assert "- 长期计划：用户正在准备考研" in fake_client.messages[1][1]["content"]

        out = pd.read_excel(output_path).fillna("")
        assert out["当前使用的模板"].tolist() == ["create", "update"]
        assert out.loc[0, "旧MEMORY.md"] == ""
        assert out.loc[1, "旧MEMORY.md"] == "- 长期计划：用户正在准备考研"
        assert out.loc[1, "MEMORY.md"] == "- 长期计划：用户计划明年考研"
        assert out.loc[1, "模型原始返回"] == "- 长期计划：用户计划明年考研"

        cases = prepare_long_memory_cases_from_run_output(
            output_path,
            model="mock",
            prompt_version="memory_v1",
            chunk_size=1,
        )
        assert [case.task_type for case in cases] == [TaskType.LONG_MEMORY, TaskType.LONG_MEMORY]
        assert cases[1].old_memory == "- 长期计划：用户正在准备考研"
        assert cases[1].candidate_output == "- 长期计划：用户计划明年考研"


def test_long_memory_empty_success_keeps_previous_document_as_no_change():
    df = pd.DataFrame({
        "轮次": [1, 2],
        "query": ["我准备考研", "今天没有新的长期事项"],
        "answer": ["好的", "好的"],
        "评测人": ["张三", "张三"],
    })
    fake_client = FakeExtractionClient([
        (True, "- 长期计划：用户正在准备考研", "r1", ""),
        (True, "# 输出\n", "r2", ""),
    ])

    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp) / "input.xlsx"
        output_path = Path(tmp) / "output.xlsx"
        df.to_excel(input_path, index=False)

        runner = MemoryExtractionRunner(
            config=MemoryExtractionConfig(model="mock", request_interval=0),
            prompt_text="update",
            create_prompt_text="create",
            update_prompt_text="update",
            client=fake_client,
            task_type=TaskType.LONG_MEMORY,
        )
        runner.process_excel(input_path, output_path, chunk_size=1)

        output = pd.read_excel(output_path).fillna("")
        second = output.iloc[1]
        assert second["status"] == "SUCCESS"
        assert second["旧MEMORY.md"] == "- 长期计划：用户正在准备考研"
        assert second["MEMORY.md"] == "- 长期计划：用户正在准备考研"
        assert "按长期记忆无变化处理" in second["parse_warnings"]


def test_long_memory_concurrency_resets_memory_for_each_reviewer_segment():
    df = pd.DataFrame({
        "轮次": [1, 1, 1],
        "query": ["Alice first", "Bob", "Alice returns"],
        "answer": ["ok", "ok", "ok"],
        "评测人": ["alice", "bob", "alice"],
    })

    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp) / "input.xlsx"
        output_path = Path(tmp) / "output.xlsx"
        df.to_excel(input_path, index=False)

        runner = MemoryExtractionRunner(
            MemoryExtractionConfig(model="mock", request_interval=0, concurrency=3),
            prompt_text="# MEMORY.md update",
            create_prompt_text="# MEMORY.md create",
            task_type=TaskType.LONG_MEMORY,
            client=LongMemoryReviewerSwitchFakeClient(),
        )
        runner.process_excel(input_path, output_path, chunk_size=1)

        out = pd.read_excel(output_path).fillna("")
        assert out["当前使用的模板"].tolist() == ["create", "create", "create"]
        assert out["旧MEMORY.md"].tolist() == ["", "", ""]


def test_long_memory_respects_preserved_source_segment_after_holdout_split():
    df = pd.DataFrame({
        "轮次": [1, 1],
        "query": ["Alice first", "Alice returns"],
        "answer": ["ok", "ok"],
        "评测人": ["alice", "alice"],
        "__source_reviewer_segment": [1, 3],
    })

    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp) / "input.xlsx"
        output_path = Path(tmp) / "output.xlsx"
        df.to_excel(input_path, index=False)
        runner = MemoryExtractionRunner(
            MemoryExtractionConfig(model="mock", request_interval=0, concurrency=1),
            prompt_text="# MEMORY.md update",
            create_prompt_text="# MEMORY.md create",
            task_type=TaskType.LONG_MEMORY,
            client=LongMemoryReviewerSwitchFakeClient(),
        )
        runner.process_excel(input_path, output_path, chunk_size=1)

        out = pd.read_excel(output_path).fillna("")
        assert out["旧MEMORY.md"].tolist() == ["", ""]
        assert "__source_reviewer_segment" not in out.columns


def test_memory_extraction_job_estimates_chunks_from_excel_dataframe():
    df = pd.DataFrame({
        "轮次": [1, 2, 1, 2, 3],
        "query": ["q1", "q2", "q3", "q4", "q5"],
        "answer": ["a1", "a2", "a3", "a4", "a5"],
        "评测人": ["张三", "张三", "李四", "李四", "李四"],
    })

    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp) / "input.xlsx"
        df.to_excel(input_path, index=False)

        session_count, estimated_chunks = estimate_total_chunks(
            MemoryExtractionJobConfig(
                job_id="test",
                input_path=str(input_path),
                output_path=str(Path(tmp) / "output.xlsx"),
                prompt_text="提取 USER.md",
                prompt_version="v1",
                chunk_size=2,
            )
        )

    assert session_count == 2
    assert estimated_chunks == 3


def test_memory_extraction_concurrency_keeps_reviewer_memory_chain_and_row_order():
    df = pd.DataFrame({
        "轮次": [1, 2, 1, 1],
        "query": ["Alice 1", "Alice 2", "Bob 1", "Alice 3"],
        "answer": ["ok", "ok", "ok", "ok"],
        "评测人": ["alice", "alice", "bob", "alice"],
    })
    fake_client = PromptAwareFakeExtractionClient()

    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp) / "input.xlsx"
        output_path = Path(tmp) / "output.xlsx"
        df.to_excel(input_path, index=False)

        runner = MemoryExtractionRunner(
            MemoryExtractionConfig(model="mock", request_interval=0, concurrency=2),
            prompt_text="提取 USER.md",
            client=fake_client,
        )
        stats = runner.process_excel(input_path, output_path, chunk_size=1)

        assert stats["chunks"] == 4
        assert stats["api_calls"] == 4
        assert stats["concurrency"] == 2
        assert stats["status_counts"] == {"SUCCESS": 4}

        out = pd.read_excel(output_path).fillna("")
        assert out["query"].tolist() == ["Alice 1", "Alice 2", "Bob 1", "Alice 3"]
        assert out["session_id"].tolist() == [1, 1, 2, 3]
        assert out.loc[0, "user.md"] == "- Alice memory 1"
        assert out.loc[1, "user.md"] == "- Alice memory 2"
        assert out.loc[2, "user.md"] == "- Bob memory 1"
        assert out.loc[3, "user.md"] == "- Alice memory 3"


def test_memory_extraction_parallel_chunk_progress_reports_live_updates():
    df = pd.DataFrame({
        "轮次": [1, 2, 1, 1],
        "query": ["Alice 1", "Alice 2", "Bob 1", "Alice 3"],
        "answer": ["ok", "ok", "ok", "ok"],
        "评测人": ["alice", "alice", "bob", "alice"],
    })
    fake_client = PromptAwareFakeExtractionClient()
    progress_events = []

    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp) / "input.xlsx"
        output_path = Path(tmp) / "output.xlsx"
        df.to_excel(input_path, index=False)

        runner = MemoryExtractionRunner(
            MemoryExtractionConfig(model="mock", request_interval=0, concurrency=2),
            prompt_text="提取 USER.md",
            client=fake_client,
        )
        runner.process_excel(
            input_path,
            output_path,
            chunk_size=1,
            progress_callback=lambda done, total, message: progress_events.append((done, total, message)),
            emit_parallel_chunk_progress=True,
        )

    completed = [done for done, _total, _message in progress_events]
    assert max(completed) == 4
    assert all(total == 4 for _done, total, _message in progress_events)
    assert any("评测人" in message and "已完成 Session" in message for _done, _total, message in progress_events)


def test_memory_extraction_runner_uses_mock_client_when_config_mock_enabled():
    runner = MemoryExtractionRunner(
        MemoryExtractionConfig(model="mock", request_interval=0, mock=True),
        prompt_text="提取 USER.md",
    )

    assert isinstance(runner.client, MockMemoryExtractionClient)
