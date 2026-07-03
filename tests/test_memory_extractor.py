import os
import sys
import tempfile
import threading
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extraction.memory_extractor import (
    MemoryExtractionConfig,
    MemoryExtractionRunner,
    MockMemoryExtractionClient,
    build_dialogue_history,
    build_user_prompt,
    extract_answer_from_response,
    extract_user_md,
    split_sessions,
)
from src.persistence import read_jsonl
from src.ui.data_service import prepare_cases_from_run_output
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


def test_extract_user_md_from_output_marker():
    text = """
# Think
分析过程

# Output
--- USER.md ---
1. 喜欢粤菜
* 常住上海
"""
    assert extract_user_md(text) == "- 喜欢粤菜\n- 常住上海"


def test_extract_user_md_rejects_reasoning_only():
    assert extract_user_md("# Reasoning\n用户说了一个临时请求") is None


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
        assert Path(stats["journal_path"]).exists()
        journal_rows = read_jsonl(stats["journal_path"])
        assert len(journal_rows) == 3
        assert [row.get("status") for row in journal_rows] == ["", "SUCCESS", "SUCCESS"]

        out = pd.read_excel(output_path).fillna("")
        assert out.loc[0, "user.md"] == ""
        assert out.loc[1, "user.md"] == "- 喜欢粤菜\n- 常住上海"
        assert out.loc[2, "user.md"] == ""
        assert out.loc[1, "reasoning"] == "r1"

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
