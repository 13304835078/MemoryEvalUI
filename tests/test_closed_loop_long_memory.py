from pathlib import Path

import pandas as pd

from src.loop import closed_loop
from src.schema import EvalConfig, TaskType, results_from_jsonl


def test_long_memory_closed_loop_runs_extraction_and_evaluation(monkeypatch, tmp_path):
    input_path = tmp_path / "input.xlsx"
    pd.DataFrame({
        "轮次": [1],
        "query": ["我准备考研"],
        "answer": ["好的"],
        "评测人": ["张三"],
    }).to_excel(input_path, index=False)

    monkeypatch.setattr(closed_loop, "CLOSED_LOOP_DIR", tmp_path / "closed_loop")
    monkeypatch.setattr(
        closed_loop,
        "save_cases",
        lambda cases, filename: str(tmp_path / filename),
    )
    monkeypatch.setattr(
        closed_loop,
        "save_prompt_version",
        lambda task_type, content, version_name, prompt_kind: version_name,
    )

    config = closed_loop.ClosedLoopConfig(
        run_id="long_memory_loop",
        input_excel_path=str(input_path),
        task_type=TaskType.LONG_MEMORY.value,
        rounds=1,
        chunk_size=1,
        extraction_model="mock-extractor",
        extraction_prompt_text="# MEMORY.md 更新规则\n- 只记录长期事项。",
        extraction_create_prompt_text="# MEMORY.md 新建规则\n- 只记录长期事项。",
        extraction_prompt_version="memory_v1",
        extraction_request_interval=0,
        judge_prompt_text="请输出长期记忆评分 JSON。",
        judge_prompt_version="judge_long_memory_v1",
        advisor_model="mock-advisor",
        eval_config=EvalConfig(mock=True, judge_model="mock-judge"),
    )

    closed_loop.run_closed_loop(config)

    state = closed_loop.read_loop_state(config.run_id)
    assert state["status"] == "completed"
    assert state["config"]["task_type"] == TaskType.LONG_MEMORY.value
    round_state = state["rounds"][0]
    assert round_state["case_stats"]["generated_cases"] == 1
    assert round_state["candidate_prompt_saved"].startswith("extract_long_memory_closed_loop_round_1_")

    extraction_path = Path(round_state["extraction_output"])
    extraction_df = pd.read_excel(extraction_path).fillna("")
    assert extraction_df.loc[0, "当前使用的模板"] == "create"
    assert extraction_df.loc[0, "MEMORY.md"]
    assert extraction_df.loc[0, "模型原始返回"]

    results = results_from_jsonl(round_state["results_path"])
    assert len(results) == 1
    assert results[0].task_type == TaskType.LONG_MEMORY.value
    assert results[0].judge_prompt_version == "judge_long_memory_v1"
