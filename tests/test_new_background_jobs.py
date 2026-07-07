from pathlib import Path

from src.schema import Case, DialogueTurn, EvalConfig, TaskType
from src.ui import judge_ab_job_runner, prompt_advisor_job_runner
from src.ui.judge_ab_job_runner import JudgeAbJobConfig, load_judge_ab_results, run_judge_ab_job
from src.ui.prompt_advisor_job_runner import PromptAdvisorJobConfig, run_prompt_advisor_job


def _case(case_id: str) -> Case:
    return Case(
        case_id=case_id,
        task_type=TaskType.USER_MD,
        session_id="s1",
        old_memory="",
        dialogue=[DialogueTurn(role="user", content="我喜欢粤菜")],
        candidate_output="- 喜欢粤菜",
        model_name="model-a",
        prompt_version="prompt-a",
    )


def test_prompt_advisor_background_job_completes_in_mock(tmp_path, monkeypatch):
    monkeypatch.setattr(prompt_advisor_job_runner, "PROMPT_ADVISOR_JOBS_DIR", tmp_path)
    config = PromptAdvisorJobConfig(
        job_id="advisor-1",
        task_type=TaskType.USER_MD.value,
        evidence=[{"case_id": "c1"}, {"case_id": "c2"}, {"case_id": "c3"}],
        current_judge_prompt="judge",
        extraction_prompt="## 规则\n- 只记录长期偏好。",
        target="extraction_prompt",
        min_evidence=3,
        eval_config=EvalConfig(mock=True),
    )

    run_prompt_advisor_job(config)

    state = prompt_advisor_job_runner.read_prompt_advisor_job_state("advisor-1")
    assert state["status"] == "completed"
    assert state["done"] == state["total"]
    assert state["total"] >= 2
    assert state["result"]["can_suggest"] is True
    assert "judge_api_bearer_token" not in state["config"]["eval_config"]


def test_judge_ab_background_job_completes_in_mock(tmp_path, monkeypatch):
    monkeypatch.setattr(judge_ab_job_runner, "JUDGE_AB_JOBS_DIR", tmp_path)
    config = JudgeAbJobConfig(
        job_id="ab-1",
        task_type=TaskType.USER_MD.value,
        prompt_a="judge_user_md_v1.md",
        prompt_b="judge_user_md_absolute_stable_with_rules_v1.md",
        cases_file=str(Path("cases.jsonl")),
        eval_config=EvalConfig(mock=True, judge_concurrency=2),
    )

    run_judge_ab_job(config, [_case("c1"), _case("c2")])

    state = judge_ab_job_runner.read_judge_ab_job_state("ab-1")
    assert state["status"] == "completed"
    assert state["done"] == 4
    assert state["summary_a"]["total"] == 2
    assert Path(state["table_path"]).exists()
    results_a, results_b = load_judge_ab_results("ab-1")
    assert len(results_a) == 2
    assert len(results_b) == 2
