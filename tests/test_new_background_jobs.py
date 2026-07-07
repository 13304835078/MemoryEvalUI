from pathlib import Path

from src.schema import Case, DialogueTurn, EvalConfig, EvalResult, TaskType
from src.ui import eval_job_runner, judge_ab_job_runner, prompt_advisor_job_runner
from src.ui.eval_job_runner import EvalJobConfig, run_eval_job
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


def test_eval_background_job_can_stop_before_writing_failure_results(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_job_runner, "EVAL_JOBS_DIR", tmp_path / "jobs")
    output_path = tmp_path / "results.jsonl"

    class FakeRunner:
        def __init__(self, *args, **kwargs):
            self.judge_client = object()

        def evaluate_one(self, case):
            raise AssertionError("STOP should be checked before evaluating the case")

    def fake_wait_for_rate_slot(_scope, _interval, *, disabled=False, should_stop=None):
        eval_job_runner.request_eval_stop("eval-stop")
        return 0.0

    monkeypatch.setattr(eval_job_runner, "EvalRunner", FakeRunner)
    monkeypatch.setattr(eval_job_runner, "wait_for_global_rate_slot", fake_wait_for_rate_slot)

    config = EvalJobConfig(
        job_id="eval-stop",
        task_type=TaskType.USER_MD.value,
        output_path=str(output_path),
        prompt_file="judge.md",
        judge_prompt_version="judge",
        eval_config=EvalConfig(mock=False, judge_concurrency=2),
    )

    run_eval_job(config, [_case("c1"), _case("c2"), _case("c3")])

    state = eval_job_runner.read_eval_job_state("eval-stop")
    assert state["status"] == "stopped"
    assert state["done"] == 0
    assert state["evaluated"] == 0
    assert state["total"] == 3
    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8").strip() == ""


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


def test_judge_ab_stop_preserves_partial_stage_results(tmp_path, monkeypatch):
    monkeypatch.setattr(judge_ab_job_runner, "JUDGE_AB_JOBS_DIR", tmp_path)

    class FakeRunner:
        def __init__(self, *args, **kwargs):
            self.judge_client = object()
            self.judge_prompt_version = kwargs.get("judge_prompt_version", "")

        def evaluate_one(self, case):
            judge_ab_job_runner.request_judge_ab_stop("ab-stop")
            return EvalResult(
                case_id=case.case_id,
                task_type=TaskType.USER_MD.value,
                score_total=5.0,
                scores={"correctness": 5.0},
                comment="ok",
                model_name=case.model_name,
                prompt_version=case.prompt_version,
                judge_prompt_version=self.judge_prompt_version,
            )

    monkeypatch.setattr(judge_ab_job_runner, "EvalRunner", FakeRunner)
    config = JudgeAbJobConfig(
        job_id="ab-stop",
        task_type=TaskType.USER_MD.value,
        prompt_a="judge_a.md",
        prompt_b="judge_b.md",
        eval_config=EvalConfig(mock=True, judge_concurrency=1),
    )

    run_judge_ab_job(config, [_case("c1"), _case("c2")])

    state = judge_ab_job_runner.read_judge_ab_job_state("ab-stop")
    results_a, results_b = load_judge_ab_results("ab-stop")
    assert state["status"] == "stopped"
    assert state["done"] == 1
    assert len(results_a) == 1
    assert results_a[0].case_id == "c1"
    assert results_b == []
