from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.schema import Case, EvalConfig, EvalResult, TaskType
from src.ui.error_diagnostics import classify_failure_text
from src.ui.preflight import (
    ERROR,
    WARNING,
    build_ab_preflight,
    build_advisor_preflight,
    build_eval_preflight,
    build_extraction_preflight,
    preflight_ok,
)
from src.ui.prompt_diff import analyze_prompt_diff
from src.ui.result_triage import result_matches_filter, result_navigation_key, triage_result_rows
from src.ui.run_presets import apply_run_preset, load_custom_presets, save_custom_preset
from src.ui import task_indicator


def _case(candidate: str = "- 用户喜欢咖啡") -> Case:
    return Case(
        case_id="case-1",
        task_type=TaskType.USER_MD,
        session_id="session-1",
        candidate_output=candidate,
    )


def _result(**overrides) -> EvalResult:
    values = {
        "case_id": "case-1",
        "task_type": TaskType.USER_MD.value,
        "score_total": 5.0,
        "model_name": "model-a",
        "prompt_version": "prompt-v1",
        "judge_model": "judge-a",
        "judge_prompt_version": "judge-v1",
    }
    values.update(overrides)
    return EvalResult(**values)


def test_eval_preflight_blocks_missing_prompt_and_warns_without_extraction_rules() -> None:
    checks = build_eval_preflight(
        cases=[_case()],
        task_type=TaskType.USER_MD.value,
        eval_config=EvalConfig(mock=True),
        judge_prompt_text="",
        extraction_prompt_selected=False,
    )

    assert not preflight_ok(checks)
    assert any(item.code == "judge_prompt" and item.status == ERROR for item in checks)
    assert any(item.code == "extraction_prompt" and item.status == WARNING for item in checks)


def test_extraction_preflight_validates_local_path_and_case_metadata(tmp_path: Path) -> None:
    input_path = tmp_path / "dialogues.xlsx"
    input_path.write_bytes(b"placeholder")
    checks = build_extraction_preflight(
        local_path=str(input_path),
        prompt_text="prompt",
        eval_config=EvalConfig(mock=True),
        model_name="model",
        concurrency=1,
        request_interval=0,
        auto_make_cases=True,
        case_model_name="",
        case_prompt_version="v1",
    )

    assert not preflight_ok(checks)
    assert any(item.code == "case_output" and item.status == ERROR for item in checks)


def test_advisor_and_ab_preflight_enforce_required_inputs() -> None:
    advisor = build_advisor_preflight(
        results_count=3,
        evidence_count=1,
        min_evidence=3,
        target="extraction_prompt",
        judge_prompt_text="judge",
        extraction_prompt_text="",
        eval_config=EvalConfig(mock=True),
    )
    ab = build_ab_preflight(
        cases=[_case()],
        task_type=TaskType.USER_MD.value,
        prompt_a_text="judge",
        prompt_b_text="judge",
        prompt_a_name="same.md",
        prompt_b_name="same.md",
        extraction_prompt_text="rules",
        eval_config=EvalConfig(mock=True),
    )

    assert not preflight_ok(advisor)
    assert any(item.code == "single_variable" and item.status == WARNING for item in ab)


def test_failure_classifier_distinguishes_rate_limit_and_json_parse() -> None:
    rate = classify_failure_text("API error: QPS limit exceeded, limit:0.10")
    parsed = classify_failure_text("JSON parsing failed: missing brace")

    assert rate.code == "rate_limit"
    assert parsed.code == "json_parse"


def test_triage_prioritizes_fatal_and_low_score_results() -> None:
    fatal = EvalResult.from_parse_failure(
        case_id="fatal",
        task_type="user_md_update",
        raw="API error: QPS limit exceeded",
    )
    low = _result(case_id="low", score_total=3.5, error_tags=["missing_key_info"])
    healthy = _result(case_id="ok")

    rows = triage_result_rows([healthy, low, fatal])

    assert [row["样本编号"] for row in rows] == ["fatal", "low"]
    assert rows[0]["失败类型"] == "接口限流"
    assert result_matches_filter(low, "低分")
    assert result_navigation_key(healthy)[0] == "ok"


def test_prompt_diff_reports_growth_and_duplicate_headings() -> None:
    summary = analyze_prompt_diff("# 原则\n- A", "# 原则\n- A\n# 原则\n- B")

    assert summary.added_lines == 2
    assert summary.removed_lines == 0
    assert summary.growth_ratio > 0
    assert summary.duplicate_headings == ("# 原则",)
    assert "+++ 对照版本" in summary.diff_text


def test_custom_run_preset_round_trip_and_apply(tmp_path: Path) -> None:
    path = tmp_path / "run_presets.json"
    save_custom_preset("真实小样本", {"judge_concurrency": 3, "api_token": "secret"}, path)

    presets = load_custom_presets(path)
    merged = apply_run_preset({"api_token": "keep", "judge_concurrency": 1}, presets["真实小样本"])

    assert merged["judge_concurrency"] == 3
    assert merged["api_token"] == "keep"
    assert "api_token" not in presets["真实小样本"]


def test_evaluation_data_page_does_not_embed_memory_extraction_runner() -> None:
    page_source = (Path(__file__).parents[1] / "pages" / "2_数据输入.py").read_text(encoding="utf-8")

    assert "MemoryExtractionRunner" not in page_source
    assert "运行 USER.md 记忆提取" not in page_source
    assert 'st.switch_page("pages/10_记忆提取.py")' in page_source


def test_sidebar_task_indicator_invokes_fragment_inside_sidebar(monkeypatch) -> None:
    events: list[str] = []

    class SidebarContext:
        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, exc_type, exc, traceback):
            events.append("exit")

    monkeypatch.setattr(task_indicator, "st", SimpleNamespace(sidebar=SidebarContext()))
    monkeypatch.setattr(
        task_indicator,
        "_render_task_indicator_fragment",
        lambda: events.append("fragment"),
    )

    task_indicator.render_sidebar_task_indicator()

    assert events == ["enter", "fragment", "exit"]
