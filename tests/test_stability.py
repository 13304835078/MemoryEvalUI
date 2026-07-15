import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.stability import compare_eval_stability, results_from_jsonl_text
from src.schema import EvalResult


def make_result(
    case_id: str,
    total: float,
    tags: list[str] | None = None,
    diagnostics_count: int = 0,
    rule_refs: list[str] | None = None,
    evidence_refs: list[str] | None = None,
    output_refs: list[str] | None = None,
    comment: str = "",
) -> EvalResult:
    return EvalResult(
        case_id=case_id,
        task_type="user_md_update",
        score_total=total,
        scores={
            "correctness": total,
            "coverage": total,
            "update_logic": total,
            "memory_boundary": total,
            "conciseness": total,
            "format": total,
        },
        error_tags=tags or [],
        diagnostics=[{"dimension": "coverage"} for _ in range(diagnostics_count)],
        rule_refs=rule_refs or [],
        evidence_refs=evidence_refs or ["用户说喜欢粤菜"],
        output_refs=output_refs or ["- 用户喜欢粤菜"],
        comment=comment,
        model_name="m1",
        prompt_version="p1",
    )


def test_compare_eval_stability_detects_case_level_differences():
    baseline = [
        make_result("c1", 5.0, [], 0, ["A. 允许记录"]),
        make_result("c2", 5.0, [], 0, ["A. 允许记录"]),
    ]
    current = [
        make_result("c1", 5.0, [], 0, ["A. 允许记录"]),
        make_result("c2", 4.5, ["missing_key_info"], 1, ["## 2. 明确性优先"]),
    ]

    report = compare_eval_stability(current, baseline)

    assert report["summary"]["common_count"] == 2
    assert report["summary"]["avg_total_abs_delta"] == 0.25
    assert report["summary"]["tag_exact_rate"] == 0.5
    assert report["summary"]["total_score_exact_rate"] == 0.5
    assert report["diff_rows"][0]["case_id"] == "c2"
    assert report["tag_rows"][0]["当前新增次数"] == 1
    assert "总分变化" in report["diff_rows"][0]["不稳定类型"]
    assert "错误标签变化" in report["diff_rows"][0]["不稳定类型"]
    assert report["summary"]["unstable_case_count"] == 1
    assert report["instability_type_rows"]


def test_compare_eval_stability_splits_explanation_instability():
    baseline = [
        make_result(
            "c1",
            4.25,
            ["missing_key_info"],
            1,
            ["### A4. 兴趣爱好"],
            evidence_refs=["用户说逐玉是下饭剧"],
            comment="遗漏下饭剧评价。",
        )
    ]
    current = [
        make_result(
            "c1",
            4.25,
            ["missing_key_info"],
            1,
            ["### A4. 兴趣爱好"],
            evidence_refs=["用户说逐玉服化道很厉害"],
            comment="遗漏服化道评价。",
        )
    ]

    report = compare_eval_stability(current, baseline)
    row = report["diff_rows"][0]

    assert row["总分变化"] == "否"
    assert row["错误标签变化"] == "否"
    assert row["证据引用变化"] == "是"
    assert row["评语变化"] == "是"
    # 评语措辞变化单独展示，但不再把同一结构化诊断误判为质量不稳定。
    assert row["不稳定类型"] == "证据引用变化"


def test_compare_eval_stability_supports_strict_key_mode():
    baseline = [make_result("c1", 5.0)]
    current = [make_result("c1", 5.0)]
    current[0].prompt_version = "p2"

    loose = compare_eval_stability(current, baseline, key_mode="case_id")
    strict = compare_eval_stability(current, baseline, key_mode="case_model_prompt")

    assert loose["summary"]["common_count"] == 1
    assert strict["summary"]["common_count"] == 0
    assert strict["summary"]["current_only_count"] == 1
    assert strict["summary"]["baseline_only_count"] == 1


def test_results_from_jsonl_text():
    text = "\n".join([
        '{"case_id":"c1","task_type":"user_md_update","score_total":5}',
        "",
        '{"case_id":"c2","task_type":"user_md_update","score_total":4}',
    ])

    results = results_from_jsonl_text(text)

    assert [r.case_id for r in results] == ["c1", "c2"]
    assert results[1].score_total == 4


def test_stability_excludes_runtime_failure_pairs():
    baseline = [make_result("c1", 4.0)]
    current = [EvalResult.from_parse_failure(
        case_id="c1", task_type="user_md_update", raw="API error: timeout"
    )]

    report = compare_eval_stability(current, baseline)

    assert report["summary"]["matched_count"] == 1
    assert report["summary"]["common_count"] == 0
    assert report["summary"]["execution_failure_pair_count"] == 1
    assert len(report["execution_failure_rows"]) == 1
