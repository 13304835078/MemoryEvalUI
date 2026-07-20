from pathlib import Path

import pandas as pd

from src.ui.extraction_prompt_ab_export import write_extraction_prompt_diff_excel


def test_diff_excel_defaults_to_compact_comparison_columns(tmp_path: Path) -> None:
    common = {
        "session_id": 1,
        "chunk_id": 1,
        "query": "query",
        "answer": "answer",
        "评测人": "reviewer",
    }
    extraction_a = tmp_path / "a.xlsx"
    extraction_b = tmp_path / "b.xlsx"
    output = tmp_path / "diff.xlsx"
    pd.DataFrame([{**common, "effective_document": "A", "reasoning": "why A"}]).to_excel(
        extraction_a,
        index=False,
    )
    pd.DataFrame([{**common, "effective_document": "B", "reasoning": ""}]).to_excel(
        extraction_b,
        index=False,
    )

    write_extraction_prompt_diff_excel(
        extraction_a_path=extraction_a,
        extraction_b_path=extraction_b,
        comparison_rows=[
            {
                "reviewer": "reviewer",
                "session_id": 1,
                "chunk_id": 1,
                "score_a": 4.0,
                "score_b": 4.5,
                "score_delta_b_minus_a": 0.5,
                "comparison": "B较优",
                "comparison_note": "B 更完整。",
                "issues_a": "遗漏稳定信息",
                "issues_b": "",
                "strengths_a": "",
                "strengths_b": "继承完整",
                "pairwise_status": "success",
            }
        ],
        output_path=output,
        model_comparison={
            "status": "success",
            "model": "comparison-model",
            "preferred_version": "B",
            "summary": "B 更稳定。",
            "reasons": ["覆盖率更高"],
        },
    )

    workbook = pd.ExcelFile(output)
    assert workbook.sheet_names == ["逐行Diff", "逐Chunk对比", "说明", "模型综合意见"]
    diff = pd.read_excel(output, sheet_name="逐行Diff")
    expected_columns = [
        "session_id",
        "chunk_id",
        "query",
        "answer",
        "评测人",
        "A提取结果",
        "B提取结果",
        "A相对问题",
        "B相对问题",
        "A相对优点",
        "B相对优点",
        "对比结论",
        "对比备注",
    ]
    assert list(diff.columns) == expected_columns
    assert diff.loc[0, "A提取结果"] == "A"
    assert diff.loc[0, "B提取结果"] == "B"
    assert diff.loc[0, "对比结论"] == "B较优"


def test_diff_excel_adds_only_selected_optional_sections(tmp_path: Path) -> None:
    source = tmp_path / "source.xlsx"
    output = tmp_path / "diff.xlsx"
    pd.DataFrame(
        [
            {
                "session_id": 1,
                "chunk_id": 1,
                "query": "query",
                "answer": "answer",
                "评测人": "reviewer",
                "effective_document": "output",
                "reasoning": "reason",
            }
        ]
    ).to_excel(source, index=False)

    write_extraction_prompt_diff_excel(
        extraction_a_path=source,
        extraction_b_path=source,
        comparison_rows=[
            {
                "reviewer": "reviewer",
                "session_id": 1,
                "chunk_id": 1,
                "comparison": "基本持平",
                "comparison_note": "无本轮质量差异。",
                "history_input_a": "已输入",
                "history_input_b": "已输入",
                "history_baseline_relation": "相同",
            }
        ],
        output_path=output,
        optional_sections=["reasoning", "history"],
    )

    diff = pd.read_excel(output, sheet_name="逐行Diff")
    assert "A_reasoning" in diff.columns
    assert "B_reasoning" in diff.columns
    assert "A历史输入" in diff.columns
    assert "B历史输入" in diff.columns
    assert "历史基线关系" in diff.columns
    assert "A错误标签" not in diff.columns
    assert "规则引用" not in diff.columns
