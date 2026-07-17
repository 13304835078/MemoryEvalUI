from pathlib import Path

import pandas as pd

from src.ui.extraction_prompt_ab_export import write_extraction_prompt_diff_excel


def test_diff_excel_contains_requested_columns_and_optional_reasoning(tmp_path: Path) -> None:
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
    for column in (
        "session_id",
        "chunk_id",
        "query",
        "answer",
        "评测人",
        "A提取结果",
        "B提取结果",
        "A_reasoning",
        "B_reasoning",
        "A总分",
        "B总分",
        "B-A",
        "对比结论",
        "对比备注",
    ):
        assert column in diff.columns
    assert diff.loc[0, "A提取结果"] == "A"
    assert diff.loc[0, "B提取结果"] == "B"
    assert diff.loc[0, "对比结论"] == "B较优"


def test_diff_excel_omits_reasoning_columns_when_no_reasoning_exists(tmp_path: Path) -> None:
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
            }
        ]
    ).to_excel(source, index=False)

    write_extraction_prompt_diff_excel(
        extraction_a_path=source,
        extraction_b_path=source,
        comparison_rows=[],
        output_path=output,
    )

    diff = pd.read_excel(output, sheet_name="逐行Diff")
    assert "A_reasoning" not in diff.columns
    assert "B_reasoning" not in diff.columns
