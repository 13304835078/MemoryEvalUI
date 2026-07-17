import json

from src.eval.extraction_prompt_comparison_advisor import (
    build_comparison_user_message,
    call_comparison_model,
)
from src.schema import EvalConfig


def _report() -> dict:
    return {
        "recommendation": "建议选择 B",
        "recommendation_reason": "B 通过统计门槛。",
        "quality_a": {"end_to_end_score": 4.0},
        "quality_b": {"end_to_end_score": 4.5},
        "validation_gate": {
            "accepted": True,
            "paired_case_count": 10,
            "paired_cluster_count": 3,
            "paired_score_delta": 0.5,
            "confidence_interval": {"lower": 0.2, "upper": 0.8},
        },
        "winner_counts": {"B较优": 8, "A较优": 2},
        "dimension_summary": [],
        "rows": [
            {
                "source_key": f"case-{index}",
                "comparison": "B较优",
                "score_a": 4.0,
                "score_b": 4.5,
                "score_delta_b_minus_a": 0.5,
                "candidate_output_a": "A output",
                "candidate_output_b": "B output",
                "comparison_note": "B improved",
            }
            for index in range(12)
        ],
    }


def test_comparison_message_limits_representative_evidence() -> None:
    message = build_comparison_user_message(
        _report(),
        prompt_a="line A",
        prompt_b="line B",
        max_evidence=3,
    )
    payload = json.loads(message.split("\n\n", 1)[1])

    assert payload["formal_statistical_conclusion"]["recommendation"] == "建议选择 B"
    assert len(payload["representative_evidence"]) == 3
    assert "prompt_A" in payload["prompt_diff"]
    assert "prompt_B" in payload["prompt_diff"]


def test_mock_comparison_model_is_auxiliary_and_tracks_model_name() -> None:
    result = call_comparison_model(
        EvalConfig(mock=True, judge_model="comparison-model"),
        _report(),
        prompt_a="A",
        prompt_b="B",
    )

    assert result["status"] == "mock"
    assert result["model"] == "comparison-model"
    assert result["preferred_version"] == "B"
    assert "统计结论" in result["summary"]
