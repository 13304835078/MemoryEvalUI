import json

from src.eval.extraction_pairwise_judge import (
    build_pairwise_user_message,
    normalize_pairwise_result,
    stable_swap_for_source,
)
from src.schema import Case, DialogueTurn, TaskType


def _case(output: str, *, reasoning: str = "") -> Case:
    return Case(
        case_id="case",
        task_type=TaskType.USER_MD,
        session_id="session-1",
        old_memory="# old",
        dialogue=[DialogueTurn(role="user", content="用户事实")],
        candidate_output=output,
        metadata={"reasoning": reasoning},
    )


def test_pairwise_result_maps_swapped_candidates_back_to_a_b() -> None:
    parsed = {
        "winner": "candidate_1",
        "confidence": "high",
        "reason": "candidate 1 better",
        "issues_candidate_1": [],
        "issues_candidate_2": ["遗漏"],
        "error_tags_candidate_2": ["missing_key_info", "invented_tag"],
        "strengths_candidate_1": ["完整"],
    }

    result = normalize_pairwise_result(parsed, swap_candidates=True)

    assert result["winner"] == "B"
    assert result["issues_a"] == ["遗漏"]
    assert result["error_tags_a"] == ["missing_key_info"]
    assert result["strengths_b"] == ["完整"]


def test_pairwise_message_marks_reasoning_as_auxiliary() -> None:
    message = build_pairwise_user_message(
        _case("A", reasoning="A reasoning"),
        _case("B", reasoning="B reasoning"),
        evaluation_rule_prompt="只记录长期稳定事实",
        task_type=TaskType.USER_MD.value,
        swap_candidates=False,
    )
    payload = json.loads(message.split("\n\n", 1)[1])

    assert payload["candidate_1"]["reasoning_auxiliary_only"] == "A reasoning"
    assert payload["candidate_2"]["reasoning_auxiliary_only"] == "B reasoning"
    assert payload["frozen_extraction_rules"] == "只记录长期稳定事实"


def test_stable_swap_is_deterministic() -> None:
    assert stable_swap_for_source("same-key") == stable_swap_for_source("same-key")
