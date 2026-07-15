from src.extraction.contracts import (
    CallStatus,
    CaseStatus,
    ParseStatus,
    coerce_extraction_state,
    get_extraction_task_profile,
    infer_legacy_extraction_state,
)
from src.schema import TaskType


def test_task_profiles_define_document_io_and_inheritance_policy():
    user_profile = get_extraction_task_profile(TaskType.USER_MD)
    memory_profile = get_extraction_task_profile(TaskType.LONG_MEMORY)

    assert user_profile.document_name == "USER.md"
    assert user_profile.candidate_columns[0] == "effective_document"
    assert user_profile.reset_on_reviewer_change is False
    assert user_profile.preserve_previous_on_empty is False
    assert "--- USER.md ---\n旧画像" in user_profile.build_user_message("旧画像", "- user: 新对话")

    assert memory_profile.document_name == "MEMORY.md"
    assert memory_profile.raw_output_columns[0] == "raw_output"
    assert memory_profile.reset_on_reviewer_change is True
    assert memory_profile.preserve_previous_on_empty is True
    assert "*现有长期记忆*\n旧记忆" in memory_profile.build_user_message("旧记忆", "- user: 新对话")


def test_legacy_parse_failed_is_reviewable_instead_of_failed():
    state = infer_legacy_extraction_state(
        "PARSE_FAILED",
        has_effective_document=False,
        has_raw_output=True,
        has_reasoning=False,
    )

    assert state.call_status == CallStatus.SUCCESS
    assert state.parse_status == ParseStatus.RAW_FALLBACK
    assert state.case_status == CaseStatus.REVIEW_REQUIRED


def test_explicit_status_fields_take_precedence_over_legacy_status():
    state = coerce_extraction_state(
        call_status="success",
        parse_status="raw_fallback",
        case_status="review_required",
        legacy_status="API_FAILED",
        has_effective_document=True,
        has_raw_output=True,
        has_reasoning=False,
    )

    assert state.call_status == CallStatus.SUCCESS
    assert state.parse_status == ParseStatus.RAW_FALLBACK
    assert state.case_status == CaseStatus.REVIEW_REQUIRED
