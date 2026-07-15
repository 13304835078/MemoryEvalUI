from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.schema import SCORING_DIMENSIONS, TaskType


class CallStatus(str, Enum):
    NOT_ATTEMPTED = "not_attempted"
    SUCCESS = "success"
    FAILED = "failed"
    STOPPED = "stopped"
    SKIPPED = "skipped"


class ParseStatus(str, Enum):
    NOT_ATTEMPTED = "not_attempted"
    STRUCTURED = "structured"
    RAW_FALLBACK = "raw_fallback"
    EMPTY = "empty"
    UNKNOWN = "unknown"


class CaseStatus(str, Enum):
    READY = "ready"
    REVIEW_REQUIRED = "review_required"
    SKIP = "skip"


class InheritanceSource(str, Enum):
    NONE = "none"
    PARSED_DOCUMENT = "parsed_document"
    RAW_OUTPUT = "raw_output"
    PREVIOUS_DOCUMENT = "previous_document"


@dataclass(frozen=True)
class ExtractionState:
    call_status: CallStatus
    parse_status: ParseStatus
    case_status: CaseStatus


@dataclass(frozen=True)
class ExtractionTaskProfile:
    profile_id: str
    task_type: TaskType
    document_name: str
    candidate_columns: tuple[str, ...]
    raw_output_columns: tuple[str, ...]
    old_document_columns: tuple[str, ...]
    legacy_candidate_column: str
    legacy_raw_output_column: str
    reset_on_reviewer_change: bool
    preserve_previous_on_empty: bool
    input_style: str
    parser_fallback_policy: str
    default_judge_prompt: str
    default_judge_prompt_version: str
    scoring_dimensions: tuple[str, ...]

    def build_user_message(self, current_document: str, formatted_history: str) -> str:
        current = str(current_document or "").strip()
        history = str(formatted_history or "").strip()
        if self.input_style == "long_memory":
            if not current:
                return f"#输入：*新增对话记录*\n{history}\n输出"
            return (
                f"#输入：\n*现有长期记忆*\n{current}\n\n"
                f"*新增对话记录*\n{history}\n\n输出"
            )

        existing_block = (
            f"--- {self.document_name} ---\n{current}"
            if current
            else f"--- {self.document_name} ---"
        )
        return (
            f"## [现有文件内容]\n"
            f"{existing_block}\n\n"
            f"## [最新对话内容]\n"
            f"{history}"
        )


_TASK_PROFILES: dict[TaskType, ExtractionTaskProfile] = {
    TaskType.USER_MD: ExtractionTaskProfile(
        profile_id="user_md_default_v1",
        task_type=TaskType.USER_MD,
        document_name="USER.md",
        candidate_columns=(
            "effective_document",
            "parsed_document",
            "user.md",
            "USER.md",
            "新USER.md",
            "新用户画像",
        ),
        raw_output_columns=("raw_output", "result", "模型原始返回"),
        old_document_columns=("old_effective_document", "old_memory", "旧USER.md", "旧用户画像"),
        legacy_candidate_column="user.md",
        legacy_raw_output_column="result",
        reset_on_reviewer_change=False,
        preserve_previous_on_empty=False,
        input_style="user_md",
        parser_fallback_policy="raw_output_requires_review",
        default_judge_prompt="judge_user_md_absolute_stable_with_rules_v1.md",
        default_judge_prompt_version="judge_user_md_absolute_stable_with_rules_v1",
        scoring_dimensions=tuple(SCORING_DIMENSIONS[TaskType.USER_MD.value]),
    ),
    TaskType.LONG_MEMORY: ExtractionTaskProfile(
        profile_id="long_memory_default_v1",
        task_type=TaskType.LONG_MEMORY,
        document_name="MEMORY.md",
        candidate_columns=(
            "effective_document",
            "parsed_document",
            "MEMORY.md",
            "生成的MEMORY.md正文",
            "memory.md",
        ),
        raw_output_columns=("raw_output", "模型原始返回", "result"),
        old_document_columns=("old_effective_document", "旧MEMORY.md", "old_memory"),
        legacy_candidate_column="MEMORY.md",
        legacy_raw_output_column="模型原始返回",
        reset_on_reviewer_change=True,
        preserve_previous_on_empty=True,
        input_style="long_memory",
        parser_fallback_policy="raw_output_requires_review",
        default_judge_prompt="judge_long_memory_v1.md",
        default_judge_prompt_version="judge_long_memory_v1",
        scoring_dimensions=tuple(SCORING_DIMENSIONS[TaskType.LONG_MEMORY.value]),
    ),
}


def get_extraction_task_profile(task_type: TaskType | str) -> ExtractionTaskProfile:
    normalized = TaskType(task_type)
    try:
        return _TASK_PROFILES[normalized]
    except KeyError as exc:
        raise ValueError(f"任务类型 {normalized.value} 暂不支持记忆提取") from exc


def normalize_state_value(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def infer_legacy_extraction_state(
    legacy_status: Any,
    *,
    has_effective_document: bool,
    has_raw_output: bool,
    has_reasoning: bool,
) -> ExtractionState:
    status = normalize_state_value(legacy_status)

    if status.startswith(("success_unstructured", "parse_failed", "parse_uncertain")):
        case_status = (
            CaseStatus.REVIEW_REQUIRED
            if (has_effective_document or has_raw_output)
            else CaseStatus.SKIP
        )
        parse_status = ParseStatus.RAW_FALLBACK if case_status != CaseStatus.SKIP else ParseStatus.EMPTY
        return ExtractionState(CallStatus.SUCCESS, parse_status, case_status)
    if status.startswith(("success", "succeeded", "completed")) or status == "ok":
        return ExtractionState(CallStatus.SUCCESS, ParseStatus.STRUCTURED, CaseStatus.READY)
    if "成功" in status:
        return ExtractionState(CallStatus.SUCCESS, ParseStatus.STRUCTURED, CaseStatus.READY)
    if status in {"output_empty"}:
        return ExtractionState(CallStatus.SUCCESS, ParseStatus.EMPTY, CaseStatus.SKIP)
    if status in {"stopped", "cancelled", "canceled"} or any(
        token in status for token in ("已终止", "已取消")
    ):
        return ExtractionState(CallStatus.STOPPED, ParseStatus.NOT_ATTEMPTED, CaseStatus.SKIP)
    if status.startswith("skipped") or "跳过" in status:
        return ExtractionState(CallStatus.SKIPPED, ParseStatus.NOT_ATTEMPTED, CaseStatus.SKIP)
    if status == "unknown":
        return ExtractionState(CallStatus.NOT_ATTEMPTED, ParseStatus.NOT_ATTEMPTED, CaseStatus.SKIP)
    if any(
        token in status
        for token in ("api_failed", "failure", "error", "timeout", "failed", "失败", "错误", "超时")
    ):
        return ExtractionState(CallStatus.FAILED, ParseStatus.NOT_ATTEMPTED, CaseStatus.SKIP)

    if has_effective_document:
        return ExtractionState(CallStatus.SUCCESS, ParseStatus.UNKNOWN, CaseStatus.READY)
    if has_raw_output or has_reasoning:
        return ExtractionState(CallStatus.SUCCESS, ParseStatus.UNKNOWN, CaseStatus.REVIEW_REQUIRED)
    return ExtractionState(CallStatus.NOT_ATTEMPTED, ParseStatus.NOT_ATTEMPTED, CaseStatus.SKIP)


def coerce_extraction_state(
    *,
    call_status: Any,
    parse_status: Any,
    case_status: Any,
    legacy_status: Any,
    has_effective_document: bool,
    has_raw_output: bool,
    has_reasoning: bool,
) -> ExtractionState:
    call_value = normalize_state_value(call_status)
    parse_value = normalize_state_value(parse_status)
    case_value = normalize_state_value(case_status)
    try:
        return ExtractionState(
            CallStatus(call_value),
            ParseStatus(parse_value),
            CaseStatus(case_value),
        )
    except ValueError:
        return infer_legacy_extraction_state(
            legacy_status,
            has_effective_document=has_effective_document,
            has_raw_output=has_raw_output,
            has_reasoning=has_reasoning,
        )
