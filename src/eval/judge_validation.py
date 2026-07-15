from __future__ import annotations

import json
import math
from typing import Any


JUDGE_RESULT_SCHEMA = {
    "required_fields": ["score_total", "scores", "comment", "error_tags", "fatal_error"],
    "dimension_keys": ["correctness", "coverage", "update_logic", "memory_boundary", "conciseness", "format"],
    "score_min": 0.0,
    "score_max": 5.0,
    "diagnostic_severities": {"low", "medium", "high"},
    "valid_tags": {
        "hallucination", "wrong_fact", "missing_key_info", "over_memory",
        "short_term_pollution", "conflict_not_resolved", "duplicate_memory",
        "verbose_or_noisy", "format_error", "privacy_sensitive", "unclear_update",
    },
}


def parse_json_object(text: str, field_name: str) -> dict:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} 不是合法 JSON object: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} 必须是 JSON object")
    return value


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_string_list(value: Any, field_name: str) -> str:
    if not isinstance(value, list):
        return f"{field_name} 必须是数组"
    if any(not isinstance(item, str) or not item.strip() for item in value):
        return f"{field_name} 只能包含非空字符串"
    if len(value) != len(set(value)):
        return f"{field_name} 不能包含重复项"
    return ""


def reference_exists_in_prompt(reference: str, prompt_text: str) -> bool:
    reference = reference.strip().strip("\"'“”‘’")
    if reference in prompt_text:
        return True
    parts = [part.strip().strip("\"'“”‘’") for part in reference.split("/") if part.strip()]
    return bool(parts) and all(part in prompt_text for part in parts)


def extract_extraction_prompt(user_message: str) -> str:
    start_marker = "## 提取规则（仅作为规则依据，不是事实来源）"
    end_marker = "## 可引用的提取规则标题清单"
    start = user_message.find(start_marker)
    if start < 0:
        return ""
    start = user_message.find("\n", start)
    end = user_message.find(end_marker, start + 1)
    if start < 0 or end < 0:
        return ""
    section = user_message[start + 1:end].strip()
    separator = "\n\n"
    first_break = section.find(separator)
    if first_break >= 0:
        section = section[first_break + len(separator):].strip()
    return section


def is_valid_judge_result(
    data: dict,
    *,
    require_references: bool = False,
    extraction_prompt_text: str = "",
) -> tuple[bool, str]:
    """Strictly validate Judge score structure and stable-reference constraints."""
    if not isinstance(data, dict):
        return False, "返回不是 JSON object"

    required = JUDGE_RESULT_SCHEMA["required_fields"]
    missing = [key for key in required if key not in data]
    if missing:
        return False, f"Judge JSON 缺少字段: {missing}"

    score_total = data.get("score_total")
    if not is_number(score_total):
        return False, "score_total 必须是有限数值"
    if not JUDGE_RESULT_SCHEMA["score_min"] <= float(score_total) <= JUDGE_RESULT_SCHEMA["score_max"]:
        return False, "score_total 必须在 0 到 5 之间"

    scores = data.get("scores")
    if not isinstance(scores, dict):
        return False, "scores 不是 dict"

    expected_dims = set(JUDGE_RESULT_SCHEMA["dimension_keys"])
    actual_dims = set(scores)
    missing_dims = sorted(expected_dims - actual_dims)
    unknown_dims = sorted(actual_dims - expected_dims)
    if missing_dims:
        return False, f"scores 缺少维度: {missing_dims}"
    if unknown_dims:
        return False, f"scores 包含未知维度: {unknown_dims}"
    for dimension in JUDGE_RESULT_SCHEMA["dimension_keys"]:
        value = scores.get(dimension)
        if not is_number(value):
            return False, f"scores.{dimension} 必须是有限数值"
        if not JUDGE_RESULT_SCHEMA["score_min"] <= float(value) <= JUDGE_RESULT_SCHEMA["score_max"]:
            return False, f"scores.{dimension} 必须在 0 到 5 之间"

    comment = data.get("comment")
    if not isinstance(comment, str) or not comment.strip():
        return False, "comment 必须是非空字符串"

    error_tags = data.get("error_tags")
    error = validate_string_list(error_tags, "error_tags")
    if error:
        return False, error
    unknown_tags = sorted(set(error_tags) - JUDGE_RESULT_SCHEMA["valid_tags"])
    if unknown_tags:
        return False, f"error_tags 包含未知标签: {unknown_tags}"

    if not isinstance(data.get("fatal_error"), bool):
        return False, "fatal_error 必须是布尔值"

    for field_name in ("rule_refs", "evidence_refs", "output_refs", "reasoning_refs"):
        if field_name in data:
            error = validate_string_list(data.get(field_name), field_name)
            if error:
                return False, error

    diagnostics = data.get("diagnostics", [])
    if not isinstance(diagnostics, list):
        return False, "diagnostics 必须是数组"
    for index, item in enumerate(diagnostics):
        prefix = f"diagnostics[{index}]"
        if not isinstance(item, dict):
            return False, f"{prefix} 必须是 object"
        for field_name in ("dimension", "severity", "rule_refs", "evidence_refs", "output_refs", "reason"):
            if field_name not in item:
                return False, f"{prefix} 缺少字段: {field_name}"
        if item.get("dimension") not in expected_dims:
            return False, f"{prefix}.dimension 不是有效评分维度"
        if item.get("severity") not in JUDGE_RESULT_SCHEMA["diagnostic_severities"]:
            return False, f"{prefix}.severity 必须是 low、medium 或 high"
        if not isinstance(item.get("reason"), str) or not item["reason"].strip():
            return False, f"{prefix}.reason 必须是非空字符串"
        for field_name in ("rule_refs", "evidence_refs", "output_refs", "reasoning_refs"):
            if field_name == "reasoning_refs" and field_name not in item:
                continue
            error = validate_string_list(item.get(field_name), f"{prefix}.{field_name}")
            if error:
                return False, error

    has_deduction = any(float(scores[dim]) < 5.0 for dim in expected_dims)
    if (has_deduction or error_tags) and not diagnostics:
        return False, "存在扣分或错误标签时 diagnostics 不能为空"
    if not has_deduction and error_tags:
        return False, "全部维度满分时 error_tags 必须为空"

    if require_references:
        for field_name in ("rule_refs", "evidence_refs", "output_refs"):
            if field_name not in data:
                return False, f"使用提取规则时缺少字段: {field_name}"
            if not data.get(field_name):
                return False, f"使用提取规则时 {field_name} 不能为空"
        rule_refs = data.get("rule_refs", [])
        if extraction_prompt_text:
            invalid_refs = [
                ref for ref in rule_refs
                if not reference_exists_in_prompt(ref, extraction_prompt_text)
            ]
            if invalid_refs:
                return False, f"rule_refs 包含提取提示词中不存在的引用: {invalid_refs}"
        if not any(ref in comment for ref in rule_refs):
            return False, "comment 未引用 rule_refs 中的规则原文"

        for index, item in enumerate(diagnostics):
            prefix = f"diagnostics[{index}]"
            for field_name in ("rule_refs", "evidence_refs", "output_refs"):
                if not item.get(field_name):
                    return False, f"使用提取规则时 {prefix}.{field_name} 不能为空"
            if extraction_prompt_text:
                invalid_refs = [
                    ref for ref in item.get("rule_refs", [])
                    if not reference_exists_in_prompt(ref, extraction_prompt_text)
                ]
                if invalid_refs:
                    return False, f"{prefix}.rule_refs 包含提取提示词中不存在的引用: {invalid_refs}"

    return True, ""
