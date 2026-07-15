from __future__ import annotations

import re
from typing import Any

from src.eval.judge_client import RealJudgeClient
from src.eval.result_status import result_is_score_eligible
from src.schema import EvalResult
from src.ui.prompt_editor import (
    infer_prompt_version,
    list_extraction_prompt_files,
    load_prompt,
    prompt_text_hash,
)


SPLIT_RE = re.compile(r"\s*(?:/|、|；|;|\n)\s*")
SPACE_RE = re.compile(r"\s+")
QUOTE_RE = re.compile(r"[“\"'「『《【]([^”\"'」』》】]{2,180})[”\"'」』》】]")
RULE_ID_RE = re.compile(r"(?<![A-Za-z0-9])R\d+(?![A-Za-z0-9])")


def _normalize_text(text: str) -> str:
    return SPACE_RE.sub(" ", str(text or "")).strip()


def _clean_ref(ref: str) -> str:
    text = _normalize_text(ref)
    return text.strip(" \t\r\n-:：,，.。；;“”\"'`「」『』《》【】[]()（）")


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _normalize_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _rule_ref_exists(ref: str, prompt_text: str) -> bool:
    cleaned = _clean_ref(ref)
    if not cleaned:
        return False

    prompt_norm = _normalize_text(prompt_text)
    cleaned_norm = _normalize_text(cleaned)
    if cleaned_norm and cleaned_norm in prompt_norm:
        return True

    parts = [_clean_ref(part) for part in SPLIT_RE.split(cleaned) if _clean_ref(part)]
    if len(parts) > 1 and all(_normalize_text(part) in prompt_norm for part in parts):
        return True

    quoted_parts = [_clean_ref(part) for part in QUOTE_RE.findall(cleaned) if _clean_ref(part)]
    if quoted_parts and all(_normalize_text(part) in prompt_norm for part in quoted_parts):
        return True

    return False


def _extract_rule_refs_from_raw_response(raw_response: str | None) -> list[dict[str, str]]:
    if not raw_response:
        return []

    parsed = RealJudgeClient._parse_json_response(raw_response)
    if not isinstance(parsed, dict):
        return []
    normalized = RealJudgeClient._normalize_judge_result(parsed)

    details: list[dict[str, str]] = []
    for ref in _normalize_str_list(normalized.get("rule_refs")):
        details.append({"source": "raw_response.rule_refs", "ref": ref})

    diagnostics = normalized.get("diagnostics") or []
    if isinstance(diagnostics, dict):
        diagnostics = [diagnostics]
    if isinstance(diagnostics, list):
        for index, item in enumerate(diagnostics, 1):
            if not isinstance(item, dict):
                continue
            for ref in _normalize_str_list(item.get("rule_refs") or item.get("rule_references") or item.get("规则引用")):
                details.append({"source": f"raw_response.diagnostics[{index}].rule_refs", "ref": ref})

    return details


def _parsed_rule_ref_details(result: EvalResult) -> list[dict[str, str]]:
    details: list[dict[str, str]] = []
    for ref in result.rule_refs or []:
        details.append({"source": "rule_refs", "ref": str(ref)})
    for index, item in enumerate(result.diagnostics or [], 1):
        if not isinstance(item, dict):
            continue
        for ref in _normalize_str_list(item.get("rule_refs")):
            details.append({"source": f"diagnostics[{index}].rule_refs", "ref": ref})
    return details


def _load_prompt_by_result(result: EvalResult) -> dict[str, Any]:
    expected_hash = result.extraction_prompt_hash or ""
    expected_version = result.extraction_prompt_version or ""
    files = list_extraction_prompt_files()

    if expected_hash:
        for filename in files:
            text = load_prompt(filename, prompt_kind="extraction")
            if prompt_text_hash(text) == expected_hash:
                return {
                    "found": True,
                    "text": text,
                    "source": filename,
                    "hash_match": True,
                }

    if expected_version:
        for filename in files:
            if filename == expected_version or infer_prompt_version(filename) == expected_version:
                text = load_prompt(filename, prompt_kind="extraction")
                actual_hash = prompt_text_hash(text)
                return {
                    "found": True,
                    "text": text,
                    "source": filename,
                    "hash_match": (actual_hash == expected_hash) if expected_hash else None,
                }

    return {
        "found": False,
        "text": "",
        "source": "",
        "hash_match": None,
    }


def validate_result_rule_refs(result: EvalResult, extraction_prompt_text: str | None = None) -> dict[str, Any]:
    if not result_is_score_eligible(result):
        return {
            "checked": False,
            "status": "runtime_failure",
            "status_label": "运行失败，未校验",
            "prompt_source": "",
            "prompt_hash_match": None,
            "missing_required": False,
            "invalid_refs": [],
            "invalid_ref_details": [],
            "raw_invalid_refs": [],
            "raw_invalid_ref_details": [],
            "valid_refs": [],
            "total_refs": 0,
        }
    prompt_source = ""
    prompt_hash_match: bool | None = None

    if extraction_prompt_text is None:
        loaded = _load_prompt_by_result(result)
        extraction_prompt_text = loaded["text"]
        prompt_source = loaded["source"]
        prompt_hash_match = loaded["hash_match"]
    else:
        prompt_source = "provided_prompt_text"
        prompt_hash_match = (
            prompt_text_hash(extraction_prompt_text) == result.extraction_prompt_hash
            if result.extraction_prompt_hash
            else None
        )

    has_prompt_reference = bool(result.extraction_prompt_hash or result.extraction_prompt_version)
    if not extraction_prompt_text:
        status = "no_prompt_found" if has_prompt_reference else "not_applicable"
        return {
            "checked": False,
            "status": status,
            "status_label": "未找到提取提示词" if has_prompt_reference else "未使用提取规则",
            "prompt_source": prompt_source,
            "prompt_hash_match": prompt_hash_match,
            "missing_required": False,
            "invalid_refs": [],
            "invalid_ref_details": [],
            "raw_invalid_refs": [],
            "raw_invalid_ref_details": [],
            "valid_refs": [],
            "total_refs": 0,
        }

    parsed_details = _parsed_rule_ref_details(result)
    raw_details = _extract_rule_refs_from_raw_response(result.raw_response)

    invalid_details = [
        detail for detail in parsed_details
        if not _rule_ref_exists(detail["ref"], extraction_prompt_text)
    ]
    raw_invalid_details = [
        detail for detail in raw_details
        if not _rule_ref_exists(detail["ref"], extraction_prompt_text)
    ]
    valid_refs = [
        detail["ref"] for detail in parsed_details
        if _rule_ref_exists(detail["ref"], extraction_prompt_text)
    ]

    missing_required = len(parsed_details) == 0
    invalid_refs = _dedupe([detail["ref"] for detail in invalid_details])
    raw_invalid_refs = _dedupe([detail["ref"] for detail in raw_invalid_details])

    comment_rule_ids = RULE_ID_RE.findall(result.comment or "")
    prompt_rule_ids = set(RULE_ID_RE.findall(extraction_prompt_text))
    comment_invalid_rule_ids = sorted({rid for rid in comment_rule_ids if rid not in prompt_rule_ids})

    if invalid_refs or raw_invalid_refs or comment_invalid_rule_ids:
        status = "invalid"
        status_label = "疑似幻觉引用"
    elif missing_required:
        status = "missing"
        status_label = "缺少规则引用"
    elif prompt_hash_match is False:
        status = "hash_mismatch"
        status_label = "Prompt hash 不一致"
    else:
        status = "ok"
        status_label = "通过"

    return {
        "checked": True,
        "status": status,
        "status_label": status_label,
        "prompt_source": prompt_source,
        "prompt_hash_match": prompt_hash_match,
        "missing_required": missing_required,
        "invalid_refs": invalid_refs,
        "invalid_ref_details": invalid_details,
        "raw_invalid_refs": raw_invalid_refs,
        "raw_invalid_ref_details": raw_invalid_details,
        "comment_invalid_rule_ids": comment_invalid_rule_ids,
        "valid_refs": _dedupe(valid_refs),
        "total_refs": len(parsed_details),
    }


def summarize_rule_ref_validation(reports: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "total": len(reports),
        "ok": sum(1 for report in reports if report.get("status") == "ok"),
        "invalid": sum(1 for report in reports if report.get("status") == "invalid"),
        "missing": sum(1 for report in reports if report.get("status") == "missing"),
        "hash_mismatch": sum(1 for report in reports if report.get("status") == "hash_mismatch"),
        "not_checked": sum(1 for report in reports if not report.get("checked")),
    }


def rule_ref_validation_rows(report: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if report.get("missing_required"):
        rows.append({
            "位置": "rule_refs",
            "引用": "（空）",
            "状态": "缺少规则引用",
            "说明": "本结果使用了提取规则辅助评测，但解析后的 rule_refs 为空。",
        })
    for detail in report.get("invalid_ref_details") or []:
        rows.append({
            "位置": detail.get("source", ""),
            "引用": detail.get("ref", ""),
            "状态": "解析后字段疑似幻觉",
            "说明": "该引用未在当前提取 prompt 中找到逐字匹配。",
        })
    for detail in report.get("raw_invalid_ref_details") or []:
        rows.append({
            "位置": detail.get("source", ""),
            "引用": detail.get("ref", ""),
            "状态": "原始响应疑似幻觉",
            "说明": "原始 Judge 输出包含该引用；如果解析后字段没有它，说明已被过滤但仍值得关注。",
        })
    for rid in report.get("comment_invalid_rule_ids") or []:
        rows.append({
            "位置": "comment",
            "引用": rid,
            "状态": "评语疑似幻觉编号",
            "说明": "comment 中出现的规则编号不在当前提取 prompt 中。",
        })
    return rows
