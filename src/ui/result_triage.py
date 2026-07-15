from __future__ import annotations

from typing import Any

from src.ui.error_diagnostics import classify_eval_result
from src.eval.result_status import result_is_score_eligible


def result_navigation_key(result: Any) -> tuple[str, str, str, str, str, str, str]:
    return (
        str(getattr(result, "case_id", "") or ""),
        str(getattr(result, "model_name", "") or "unknown"),
        str(getattr(result, "prompt_version", "") or "unknown"),
        str(getattr(result, "judge_model", "") or ""),
        str(getattr(result, "judge_prompt_version", "") or ""),
        str(getattr(result, "extraction_prompt_hash", "") or ""),
        str(getattr(result, "evaluation_fingerprint", "") or ""),
    )


def priority_reasons(result: Any, *, rule_status: str = "", low_score_threshold: float = 4.0) -> list[str]:
    reasons: list[str] = []
    eligible = result_is_score_eligible(result)
    if not eligible:
        failure = classify_eval_result(result)
        reasons.append(f"P0 {failure.label if failure else '运行失败'}")
    elif bool(getattr(result, "fatal_error", False)):
        reasons.append("P1 严重质量错误")
    if rule_status in {"invalid", "missing", "hash_mismatch", "no_prompt_found"}:
        reasons.append("P1 规则引用异常")
    if eligible and float(getattr(result, "score_total", 0.0) or 0.0) < float(low_score_threshold):
        reasons.append("P1 低分")
    if getattr(result, "error_tags", None):
        reasons.append("P2 有错误标签")
    if getattr(result, "diagnostics", None):
        reasons.append("P2 有结构化诊断")
    return reasons


def triage_result_rows(
    results: list[Any],
    *,
    rule_status_by_key: dict[tuple[str, str, str, str, str, str, str], str] | None = None,
    low_score_threshold: float = 4.0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    statuses = rule_status_by_key or {}
    for index, result in enumerate(results):
        key = result_navigation_key(result)
        rule_status = statuses.get(key, "")
        reasons = priority_reasons(result, rule_status=rule_status, low_score_threshold=low_score_threshold)
        if not reasons:
            continue
        failure = classify_eval_result(result)
        rows.append({
            "_result_index": index,
            "优先级/原因": "；".join(reasons),
            "样本编号": getattr(result, "case_id", ""),
            "总分": float(getattr(result, "score_total", 0.0) or 0.0) if result_is_score_eligible(result) else None,
            "失败类型": failure.label if failure else "",
            "错误标签": "; ".join(getattr(result, "error_tags", []) or []),
            "规则引用状态": rule_status,
            "评语": getattr(result, "comment", ""),
        })
    rows.sort(key=lambda item: (0 if str(item["优先级/原因"]).startswith("P0") else 1, item["总分"]))
    return rows


def result_matches_filter(result: Any, mode: str, *, low_score_threshold: float = 4.0) -> bool:
    if mode == "严重失败":
        return result_is_score_eligible(result) and bool(getattr(result, "fatal_error", False))
    if mode == "运行失败":
        return not result_is_score_eligible(result)
    if mode == "低分":
        return result_is_score_eligible(result) and float(getattr(result, "score_total", 0.0) or 0.0) < float(low_score_threshold)
    if mode == "有错误标签":
        return bool(getattr(result, "error_tags", None))
    if mode == "有结构化诊断":
        return bool(getattr(result, "diagnostics", None))
    return True
