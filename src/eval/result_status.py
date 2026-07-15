from __future__ import annotations

from typing import Any


EVAL_STATUS_SUCCESS = "success"
EVAL_STATUS_API_FAILED = "judge_api_failed"
EVAL_STATUS_PARSE_FAILED = "judge_parse_failed"
EVAL_STATUS_RUNTIME_FAILED = "judge_runtime_failed"
EVAL_STATUS_STOPPED = "stopped"

SCORE_INELIGIBLE_STATUSES = {
    EVAL_STATUS_API_FAILED,
    EVAL_STATUS_PARSE_FAILED,
    EVAL_STATUS_RUNTIME_FAILED,
    EVAL_STATUS_STOPPED,
}

STATUS_LABELS = {
    EVAL_STATUS_SUCCESS: "评分成功",
    EVAL_STATUS_API_FAILED: "Judge 接口失败",
    EVAL_STATUS_PARSE_FAILED: "Judge 输出解析失败",
    EVAL_STATUS_RUNTIME_FAILED: "评测运行失败",
    EVAL_STATUS_STOPPED: "任务已终止",
}


def classify_failure_status(raw: str) -> tuple[str, str]:
    """Classify a failed Judge attempt without treating it as a quality score."""
    text = str(raw or "")
    lower = text.lower()

    parse_markers = (
        "不是可解析 json",
        "无法解析",
        "json 解析",
        "json解析",
        "json decode",
        "jsondecode",
        "不符合评分格式",
        "missing required",
        "schema validation",
    )
    api_markers = (
        "api error",
        "qps limit",
        "rate limit",
        "too many requests",
        "connection idle timeout",
        "idle timeout",
        "websocket",
        "connection reset",
        "connection aborted",
        "going away",
        "timed out",
        "timeout",
        "request error",
        "requestexception",
        "http error",
        "status code",
        "429",
        "502",
        "503",
        "504",
    )
    if any(marker in lower or marker in text for marker in parse_markers):
        return EVAL_STATUS_PARSE_FAILED, "judge_output_invalid"
    if any(marker in lower or marker in text for marker in api_markers):
        return EVAL_STATUS_API_FAILED, "judge_api_error"
    return EVAL_STATUS_RUNTIME_FAILED, "judge_runtime_error"


def result_is_score_eligible(result: Any) -> bool:
    explicit = getattr(result, "score_eligible", None)
    if explicit is not None:
        return bool(explicit)
    status = str(getattr(result, "evaluation_status", "") or "")
    return not status or status not in SCORE_INELIGIBLE_STATUSES


def result_evaluation_status(result: Any) -> str:
    status = str(getattr(result, "evaluation_status", "") or "")
    if status:
        return status
    return EVAL_STATUS_SUCCESS if result_is_score_eligible(result) else EVAL_STATUS_RUNTIME_FAILED


def infer_legacy_result_status(data: dict[str, Any]) -> tuple[str, bool, str]:
    """Infer execution status for results written before the status fields existed."""
    raw = str(data.get("raw_response") or "")
    comment = str(data.get("comment") or "")
    scores = data.get("scores")
    fatal = bool(data.get("fatal_error", False))
    known_failure_comment = "judge 调用失败" in comment.lower() or "json 解析失败" in comment.lower()
    looks_unscored = fatal and not scores and float(data.get("score_total") or 0.0) == 0.0
    if known_failure_comment or looks_unscored:
        status, failure_type = classify_failure_status(raw or comment)
        return status, False, failure_type
    return EVAL_STATUS_SUCCESS, True, ""
