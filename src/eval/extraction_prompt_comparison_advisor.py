from __future__ import annotations

import difflib
import json
import re
import time
from typing import Any, Callable

import requests

from src.llm_api import (
    ChatPayloadOptions,
    LLMChatClient,
    build_chat_payload,
    retry_wait_seconds,
)
from src.schema import EvalConfig


COMPARISON_SYSTEM_PROMPT = """你是提取提示词 A/B 实验的结果分析员。

你的职责是总结已经完成的绝对评测与统计比较，不是重新发明评分标准。

约束：
1. 统计门槛给出的结论是正式版本选择依据，你不能修改分数、覆盖率、置信区间或正式结论。
2. 只能使用输入中的提示词差异、统计量和代表性样本，不得补充不存在的事实或规则。
3. API/Judge 失败表示证据缺失，不表示模型质量为 0。
4. A/B 输出相同时，不得把 Judge 波动归因于提取提示词。
5. 修改方向必须是通用规则，不得针对单个 case 打具体补丁，也不得建议无依据的大规模重写。
6. 输出严格 JSON object，不要 Markdown 代码块或额外文字。

输出格式：
{
  "preferred_version": "A|B|TIE|INSUFFICIENT",
  "confidence": "low|medium|high",
  "summary": "不超过300字的综合说明",
  "reasons": ["最多5条主要依据"],
  "risks": ["最多5条风险或证据不足"],
  "prompt_suggestions": ["最多5条通用、精简的提示词修改方向"]
}
"""


def _truncate(value: Any, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 12)] + "\n[已截断]"


def _prompt_diff(prompt_a: str, prompt_b: str, limit: int = 12_000) -> str:
    diff = "\n".join(
        difflib.unified_diff(
            str(prompt_a or "").splitlines(),
            str(prompt_b or "").splitlines(),
            fromfile="prompt_A",
            tofile="prompt_B",
            lineterm="",
        )
    )
    return _truncate(diff, limit)


def _evidence_priority(row: dict[str, Any]) -> tuple[int, float, str]:
    comparison = str(row.get("comparison") or "")
    priority = {
        "A独有": 0,
        "B独有": 0,
        "A较优": 1,
        "B较优": 1,
        "不可比较": 2,
        "基本持平": 3,
        "输出相同": 4,
    }.get(comparison, 3)
    try:
        delta = abs(float(row.get("score_delta_b_minus_a") or 0.0))
    except (TypeError, ValueError):
        delta = 0.0
    return priority, -delta, str(row.get("source_key") or "")


def _representative_evidence(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    selected = sorted(rows, key=_evidence_priority)[: max(1, int(limit))]
    return [
        {
            "source_key": row.get("source_key", ""),
            "reviewer": row.get("reviewer", ""),
            "session_id": row.get("session_id", ""),
            "chunk_id": row.get("chunk_id", ""),
            "comparison": row.get("comparison", ""),
            "score_a": row.get("score_a"),
            "score_b": row.get("score_b"),
            "score_delta_b_minus_a": row.get("score_delta_b_minus_a"),
            "extraction_a": row.get("extraction_a", ""),
            "extraction_b": row.get("extraction_b", ""),
            "candidate_output_a": _truncate(row.get("candidate_output_a"), 2_500),
            "candidate_output_b": _truncate(row.get("candidate_output_b"), 2_500),
            "error_tags_a": row.get("error_tags_a", ""),
            "error_tags_b": row.get("error_tags_b", ""),
            "comment_a": _truncate(row.get("comment_a"), 1_000),
            "comment_b": _truncate(row.get("comment_b"), 1_000),
            "rule_refs_a": _truncate(row.get("rule_refs_a"), 800),
            "rule_refs_b": _truncate(row.get("rule_refs_b"), 800),
            "deterministic_note": _truncate(row.get("comparison_note"), 800),
        }
        for row in selected
    ]


def build_comparison_user_message(
    report: dict[str, Any],
    *,
    prompt_a: str,
    prompt_b: str,
    max_evidence: int = 8,
) -> str:
    gate = report.get("validation_gate") if isinstance(report.get("validation_gate"), dict) else {}
    payload = {
        "formal_statistical_conclusion": {
            "recommendation": report.get("recommendation"),
            "reason": report.get("recommendation_reason"),
        },
        "quality_a": report.get("quality_a") or {},
        "quality_b": report.get("quality_b") or {},
        "validation_gate": {
            key: gate.get(key)
            for key in (
                "accepted",
                "paired_case_count",
                "paired_cluster_count",
                "paired_score_delta",
                "confidence_level",
                "confidence_interval",
                "extraction_coverage_drop",
                "case_regression_rate",
                "critical_error_delta",
                "reasons",
            )
        },
        "winner_counts": report.get("winner_counts") or {},
        "dimension_summary": report.get("dimension_summary") or [],
        "identical_output_count": report.get("identical_output_count", 0),
        "judge_disagreement_on_identical_output_count": report.get(
            "judge_disagreement_on_identical_output_count",
            0,
        ),
        "prompt_diff": _prompt_diff(prompt_a, prompt_b),
        "representative_evidence": _representative_evidence(
            list(report.get("rows") or []),
            max_evidence,
        ),
    }
    return (
        "请基于以下 A/B 实验结果生成补充分析。正式版本结论必须服从输入中的统计结论。\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def _extract_content(data: dict[str, Any]) -> tuple[str, str]:
    choices = data.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        choice = choices[0]
        message = choice.get("message")
        if isinstance(message, dict):
            return (
                str(message.get("content") or message.get("reasoning_content") or message.get("reasoning") or ""),
                str(message.get("reasoning") or message.get("reasoning_content") or ""),
            )
        if "text" in choice:
            return str(choice.get("text") or ""), ""
    for key in ("content", "result", "answer"):
        if key in data:
            return str(data.get(key) or ""), ""
    return json.dumps(data, ensure_ascii=False), ""


def _parse_json_object(text: str) -> dict[str, Any]:
    value = str(text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?", "", value, flags=re.IGNORECASE).strip()
        value = re.sub(r"```$", "", value).strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        start, end = value.find("{"), value.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("对比模型输出中没有 JSON object")
        parsed = json.loads(value[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("对比模型输出必须是 JSON object")
    return parsed


def _string_list(value: Any, limit: int = 5) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()][:limit]


def _normalize_result(parsed: dict[str, Any]) -> dict[str, Any]:
    raw_preference = str(parsed.get("preferred_version") or "INSUFFICIENT").strip().upper()
    preference_aliases = {
        "建议选择A": "A",
        "建议保留A": "A",
        "建议选择B": "B",
        "持平": "TIE",
        "证据不足": "INSUFFICIENT",
    }
    preferred = preference_aliases.get(raw_preference.replace(" ", ""), raw_preference)
    if preferred not in {"A", "B", "TIE", "INSUFFICIENT"}:
        preferred = "INSUFFICIENT"
    confidence = str(parsed.get("confidence") or "low").strip().lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "low"
    return {
        "preferred_version": preferred,
        "confidence": confidence,
        "summary": _truncate(parsed.get("summary"), 1_200),
        "reasons": _string_list(parsed.get("reasons")),
        "risks": _string_list(parsed.get("risks")),
        "prompt_suggestions": _string_list(parsed.get("prompt_suggestions")),
    }


def _mock_result(config: EvalConfig, report: dict[str, Any]) -> dict[str, Any]:
    recommendation = str(report.get("recommendation") or "")
    preferred = "B" if recommendation == "建议选择 B" else (
        "A" if recommendation in {"建议保留 A", "建议选择 A"} else "INSUFFICIENT"
    )
    return {
        "status": "mock",
        "model": config.judge_model or "mock-comparison-model",
        "preferred_version": preferred,
        "confidence": "low",
        "summary": f"[MOCK] 统计结论为“{recommendation or '暂不定版'}”，未调用真实对比模型。",
        "reasons": [str(report.get("recommendation_reason") or "模拟结果")],
        "risks": [],
        "prompt_suggestions": [],
        "reasoning": "",
        "raw_response": "",
        "error": "",
    }


def call_comparison_model(
    config: EvalConfig,
    report: dict[str, Any],
    *,
    prompt_a: str,
    prompt_b: str,
    max_evidence: int = 8,
    rate_limit_wait_callback: Callable[[], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    if config.mock:
        return _mock_result(config, report)

    user_message = build_comparison_user_message(
        report,
        prompt_a=prompt_a,
        prompt_b=prompt_b,
        max_evidence=max_evidence,
    )
    payload = build_chat_payload(
        [
            {"role": "system", "content": COMPARISON_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        ChatPayloadOptions(
            model=config.judge_model,
            max_tokens=config.judge_max_tokens,
            temperature=float(config.judge_temperature),
            top_p=float(config.judge_top_p),
            top_k=config.judge_top_k,
            stream=False,
            enable_thinking=bool(config.judge_enable_thinking),
            send_enable_thinking=bool(config.judge_send_enable_thinking),
            skip_special_tokens=bool(config.judge_skip_special_tokens),
            prompt_cache_id=str(config.judge_prompt_cache_id or ""),
            prompt_cache_location=str(config.judge_prompt_cache_location or "none"),
        ),
    )
    client = LLMChatClient(
        config.judge_api_base_url,
        config.judge_api_bearer_token,
        timeout=config.judge_timeout,
    )
    last_error = ""
    last_raw = ""
    attempts = max(1, int(config.judge_max_retries or 1))
    for attempt in range(1, attempts + 1):
        if should_stop is not None and should_stop():
            return {
                "status": "stopped",
                "model": config.judge_model,
                "error": "收到终止请求",
            }
        if rate_limit_wait_callback is not None:
            rate_limit_wait_callback()
        try:
            completion = client.post_json(payload, stream=False)
            content, reasoning = _extract_content(completion.data)
            last_raw = content
            normalized = _normalize_result(_parse_json_object(content))
            return {
                "status": "success",
                "model": config.judge_model,
                **normalized,
                "reasoning": _truncate(reasoning, 8_000),
                "raw_response": _truncate(content, 20_000),
                "error": "",
            }
        except (RuntimeError, ValueError, requests.exceptions.RequestException) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < attempts:
            time.sleep(
                retry_wait_seconds(
                    last_error,
                    attempt,
                    float(config.judge_qps_backoff or 12.0),
                )
            )

    return {
        "status": "failed",
        "model": config.judge_model,
        "preferred_version": "INSUFFICIENT",
        "confidence": "low",
        "summary": "对比模型调用或 JSON 解析失败，统计结论不受影响。",
        "reasons": [],
        "risks": [last_error] if last_error else [],
        "prompt_suggestions": [],
        "reasoning": "",
        "raw_response": _truncate(last_raw, 20_000),
        "error": last_error or "对比模型未返回可用结果",
    }
