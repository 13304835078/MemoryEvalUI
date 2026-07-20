from __future__ import annotations

import json
import re
import time
from typing import Any, Callable

import requests

from src.llm_api import ChatPayloadOptions, LLMChatClient, build_chat_payload, retry_wait_seconds
from src.schema import EvalConfig


PROTOCOL_VERSION = "candidate_neutral_common_core_v2"

UNIVERSAL_QUALITY_RULES = [
    "事实必须能由本次源对话或候选对应的历史记忆支持；历史记忆是本轮更新的合法基线，不要求在本轮对话中重新举证。",
    "只评价本轮从历史记忆到候选记忆的状态变化；历史基线中的内容即使本身有误，原样继承也不得算作本轮错误。",
    "双方历史基线中已经存在的差异不得在后续 chunk 重复计错；只有当前对话明确触发且双方共同规则要求纠错时，未纠错才属于本轮问题。",
    "新增或改写内容不能把提示词、示例、assistant、tool 或 reasoning 当作事实来源。",
    "候选不得捏造、错误归因、错误改写或自相矛盾；reasoning 只能辅助排查，不能弥补正文。",
    "对于双方规则都要求保留、更新或删除的信息，应检查遗漏、错误删除、冲突覆盖、重复和过度泛化。",
    "只有共同质量要求造成的差异才能决定胜负；双方准入范围、数据源或输出结构不同造成的差异属于策略差异。",
    "文本更长、字段更多、结构更复杂或更接近任一候选提示词，均不能单独证明质量更高。",
    "API、网络、解析和源数据对齐失败不计为任一候选的质量损失。",
]


PROTOCOL_SYSTEM_PROMPT = """你是候选无关的提取规则分析器。你的任务不是选择 A 或 B，
而是把两份可能差异很大的提取提示词整理成可用于公平 A/B 的评测协议。

必须遵守：
1. 只提取双方明确共享的语义要求作为 common_rules，不得把 A 独有或 B 独有的规则放入共同规则。
2. 对准入范围、禁止范围、信息来源、更新策略、隐私边界、长度和输出结构的冲突，放入 policy_conflicts。
3. 纯标题、字段名、Markdown/JSON 结构、变更日志和元数据要求差异，放入 format_differences。
4. 不评价哪份提示词更合理，不根据详尽程度、篇幅或示例数量选择一侧。
5. 如果两份提示词任务目标差异很大，compatibility 返回 low；仍尽量给出事实可靠性等真正共同规则。
6. 分别评价两份提示词本身的设计质量。只评价清晰度、内部一致性、规则可执行性、边界定义、
   输出契约可解析性、对模型的指导作用和上下文效率；不得因为某一版更长、字段更多或策略更保守就给高分。
7. 策略取舍本身不分优劣，但自相矛盾、定义含混、无法执行、格式要求冲突和明显重复属于提示词缺陷。
8. 只输出一个 JSON object，不要 Markdown 或额外文字。

输出格式：
{
  "compatibility": "high|medium|low",
  "common_rules": ["双方共同规则，使用中性表述"],
  "policy_conflicts": [
    {"topic": "差异主题", "policy_a": "A 的规则", "policy_b": "B 的规则"}
  ],
  "format_differences": ["格式或审计结构差异"],
  "prompt_quality_a": {
    "clarity": 1,
    "consistency": 1,
    "executability": 1,
    "boundary_definition": 1,
    "output_contract": 1,
    "model_guidance": 1,
    "context_efficiency": 1,
    "issues": ["提示词本身的问题"],
    "strengths": ["提示词本身的优点"]
  },
  "prompt_quality_b": {
    "clarity": 1,
    "consistency": 1,
    "executability": 1,
    "boundary_definition": 1,
    "output_contract": 1,
    "model_guidance": 1,
    "context_efficiency": 1,
    "issues": [],
    "strengths": []
  },
  "notes": ["影响可比性的简短说明"]
}
"""


def _truncate(value: Any, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 12)] + "\n[已截断]"


def _contract_view(prompt: str, limit: int = 18_000) -> str:
    """Keep rule-bearing lines first so long prompts fit a bounded one-time request."""
    text = str(prompt or "").replace("\r\n", "\n")
    if len(text) <= limit:
        return text
    lines = text.splitlines()
    selected: list[str] = []
    seen: set[int] = set()
    rule_markers = (
        "允许", "禁止", "必须", "不得", "只", "仅", "来源", "输入", "输出",
        "更新", "冲突", "隐私", "长度", "格式", "记录", "保留", "删除", "合并",
    )
    for index, line in enumerate(lines):
        stripped = line.strip()
        if (
            stripped.startswith(("#", "-", "*"))
            or re.match(r"^\d+[.、)]", stripped)
            or any(marker in stripped for marker in rule_markers)
        ):
            seen.add(index)
            selected.append(line)
    compact = "\n".join(selected)
    if len(compact) < limit // 2:
        remaining = [line for index, line in enumerate(lines) if index not in seen]
        compact += "\n" + "\n".join(remaining)
    return _truncate(compact, limit)


def default_evaluation_protocol(*, status: str = "fallback", error: str = "") -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": status,
        "compatibility": "unknown",
        "universal_rules": list(UNIVERSAL_QUALITY_RULES),
        "common_rules": [],
        "policy_conflicts": [],
        "format_differences": [],
        "prompt_quality_a": {},
        "prompt_quality_b": {},
        "notes": ["未能自动整理双方规则，本次仅使用候选无关的通用质量规则。"] if error else [],
        "error": error,
        "raw_response": "",
    }


def _extract_content(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict):
            return str(message.get("content") or message.get("reasoning_content") or "")
        return str(choices[0].get("text") or "")
    for key in ("content", "result", "answer"):
        if key in data:
            return str(data.get(key) or "")
    return json.dumps(data, ensure_ascii=False)


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
            raise ValueError("规则分析结果中没有 JSON object")
        parsed = json.loads(value[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("规则分析结果必须是 JSON object")
    return parsed


def _string_list(value: Any, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_truncate(item, 600).strip() for item in value if str(item or "").strip()][:limit]


def _normalize_protocol(parsed: dict[str, Any]) -> dict[str, Any]:
    compatibility = str(parsed.get("compatibility") or "unknown").strip().lower()
    if compatibility not in {"high", "medium", "low"}:
        compatibility = "unknown"
    conflicts: list[dict[str, str]] = []
    for item in parsed.get("policy_conflicts") or []:
        if not isinstance(item, dict):
            continue
        conflicts.append(
            {
                "topic": _truncate(item.get("topic"), 300).strip(),
                "policy_a": _truncate(item.get("policy_a"), 600).strip(),
                "policy_b": _truncate(item.get("policy_b"), 600).strip(),
            }
        )
        if len(conflicts) >= 20:
            break
    def prompt_quality(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        dimensions: dict[str, float] = {}
        for key in (
            "clarity",
            "consistency",
            "executability",
            "boundary_definition",
            "output_contract",
            "model_guidance",
            "context_efficiency",
        ):
            try:
                dimensions[key] = min(5.0, max(1.0, float(value.get(key))))
            except (TypeError, ValueError):
                continue
        overall = sum(dimensions.values()) / len(dimensions) if dimensions else None
        return {
            **dimensions,
            "overall": round(overall, 3) if overall is not None else None,
            "issues": _string_list(value.get("issues"), 10),
            "strengths": _string_list(value.get("strengths"), 10),
        }

    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "success",
        "compatibility": compatibility,
        "universal_rules": list(UNIVERSAL_QUALITY_RULES),
        "common_rules": _string_list(parsed.get("common_rules"), 24),
        "policy_conflicts": conflicts,
        "format_differences": _string_list(parsed.get("format_differences"), 16),
        "prompt_quality_a": prompt_quality(parsed.get("prompt_quality_a")),
        "prompt_quality_b": prompt_quality(parsed.get("prompt_quality_b")),
        "notes": _string_list(parsed.get("notes"), 10),
        "error": "",
    }


def compile_evaluation_protocol(
    config: EvalConfig,
    *,
    prompt_a: str,
    prompt_b: str,
    task_type: str,
    rate_limit_wait_callback: Callable[[], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    if config.mock:
        protocol = default_evaluation_protocol(status="mock")
        protocol["notes"] = ["模拟模式未调用规则分析模型，仅使用候选无关通用规则。"]
        return protocol

    payload = build_chat_payload(
        [
            {"role": "system", "content": PROTOCOL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task_type": task_type,
                        "prompt_A": _contract_view(prompt_a),
                        "prompt_B": _contract_view(prompt_b),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ],
        ChatPayloadOptions(
            model=config.judge_model,
            max_tokens=min(max(1_500, int(config.judge_max_tokens or 2_000)), 4_000),
            temperature=0.0,
            top_p=1.0,
            top_k=None,
            stream=False,
            enable_thinking=False,
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
    attempts = max(1, int(config.judge_max_retries or 1))
    last_error = ""
    last_raw = ""
    for attempt in range(1, attempts + 1):
        if should_stop is not None and should_stop():
            return default_evaluation_protocol(status="stopped", error="收到终止请求")
        if rate_limit_wait_callback is not None:
            rate_limit_wait_callback()
        try:
            completion = client.post_json(payload, stream=False)
            last_raw = _extract_content(completion.data)
            protocol = _normalize_protocol(_parse_json_object(last_raw))
            protocol["raw_response"] = _truncate(last_raw, 20_000)
            return protocol
        except (RuntimeError, ValueError, requests.exceptions.RequestException) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < attempts:
            time.sleep(retry_wait_seconds(last_error, attempt, float(config.judge_qps_backoff or 12.0)))
    protocol = default_evaluation_protocol(status="fallback", error=last_error or "规则分析失败")
    protocol["raw_response"] = _truncate(last_raw, 20_000)
    return protocol
