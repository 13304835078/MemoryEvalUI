from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from typing import Any, Callable

import requests

from src.llm_api import ChatPayloadOptions, LLMChatClient, build_chat_payload, retry_wait_seconds
from src.schema import Case, EvalConfig, VALID_ERROR_TAGS


PAIRWISE_SYSTEM_PROMPT = """你是记忆提取结果的成对比较裁判。

你会收到同一段源对话在两个提取版本下产生的候选记忆，以及候选无关的评测协议。
请直接比较候选 1 和候选 2，
不要分别打绝对分，也不要根据模型名称、提示词版本名称或文本长短猜测优劣。

强制规则：
1. 两侧 old_memory 都是各自本轮更新前的合法历史基线。候选保留其中已有内容时，不要求该内容在本轮对话中再次出现；
   即使某条历史内容本身可能不正确，原样继承也不能算作当前 chunk 的捏造、事实错误或越界。
2. 只评价“各自 old_memory → 各自 output”的本轮状态变化。新增或改写事实需由本轮源对话支持；
   当前对话未否定的历史内容不能仅因本轮未提及而判错，并且必须检查候选是否无故丢失应继承的历史内容。
3. 如果 A/B 的差异在两侧 old_memory 中已经存在，且本轮只是各自原样继承，判 HISTORICAL_DIFFERENCE，
   不得在后续 chunk 重复计为任一侧错误。只有当前对话明确给出纠正信息，且双方共同提取规则明确要求纠错时，
   未纠错才算本轮问题；若只有一侧提示词支持纠错，则属于策略差异。若本轮主动做了新的错误增删改，仍按共同质量问题判断。
4. reasoning 只用于排查提取过程，不能单独证明候选正文正确，也不能弥补正文遗漏。
5. 只有通用质量规则或双方共同规则中的错误可以决定 A/B 胜负。若差异仅来自双方准入范围、
   数据源、长度或输出结构不同，必须判 POLICY_DIFFERENCE，不能选择更符合自身提示词的一侧。
6. 同类错误必须使用一致标准。内容等价时判 TIE；证据不足或无法可靠判断时判 INSUFFICIENT。
7. 规则引用必须能在评测协议的 universal_rules 或 common_rules 中找到，禁止编造 R1/R2 等编号。
8. policy_conflicts 和 format_differences 只用于识别不可直接定胜负的策略差异，不能作为偏向任一候选的依据。
9. 团队裁判提示词只作为候选无关的质量原则来源；若与本协议冲突，以本协议为准。
10. 只输出一个 JSON object，不要 Markdown 代码块或额外文字。

输出格式：
{
  "winner": "candidate_1|candidate_2|TIE|POLICY_DIFFERENCE|HISTORICAL_DIFFERENCE|INSUFFICIENT",
  "decision_basis": "common_quality|policy_difference|historical_baseline_difference|equivalent|insufficient",
  "confidence": "low|medium|high",
  "reason": "简明说明直接比较依据",
  "rule_refs": ["通用质量规则或共同规则中的原文短句"],
  "policy_differences": ["导致本条不可直接定胜负的策略差异"],
  "evidence_refs": ["源对话或旧记忆中的短引用"],
  "issues_candidate_1": ["候选1的问题"],
  "issues_candidate_2": ["候选2的问题"],
  "error_tags_candidate_1": ["允许的错误标签"],
  "error_tags_candidate_2": ["允许的错误标签"],
  "strengths_candidate_1": ["候选1相对优点"],
  "strengths_candidate_2": ["候选2相对优点"]
}
"""


def _truncate(value: Any, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 12)] + "\n[已截断]"


def _truncate_document(value: Any, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    marker = "\n[中间内容因上下文预算已截断]\n"
    available = max(0, limit - len(marker))
    head = int(available * 0.55)
    return text[:head] + marker + text[-(available - head):]


def _dialogue(case: Case) -> list[dict[str, str]]:
    return [
        {"role": str(turn.role or ""), "content": _truncate(turn.content, 4_000)}
        for turn in (case.dialogue or [])
        if str(turn.content or "").strip()
    ]


def _reasoning(case: Case) -> str:
    metadata = case.metadata if isinstance(case.metadata, dict) else {}
    return _truncate(metadata.get("reasoning"), 6_000)


def _transition_view(old_memory: Any, candidate_output: Any) -> dict[str, Any]:
    old_lines = [line.strip() for line in str(old_memory or "").splitlines() if line.strip()]
    new_lines = [line.strip() for line in str(candidate_output or "").splitlines() if line.strip()]
    old_counts = Counter(old_lines)
    new_counts = Counter(new_lines)
    retained: list[str] = []
    added: list[str] = []
    removed: list[str] = []
    for line, count in new_counts.items():
        retained.extend([line] * min(count, old_counts.get(line, 0)))
        added.extend([line] * max(0, count - old_counts.get(line, 0)))
    for line, count in old_counts.items():
        removed.extend([line] * max(0, count - new_counts.get(line, 0)))
    return {
        "historical_line_count": len(old_lines),
        "output_line_count": len(new_lines),
        "exactly_retained_line_count": len(retained),
        "exactly_retained_lines": _truncate("\n".join(retained), 6_000),
        "retained_lines_evaluation_rule": "这些行来自历史基线，禁止因本轮对话未提及或历史本身可能有误而判为本轮捏造。",
        "added_or_rewritten_lines": _truncate("\n".join(added), 6_000),
        "removed_or_rewritten_historical_lines": _truncate("\n".join(removed), 6_000),
        "note": "逐行差异仅辅助定位；语义等价改写仍需结合完整上下文判断。",
    }


def stable_swap_for_source(source_key: str) -> bool:
    """Deterministically balance which real version appears as candidate_1."""
    digest = hashlib.sha256(str(source_key or "").encode("utf-8")).digest()
    return bool(digest[0] & 1)


def build_pairwise_user_message(
    case_a: Case,
    case_b: Case,
    *,
    evaluation_rule_prompt: str = "",
    evaluation_protocol: dict[str, Any] | None = None,
    task_type: str,
    swap_candidates: bool,
) -> str:
    first, second = (case_b, case_a) if swap_candidates else (case_a, case_b)
    source = case_a if case_a.dialogue else case_b
    payload = {
        "task_type": task_type,
        "evaluation_scope": "只评价本轮状态变化；old_memory 是合法历史基线，历史基线已有差异不重复计错。",
        "candidate_neutral_evaluation_protocol": evaluation_protocol or {
            "universal_rules": [_truncate(evaluation_rule_prompt, 28_000)] if evaluation_rule_prompt else [],
            "common_rules": [],
            "policy_conflicts": [],
            "format_differences": [],
        },
        "source_dialogue": _dialogue(source),
        "candidate_1": {
            "old_memory_historical_baseline": _truncate_document(first.old_memory, 20_000),
            "output_after_current_dialogue": _truncate_document(first.candidate_output, 24_000),
            "transition_view": _transition_view(first.old_memory, first.candidate_output),
            "reasoning_auxiliary_only": _reasoning(first),
        },
        "candidate_2": {
            "old_memory_historical_baseline": _truncate_document(second.old_memory, 20_000),
            "output_after_current_dialogue": _truncate_document(second.candidate_output, 24_000),
            "transition_view": _transition_view(second.old_memory, second.candidate_output),
            "reasoning_auxiliary_only": _reasoning(second),
        },
    }
    return (
        "请按候选无关评测协议直接比较两个候选。候选编号已做稳定化换位，不代表 A/B 或新旧关系。\n\n"
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
            raise ValueError("成对比较输出中没有 JSON object")
        parsed = json.loads(value[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("成对比较输出必须是 JSON object")
    return parsed


def _string_list(value: Any, limit: int = 8) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_truncate(item, 500).strip() for item in value if str(item or "").strip()][:limit]


def _error_tags(value: Any) -> list[str]:
    return [item for item in _string_list(value) if item in VALID_ERROR_TAGS]


def normalize_pairwise_result(parsed: dict[str, Any], *, swap_candidates: bool) -> dict[str, Any]:
    raw_winner = str(parsed.get("winner") or "INSUFFICIENT").strip().upper().replace("-", "_")
    aliases = {
        "1": "CANDIDATE_1",
        "2": "CANDIDATE_2",
        "候选1": "CANDIDATE_1",
        "候选2": "CANDIDATE_2",
        "CANDIDATE1": "CANDIDATE_1",
        "CANDIDATE2": "CANDIDATE_2",
        "持平": "TIE",
        "策略差异": "POLICY_DIFFERENCE",
        "POLICYDIFFERENCE": "POLICY_DIFFERENCE",
        "历史基线差异": "HISTORICAL_DIFFERENCE",
        "HISTORICALDIFFERENCE": "HISTORICAL_DIFFERENCE",
        "证据不足": "INSUFFICIENT",
    }
    winner = aliases.get(raw_winner, raw_winner)
    if winner == "CANDIDATE_1":
        mapped_winner = "B" if swap_candidates else "A"
    elif winner == "CANDIDATE_2":
        mapped_winner = "A" if swap_candidates else "B"
    elif winner in {"TIE", "POLICY_DIFFERENCE", "HISTORICAL_DIFFERENCE", "INSUFFICIENT"}:
        mapped_winner = winner
    else:
        mapped_winner = "INSUFFICIENT"

    confidence = str(parsed.get("confidence") or "low").strip().lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "low"

    first_suffix, second_suffix = ("b", "a") if swap_candidates else ("a", "b")
    by_side: dict[str, dict[str, list[str]]] = {
        "a": {"issues": [], "error_tags": [], "strengths": []},
        "b": {"issues": [], "error_tags": [], "strengths": []},
    }
    for candidate_number, side in ((1, first_suffix), (2, second_suffix)):
        by_side[side] = {
            "issues": _string_list(parsed.get(f"issues_candidate_{candidate_number}")),
            "error_tags": _error_tags(parsed.get(f"error_tags_candidate_{candidate_number}")),
            "strengths": _string_list(parsed.get(f"strengths_candidate_{candidate_number}")),
        }

    policy_differences = _string_list(parsed.get("policy_differences"))
    decision_basis = str(parsed.get("decision_basis") or "").strip().lower()
    if decision_basis not in {"common_quality", "policy_difference", "historical_baseline_difference", "equivalent", "insufficient"}:
        decision_basis = (
            "policy_difference" if mapped_winner == "POLICY_DIFFERENCE"
            else "historical_baseline_difference" if mapped_winner == "HISTORICAL_DIFFERENCE"
            else "equivalent" if mapped_winner == "TIE"
            else "insufficient" if mapped_winner == "INSUFFICIENT"
            else "common_quality"
        )
    if policy_differences and not str(parsed.get("decision_basis") or "").strip():
        decision_basis = "policy_difference"
    if decision_basis == "policy_difference":
        mapped_winner = "POLICY_DIFFERENCE"
    elif decision_basis == "historical_baseline_difference":
        mapped_winner = "HISTORICAL_DIFFERENCE"

    return {
        "winner": mapped_winner,
        "decision_basis": decision_basis,
        "confidence": confidence,
        "reason": _truncate(parsed.get("reason"), 1_500),
        "rule_refs": _string_list(parsed.get("rule_refs")),
        "policy_differences": policy_differences,
        "evidence_refs": _string_list(parsed.get("evidence_refs")),
        "issues_a": by_side["a"]["issues"],
        "issues_b": by_side["b"]["issues"],
        "error_tags_a": by_side["a"]["error_tags"],
        "error_tags_b": by_side["b"]["error_tags"],
        "strengths_a": by_side["a"]["strengths"],
        "strengths_b": by_side["b"]["strengths"],
    }


def call_pairwise_judge(
    config: EvalConfig,
    case_a: Case,
    case_b: Case,
    *,
    source_key: str,
    judge_prompt_text: str,
    evaluation_rule_prompt: str,
    evaluation_protocol: dict[str, Any] | None = None,
    task_type: str,
    rate_limit_wait_callback: Callable[[], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    swap_candidates = stable_swap_for_source(source_key)
    if config.mock:
        return {
            "source_key": source_key,
            "status": "mock",
            "model": config.judge_model or "mock-pairwise-model",
            "winner": "TIE",
            "decision_basis": "equivalent",
            "confidence": "low",
            "reason": "[MOCK] 两侧正文不同，本次未调用真实成对比较模型。",
            "rule_refs": [],
            "policy_differences": [],
            "evidence_refs": [],
            "issues_a": [],
            "issues_b": [],
            "error_tags_a": [],
            "error_tags_b": [],
            "strengths_a": [],
            "strengths_b": [],
            "reasoning": "",
            "raw_response": "",
            "error": "",
            "swapped": swap_candidates,
        }

    user_message = build_pairwise_user_message(
        case_a,
        case_b,
        evaluation_rule_prompt=evaluation_rule_prompt,
        evaluation_protocol=evaluation_protocol,
        task_type=task_type,
        swap_candidates=swap_candidates,
    )
    system_prompt = (
        PAIRWISE_SYSTEM_PROMPT
        + "\n\n【团队裁判提示词，仅采纳其中与质量边界有关的规则】\n"
        + _truncate(judge_prompt_text, 24_000)
    )
    payload = build_chat_payload(
        [
            {"role": "system", "content": system_prompt},
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
            return {"source_key": source_key, "status": "stopped", "winner": "INSUFFICIENT"}
        if rate_limit_wait_callback is not None:
            rate_limit_wait_callback()
        try:
            completion = client.post_json(payload, stream=False)
            content, reasoning = _extract_content(completion.data)
            last_raw = content
            normalized = normalize_pairwise_result(
                _parse_json_object(content),
                swap_candidates=swap_candidates,
            )
            return {
                "source_key": source_key,
                "status": "success",
                "model": config.judge_model,
                **normalized,
                "reasoning": _truncate(reasoning, 8_000),
                "raw_response": _truncate(content, 20_000),
                "error": "",
                "swapped": swap_candidates,
            }
        except (RuntimeError, ValueError, requests.exceptions.RequestException) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < attempts:
            time.sleep(retry_wait_seconds(last_error, attempt, float(config.judge_qps_backoff or 12.0)))

    return {
        "source_key": source_key,
        "status": "failed",
        "model": config.judge_model,
        "winner": "INSUFFICIENT",
        "decision_basis": "insufficient",
        "confidence": "low",
        "reason": "对比模型调用或 JSON 解析失败，本条不进入胜负统计。",
        "rule_refs": [],
        "policy_differences": [],
        "evidence_refs": [],
        "issues_a": [],
        "issues_b": [],
        "error_tags_a": [],
        "error_tags_b": [],
        "strengths_a": [],
        "strengths_b": [],
        "reasoning": "",
        "raw_response": _truncate(last_raw, 20_000),
        "error": last_error or "成对比较模型未返回可用结果",
        "swapped": swap_candidates,
    }
