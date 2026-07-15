from __future__ import annotations

import json
import time
from typing import Any

import requests

from src.eval.judge_client import RealJudgeClient
from src.llm_api import apply_prompt_cache, make_prompt_cache_id, retry_wait_seconds
from src.schema import EvalConfig
from src.ui.global_rate_limiter import api_rate_scope, wait_for_global_rate_slot


def _compact_json_value(value: Any, text_limit: int = 500, list_limit: int = 5) -> Any:
    if isinstance(value, str):
        return _truncate(value, text_limit)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _compact_json_value(v, text_limit=text_limit, list_limit=list_limit) for k, v in value.items()}
    if isinstance(value, list):
        return [_compact_json_value(item, text_limit=text_limit, list_limit=list_limit) for item in value[:list_limit]]
    return _truncate(value, text_limit)


def _compact_advisor_evidence(
    evidence: list[dict[str, Any]],
    *,
    max_items: int,
    text_limit: int,
    diagnostics_limit: int = 2,
    refs_limit: int = 5,
) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for item in evidence[:max_items]:
        if not isinstance(item, dict):
            continue
        row: dict[str, Any] = {}
        for key, value in item.items():
            if key == "diagnostics" and isinstance(value, list):
                row[key] = [_compact_json_value(v, text_limit=text_limit, list_limit=refs_limit) for v in value[:diagnostics_limit]]
            elif key in {"rule_refs", "evidence_refs", "output_refs", "error_tags"} and isinstance(value, list):
                row[key] = [_truncate(v, min(300, text_limit)) for v in value[:refs_limit]]
            else:
                row[key] = _compact_json_value(value, text_limit=text_limit, list_limit=refs_limit)
        compacted.append(row)
    return compacted


def _compact_extraction_advisor_evidence(
    evidence: list[dict[str, Any]],
    *,
    max_items: int,
    text_limit: int = 260,
) -> list[dict[str, Any]]:
    scalar_keys = (
        "case_id", "row_id", "pair_id", "evidence_mode", "score_total",
        "fatal_error", "comment", "human_comment", "human_score",
        "llm_comment", "llm_error_tags", "human_error_tags",
    )
    list_keys = ("error_tags", "rule_refs", "evidence_refs", "output_refs")
    rows: list[dict[str, Any]] = []
    for item in evidence[:max_items]:
        if not isinstance(item, dict):
            continue
        row: dict[str, Any] = {}
        for key in scalar_keys:
            if key in item and item.get(key) not in (None, "", [], {}):
                row[key] = _compact_json_value(item.get(key), text_limit=text_limit, list_limit=3)
        for key in list_keys:
            values = item.get(key)
            if isinstance(values, list) and values:
                row[key] = [_truncate(value, min(text_limit, 180)) for value in values[:3]]
        diagnostics = []
        for diagnostic in (item.get("diagnostics") or [])[:1]:
            if not isinstance(diagnostic, dict):
                continue
            diagnostics.append({
                key: _compact_json_value(diagnostic.get(key), text_limit=text_limit, list_limit=3)
                for key in ("dimension", "severity", "reason", "rule_refs", "evidence_refs", "output_refs")
                if diagnostic.get(key) not in (None, "", [], {})
            })
        if diagnostics:
            row["diagnostics"] = diagnostics
        if _evidence_ref_id(row):
            rows.append(row)
    return rows


def _advisor_retry_wait_seconds(config: EvalConfig, last_error: str, attempt: int) -> float:
    return retry_wait_seconds(
        last_error,
        attempt,
        float(getattr(config, "judge_qps_backoff", 12.0) or 12.0),
    )


def _advisor_attempt_profile(
    *,
    attempt: int,
    target: str,
    evidence_count: int,
    min_evidence: int,
) -> dict[str, int]:
    is_extraction_target = target in {"extraction_prompt", "both"}
    floor = max(1, int(min_evidence or 0))

    if attempt <= 1:
        cap = max(floor, evidence_count)
        profile = {
            "max_items": cap,
            "text_limit": 420,
            "diagnostics_limit": 2,
            "refs_limit": 4,
            "judge_prompt_limit": 3500 if is_extraction_target else 6000,
            "extraction_prompt_limit": 0 if is_extraction_target else 2500,
            "extraction_section_limit": 70,
            "extraction_section_preview_chars": 280,
            "max_tokens_cap": 3200 if is_extraction_target else 5000,
        }
    elif attempt == 2:
        cap = 8 if is_extraction_target else 12
        profile = {
            "max_items": cap,
            "text_limit": 280,
            "diagnostics_limit": 2,
            "refs_limit": 3,
            "judge_prompt_limit": 2500 if is_extraction_target else 4500,
            "extraction_prompt_limit": 0 if is_extraction_target else 1500,
            "extraction_section_limit": 45,
            "extraction_section_preview_chars": 220,
            "max_tokens_cap": 2400 if is_extraction_target else 4000,
        }
    else:
        cap = 5 if is_extraction_target else 8
        profile = {
            "max_items": cap,
            "text_limit": 180,
            "diagnostics_limit": 1,
            "refs_limit": 2,
            "judge_prompt_limit": 1600 if is_extraction_target else 3000,
            "extraction_prompt_limit": 0 if is_extraction_target else 900,
            "extraction_section_limit": 28,
            "extraction_section_preview_chars": 160,
            "max_tokens_cap": 1600 if is_extraction_target else 3000,
        }

    profile["max_items"] = min(evidence_count, max(floor, profile["max_items"]))
    return profile


def _evidence_usage(
    selected_count: int,
    used_count: int,
    *,
    request_metrics: list[dict[str, Any]] | None = None,
    attempts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "selected_count": int(selected_count),
        "initial_used_count": int(used_count),
        "all_selected_used_initially": int(selected_count) == int(used_count),
        "request_count": len(request_metrics or attempts or []),
        "request_metrics": request_metrics or attempts or [],
    }


def _advisor_max_tokens(config: EvalConfig, profile: dict[str, int]) -> int:
    configured = int(getattr(config, "judge_max_tokens", 2000) or 2000)
    cap = int(profile.get("max_tokens_cap", 3000) or 3000)
    if configured <= 0:
        configured = 2000
    return max(800, min(configured, cap))


def _shrink_advisor_message_for_retry(user_message: str, attempt: int) -> str:
    if attempt <= 1:
        return user_message
    try:
        payload = json.loads(user_message)
    except Exception:
        return user_message
    if not isinstance(payload, dict):
        return user_message

    item_limit = 2 if attempt == 2 else 1
    evidence = payload.get("evidence")
    if isinstance(evidence, list):
        payload["evidence"] = evidence[:item_limit]

    outline = payload.get("prompt_global_outline")
    if isinstance(outline, list):
        outline_limit = 80 if attempt == 2 else 40
        payload["prompt_global_outline"] = [
            {
                key: value
                for key, value in item.items()
                if key != "preview" or attempt == 2
            }
            for item in outline[:outline_limit]
            if isinstance(item, dict)
        ]
        if attempt == 2:
            for item in payload["prompt_global_outline"]:
                if "preview" in item:
                    item["preview"] = _truncate(item["preview"], 60)
    if "current_judge_prompt_excerpt" in payload:
        payload["current_judge_prompt_excerpt"] = ""

    section_group = payload.get("section_group")
    if isinstance(section_group, dict):
        section_group["patch_intents"] = list(section_group.get("patch_intents") or [])[:item_limit]
        section_group["evidence_refs"] = list(section_group.get("evidence_refs") or [])[:item_limit]

    block_context = payload.get("target_section_blocks")
    if isinstance(block_context, dict):
        block_context["editable_blocks"] = list(block_context.get("editable_blocks") or [])[:1]
        outlines = list(block_context.get("block_outline") or [])
        block_context["block_outline"] = [
            {
                **item,
                "preview": _truncate(item.get("preview"), 80 if attempt == 2 else 40),
            }
            for item in outlines[:(20 if attempt == 2 else 10)]
            if isinstance(item, dict)
        ]
    if attempt >= 2 and "neighbor_section_outline" in payload:
        payload["neighbor_section_outline"] = []

    payload["retry_compaction_level"] = attempt - 1
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _call_advisor_json(
    *,
    config: EvalConfig,
    url: str,
    headers: dict[str, str],
    client: RealJudgeClient,
    system_prompt: str,
    user_message: str,
    max_tokens: int,
) -> tuple[dict[str, Any] | None, str, str]:
    max_attempts = max(1, int(config.judge_max_retries or 1))
    last_error = ""
    raw_text = ""
    rate_scope = api_rate_scope(config.judge_api_base_url, config.judge_api_bearer_token)
    request_interval = float(getattr(config, "judge_request_interval", 0.0) or 0.0)
    for attempt in range(1, max_attempts + 1):
        raw_text = ""
        attempt_message = _shrink_advisor_message_for_retry(user_message, attempt)
        attempt_max_tokens = max(800, int(max_tokens or 2000) - ((attempt - 1) * 400))
        payload = {
            "model": config.judge_model,
            "max_tokens": attempt_max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": True,
            "stream_options": {"include_usage": False},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": attempt_message},
            ],
            "extra_body": {
                "enable_thinking": False,
                "skip_special_tokens": False,
            },
        }
        cache_location = str(getattr(config, "judge_prompt_cache_location", "none") or "none")
        cache_id = str(getattr(config, "judge_prompt_cache_id", "") or "")
        if cache_location != "none" and not cache_id:
            cache_id = make_prompt_cache_id("memory_eval_advisor", config.judge_model, system_prompt)
        apply_prompt_cache(payload, cache_id, cache_location)
        try:
            wait_for_global_rate_slot(rate_scope, request_interval, disabled=bool(config.mock))
            response = requests.post(
                url,
                headers=headers,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                timeout=config.judge_timeout,
                stream=True,
            )
            if hasattr(response, "iter_lines"):
                response.raise_for_status()
                content, stream_error = client._extract_stream_content(response)
                raw_text = content
                parsed = RealJudgeClient._parse_json_response(content)
                if isinstance(parsed, dict):
                    return parsed, content or raw_text, stream_error
                last_error = stream_error or f"提示词建议流式输出不是可解析 JSON: {content[:1000]}"
                if attempt < max_attempts:
                    time.sleep(_advisor_retry_wait_seconds(config, last_error, attempt))
                continue
            raw_text = response.text
            try:
                data = response.json()
            except Exception:
                response.raise_for_status()
                last_error = f"响应不是 JSON: {raw_text[:1000]}"
                data = None

            if isinstance(data, dict):
                is_err, err_msg = RealJudgeClient._is_api_error(data)
                if is_err:
                    last_error = f"API error: {err_msg}. raw={raw_text[:1000]}"
                else:
                    response.raise_for_status()
                    content = client._extract_content(data)
                    parsed = RealJudgeClient._parse_json_response(content)
                    if isinstance(parsed, dict):
                        return parsed, content or raw_text, ""
                    last_error = f"提示词建议输出不是可解析 JSON: {content[:1000]}"
        except requests.exceptions.Timeout:
            last_error = f"请求超时 ({attempt}/{max_attempts})"
        except requests.exceptions.RequestException as exc:
            last_error = f"请求异常 ({attempt}/{max_attempts}): {exc}"
        except Exception as exc:
            last_error = f"未知错误 ({attempt}/{max_attempts}): {exc}"

        if attempt < max_attempts:
            time.sleep(_advisor_retry_wait_seconds(config, last_error, attempt))

    return None, raw_text or last_error, last_error


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _truncate(value: Any, max_len: int) -> str:
    text = _clean(value)
    return text[:max_len] + ("..." if len(text) > max_len else "")


def _evidence_ref_id(item: dict[str, Any]) -> str:
    for key in ("case_id", "row_id", "pair_id"):
        value = _clean(item.get(key))
        if value:
            return value
    return ""
