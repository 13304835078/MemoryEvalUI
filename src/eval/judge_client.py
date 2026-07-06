import json
import re
import time
import random
import requests
from typing import Any, Callable, Optional

from ..llm_api import (
    ChatPayloadOptions,
    build_auth_header,
    build_chat_payload,
    build_headers,
    is_api_error,
    is_rate_limit_error,
    is_retryable_transient_error,
    LLMChatClient,
    normalize_chat_completions_url,
    parse_qps_limit,
    rate_limit_backoff,
)
from ..schema import EvalConfig
from .judge_validation import (
    JUDGE_RESULT_SCHEMA,
    extract_extraction_prompt,
    is_number,
    is_valid_judge_result,
    parse_json_object,
    reference_exists_in_prompt,
    validate_string_list,
)


class JudgeClient:
    def __init__(self, config: EvalConfig):
        self.config = config
        self.rate_limit_wait_callback: Callable[[], None] | None = None

    def judge(self, system_prompt: str, user_message: str) -> tuple[Optional[dict], Optional[str]]:
        raise NotImplementedError


class RealJudgeClient(JudgeClient):
    """通过 OpenAI-compatible API 调用 LLM Judge"""

    def __init__(self, config: EvalConfig):
        super().__init__(config)
        self.chat_client = LLMChatClient(
            config.judge_api_base_url,
            getattr(config, "judge_api_bearer_token", ""),
            timeout=getattr(config, "judge_timeout", 120),
        )

    @staticmethod
    def _normalize_chat_completions_url(url: str) -> str:
        return normalize_chat_completions_url(url)

    @staticmethod
    def _build_auth_header(token: str) -> str:
        return build_auth_header(token)

    @staticmethod
    def _parse_json_object(text: str, field_name: str) -> dict:
        return parse_json_object(text, field_name)

    def _build_headers(self) -> dict:
        return build_headers(getattr(self.config, "judge_api_bearer_token", ""))

    def _build_payload(self, system_prompt: str, user_message: str) -> dict:
        return build_chat_payload(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            ChatPayloadOptions(
                model=self.config.judge_model,
                max_tokens=self.config.judge_max_tokens,
                temperature=float(getattr(self.config, "judge_temperature", 0.0) or 0.0),
                top_p=float(getattr(self.config, "judge_top_p", 1.0) or 1.0),
                top_k=getattr(self.config, "judge_top_k", None),
                stream=False,
                enable_thinking=bool(getattr(self.config, "judge_enable_thinking", False)),
                send_enable_thinking=bool(getattr(self.config, "judge_send_enable_thinking", True)),
                skip_special_tokens=False,
            ),
        )

    @staticmethod
    def _is_api_error(data: dict) -> tuple[bool, str]:
        """判断 API 是否返回了错误 JSON。"""
        return is_api_error(data)

    @staticmethod
    def _is_number(value: Any) -> bool:
        return is_number(value)

    @classmethod
    def _validate_string_list(cls, value: Any, field_name: str) -> str:
        return validate_string_list(value, field_name)

    @classmethod
    def _is_valid_judge_result(
        cls,
        data: dict,
        *,
        require_references: bool = False,
        extraction_prompt_text: str = "",
    ) -> tuple[bool, str]:
        return is_valid_judge_result(
            data,
            require_references=require_references,
            extraction_prompt_text=extraction_prompt_text,
        )

    @staticmethod
    def _reference_exists_in_prompt(reference: str, prompt_text: str) -> bool:
        return reference_exists_in_prompt(reference, prompt_text)

    @staticmethod
    def _extract_extraction_prompt(user_message: str) -> str:
        return extract_extraction_prompt(user_message)

    def judge(self, system_prompt: str, user_message: str) -> tuple[Optional[dict], Optional[str]]:
        payload = self._build_payload(system_prompt, user_message)

        last_error = None

        for attempt in range(1, self.config.judge_max_retries + 1):
            try:
                completion = self.chat_client.post_json(payload, stream=False)
                data = completion.data

                content = self._extract_content(data)
                parsed = self._parse_json_response(content)

                if parsed is None:
                    last_error = f"Judge 输出不是可解析 JSON: {content[:1000]}"
                else:
                    parsed = self._normalize_judge_result(parsed)
                    extraction_prompt_text = self._extract_extraction_prompt(user_message)
                    valid, reason = self._is_valid_judge_result(
                        parsed,
                        require_references=bool(extraction_prompt_text),
                        extraction_prompt_text=extraction_prompt_text,
                    )
                    if valid:
                        return parsed, content
                    last_error = f"Judge JSON 不符合评分格式: {reason}. raw={content[:1000]}"

            except RuntimeError as e:
                last_error = str(e)
                if attempt < self.config.judge_max_retries:
                    if self._is_rate_limit_error(last_error):
                        time.sleep(self._get_rate_limit_backoff(last_error))
                        if self.rate_limit_wait_callback is not None:
                            self.rate_limit_wait_callback()
                    elif self._is_retryable_transient_error(last_error):
                        time.sleep(max(float(getattr(self.config, "judge_qps_backoff", 12.0) or 12.0), 2 ** attempt))
                    else:
                        time.sleep(2 ** attempt)
                    continue
                return None, last_error
            except ValueError as e:
                last_error = str(e)
            except requests.exceptions.Timeout:
                last_error = f"请求超时 ({attempt}/{self.config.judge_max_retries})"
            except requests.exceptions.RequestException as e:
                last_error = f"请求异常 ({attempt}/{self.config.judge_max_retries}): {e}"
            except Exception as e:
                last_error = f"未知错误 ({attempt}/{self.config.judge_max_retries}): {e}"

            if attempt < self.config.judge_max_retries:
                time.sleep(2 ** attempt)

        return None, last_error

    def _extract_stream_content(self, response: requests.Response) -> tuple[str, str]:
        answer_parts: list[str] = []
        reasoning_parts: list[str] = []
        last_error = ""

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            line = line.strip()
            if line.startswith("data:"):
                line = line[len("data:"):].strip()
            if not line:
                continue
            if line == "[DONE]":
                break
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                last_error = f"流式响应 chunk 不是 JSON: {line[:500]}"
                continue

            is_err, err_msg = self._is_api_error(chunk)
            if is_err:
                return "", f"API error: {err_msg}. raw={line[:1000]}"

            choices = chunk.get("choices") or []
            if not choices:
                continue

            choice = choices[0]
            delta = choice.get("delta") or choice.get("message") or {}
            answer_parts.append(delta.get("content") or "")
            reasoning_parts.append(delta.get("reasoning_content") or delta.get("reasoning") or "")

            finish_reason = choice.get("finish_reason")
            if finish_reason and finish_reason not in {"", "stop"}:
                last_error = f"流式响应异常结束: finish_reason={finish_reason}"
            if finish_reason == "stop":
                break

        answer = "".join(answer_parts).strip()
        if answer:
            return answer, last_error
        reasoning = "".join(reasoning_parts).strip()
        return reasoning, last_error or ("流式响应没有正文 content" if not reasoning else "")

    def test_connection(self) -> tuple[bool, str]:
        """测试真实 API 是否可用，并确认能返回符合 JudgeResult 格式的 JSON。"""
        test_system_prompt = (
            "你是一个严格的 JSON 输出助手。"
            "请不要输出解释，不要输出 Markdown，只输出用户要求的 JSON。"
        )
        test_user_message = (
            "请原样输出以下 JSON："
            '{"score_total":5,"scores":{"correctness":5,"coverage":5,'
            '"update_logic":5,"memory_boundary":5,"conciseness":5,"format":5},'
            '"comment":"ok","error_tags":[],"fatal_error":false}'
        )

        parsed, raw = self.judge(test_system_prompt, test_user_message)

        if parsed is None:
            return False, raw or "连接失败"

        valid, reason = self._is_valid_judge_result(parsed)
        if not valid:
            return False, f"连接返回了 JSON，但不是合法 JudgeResult: {reason}. raw={raw}"

        return True, "连接成功，并返回合法 JudgeResult JSON"

    def _extract_content(self, data: dict) -> str:
        if "choices" in data and len(data["choices"]) > 0:
            choice = data["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                message = choice["message"]
                return message.get("content") or message.get("reasoning_content") or message.get("reasoning") or ""
            if "text" in choice:
                return choice["text"]

        # 兼容部分内部接口
        if "content" in data:
            return str(data["content"])
        if "result" in data:
            return str(data["result"])
        if "answer" in data:
            return str(data["answer"])

        return json.dumps(data, ensure_ascii=False)

    @staticmethod
    def _parse_json_response(text: str) -> Optional[dict]:
        if not text:
            return None

        text = text.strip()

        # 去掉 markdown code fence
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
            text = re.sub(r"```$", "", text).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                return None

        return None

    @staticmethod
    def _normalize_judge_result(data: dict) -> dict:
        """兼容候选 prompt 常见的字段名漂移，但仍收敛为内部标准 schema。"""
        if not isinstance(data, dict):
            return data

        for key in ("result", "evaluation", "judge_result", "评估结果", "评分结果"):
            nested = data.get(key)
            if isinstance(nested, dict):
                data = nested
                break

        dim_keys = JUDGE_RESULT_SCHEMA["dimension_keys"]
        normalized = dict(data)

        if "score_total" not in normalized:
            for key in ("total_score", "score", "overall_score", "总分", "综合得分"):
                if key in normalized:
                    normalized["score_total"] = normalized.get(key)
                    break

        scores = normalized.get("scores")
        if not isinstance(scores, dict):
            for key in ("dimension_scores", "score_detail", "score_details", "维度分", "维度评分"):
                if isinstance(normalized.get(key), dict):
                    scores = normalized.get(key)
                    break
        if not isinstance(scores, dict):
            scores = {dim: normalized.get(dim) for dim in dim_keys if dim in normalized}

        score_aliases = {
            "correctness": ["correctness", "正确性", "事实正确性"],
            "coverage": ["coverage", "完整性", "覆盖度"],
            "update_logic": ["update_logic", "update", "更新合理性", "更新逻辑"],
            "memory_boundary": ["memory_boundary", "boundary", "记忆边界", "边界"],
            "conciseness": ["conciseness", "简洁性", "凝练性"],
            "format": ["format", "格式", "格式合规"],
        }
        normalized_scores = {}
        for dim, aliases in score_aliases.items():
            for alias in aliases:
                if isinstance(scores, dict) and alias in scores:
                    normalized_scores[dim] = scores.get(alias)
                    break
        normalized["scores"] = normalized_scores

        if "comment" not in normalized:
            for key in ("reason", "rationale", "explanation", "备注", "理由", "评语"):
                if key in normalized:
                    normalized["comment"] = normalized.get(key)
                    break

        if "error_tags" not in normalized:
            for key in ("errors", "tags", "errorTags", "错误标签"):
                if key in normalized:
                    normalized["error_tags"] = normalized.get(key)
                    break
        if isinstance(normalized.get("error_tags"), str):
            tag_text = normalized["error_tags"].strip()
            normalized["error_tags"] = [] if tag_text in {"", "无", "none", "None", "[]"} else [
                t.strip() for t in re.split(r"[,，、\s]+", tag_text) if t.strip()
            ]

        if "fatal_error" not in normalized:
            for key in ("fatal", "is_fatal", "fatalError", "严重错误"):
                if key in normalized:
                    normalized["fatal_error"] = normalized.get(key)
                    break
        if isinstance(normalized.get("fatal_error"), str):
            normalized["fatal_error"] = normalized["fatal_error"].strip().lower() in {"true", "1", "yes", "是"}

        optional_aliases = {
            "diagnostics": ("diagnostics", "diagnoses", "issues", "扣分点", "诊断", "问题诊断"),
            "rule_refs": ("rule_refs", "rule_references", "rules", "规则引用", "规则依据"),
            "evidence_refs": ("evidence_refs", "evidence_references", "evidence", "证据引用", "事实证据"),
            "output_refs": ("output_refs", "output_references", "output_evidence", "输出引用", "候选输出引用"),
        }
        for target, aliases in optional_aliases.items():
            if target in normalized:
                continue
            for alias in aliases:
                if alias in normalized:
                    normalized[target] = normalized.get(alias)
                    break

        return normalized

    @staticmethod
    def _is_rate_limit_error(message: str) -> bool:
        return is_rate_limit_error(message)

    @staticmethod
    def _is_retryable_transient_error(message: str) -> bool:
        return is_retryable_transient_error(message)

    @staticmethod
    def _parse_qps_limit(message: str) -> float | None:
        return parse_qps_limit(message)

    def _get_rate_limit_backoff(self, message: str = "") -> float:
        configured = float(getattr(self.config, "judge_qps_backoff", 12.0) or 12.0)
        return rate_limit_backoff(message, configured)


class MockJudgeClient(JudgeClient):
    """离线评测模拟裁判，不需要 API"""

    def __init__(self, config: EvalConfig, strategy: str = "pass_all"):
        super().__init__(config)
        self.strategy = strategy

    def judge(self, system_prompt: str, user_message: str) -> tuple[Optional[dict], Optional[str]]:
        if self.strategy == "pass_all":
            result = self._make_pass_all(user_message)
        elif self.strategy == "random":
            result = self._make_random()
        else:
            result = self._make_pass_all(user_message)

        return result, json.dumps(result, ensure_ascii=False)

    def _make_pass_all(self, user_message: str) -> dict:
        has_candidate = _detect_content(user_message)
        if not has_candidate:
            return {
                "score_total": 0.0,
                "scores": {d: 0 for d in JUDGE_RESULT_SCHEMA["dimension_keys"]},
                "comment": "候选输出为空",
                "error_tags": ["format_error"],
                "fatal_error": True,
            }
        return {
            "score_total": 5.0,
            "scores": {d: 5 for d in JUDGE_RESULT_SCHEMA["dimension_keys"]},
            "comment": "[MOCK] 全满分模拟评分",
            "error_tags": [],
            "fatal_error": False,
        }

    def _make_random(self) -> dict:
        scores = {d: random.randint(3, 5) for d in JUDGE_RESULT_SCHEMA["dimension_keys"]}
        total = sum(scores[d] * w for d, w in zip(
            JUDGE_RESULT_SCHEMA["dimension_keys"],
            [0.30, 0.20, 0.20, 0.15, 0.10, 0.05],
        ))
        return {
            "score_total": round(total, 1),
            "scores": scores,
            "comment": "[MOCK] 随机模拟评分",
            "error_tags": [],
            "fatal_error": False,
        }

    
    


def _detect_content(user_message: str) -> bool:
    markers = ("## 新 USER.md", "## 新 MEMORY.md", "## 候选输出")
    marker = next((item for item in markers if item in user_message), "")
    idx = user_message.rfind(marker) if marker else -1
    if idx == -1:
        return True
    after = user_message[idx + len(marker):].strip()
    if len(after) > 5:
        return True

    reasoning_marker = "## 模型 reasoning"
    reasoning_idx = user_message.rfind(reasoning_marker, 0, idx)
    if reasoning_idx == -1:
        return False
    reasoning = user_message[reasoning_idx + len(reasoning_marker):idx].strip()
    return bool(reasoning and reasoning != "（空）" and len(reasoning) > 5)
