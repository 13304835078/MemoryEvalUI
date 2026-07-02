import json
import re
import time
import random
import requests
from typing import Any, Callable, Optional

from ..schema import EvalConfig

JUDGE_RESULT_SCHEMA = {
    "required_fields": ["score_total", "scores", "comment", "error_tags", "fatal_error"],
    "dimension_keys": ["correctness", "coverage", "update_logic", "memory_boundary", "conciseness", "format"],
    "valid_tags": {
        "hallucination", "wrong_fact", "missing_key_info", "over_memory",
        "short_term_pollution", "conflict_not_resolved", "duplicate_memory",
        "verbose_or_noisy", "format_error", "privacy_sensitive", "unclear_update",
    },
}


class JudgeClient:
    def __init__(self, config: EvalConfig):
        self.config = config
        self.rate_limit_wait_callback: Callable[[], None] | None = None

    def judge(self, system_prompt: str, user_message: str) -> tuple[Optional[dict], Optional[str]]:
        raise NotImplementedError


class RealJudgeClient(JudgeClient):
    """通过 OpenAI-compatible API 调用 LLM Judge"""

    @staticmethod
    def _normalize_chat_completions_url(url: str) -> str:
        url = (url or "").strip().rstrip("/")
        if not url:
            return url
        if url.endswith("/chat/completions"):
            return url
        return url + "/chat/completions"

    @staticmethod
    def _build_auth_header(token: str) -> str:
        token = (token or "").strip()
        if not token:
            return ""
        if token.lower().startswith("bearer "):
            return token
        return f"Bearer {token}"

    @staticmethod
    def _parse_json_object(text: str, field_name: str) -> dict:
        if not text:
            return {}
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field_name} 不是合法 JSON object: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{field_name} 必须是 JSON object")
        return value

    def _build_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        token = getattr(self.config, "judge_api_bearer_token", "")
        if token:
            headers["Authorization"] = self._build_auth_header(token)
        return headers

    def _build_payload(self, system_prompt: str, user_message: str) -> dict:
        payload = {
            "model": self.config.judge_model,
            "max_tokens": self.config.judge_max_tokens,
            "temperature": float(getattr(self.config, "judge_temperature", 0.0) or 0.0),
            "top_p": float(getattr(self.config, "judge_top_p", 1.0) or 1.0),
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }

        top_k = getattr(self.config, "judge_top_k", None)
        if top_k not in (None, ""):
            payload["top_k"] = int(top_k)

        extra_body = {
            "skip_special_tokens": False,
        }
        if getattr(self.config, "judge_send_enable_thinking", True):
            extra_body["enable_thinking"] = bool(getattr(self.config, "judge_enable_thinking", False))
        payload["extra_body"] = extra_body

        return payload

    @staticmethod
    def _is_api_error(data: dict) -> tuple[bool, str]:
        """判断 API 是否返回了错误 JSON。"""
        if not isinstance(data, dict):
            return False, ""

        if "error" in data:
            err = data.get("error")
            if isinstance(err, dict):
                msg = err.get("message") or err.get("msg") or str(err)
            else:
                msg = str(err)
            return True, msg

        # 兼容一些内部服务可能用 code/msg 表示错误
        code = data.get("code")
        if code not in (None, 0, "0", "success", "SUCCESS"):
            msg = data.get("message") or data.get("msg") or str(data)
            return True, msg

        return False, ""

    @staticmethod
    def _is_valid_judge_result(data: dict) -> tuple[bool, str]:
        """判断返回的 JSON 是否真的是 Judge 评分结果。"""
        if not isinstance(data, dict):
            return False, "返回不是 JSON object"

        required = ["score_total", "scores", "comment", "error_tags", "fatal_error"]
        missing = [k for k in required if k not in data]
        if missing:
            return False, f"Judge JSON 缺少字段: {missing}"

        if not isinstance(data.get("scores"), dict):
            return False, "scores 不是 dict"

        return True, ""

    def judge(self, system_prompt: str, user_message: str) -> tuple[Optional[dict], Optional[str]]:
        url = self._normalize_chat_completions_url(self.config.judge_api_base_url)
        payload = json.dumps(self._build_payload(system_prompt, user_message), ensure_ascii=False)
        headers = self._build_headers()

        last_error = None

        for attempt in range(1, self.config.judge_max_retries + 1):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    data=payload.encode("utf-8"),
                    timeout=self.config.judge_timeout,
                    stream=False,
                )

                raw_text = response.text

                # 有些内部服务即使鉴权失败也可能返回 200，所以不能只靠 status_code
                try:
                    data = response.json()
                except Exception:
                    response.raise_for_status()
                    last_error = f"响应不是 JSON: {raw_text[:500]}"
                    if attempt < self.config.judge_max_retries:
                        time.sleep(2 ** attempt)
                        continue
                    return None, last_error

                is_err, err_msg = self._is_api_error(data)
                if is_err:
                    last_error = f"API error: {err_msg}. raw={raw_text[:1000]}"

                    if attempt < self.config.judge_max_retries:
                        if self._is_rate_limit_error(err_msg):
                            time.sleep(self._get_rate_limit_backoff(err_msg))
                            if self.rate_limit_wait_callback is not None:
                                self.rate_limit_wait_callback()
                        elif self._is_retryable_transient_error(err_msg):
                            time.sleep(max(float(getattr(self.config, "judge_qps_backoff", 12.0) or 12.0), 2 ** attempt))
                        else:
                            time.sleep(2 ** attempt)
                        continue

                    return None, last_error

                response.raise_for_status()

                content = self._extract_content(data)
                parsed = self._parse_json_response(content)

                if parsed is None:
                    last_error = f"Judge 输出不是可解析 JSON: {content[:1000]}"
                else:
                    parsed = self._normalize_judge_result(parsed)
                    valid, reason = self._is_valid_judge_result(parsed)
                    if valid:
                        return parsed, content
                    last_error = f"Judge JSON 不符合评分格式: {reason}. raw={content[:1000]}"

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
            value = None
            for alias in aliases:
                if isinstance(scores, dict) and alias in scores:
                    value = scores.get(alias)
                    break
            try:
                normalized_scores[dim] = float(value)
            except (TypeError, ValueError):
                normalized_scores[dim] = 0.0
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

        normalized.setdefault("comment", "")
        normalized.setdefault("error_tags", [])
        normalized.setdefault("fatal_error", False)
        return normalized

    @staticmethod
    def _is_rate_limit_error(message: str) -> bool:
        msg = (message or "").lower()
        return (
            "qps limit" in msg
            or "rate limit" in msg
            or "too many requests" in msg
            or "429" in msg
        )

    @staticmethod
    def _is_retryable_transient_error(message: str) -> bool:
        msg = (message or "").lower()
        return (
            "idle timeout" in msg
            or "connection idle" in msg
            or "websocket" in msg
            or "going away" in msg
            or "connection reset" in msg
            or "temporarily unavailable" in msg
            or "bad gateway" in msg
            or "gateway timeout" in msg
            or " 502" in msg
            or " 503" in msg
            or " 504" in msg
        )

    @staticmethod
    def _parse_qps_limit(message: str) -> float | None:
        match = re.search(r"limit\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", message or "", flags=re.IGNORECASE)
        if not match:
            return None
        try:
            value = float(match.group(1))
        except ValueError:
            return None
        return value if value > 0 else None

    def _get_rate_limit_backoff(self, message: str = "") -> float:
        configured = float(getattr(self.config, "judge_qps_backoff", 12.0) or 12.0)
        qps_limit = self._parse_qps_limit(message)
        if not qps_limit:
            return configured
        # Example: "QPS limit exceeded, limit:0.10" means one request per 10 seconds.
        return max(configured, (1.0 / qps_limit) + 1.0)


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
    marker = "## 新 USER.md"
    idx = user_message.rfind(marker)
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
