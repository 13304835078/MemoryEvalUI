from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Callable

import requests

from src.llm_api import (
    ChatPayloadOptions,
    build_chat_payload,
    LLMChatClient,
    retry_wait_seconds,
)
from src.schema import EvalConfig


logger = logging.getLogger(__name__)


@dataclass
class MemoryExtractionConfig:
    api_base: str = ""
    api_token: str = ""
    model: str = ""
    max_tokens: int = 50000
    timeout: int = 100
    request_interval: float = 10.0
    max_retries: int = 2
    retry_sleep: float = 15.0
    enable_thinking: bool = True
    send_enable_thinking: bool = True
    skip_special_tokens: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    concurrency: int = 1
    priority: int = 5
    prompt_cache_id: str = ""
    prompt_cache_location: str = "none"
    mock: bool = False

    @classmethod
    def from_eval_config(
        cls,
        config: EvalConfig,
        model: str = "",
        max_tokens: int = 50000,
        request_interval: float | None = None,
        max_retries: int | None = None,
        retry_sleep: float | None = None,
        enable_thinking: bool = True,
        timeout: int | None = None,
    ) -> "MemoryExtractionConfig":
        return cls(
            api_base=config.judge_api_base_url,
            api_token=config.judge_api_bearer_token,
            model=model or config.judge_model,
            max_tokens=max_tokens,
            timeout=int(timeout if timeout is not None else (config.judge_timeout or 100)),
            request_interval=float(
                request_interval if request_interval is not None else getattr(config, "judge_request_interval", 10.0)
            ),
            max_retries=int(max_retries if max_retries is not None else max(0, (config.judge_max_retries or 1) - 1)),
            retry_sleep=float(retry_sleep if retry_sleep is not None else getattr(config, "judge_qps_backoff", 15.0)),
            enable_thinking=enable_thinking,
            send_enable_thinking=True,
            skip_special_tokens=False,
            temperature=getattr(config, "judge_temperature", None),
            top_p=getattr(config, "judge_top_p", None),
            top_k=getattr(config, "judge_top_k", None),
            concurrency=min(100, max(1, int(getattr(config, "judge_concurrency", 1) or 1))),
            prompt_cache_id=str(getattr(config, "judge_prompt_cache_id", "") or ""),
            prompt_cache_location=str(getattr(config, "judge_prompt_cache_location", "none") or "none"),
            mock=bool(getattr(config, "mock", False)),
        )


def extract_answer_from_response(response_data: dict) -> tuple[str | None, str]:
    if not isinstance(response_data, dict):
        return None, ""

    choices = response_data.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            message = choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
                if content is not None:
                    return str(content), str(reasoning or "")
                if reasoning:
                    return str(reasoning), str(reasoning)
            if "text" in choice:
                return str(choice.get("text") or ""), ""

    for key in ("result", "content", "message"):
        if key in response_data:
            return str(response_data.get(key) or ""), ""

    data = response_data.get("data")
    if isinstance(data, dict):
        for key in ("content", "answer", "response"):
            if key in data:
                return str(data.get(key) or ""), ""
    elif data is not None:
        return str(data), ""

    return None, ""


class MemoryExtractionClient:
    def __init__(self, config: MemoryExtractionConfig):
        self.config = config
        self.rate_limit_wait_callback: Callable[[], None] | None = None
        self.should_stop_callback: Callable[[], bool] | None = None
        self.chat_client = LLMChatClient(
            config.api_base,
            config.api_token,
            timeout=int(config.timeout),
        )

    def _build_payload(self, messages: list[dict[str, str]]) -> dict:
        return build_chat_payload(
            messages,
            ChatPayloadOptions(
                model=self.config.model,
                max_tokens=int(self.config.max_tokens),
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                stream=False,
                enable_thinking=bool(self.config.enable_thinking),
                send_enable_thinking=bool(self.config.send_enable_thinking),
                skip_special_tokens=bool(self.config.skip_special_tokens),
                prompt_cache_id=self.config.prompt_cache_id,
                prompt_cache_location=self.config.prompt_cache_location,
            ),
        )

    def _request_once(self, messages: list[dict[str, str]]) -> tuple[bool, str, str, str]:
        completion = self.chat_client.post_json(self._build_payload(messages), stream=False)
        data = completion.data
        raw_text = completion.raw_text
        answer, reasoning = extract_answer_from_response(data)
        if answer is None:
            return False, f"无法从响应中提取答案。完整响应: {raw_text[:1000]}", "", raw_text
        return True, answer, reasoning or "", raw_text

    def chat_with_retry(self, messages: list[dict[str, str]]) -> tuple[bool, str, str, str]:
        last_error = ""
        attempts = int(self.config.max_retries) + 1
        for attempt in range(1, attempts + 1):
            try:
                success, answer, reasoning, raw = self._request_once(messages)
                if success:
                    return True, answer or "", reasoning or "", ""
                last_error = answer or raw or "API 返回失败"
            except requests.exceptions.RequestException as exc:
                last_error = f"请求错误: {exc}"
            except json.JSONDecodeError as exc:
                last_error = f"JSON解析错误: {exc}"
            except (RuntimeError, ValueError) as exc:
                last_error = str(exc)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"

            logger.warning("记忆提取 API 调用失败，第 %s/%s 次：%s", attempt, attempts, last_error)
            if attempt < attempts:
                if self.should_stop_callback is not None and self.should_stop_callback():
                    return False, "STOP REQUESTED", "", "STOP_REQUESTED"
                if self.config.retry_sleep > 0:
                    time.sleep(retry_wait_seconds(last_error, attempt, float(self.config.retry_sleep)))
                if self.rate_limit_wait_callback is not None:
                    self.rate_limit_wait_callback()
                if self.should_stop_callback is not None and self.should_stop_callback():
                    return False, "STOP REQUESTED", "", "STOP_REQUESTED"

        return False, f"API CALL FAILED: {last_error}", "", last_error


class MockMemoryExtractionClient(MemoryExtractionClient):
    def __init__(self, config: MemoryExtractionConfig):
        super().__init__(config)
        self.counter = 0

    def chat_with_retry(self, messages: list[dict[str, str]]) -> tuple[bool, str, str, str]:
        self.counter += 1
        user_message = messages[-1]["content"] if messages else ""
        system_message = messages[0]["content"] if messages else ""
        is_long_memory = "MEMORY.md" in system_message or "*现有长期记忆*" in user_message
        document_name = "MEMORY.md" if is_long_memory else "USER.md"
        existing = ""
        marker = "## [现有文件内容]"
        dialogue_marker = "## [最新对话内容]"
        if marker in user_message and dialogue_marker in user_message:
            existing = user_message.split(marker, 1)[1].split(dialogue_marker, 1)[0].strip()
            existing = existing.replace(f"--- {document_name} ---", "").strip()
        elif "*现有长期记忆*" in user_message and "*新增对话记录*" in user_message:
            existing = user_message.split("*现有长期记忆*", 1)[1].split("*新增对话记录*", 1)[0].strip()
        lines = [line.strip() for line in existing.splitlines() if line.strip()]
        lines.append(f"- [MOCK] 已处理第 {self.counter} 个记忆提取 chunk")
        content = f"# Output\n--- {document_name} ---\n" + "\n".join(lines)
        return True, content, "[MOCK] 未调用真实接口。", ""
