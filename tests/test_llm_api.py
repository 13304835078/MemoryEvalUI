from __future__ import annotations

import json
import pytest

from src.llm_api import (
    ChatPayloadOptions,
    LLMChatClient,
    build_auth_header,
    build_chat_payload,
    is_api_error,
    is_retryable_transient_error,
    normalize_chat_completions_url,
    parse_qps_limit,
    retry_wait_seconds,
)


def test_llm_api_normalizes_endpoint_and_auth():
    assert normalize_chat_completions_url("http://example.com/v1") == "http://example.com/v1/chat/completions"
    assert normalize_chat_completions_url("http://example.com/v1/chat/completions") == (
        "http://example.com/v1/chat/completions"
    )
    assert build_auth_header("token") == "Bearer token"
    assert build_auth_header("Bearer token") == "Bearer token"


def test_llm_api_build_chat_payload_core_options():
    payload = build_chat_payload(
        [{"role": "user", "content": "hi"}],
        ChatPayloadOptions(
            model="m",
            max_tokens=100,
            temperature=0,
            top_p=1,
            top_k=1,
            enable_thinking=False,
            send_enable_thinking=True,
            skip_special_tokens=False,
        ),
    )

    assert payload["model"] == "m"
    assert payload["temperature"] == 0
    assert payload["top_p"] == 1
    assert payload["top_k"] == 1
    assert payload["extra_body"] == {"skip_special_tokens": False, "enable_thinking": False}


def test_llm_api_error_and_retry_wait_are_shared():
    is_err, message = is_api_error({"error": {"message": "QPS limit exceeded, limit:0.10"}})

    assert is_err
    assert "QPS" in message
    assert parse_qps_limit(message) == 0.10
    assert retry_wait_seconds(message, attempt=1, configured_backoff=1.0) >= 11.0
    assert is_retryable_transient_error("websocket: close 1001 (going away): Connection Idle Timeout")


class _Response:
    def __init__(self, data):
        self._data = data
        self.text = json.dumps(data, ensure_ascii=False)

    def json(self):
        return self._data

    def raise_for_status(self):
        return None


class _Session:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.response


def test_llm_chat_client_posts_json_with_shared_session():
    session = _Session(_Response({"choices": [{"message": {"content": "ok"}}]}))
    client = LLMChatClient("http://example.com/v1", "token", timeout=7, session=session)

    result = client.post_json({"model": "m", "messages": []})

    assert result.data["choices"][0]["message"]["content"] == "ok"
    args, kwargs = session.calls[0]
    assert args[0] == "http://example.com/v1/chat/completions"
    assert kwargs["headers"]["Authorization"] == "Bearer token"
    assert kwargs["timeout"] == 7
    assert json.loads(kwargs["data"].decode("utf-8"))["model"] == "m"


def test_llm_chat_client_raises_api_error():
    session = _Session(_Response({"error": {"message": "QPS limit exceeded, limit:0.10"}}))
    client = LLMChatClient("http://example.com/v1", "token", session=session)

    with pytest.raises(RuntimeError, match="QPS limit exceeded"):
        client.post_json({"model": "m", "messages": []})
