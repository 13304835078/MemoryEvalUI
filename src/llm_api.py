from __future__ import annotations

import re
import json
import hashlib
import threading
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class ChatPayloadOptions:
    model: str
    max_tokens: int
    temperature: float | None = 0.0
    top_p: float | None = 1.0
    top_k: int | None = None
    stream: bool = False
    enable_thinking: bool = False
    send_enable_thinking: bool = True
    skip_special_tokens: bool = False
    prompt_cache_id: str = ""
    prompt_cache_location: str = "none"


@dataclass(frozen=True)
class ChatCompletionData:
    data: dict[str, Any]
    raw_text: str


def normalize_chat_completions_url(url: str) -> str:
    url = (url or "").strip().rstrip("/")
    if not url:
        return url
    if url.endswith("/chat/completions"):
        return url
    return url + "/chat/completions"


def build_auth_header(token: str) -> str:
    token = (token or "").strip()
    if not token:
        return ""
    if token.lower().startswith("bearer "):
        return token
    return f"Bearer {token}"


def build_headers(token: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    auth = build_auth_header(token)
    if auth:
        headers["Authorization"] = auth
    return headers


def build_chat_payload(messages: list[dict[str, str]], options: ChatPayloadOptions) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": options.model,
        "max_tokens": int(options.max_tokens),
        "stream": bool(options.stream),
        "messages": messages,
        "extra_body": {
            "skip_special_tokens": bool(options.skip_special_tokens),
        },
    }
    if options.temperature is not None:
        payload["temperature"] = float(options.temperature)
    if options.top_p is not None:
        payload["top_p"] = float(options.top_p)
    if options.top_k not in (None, ""):
        payload["top_k"] = int(options.top_k)
    if options.send_enable_thinking:
        payload["extra_body"]["enable_thinking"] = bool(options.enable_thinking)
    apply_prompt_cache(
        payload,
        options.prompt_cache_id,
        options.prompt_cache_location,
    )
    return payload


def make_prompt_cache_id(namespace: str, *parts: str) -> str:
    prefix = re.sub(r"[^0-9a-z_-]+", "_", str(namespace or "memory_eval").strip().lower()).strip("_")
    prefix = prefix or "memory_eval"
    digest = hashlib.sha256("|".join(str(part or "") for part in parts).encode("utf-8")).hexdigest()[:24]
    return f"{prefix[:48]}_{digest}"


def apply_prompt_cache(payload: dict[str, Any], prompt_cache_id: str, location: str = "none") -> dict[str, Any]:
    cache_id = str(prompt_cache_id or "").strip().lower()
    cache_location = str(location or "none").strip().lower()
    if cache_id:
        if not re.fullmatch(r"[0-9a-z_-]+", cache_id):
            raise ValueError("promptCacheId 只能包含 0-9、a-z、下划线和连字符")
        if cache_location not in {"top", "extra", "both"}:
            raise ValueError("promptCacheId 已设置，但 prompt_cache_location 不是 top/extra/both")
        if cache_location in {"top", "both"}:
            payload["promptCacheId"] = cache_id
        if cache_location in {"extra", "both"}:
            payload.setdefault("extra_body", {})["promptCacheId"] = cache_id
    return payload


def is_api_error(data: dict) -> tuple[bool, str]:
    if not isinstance(data, dict):
        return False, ""

    if "error" in data:
        err = data.get("error")
        if isinstance(err, dict):
            msg = err.get("message") or err.get("msg") or str(err)
        else:
            msg = str(err)
        return True, msg

    code = data.get("code")
    if code not in (None, 0, "0", "success", "SUCCESS"):
        msg = data.get("message") or data.get("msg") or str(data)
        return True, msg

    return False, ""


def is_rate_limit_error(message: str) -> bool:
    msg = (message or "").lower()
    return (
        "qps limit" in msg
        or "rate limit" in msg
        or "too many requests" in msg
        or "429" in msg
    )


def is_retryable_transient_error(message: str) -> bool:
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


def parse_qps_limit(message: str) -> float | None:
    match = re.search(r"limit\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", message or "", flags=re.IGNORECASE)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    return value if value > 0 else None


def rate_limit_backoff(message: str = "", configured_backoff: float = 12.0) -> float:
    configured = float(configured_backoff or 12.0)
    qps_limit = parse_qps_limit(message)
    if not qps_limit:
        return configured
    return max(configured, (1.0 / qps_limit) + 1.0)


def retry_wait_seconds(message: str, attempt: int, configured_backoff: float = 12.0) -> float:
    if is_rate_limit_error(message):
        return rate_limit_backoff(message, configured_backoff)
    if is_retryable_transient_error(message):
        return max(float(configured_backoff or 12.0), float(2 ** attempt))
    return float(2 ** attempt)


class LLMChatClient:
    """Small shared HTTP client for OpenAI-compatible chat/completions APIs."""

    def __init__(
        self,
        api_base: str,
        api_token: str = "",
        *,
        timeout: int | float = 120,
        session: requests.Session | None = None,
    ):
        self.url = normalize_chat_completions_url(api_base)
        self.headers = build_headers(api_token)
        self.timeout = timeout
        self._provided_session = session
        self._thread_local = threading.local()

    def _session(self) -> requests.Session:
        if self._provided_session is not None:
            return self._provided_session
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            self._thread_local.session = session
        return session

    def post_json(self, payload: dict[str, Any], *, stream: bool = False) -> ChatCompletionData:
        response = self._session().post(
            self.url,
            headers=self.headers,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout=self.timeout,
            stream=stream,
        )
        raw_text = response.text
        try:
            data = response.json()
        except Exception:
            response.raise_for_status()
            raise ValueError(f"响应不是 JSON: {raw_text[:500]}")

        is_err, err_msg = is_api_error(data)
        if is_err:
            raise RuntimeError(f"API error: {err_msg}. raw={raw_text[:1000]}")

        response.raise_for_status()
        return ChatCompletionData(data=data, raw_text=raw_text)
