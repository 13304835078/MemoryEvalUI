from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.schema import EvalConfig
from src.runtime_paths import APP_HOME, CONFIG_DIR, ensure_writable_layout
from src.persistence import atomic_write_text, backup_corrupt_file


PROJECT_ROOT = APP_HOME
CONFIG_PATH = CONFIG_DIR / "local_config.json"


DEFAULT_CONFIG: dict[str, Any] = {
    "api_base": "",
    "api_token": "",
    "judge_model": "",
    "judge_max_tokens": 2000,
    "judge_timeout": 120,
    "judge_max_retries": 3,

    # 新增
    "judge_request_interval": 10.5,
    "judge_concurrency": 1,
    "judge_qps_backoff": 12.0,
    "judge_enable_thinking": False,
    "judge_send_enable_thinking": True,
    "judge_send_skip_special_tokens": True,
    "judge_skip_special_tokens": False,
    "judge_temperature": 0.0,
    "judge_top_p": 1.0,
    "judge_top_k": None,
    "judge_stop": [],
    "judge_stream": False,
    "judge_stream_include_usage": True,
    "judge_prompt_cache_id": "",
    "judge_prompt_cache_location": "none",

    "judge_auth_type": "bearer",
    "judge_bearer_header_name": "Authorization",
    "judge_hmac_access_key": "",
    "judge_hmac_secret_key": "",
    "judge_hmac_access_key_header": "accessKey",
    "judge_hmac_timestamp_header": "ts",
    "judge_hmac_sign_header": "sign",

    "judge_call_from": "default",
    "judge_session_id": "",
    "judge_interaction_id": None,
    "judge_moderation_action": "",
    "judge_extra_body_json": "{}",
    "judge_custom_headers_json": "{}",

    "mock": True,
}


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """读取本地 UI 配置。不存在时返回默认配置。"""
    ensure_writable_layout()
    path = Path(path)
    config = dict(DEFAULT_CONFIG)

    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if not isinstance(loaded, dict):
                raise ValueError("配置文件 JSON 顶层不是 object")
            config.update(loaded)
        except Exception as exc:
            backup_path = ""
            try:
                backup = backup_corrupt_file(path)
                backup_path = str(backup) if backup else ""
            except Exception as backup_exc:
                backup_path = f"备份失败：{type(backup_exc).__name__}: {backup_exc}"
            config["_config_error"] = f"配置文件损坏，已使用默认配置：{type(exc).__name__}: {exc}"
            config["_config_corrupt_path"] = backup_path

    # 环境变量兜底，UI 输入优先级仍然更高
    config["api_base"] = config.get("api_base") or os.environ.get("EVAL_API_BASE_URL", "")
    config["api_token"] = config.get("api_token") or os.environ.get("EVAL_API_BEARER_TOKEN", "")
    config["judge_model"] = config.get("judge_model") or os.environ.get("EVAL_MODEL_NAME", "")

    return config


def save_config(config: dict[str, Any], path: str | Path = CONFIG_PATH) -> None:
    """保存本地 UI 配置。注意 config/local_config.json 应加入 .gitignore。"""
    ensure_writable_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    safe_config = dict(config)
    safe_config.pop("_config_error", None)
    safe_config.pop("_config_corrupt_path", None)
    atomic_write_text(path, json.dumps(safe_config, ensure_ascii=False, indent=2))


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "是"}


def _as_optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _as_stop_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v)]
    text = str(value).strip()
    if not text:
        return []
    try:
        loaded = json.loads(text)
        if isinstance(loaded, list):
            return [str(v) for v in loaded if str(v)]
    except Exception:
        pass
    return [line.strip() for line in text.splitlines() if line.strip()]


def build_eval_config(config: dict[str, Any], mock: bool | None = None) -> EvalConfig:
    """把 UI 配置转换为 EvalConfig。"""
    use_mock = config.get("mock", True) if mock is None else mock

    return EvalConfig(
        judge_model=config.get("judge_model", ""),
        judge_api_base_url=config.get("api_base", ""),
        judge_api_bearer_token=config.get("api_token", ""),
        judge_max_tokens=int(config.get("judge_max_tokens", 2000) or 2000),
        judge_timeout=int(config.get("judge_timeout", 120) or 120),
        judge_max_retries=int(config.get("judge_max_retries", 3) or 3),

        # 新增
        judge_request_interval=float(config.get("judge_request_interval", 0.0) or 0.0),
        judge_concurrency=min(100, max(1, int(config.get("judge_concurrency", 1) or 1))),
        judge_qps_backoff=float(config.get("judge_qps_backoff", 12.0) or 12.0),
        judge_enable_thinking=_as_bool(config.get("judge_enable_thinking", False)),
        judge_send_enable_thinking=_as_bool(config.get("judge_send_enable_thinking", True), True),
        judge_send_skip_special_tokens=_as_bool(config.get("judge_send_skip_special_tokens", True), True),
        judge_skip_special_tokens=_as_bool(config.get("judge_skip_special_tokens", False)),
        judge_temperature=float(config.get("judge_temperature", 0.0) or 0.0),
        judge_top_p=float(config.get("judge_top_p", 1.0) or 1.0),
        judge_top_k=_as_optional_int(config.get("judge_top_k")),
        judge_stop=_as_stop_list(config.get("judge_stop", [])),
        judge_stream=_as_bool(config.get("judge_stream", False)),
        judge_stream_include_usage=_as_bool(config.get("judge_stream_include_usage", True), True),
        judge_prompt_cache_id=str(config.get("judge_prompt_cache_id", "") or ""),
        judge_prompt_cache_location=str(config.get("judge_prompt_cache_location", "none") or "none"),

        judge_auth_type="bearer",
        judge_bearer_header_name=str(config.get("judge_bearer_header_name", "Authorization") or "Authorization"),
        judge_hmac_access_key=str(config.get("judge_hmac_access_key", "") or ""),
        judge_hmac_secret_key=str(config.get("judge_hmac_secret_key", "") or ""),
        judge_hmac_access_key_header=str(config.get("judge_hmac_access_key_header", "accessKey") or "accessKey"),
        judge_hmac_timestamp_header=str(config.get("judge_hmac_timestamp_header", "ts") or "ts"),
        judge_hmac_sign_header=str(config.get("judge_hmac_sign_header", "sign") or "sign"),

        judge_call_from=str(config.get("judge_call_from", "default") or "default"),
        judge_session_id=str(config.get("judge_session_id", "") or ""),
        judge_interaction_id=_as_optional_int(config.get("judge_interaction_id")),
        judge_moderation_action=str(config.get("judge_moderation_action", "") or ""),
        judge_extra_body_json=str(config.get("judge_extra_body_json", "{}") or "{}"),
        judge_custom_headers_json=str(config.get("judge_custom_headers_json", "{}") or "{}"),

        mock=bool(use_mock),
    )


def mask_token(token: str) -> str:
    """用于 UI 展示，不泄露完整 token。"""
    if not token:
        return ""
    if len(token) <= 8:
        return "*" * len(token)
    return token[:4] + "*" * (len(token) - 8) + token[-4:]
