from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from src.runtime_paths import CONFIG_DIR
from src.ui.config_store import save_config
from src.ui.state_io import atomic_write_json


RUN_PRESETS_PATH = CONFIG_DIR / "run_presets.json"
PRESET_FIELDS = (
    "judge_max_tokens",
    "judge_timeout",
    "judge_max_retries",
    "judge_request_interval",
    "judge_concurrency",
    "judge_qps_backoff",
    "judge_temperature",
    "judge_top_p",
    "judge_top_k",
    "judge_send_enable_thinking",
    "judge_enable_thinking",
    "mock",
)

BUILTIN_PRESETS: dict[str, dict[str, Any]] = {
    "稳定评测": {
        "judge_max_tokens": 2000,
        "judge_timeout": 120,
        "judge_max_retries": 3,
        "judge_request_interval": 10.5,
        "judge_concurrency": 1,
        "judge_qps_backoff": 12.0,
        "judge_temperature": 0.0,
        "judge_top_p": 1.0,
        "judge_top_k": None,
        "judge_send_enable_thinking": True,
        "judge_enable_thinking": False,
        "mock": False,
    },
    "小样本调试": {
        "judge_max_tokens": 2000,
        "judge_timeout": 60,
        "judge_max_retries": 2,
        "judge_request_interval": 0.0,
        "judge_concurrency": 2,
        "judge_qps_backoff": 12.0,
        "judge_temperature": 0.0,
        "judge_top_p": 1.0,
        "judge_top_k": None,
        "judge_send_enable_thinking": True,
        "judge_enable_thinking": False,
        "mock": True,
    },
    "闭环实验": {
        "judge_max_tokens": 4000,
        "judge_timeout": 180,
        "judge_max_retries": 3,
        "judge_request_interval": 10.5,
        "judge_concurrency": 1,
        "judge_qps_backoff": 15.0,
        "judge_temperature": 0.0,
        "judge_top_p": 1.0,
        "judge_top_k": None,
        "judge_send_enable_thinking": True,
        "judge_enable_thinking": False,
        "mock": False,
    },
}


def capture_run_preset(config: dict[str, Any]) -> dict[str, Any]:
    return {field: config.get(field) for field in PRESET_FIELDS}


def load_custom_presets(path: str | Path = RUN_PRESETS_PATH) -> dict[str, dict[str, Any]]:
    preset_path = Path(path)
    if not preset_path.exists():
        return {}
    try:
        import json

        payload = json.loads(preset_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    values = payload.get("presets", payload) if isinstance(payload, dict) else {}
    return {
        str(name): capture_run_preset(value)
        for name, value in values.items()
        if isinstance(value, dict)
    }


def load_run_presets(path: str | Path = RUN_PRESETS_PATH) -> dict[str, dict[str, Any]]:
    return {**BUILTIN_PRESETS, **load_custom_presets(path)}


def save_custom_preset(name: str, config: dict[str, Any], path: str | Path = RUN_PRESETS_PATH) -> str:
    normalized = str(name or "").strip()
    if not normalized:
        raise ValueError("预设名称不能为空。")
    if normalized in BUILTIN_PRESETS:
        raise ValueError("内置预设不能覆盖，请使用其他名称。")
    custom = load_custom_presets(path)
    custom[normalized] = capture_run_preset(config)
    atomic_write_json(Path(path), {"version": 1, "presets": custom})
    return normalized


def apply_run_preset(config: dict[str, Any], preset: dict[str, Any]) -> dict[str, Any]:
    merged = dict(config)
    for field in PRESET_FIELDS:
        if field in preset:
            merged[field] = preset[field]
    return merged


def render_run_preset_selector(config: dict[str, Any], *, key: str) -> None:
    presets = load_run_presets()
    with st.expander("运行预设", expanded=False):
        selected = st.selectbox("选择预设", list(presets), key=f"{key}_run_preset")
        st.caption("预设只覆盖运行参数，不修改 API、Token、模型名和提示词。")
        if st.button("应用预设", width="stretch", key=f"{key}_apply_run_preset"):
            updated = apply_run_preset(config, presets[selected])
            st.session_state.ui_config = updated
            save_config(updated)
            st.success(f"已应用运行预设：{selected}")
            st.rerun()
        with st.expander("查看预设参数", expanded=False):
            st.json(presets[selected])


def render_save_current_preset(config: dict[str, Any], *, key: str) -> None:
    with st.expander("保存当前参数为预设", expanded=False):
        name = st.text_input("预设名称", key=f"{key}_preset_name", placeholder="例如：真实接口小样本")
        if st.button("保存运行预设", width="stretch", key=f"{key}_save_run_preset"):
            try:
                saved = save_custom_preset(name, config)
                st.success(f"已保存运行预设：{saved}")
            except Exception as exc:
                st.error(str(exc))

