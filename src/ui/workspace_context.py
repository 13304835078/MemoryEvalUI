from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any, Iterable

import streamlit as st

from src.schema import TASK_TYPE_LABELS, TaskType
from src.ui.prompt_editor import infer_prompt_version


def _task_value(task_type: Any) -> str:
    if isinstance(task_type, TaskType):
        return task_type.value
    return str(task_type or "")


def _short_file(value: str) -> str:
    text = str(value or "").strip()
    return Path(text).name if text else "未选择"


def summarize_values(values: Iterable[Any], *, empty: str = "未识别") -> str:
    normalized = sorted({str(value).strip() for value in values if str(value).strip()})
    if not normalized:
        return empty
    if len(normalized) == 1:
        return normalized[0]
    return f"多个（{len(normalized)}）"


def render_workspace_context(
    *,
    task_type: Any = "",
    case_count: int | None = None,
    cases_file: str = "",
    model_name: str = "",
    judge_prompt: str = "",
    extraction_prompt: str = "",
    mock: bool | None = None,
    title: str = "当前运行上下文",
) -> None:
    task_value = _task_value(task_type)
    task_label = TASK_TYPE_LABELS.get(task_value, task_value or "未选择")
    count_label = "未加载" if case_count is None else f"{int(case_count)} 条"
    if cases_file:
        count_label = f"{count_label} · {_short_file(cases_file)}"

    judge_version = infer_prompt_version(judge_prompt) if judge_prompt else "未选择"
    extraction_version = infer_prompt_version(extraction_prompt) if extraction_prompt else "未使用"
    mode_label = "未指定" if mock is None else ("模拟" if mock else "真实调用")

    items = [
        ("任务", task_label),
        ("样本", count_label),
        ("模型", model_name or "未配置"),
        ("裁判提示词", judge_version),
        ("提取提示词", extraction_version),
        ("运行模式", mode_label),
    ]
    cells = "".join(
        '<div class="me-context-item">'
        f'<span class="me-context-label">{escape(label)}</span>'
        f'<span class="me-context-value" title="{escape(value)}">{escape(value)}</span>'
        "</div>"
        for label, value in items
    )
    st.markdown(
        '<section class="me-context-bar">'
        f'<div class="me-context-heading">{escape(title)}</div>'
        f'<div class="me-context-grid">{cells}</div>'
        "</section>",
        unsafe_allow_html=True,
    )

