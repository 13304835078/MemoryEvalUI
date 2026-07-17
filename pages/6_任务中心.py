from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.user_identity import require_page_identity
require_page_identity()

from src.loop.closed_loop import (
    CLOSED_LOOP_DIR,
    loop_state_is_stale,
    mark_loop_interrupted,
    read_loop_controls,
    read_loop_state,
    request_stop as request_loop_stop,
    update_loop_controls,
)
from src.ui.eval_job_runner import (
    eval_job_is_stale,
    list_eval_job_ids,
    mark_eval_job_interrupted,
    read_eval_job_controls,
    read_eval_job_state,
    request_eval_stop,
    update_eval_job_controls,
)
from src.ui.extraction_prompt_ab_job_runner import (
    extraction_prompt_ab_job_is_stale,
    list_extraction_prompt_ab_job_ids,
    mark_extraction_prompt_ab_job_interrupted,
    read_extraction_prompt_ab_job_controls,
    read_extraction_prompt_ab_job_state,
    request_extraction_prompt_ab_stop,
    update_extraction_prompt_ab_job_controls,
)
from src.ui.judge_ab_job_runner import (
    judge_ab_job_is_stale,
    list_judge_ab_job_ids,
    mark_judge_ab_job_interrupted,
    read_judge_ab_job_controls,
    read_judge_ab_job_state,
    request_judge_ab_stop,
    update_judge_ab_job_controls,
)
from src.ui.memory_extraction_job_runner import (
    list_memory_extraction_job_ids,
    mark_memory_extraction_job_interrupted,
    memory_extraction_job_is_stale,
    read_memory_extraction_job_controls,
    read_memory_extraction_job_state,
    request_memory_extraction_stop,
    update_memory_extraction_job_controls,
)
from src.ui.prompt_advisor_job_runner import (
    list_prompt_advisor_job_ids,
    mark_prompt_advisor_job_interrupted,
    prompt_advisor_job_is_stale,
    read_prompt_advisor_job_controls,
    read_prompt_advisor_job_state,
    request_prompt_advisor_stop,
    update_prompt_advisor_job_controls,
)
from src.ui.components import render_state_file_notice
from src.ui.theme import render_page_header


render_page_header(
    "任务中心",
    "集中查看后台任务状态、调整可变参数与提交终止请求。",
    category="运行管理",
)
st.caption("后台任务由独立进程执行，切换页面或普通页面重载不会中断；关闭整套服务或结束后台进程会中断任务。")


def _list_loop_ids() -> list[str]:
    if not CLOSED_LOOP_DIR.exists():
        return []
    paths = [path for path in CLOSED_LOOP_DIR.iterdir() if path.is_dir()]
    paths.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return [path.name for path in paths]


def _read_state(task_type: str, job_id: str) -> dict[str, Any]:
    if task_type == "执行评测":
        state = read_eval_job_state(job_id)
        return mark_eval_job_interrupted(job_id) if eval_job_is_stale(state) else state
    if task_type == "记忆提取":
        state = read_memory_extraction_job_state(job_id)
        return mark_memory_extraction_job_interrupted(job_id) if memory_extraction_job_is_stale(state) else state
    if task_type == "闭环实验":
        state = read_loop_state(job_id)
        return mark_loop_interrupted(job_id) if loop_state_is_stale(state) else state
    if task_type == "提示词建议":
        state = read_prompt_advisor_job_state(job_id)
        return mark_prompt_advisor_job_interrupted(job_id) if prompt_advisor_job_is_stale(state) else state
    if task_type == "A/B 对比":
        state = read_judge_ab_job_state(job_id)
        return mark_judge_ab_job_interrupted(job_id) if judge_ab_job_is_stale(state) else state
    if task_type == "提取提示词 A/B":
        state = read_extraction_prompt_ab_job_state(job_id)
        return (
            mark_extraction_prompt_ab_job_interrupted(job_id)
            if extraction_prompt_ab_job_is_stale(state)
            else state
        )
    return {}


def _request_stop(task_type: str, job_id: str) -> bool:
    if task_type == "执行评测":
        request_eval_stop(job_id)
        return True
    if task_type == "记忆提取":
        request_memory_extraction_stop(job_id)
        return True
    if task_type == "闭环实验":
        request_loop_stop(job_id)
        return True
    if task_type == "提示词建议":
        request_prompt_advisor_stop(job_id)
        return True
    if task_type == "A/B 对比":
        request_judge_ab_stop(job_id)
        return True
    if task_type == "提取提示词 A/B":
        request_extraction_prompt_ab_stop(job_id)
        return True
    return False


def _read_controls(task_type: str, job_id: str) -> dict[str, Any]:
    if task_type == "执行评测":
        return read_eval_job_controls(job_id)
    if task_type == "记忆提取":
        return read_memory_extraction_job_controls(job_id)
    if task_type == "闭环实验":
        return read_loop_controls(job_id)
    if task_type == "提示词建议":
        return read_prompt_advisor_job_controls(job_id)
    if task_type == "A/B 对比":
        return read_judge_ab_job_controls(job_id)
    if task_type == "提取提示词 A/B":
        return read_extraction_prompt_ab_job_controls(job_id)
    return {}


def _update_controls(task_type: str, job_id: str, updates: dict[str, Any]) -> bool:
    if task_type == "执行评测":
        update_eval_job_controls(job_id, updates)
        return True
    if task_type == "记忆提取":
        update_memory_extraction_job_controls(job_id, updates)
        return True
    if task_type == "闭环实验":
        update_loop_controls(job_id, updates)
        return True
    if task_type == "提示词建议":
        update_prompt_advisor_job_controls(job_id, updates)
        return True
    if task_type == "A/B 对比":
        update_judge_ab_job_controls(job_id, updates)
        return True
    if task_type == "提取提示词 A/B":
        update_extraction_prompt_ab_job_controls(job_id, updates)
        return True
    return False


def _as_int(value: Any, default: int, *, min_value: int, max_value: int) -> int:
    try:
        current = int(value)
    except (TypeError, ValueError):
        current = int(default)
    return min(max_value, max(min_value, current))


def _as_float(value: Any, default: float, *, min_value: float, max_value: float) -> float:
    try:
        current = float(value)
    except (TypeError, ValueError):
        current = float(default)
    return min(max_value, max(min_value, current))


def _render_runtime_controls(task_type: str, job_id: str, state: dict[str, Any]) -> None:
    if state.get("status") != "running":
        return

    controls = _read_controls(task_type, job_id)
    config = state.get("config") if isinstance(state.get("config"), dict) else {}
    eval_config = config.get("eval_config") if isinstance(config.get("eval_config"), dict) else {}
    extraction_config = config.get("extraction_config") if isinstance(config.get("extraction_config"), dict) else {}

    with st.expander("运行中可调整参数", expanded=False):
        st.caption(
            "只允许调整后续调度参数：已发出的 API 请求不会被强制取消；模型、prompt、输入文件等会影响结果解释的参数不允许运行中修改。"
        )
        with st.form(f"runtime_controls_{task_type}_{job_id}"):
            updates: dict[str, Any] = {}
            c1, c2, c3 = st.columns(3)
            with c1:
                priority = st.number_input(
                    "任务优先级（1低-10高）",
                    min_value=1,
                    max_value=10,
                    value=_as_int(controls.get("priority"), 5, min_value=1, max_value=10),
                    step=1,
                )
                updates["priority"] = int(priority)

            if task_type == "闭环实验":
                max_rounds = max(1, int(config.get("rounds") or 1))
                with c1:
                    updates["target_rounds"] = int(st.number_input(
                        "目标总轮数",
                        min_value=1,
                        max_value=max_rounds,
                        value=_as_int(controls.get("target_rounds"), max_rounds, min_value=1, max_value=max_rounds),
                        step=1,
                    ))
                with c2:
                    updates["extraction_concurrency"] = int(st.number_input(
                        "后续提取并发",
                        min_value=1,
                        max_value=100,
                        value=_as_int(controls.get("extraction_concurrency"), config.get("extraction_concurrency") or 1, min_value=1, max_value=100),
                        step=1,
                    ))
                    updates["judge_concurrency"] = int(st.number_input(
                        "后续评测并发",
                        min_value=1,
                        max_value=100,
                        value=_as_int(controls.get("judge_concurrency"), eval_config.get("judge_concurrency") or 1, min_value=1, max_value=100),
                        step=1,
                    ))
                with c3:
                    updates["judge_request_interval"] = float(st.number_input(
                        "后续评测请求间隔",
                        min_value=0.0,
                        max_value=300.0,
                        value=_as_float(controls.get("judge_request_interval"), eval_config.get("judge_request_interval") or 0.0, min_value=0.0, max_value=300.0),
                        step=0.5,
                    ))
            elif task_type == "提取提示词 A/B":
                comparison_config = (
                    config.get("comparison_config")
                    if isinstance(config.get("comparison_config"), dict)
                    else {}
                )
                with c2:
                    updates["extraction_concurrency"] = int(st.number_input(
                        "后续提取并发",
                        min_value=1,
                        max_value=100,
                        value=_as_int(controls.get("extraction_concurrency"), extraction_config.get("concurrency") or 1, min_value=1, max_value=100),
                        step=1,
                    ))
                with c3:
                    updates["judge_concurrency"] = int(st.number_input(
                        "后续评测并发",
                        min_value=1,
                        max_value=100,
                        value=_as_int(controls.get("judge_concurrency"), eval_config.get("judge_concurrency") or 1, min_value=1, max_value=100),
                        step=1,
                    ))
                c2, c3 = st.columns(2)
                with c2:
                    updates["judge_request_interval"] = float(st.number_input(
                        "后续评测请求间隔（秒）",
                        min_value=0.0,
                        max_value=300.0,
                        value=_as_float(controls.get("judge_request_interval"), eval_config.get("judge_request_interval") or 0.0, min_value=0.0, max_value=300.0),
                        step=0.5,
                    ))
                with c3:
                    updates["comparison_request_interval"] = float(st.number_input(
                        "最终对比请求间隔（秒）",
                        min_value=0.0,
                        max_value=300.0,
                        value=_as_float(
                            controls.get("comparison_request_interval"),
                            comparison_config.get("judge_request_interval") or 0.0,
                            min_value=0.0,
                            max_value=300.0,
                        ),
                        step=0.5,
                    ))
            elif task_type in {"执行评测", "A/B 对比"}:
                with c2:
                    updates["judge_concurrency"] = int(st.number_input(
                        "后续评测并发",
                        min_value=1,
                        max_value=100,
                        value=_as_int(controls.get("judge_concurrency"), eval_config.get("judge_concurrency") or 1, min_value=1, max_value=100),
                        step=1,
                    ))
                with c3:
                    updates["judge_request_interval"] = float(st.number_input(
                        "后续评测请求间隔",
                        min_value=0.0,
                        max_value=300.0,
                        value=_as_float(controls.get("judge_request_interval"), eval_config.get("judge_request_interval") or 0.0, min_value=0.0, max_value=300.0),
                        step=0.5,
                    ))
            elif task_type == "记忆提取":
                with c2:
                    updates["extraction_concurrency"] = int(st.number_input(
                        "后续提取并发",
                        min_value=1,
                        max_value=100,
                        value=_as_int(controls.get("extraction_concurrency"), extraction_config.get("concurrency") or 1, min_value=1, max_value=100),
                        step=1,
                    ))
            else:
                c2.info("该任务当前只支持调整优先级。")

            submitted = st.form_submit_button("保存运行中参数", type="primary", width="stretch")
            if submitted:
                if _update_controls(task_type, job_id, updates):
                    st.success("已保存运行中参数，后台任务会在后续调度点读取。")
                    st.rerun()


def _job_rows() -> list[dict[str, Any]]:
    sources = [
        ("执行评测", list_eval_job_ids()),
        ("记忆提取", list_memory_extraction_job_ids()),
        ("闭环实验", _list_loop_ids()),
        ("提示词建议", list_prompt_advisor_job_ids()),
        ("A/B 对比", list_judge_ab_job_ids()),
        ("提取提示词 A/B", list_extraction_prompt_ab_job_ids()),
    ]
    rows = []
    for task_type, job_ids in sources:
        for job_id in job_ids:
            state = _read_state(task_type, job_id)
            if not state:
                continue
            done = int(state.get("done", 0) or 0)
            total = int(state.get("total", 0) or 0)
            progress = f"{done}/{total}" if total else "-"
            if task_type == "闭环实验":
                rounds = state.get("rounds") if isinstance(state.get("rounds"), list) else []
                progress = f"{len(rounds)}轮记录"
            rows.append({
                "任务类型": task_type,
                "任务ID": job_id,
                "状态": state.get("status", ""),
                "阶段": state.get("stage", ""),
                "进度": progress,
                "消息": state.get("message", state.get("stage", "")),
                "更新时间": str(state.get("updated_at") or state.get("heartbeat_at") or "")[:19],
            })
    return rows


with st.expander("使用说明", expanded=False):
    st.markdown(
        """
- 这里展示所有后台任务：执行评测、记忆提取、闭环实验、提示词建议、裁判提示词 A/B 对比、提取提示词 A/B 对比。
- 状态来自本地 `data/` 下的任务状态文件，因此切换页面不会清空进度。
- 后台任务使用独立进程；关闭整套服务、重启虚拟机或手动结束后台进程后，未完成任务会在超时后标记为中断。
- 多个任务可以同时运行；同一 API/Token 的请求会共用全局请求启动间隔，降低叠加超 QPS 的概率。
        """.strip()
    )


auto_refresh = st.checkbox("每10秒自动刷新任务列表", value=False)


@st.fragment(run_every="10s")
def render_task_table_auto() -> None:
    require_page_identity()
    render_task_table()


def render_task_table() -> None:
    rows = _job_rows()
    if not rows:
        st.info("暂无后台任务。")
        return

    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)

    running_count = int((df["状态"] == "running").sum()) if "状态" in df else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("任务总数", len(df))
    c2.metric("运行中", running_count)
    c3.metric("最近更新时间", str(df.iloc[0]["更新时间"]) if len(df) else "-")


if auto_refresh:
    render_task_table_auto()
else:
    render_task_table()


rows = _job_rows()
if rows:
    st.divider()
    st.subheader("任务详情与终止")
    labels = [f"{row['任务类型']} | {row['任务ID']}" for row in rows]
    selected = st.selectbox("选择任务", labels)
    selected_row = rows[labels.index(selected)]
    selected_type = selected_row["任务类型"]
    selected_id = selected_row["任务ID"]
    selected_state = _read_state(selected_type, selected_id)
    render_state_file_notice(selected_state)
    st.json(selected_state)
    _render_runtime_controls(selected_type, selected_id, selected_state)
    if selected_state.get("status") == "running":
        if st.button("请求终止该任务", type="secondary", width="stretch"):
            if _request_stop(selected_type, selected_id):
                st.warning("已写入终止请求。已发出的单次 API 调用无法立即强制中断，会在下一个检查点停止。")
                st.rerun()
