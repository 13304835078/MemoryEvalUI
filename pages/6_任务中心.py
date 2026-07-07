from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.loop.closed_loop import (
    CLOSED_LOOP_DIR,
    loop_state_is_stale,
    mark_loop_interrupted,
    read_loop_state,
    request_stop as request_loop_stop,
)
from src.ui.eval_job_runner import (
    eval_job_is_stale,
    list_eval_job_ids,
    mark_eval_job_interrupted,
    read_eval_job_state,
    request_eval_stop,
)
from src.ui.judge_ab_job_runner import (
    judge_ab_job_is_stale,
    list_judge_ab_job_ids,
    mark_judge_ab_job_interrupted,
    read_judge_ab_job_state,
    request_judge_ab_stop,
)
from src.ui.memory_extraction_job_runner import (
    list_memory_extraction_job_ids,
    mark_memory_extraction_job_interrupted,
    memory_extraction_job_is_stale,
    read_memory_extraction_job_state,
    request_memory_extraction_stop,
)
from src.ui.prompt_advisor_job_runner import (
    list_prompt_advisor_job_ids,
    mark_prompt_advisor_job_interrupted,
    prompt_advisor_job_is_stale,
    read_prompt_advisor_job_state,
    request_prompt_advisor_stop,
)
from src.ui.components import render_state_file_notice


st.title("任务中心")
st.caption("集中查看后台任务状态。切换页面不会中断后台任务，但关闭应用进程会中断正在运行的线程。")


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
    return False


def _job_rows() -> list[dict[str, Any]]:
    sources = [
        ("执行评测", list_eval_job_ids()),
        ("记忆提取", list_memory_extraction_job_ids()),
        ("闭环实验", _list_loop_ids()),
        ("提示词建议", list_prompt_advisor_job_ids()),
        ("A/B 对比", list_judge_ab_job_ids()),
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


with st.expander("使用说明", expanded=True):
    st.markdown(
        """
- 这里展示所有后台任务：执行评测、记忆提取、闭环实验、提示词建议、裁判提示词 A/B 对比。
- 状态来自本地 `data/` 下的任务状态文件，因此切换页面不会清空进度。
- 任务仍依赖当前应用进程；关闭 exe 或 Streamlit 进程后，运行中的线程会停止，之后会被标记为中断。
- 多个任务可以同时运行；同一 API/Token 的请求会共用全局请求启动间隔，降低叠加超 QPS 的概率。
        """.strip()
    )


auto_refresh = st.checkbox("每10秒自动刷新任务列表", value=False)


@st.fragment(run_every="10s")
def render_task_table_auto() -> None:
    render_task_table()


def render_task_table() -> None:
    rows = _job_rows()
    if not rows:
        st.info("暂无后台任务。")
        return

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

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
    if selected_state.get("status") == "running":
        if st.button("请求终止该任务", type="secondary", use_container_width=True):
            if _request_stop(selected_type, selected_id):
                st.warning("已写入终止请求。已发出的单次 API 调用无法立即强制中断，会在下一个检查点停止。")
                st.rerun()
