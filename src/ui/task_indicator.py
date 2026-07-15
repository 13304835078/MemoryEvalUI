from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import streamlit as st

from src.loop.closed_loop import CLOSED_LOOP_DIR, read_loop_state, request_stop as request_loop_stop
from src.ui.eval_job_runner import list_eval_job_ids, read_eval_job_state, request_eval_stop
from src.ui.judge_ab_job_runner import list_judge_ab_job_ids, read_judge_ab_job_state, request_judge_ab_stop
from src.ui.memory_extraction_job_runner import (
    list_memory_extraction_job_ids,
    read_memory_extraction_job_state,
    request_memory_extraction_stop,
)
from src.ui.prompt_advisor_job_runner import (
    list_prompt_advisor_job_ids,
    read_prompt_advisor_job_state,
    request_prompt_advisor_stop,
)


@dataclass(frozen=True)
class TaskSource:
    label: str
    ids: Callable[[], list[str]]
    read: Callable[[str], dict[str, Any]]
    stop: Callable[[str], None]


def _loop_ids() -> list[str]:
    if not CLOSED_LOOP_DIR.exists():
        return []
    paths = [item for item in CLOSED_LOOP_DIR.iterdir() if item.is_dir()]
    paths.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return [item.name for item in paths]


TASK_SOURCES = (
    TaskSource("执行评测", list_eval_job_ids, read_eval_job_state, request_eval_stop),
    TaskSource("记忆提取", list_memory_extraction_job_ids, read_memory_extraction_job_state, request_memory_extraction_stop),
    TaskSource("闭环实验", _loop_ids, read_loop_state, request_loop_stop),
    TaskSource("提示词建议", list_prompt_advisor_job_ids, read_prompt_advisor_job_state, request_prompt_advisor_stop),
    TaskSource("裁判 A/B", list_judge_ab_job_ids, read_judge_ab_job_state, request_judge_ab_stop),
)


def collect_task_summaries(*, per_type: int = 3) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in TASK_SOURCES:
        for job_id in source.ids()[:per_type]:
            state = source.read(job_id)
            if not state:
                continue
            rows.append({
                "type": source.label,
                "job_id": job_id,
                "status": str(state.get("status") or "unknown"),
                "stage": str(state.get("stage") or ""),
                "done": int(state.get("done", 0) or 0),
                "total": int(state.get("total", 0) or 0),
                "message": str(state.get("message") or ""),
                "updated_at": str(state.get("updated_at") or state.get("created_at") or ""),
                "stop": source.stop,
            })
    rows.sort(key=lambda item: item["updated_at"], reverse=True)
    return rows


@st.fragment(run_every="10s")
def render_sidebar_task_indicator() -> None:
    rows = collect_task_summaries()
    running = [item for item in rows if item["status"] == "running"]
    recent = [item for item in rows if item["status"] != "running"][:3]
    label = f"后台任务 · {len(running)} 个运行中" if running else "后台任务"
    with st.sidebar.expander(label, expanded=bool(running)):
        if not running and not recent:
            st.caption("暂无后台任务。")
        for item in running:
            st.markdown(f"**{item['type']}**")
            st.caption(f"{item['stage'] or '运行中'} · {item['job_id']}")
            fraction = item["done"] / item["total"] if item["total"] else 0.0
            st.progress(fraction)
            st.caption(f"{item['done']}/{item['total']}" if item["total"] else item["message"][:80])
            if st.button("请求终止", key=f"sidebar_stop_{item['type']}_{item['job_id']}", width="stretch"):
                item["stop"](item["job_id"])
                st.warning("已提交终止请求。")
                st.rerun()
        if recent:
            st.markdown("**最近任务**")
            for item in recent:
                st.caption(f"{item['type']} · {item['status']} · {item['job_id']}")
        st.page_link("pages/6_任务中心.py", label="打开任务中心", icon=":material/task_alt:", width="stretch")
