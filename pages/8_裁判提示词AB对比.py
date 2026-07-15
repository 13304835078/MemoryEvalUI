from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.user_identity import require_page_identity
require_page_identity()

from src.schema import TaskType
from src.ui.config_store import build_eval_config, load_config
from src.ui.components import render_state_file_notice
from src.ui.data_service import list_case_files, load_cases
from src.ui.preflight import build_ab_preflight, render_preflight
from src.ui.run_presets import render_run_preset_selector
from src.ui.judge_ab_job_runner import (
    JudgeAbJobConfig,
    avg_dimension_scores,
    judge_ab_job_is_stale,
    list_judge_ab_job_ids,
    load_judge_ab_results,
    mark_judge_ab_job_interrupted,
    read_judge_ab_job_state,
    request_judge_ab_stop,
    result_table,
    summarize_results,
)
from src.ui.task_worker import launch_background_task
from src.ui.prompt_editor import (
    get_default_extraction_prompt_file,
    get_default_prompt_file,
    infer_prompt_version,
    list_extraction_prompt_files,
    list_prompt_files,
    load_prompt,
    prompt_text_hash,
)
from src.ui.theme import render_page_header
from src.ui.workspace_context import render_workspace_context, summarize_values


NO_EXTRACTION_PROMPT = "不使用提取规则辅助评测"


render_page_header(
    "裁判提示词 A/B 对比",
    "保持样本与裁判模型不变，仅比较两版裁判提示词的评分表现。",
    category="优化实验",
)

if "ui_config" not in st.session_state:
    st.session_state.ui_config = load_config()


def get_task_choices() -> list[str]:
    return [item.value for item in TaskType if item.value != "raw_dialogue"]


def render_judge_ab_result(job_id: str, state: dict) -> None:
    results_a, results_b = load_judge_ab_results(job_id)
    if not results_a or not results_b:
        return

    st.divider()
    st.subheader("4. 对比结果")
    summary_a = state.get("summary_a") or summarize_results(results_a)
    summary_b = state.get("summary_b") or summarize_results(results_b)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("A 平均分", f"{float(summary_a.get('avg_score') or 0):.4f}")
    c2.metric("B 平均分", f"{float(summary_b.get('avg_score') or 0):.4f}")
    c3.metric("B-A 平均分差", f"{float(summary_b.get('avg_score') or 0) - float(summary_a.get('avg_score') or 0):.4f}")
    c4.metric("A/B 记录数", int(summary_a.get("total") or 0))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("A Judge失败", int(summary_a.get("judge_failure_count") or 0))
    c2.metric("B Judge失败", int(summary_b.get("judge_failure_count") or 0))
    c3.metric("A 成功评分", int(summary_a.get("scored_count") or 0))
    c4.metric("B 成功评分", int(summary_b.get("scored_count") or 0))
    st.caption("任一侧 Judge 运行失败的样本不计算分差，也不按 0 分进入平均分。fatal 表示已评分后的严重质量错误。")

    dim_a = avg_dimension_scores(results_a)
    dim_b = avg_dimension_scores(results_b)
    dim_rows = []
    for dim in sorted(set(dim_a) | set(dim_b)):
        dim_rows.append({
            "dimension": dim,
            "avg_A": dim_a.get(dim, 0.0),
            "avg_B": dim_b.get(dim, 0.0),
            "delta_B_minus_A": round(dim_b.get(dim, 0.0) - dim_a.get(dim, 0.0), 4),
        })
    if dim_rows:
        st.markdown("**维度平均分对比**")
        st.dataframe(pd.DataFrame(dim_rows), width="stretch", hide_index=True)

    preview_rows = state.get("table_preview") or []
    table = pd.DataFrame(preview_rows) if preview_rows else result_table(results_a, results_b)
    st.markdown("**样本明细**")
    st.dataframe(table, width="stretch", hide_index=True)
    table_file = str(state.get("table_path") or "")
    if table_file and Path(table_file).exists():
        st.download_button(
            "下载 A/B 对比结果",
            data=Path(table_file).read_bytes(),
            file_name=Path(table_file).name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
            key=f"{job_id}_download_ab",
        )

    with st.expander("运行元信息", expanded=False):
        st.json({
            "job_id": job_id,
            "created_at": state.get("started_at", ""),
            "prompt_A": state.get("stats_a", {}),
            "prompt_B": state.get("stats_b", {}),
            "config": state.get("config", {}),
        })


def render_judge_ab_job_state(job_id: str) -> None:
    state = read_judge_ab_job_state(job_id)
    if judge_ab_job_is_stale(state):
        state = mark_judge_ab_job_interrupted(job_id)
    if not state:
        st.info("暂无这个 A/B 对比任务的状态。")
        return
    render_state_file_notice(state)

    status = str(state.get("status") or "")
    done = int(state.get("done", 0) or 0)
    total = int(state.get("total", 0) or 0)
    progress = done / total if total else 0.0

    st.subheader("后台 A/B 对比进度")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("状态", status or "-")
    c2.metric("阶段", state.get("stage", "-"))
    c3.metric("进度", f"{done}/{total}" if total else "准备中")
    c4.metric("更新时间", str(state.get("updated_at", ""))[:19])
    st.progress(progress)
    st.write(state.get("message", ""))
    if state.get("effective_request_interval") is not None:
        st.caption(
            f"实际请求启动间隔：{float(state.get('effective_request_interval') or 0):.1f}s"
            f"（配置请求间隔：{float(state.get('configured_request_interval') or 0):.1f}s）"
        )
    if status == "running":
        st.info("任务仍在后台运行。切换页面后再回来，进度会从状态文件恢复。")
        if st.button("请求终止 A/B 对比", type="secondary", width="stretch", key=f"{job_id}_stop"):
            request_judge_ab_stop(job_id)
            st.warning("已写入终止请求。已发出的 API 调用会先返回，后续样本会停止提交。")
            st.rerun()
    elif status == "interrupted":
        st.warning("任务状态为已中断。通常是程序关闭或后台线程退出导致；可以重新启动任务。")
    elif status == "stopped":
        st.warning("任务已终止。")

    if status in {"completed", "stopped", "interrupted", "failed"}:
        render_judge_ab_result(job_id, state)
    if state.get("traceback"):
        with st.expander("错误堆栈", expanded=True):
            st.code(state.get("traceback", ""), language="text")


@st.fragment(run_every="10s")
def render_judge_ab_job_state_auto(job_id: str) -> None:
    render_judge_ab_job_state(job_id)


def render_judge_ab_job_panel() -> str:
    job_ids = list_judge_ab_job_ids()
    if not job_ids:
        return ""
    last_job_id = st.session_state.get("judge_ab_job_id", "") or job_ids[0]
    index = job_ids.index(last_job_id) if last_job_id in job_ids else 0
    selected_job_id = st.selectbox("查看后台 A/B 对比任务", job_ids, index=index)
    st.session_state.judge_ab_job_id = selected_job_id
    state = read_judge_ab_job_state(selected_job_id)
    if state.get("status") == "running":
        auto_refresh = st.checkbox(
            "运行中每10秒自动刷新进度区",
            value=False,
            key=f"{selected_job_id}_ab_auto_refresh",
            help="只刷新下面的进度区域，不刷新整个页面。",
        )
        if auto_refresh:
            render_judge_ab_job_state_auto(selected_job_id)
        else:
            render_judge_ab_job_state(selected_job_id)
    else:
        render_judge_ab_job_state(selected_job_id)
    return selected_job_id


with st.expander("使用说明", expanded=False):
    st.markdown(
        """
这个页面用于比较两个裁判提示词，而不是比较两个被评测模型。

- 固定不变：case 文件、被评测模型输出、裁判模型、温度、top_p、并发、请求间隔、提取规则。
- 唯一变化：裁判提示词 A 和裁判提示词 B。
- 适合观察：分数分布、错误标签、fatal、diagnostics、comment 和规则引用是否更稳定、更清晰。
- 如果接口提示 `QPS limit exceeded, limit:0.10`，建议请求间隔设为 10.5 秒以上，并发设为 1-2。
        """.strip()
    )


st.subheader("1. 选择样本")
case_files = list_case_files()
if not case_files:
    st.warning("没有可用 case 文件，请先到“评测数据”导入，或在“记忆提取”完成后生成。")
    st.stop()

labels = [Path(path).name for path in case_files]
selected_case_label = st.selectbox("case 文件", labels)
selected_case_path = case_files[labels.index(selected_case_label)]
cases = load_cases(selected_case_path)

model_names = sorted({case.model_name or "unknown" for case in cases})
selected_models = st.multiselect(
    "被评测模型筛选",
    model_names,
    default=model_names[:1] if len(model_names) > 1 else model_names,
    help="为了保证变量唯一，建议一次只选择一个被评测模型。",
)
if selected_models:
    cases = [case for case in cases if (case.model_name or "unknown") in selected_models]

limit = st.number_input("对比 case 数量（0 表示全部）", min_value=0, value=0, step=1)
run_cases = cases[: int(limit)] if int(limit) > 0 else cases

c1, c2, c3 = st.columns(3)
c1.metric("可用 case", len(cases))
c2.metric("本次对比 case", len(run_cases))
c3.metric("模型数", len({case.model_name for case in run_cases}))
if len({case.model_name for case in run_cases}) > 1:
    st.warning("当前选择了多个被评测模型，A/B 结果会混入模型差异。建议只保留一个模型。")

with st.expander("样本预览", expanded=False):
    st.dataframe(
        pd.DataFrame([
            {
                "case_id": case.case_id,
                "model_name": case.model_name,
                "prompt_version": case.prompt_version,
                "candidate_preview": (case.candidate_output or "")[:120],
            }
            for case in run_cases[:50]
        ]),
        width="stretch",
        hide_index=True,
    )


st.subheader("2. 选择提示词")
task_choices = get_task_choices()
task_type = st.selectbox(
    "任务类型",
    task_choices,
    index=task_choices.index("user_md_update") if "user_md_update" in task_choices else 0,
)

prompt_files = list_prompt_files()
if not prompt_files:
    st.warning("prompts/judge 下没有裁判提示词文件。")
    st.stop()

default_prompt = get_default_prompt_file(task_type)
prompt_a = st.selectbox(
    "裁判提示词 A（基线）",
    prompt_files,
    index=prompt_files.index(default_prompt) if default_prompt in prompt_files else 0,
)
prompt_b_default = prompt_files[1] if len(prompt_files) > 1 and prompt_files[0] == prompt_a else prompt_files[0]
prompt_b = st.selectbox(
    "裁判提示词 B（候选）",
    prompt_files,
    index=prompt_files.index(prompt_b_default) if prompt_b_default in prompt_files else 0,
)
if prompt_a == prompt_b:
    st.warning("A 和 B 当前选择的是同一个裁判提示词，对比结果理论上不应有系统差异。")

extraction_prompt_files = list_extraction_prompt_files()
default_extraction_prompt = get_default_extraction_prompt_file(task_type)
extraction_options = [NO_EXTRACTION_PROMPT] + extraction_prompt_files
default_extraction_index = (
    extraction_options.index(default_extraction_prompt)
    if default_extraction_prompt in extraction_options else 0
)
selected_extraction_prompt = st.selectbox(
    "提取提示词规则（A/B 共用）",
    extraction_options,
    index=default_extraction_index,
    help="用于把提取规则同时提供给两个裁判提示词，确保对比变量唯一。",
)
if selected_extraction_prompt == NO_EXTRACTION_PROMPT:
    extraction_prompt_text = ""
    extraction_prompt_version = ""
    extraction_prompt_hash = ""
else:
    extraction_prompt_text = load_prompt(selected_extraction_prompt, prompt_kind="extraction")
    extraction_prompt_version = infer_prompt_version(selected_extraction_prompt)
    extraction_prompt_hash = prompt_text_hash(extraction_prompt_text)

with st.expander("查看 A/B 提示词全文", expanded=False):
    prompt_a_text = load_prompt(prompt_a)
    prompt_b_text = load_prompt(prompt_b)
    left, right = st.columns(2)
    with left:
        st.text_area("裁判提示词 A", value=prompt_a_text, height=300, disabled=True)
    with right:
        st.text_area("裁判提示词 B", value=prompt_b_text, height=300, disabled=True)
    if extraction_prompt_text:
        st.text_area("共用提取提示词规则", value=extraction_prompt_text, height=220, disabled=True)


st.subheader("3. 运行配置")
cfg = dict(st.session_state.ui_config)
render_run_preset_selector(cfg, key="judge_ab")
mock = st.checkbox("模拟模式", value=bool(cfg.get("mock", False)))
cfg["judge_concurrency"] = st.number_input(
    "并发数",
    min_value=1,
    max_value=100,
    value=min(100, max(1, int(cfg.get("judge_concurrency", 1) or 1))),
    step=1,
)
configured_interval = float(cfg.get("judge_request_interval", 0) or 0)
if int(cfg["judge_concurrency"]) > 1 and configured_interval < 10:
    st.warning(
        "并发数较高且请求间隔小于 10 秒。如果接口限制是 `limit:0.10`，建议请求间隔设为 10.5 秒以上。"
        "未设置请求间隔时，本页会用“限流等待”作为保底启动间隔。"
    )

with st.expander("当前共用裁判模型配置", expanded=False):
    st.write({
        "模拟模式": mock,
        "裁判模型": cfg.get("judge_model", ""),
        "温度": cfg.get("judge_temperature", 0),
        "top_p": cfg.get("judge_top_p", 1.0),
        "top_k": cfg.get("judge_top_k", None),
        "请求间隔": cfg.get("judge_request_interval", 0),
        "并发数": cfg.get("judge_concurrency", 1),
        "限流退避": cfg.get("judge_qps_backoff", 12),
        "最大尝试（含首次）": cfg.get("judge_max_retries", 3),
        "提取提示词版本": extraction_prompt_version or "未使用",
        "提取提示词 Hash": extraction_prompt_hash[:12] if extraction_prompt_hash else "",
    })

config = build_eval_config(cfg, mock=mock)
render_workspace_context(
    task_type=task_type,
    case_count=len(run_cases),
    cases_file=selected_case_path,
    model_name=summarize_values(case.model_name for case in run_cases),
    judge_prompt=f"A {infer_prompt_version(prompt_a)} · B {infer_prompt_version(prompt_b)}",
    extraction_prompt=selected_extraction_prompt if extraction_prompt_text else "",
    mock=mock,
    title="本次 A/B 上下文",
)
preflight_checks = build_ab_preflight(
    cases=run_cases,
    task_type=task_type,
    prompt_a_text=prompt_a_text,
    prompt_b_text=prompt_b_text,
    prompt_a_name=prompt_a,
    prompt_b_name=prompt_b,
    extraction_prompt_text=extraction_prompt_text,
    eval_config=config,
)
preflight_ready = render_preflight(preflight_checks)

if st.button(
    "开始 A/B 对比",
    type="primary",
    width="stretch",
    disabled=not bool(run_cases) or not preflight_ready,
):

    job_id = f"judge_ab_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    job_config = JudgeAbJobConfig(
        job_id=job_id,
        task_type=task_type,
        prompt_a=prompt_a,
        prompt_b=prompt_b,
        cases_file=selected_case_path,
        extraction_prompt_text=extraction_prompt_text,
        extraction_prompt_version=extraction_prompt_version,
        extraction_prompt_hash=extraction_prompt_hash,
        eval_config=config,
    )
    launch_background_task("judge_ab", job_config, cases=run_cases)
    st.session_state.judge_ab_job_id = job_id
    st.success(f"已启动独立后台 A/B 对比进程：{job_id}")
    st.rerun()


st.divider()
render_judge_ab_job_panel()
