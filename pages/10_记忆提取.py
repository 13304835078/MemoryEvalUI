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

from src.extraction.memory_extractor import (
    EXTRACTION_OUTPUT_DIR,
    MemoryExtractionConfig,
    load_generation_prompt_templates,
    sanitize_filename,
)
from src.schema import TASK_TYPE_LABELS, TaskType, cases_from_jsonl
from src.ui.config_store import build_eval_config, load_config
from src.ui.components import render_state_file_notice
from src.ui.next_actions import NextAction, render_next_actions
from src.ui.preflight import build_extraction_preflight, render_preflight
from src.ui.run_presets import render_run_preset_selector
from src.ui.data_service import (
    save_uploaded_file,
)
from src.ui.prompt_editor import (
    get_default_extraction_prompt_file,
    infer_prompt_version,
    list_extraction_prompt_files,
    load_extraction_prompt_templates,
    load_prompt,
    prompt_text_hash,
)
from src.ui.memory_extraction_job_runner import (
    MemoryExtractionJobConfig,
    list_memory_extraction_job_ids,
    mark_memory_extraction_job_interrupted,
    memory_extraction_job_is_running,
    memory_extraction_job_is_stale,
    read_memory_extraction_job_state,
    request_memory_extraction_stop,
)
from src.ui.task_worker import launch_background_task
from src.ui.theme import render_page_header
from src.ui.workspace_context import render_workspace_context


CONFIG_PROMPT = "使用配置页当前编辑文本"
SAVED_PROMPT = "选择已保存的提取提示词文件"
LOCAL_PROMPT = "读取本地提示词文件"

EXTRACTION_CORE_COLUMNS = [
    "评测人",
    "session_id",
    "chunk_id",
    "轮次",
    "query",
    "answer",
    "call_status",
    "parse_status",
    "case_status",
    "propagation_status",
    "error",
    "user.md",
    "旧MEMORY.md",
    "MEMORY.md",
    "当前使用的模板",
]
EXTRACTION_DETAIL_COLUMNS = EXTRACTION_CORE_COLUMNS + [
    "status",
    "task_profile_id",
    "inheritance_source",
    "parse_method",
    "parse_confidence",
    "parse_warnings",
    "old_effective_document",
    "raw_output",
    "parsed_document",
    "effective_document",
    "reasoning",
    "result",
    "模型原始返回",
]


def existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def chunk_result_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    markers = [
        col for col in [
            "case_status",
            "status",
            "effective_document",
            "user.md",
            "MEMORY.md",
            "error",
            "raw_output",
            "result",
            "模型原始返回",
            "reasoning",
        ]
        if col in df.columns
    ]
    if not markers:
        return df
    mask = pd.Series(False, index=df.index)
    for col in markers:
        mask = mask | df[col].fillna("").astype(str).str.strip().ne("")
    filtered = df[mask].copy()
    return filtered if not filtered.empty else df


def core_extraction_view(df: pd.DataFrame, *, detail: bool = False) -> pd.DataFrame:
    columns = existing_columns(df, EXTRACTION_DETAIL_COLUMNS if detail else EXTRACTION_CORE_COLUMNS)
    return df[columns].copy() if columns else df.copy()


def render_extraction_dataframe_preview(df: pd.DataFrame, *, max_rows: int = 80) -> None:
    result_df = chunk_result_rows(df.fillna(""))
    st.dataframe(
        core_extraction_view(result_df.head(max_rows)),
        width="stretch",
        hide_index=True,
    )


def render_extraction_detail_view(output_path: str, key_prefix: str) -> None:
    path = Path(output_path)
    if not output_path or not path.exists():
        return

    with st.expander("详细查看提取结果", expanded=False):
        full_df = pd.read_excel(path).fillna("")
        result_df = chunk_result_rows(full_df)
        if "propagation_status" in result_df.columns:
            blocked = int((result_df["propagation_status"].astype(str) == "blocked_low_confidence").sum())
            if blocked:
                st.warning(
                    f"有 {blocked} 个低置信解析结果被保留为待复核候选，但未继承到后续 chunk。"
                    "后续提取继续使用最近一次可靠正文，避免错误传播。"
                )

        c1, c2, c3 = st.columns(3)
        c1.metric("Excel总行数", len(full_df))
        c2.metric("chunk结果行", len(result_df))
        if "call_status" in result_df.columns:
            c3.metric("调用成功chunk", int((result_df["call_status"].astype(str) == "success").sum()))
        elif "status" in result_df.columns:
            c3.metric("调用成功chunk", int(result_df["status"].astype(str).str.startswith("SUCCESS").sum()))
        else:
            c3.metric("调用成功chunk", "-")

        filter_df = result_df.copy()
        f1, f2 = st.columns(2)
        if "评测人" in filter_df.columns:
            reviewers = ["全部"] + sorted([x for x in filter_df["评测人"].astype(str).unique() if x])
            selected_reviewer = f1.selectbox("筛选评测人", reviewers, key=f"{key_prefix}_reviewer")
            if selected_reviewer != "全部":
                filter_df = filter_df[filter_df["评测人"].astype(str) == selected_reviewer]
        filter_status_column = "case_status" if "case_status" in filter_df.columns else "status"
        if filter_status_column in filter_df.columns:
            statuses = ["全部"] + sorted([x for x in filter_df[filter_status_column].astype(str).unique() if x])
            selected_status = f2.selectbox("筛选Case状态", statuses, key=f"{key_prefix}_status")
            if selected_status != "全部":
                filter_df = filter_df[filter_df[filter_status_column].astype(str) == selected_status]

        st.markdown("**chunk结果列表**")
        st.dataframe(core_extraction_view(filter_df, detail=False), width="stretch", hide_index=True)

        if not filter_df.empty:
            labels = []
            index_values = list(filter_df.index)
            for idx, row in filter_df.iterrows():
                labels.append(
                    f"行{idx + 2} | {row.get('评测人', '')} | session {row.get('session_id', '')} | "
                    f"chunk {row.get('chunk_id', '')} | {row.get('case_status', '') or row.get('status', '')}"
                )
            selected_label = st.selectbox("查看单个chunk详情", labels, key=f"{key_prefix}_chunk_select")
            selected_index = index_values[labels.index(selected_label)]
            row = full_df.loc[selected_index]

            st.markdown("**本chunk对话**")
            chunk_df = full_df.copy()
            for col in ["评测人", "session_id", "chunk_id"]:
                if col in chunk_df.columns and col in row.index:
                    chunk_df = chunk_df[chunk_df[col].astype(str) == str(row.get(col, ""))]
            dialogue_cols = existing_columns(chunk_df, ["轮次", "query", "answer"])
            if dialogue_cols:
                st.dataframe(chunk_df[dialogue_cols], width="stretch", hide_index=True)

            t1, t2 = st.columns(2)
            output_column = "MEMORY.md" if "MEMORY.md" in row.index else "user.md"
            document_name = "MEMORY.md" if output_column == "MEMORY.md" else "USER.md"
            raw_column = "模型原始返回" if "模型原始返回" in row.index else "result"
            old_document = str(row.get("old_effective_document", "") or row.get("旧MEMORY.md", ""))
            effective_document = str(row.get("effective_document", "") or row.get(output_column, ""))
            raw_output = str(row.get("raw_output", "") or row.get(raw_column, ""))
            with t1:
                st.text_area(
                    f"提取前的 {document_name}",
                    value=old_document,
                    height=140,
                    disabled=True,
                    key=f"{key_prefix}_old_memory",
                )
                st.text_area(
                    f"本轮生效的 {document_name}",
                    value=effective_document,
                    height=220,
                    disabled=True,
                    key=f"{key_prefix}_document",
                )
                if str(row.get("error", "")).strip():
                    st.text_area("错误信息", value=str(row.get("error", "")), height=120, disabled=True, key=f"{key_prefix}_error")
            with t2:
                if document_name == "MEMORY.md":
                    st.caption(f"本次模板：{row.get('当前使用的模板', '') or '未记录'}")
                if str(row.get("call_status", "")).strip():
                    st.caption(
                        "处理状态："
                        f"调用 {row.get('call_status', '')} / "
                        f"解析 {row.get('parse_status', '') or '未记录'} / "
                        f"Case {row.get('case_status', '') or '未记录'} / "
                        f"继承来源 {row.get('inheritance_source', '') or '未记录'}"
                    )
                if str(row.get("parse_method", "")).strip():
                    confidence = row.get("parse_confidence", "")
                    st.caption(
                        f"正文解析：{row.get('parse_method', '')}"
                        + (f"，置信度 {confidence}" if str(confidence).strip() else "")
                    )
                if str(row.get("parse_warnings", "")).strip():
                    st.warning(f"解析提示：{row.get('parse_warnings', '')}")
                st.text_area("模型 reasoning", value=str(row.get("reasoning", "")), height=160, disabled=True, key=f"{key_prefix}_reasoning")
                st.text_area("模型原始输出", value=raw_output, height=180, disabled=True, key=f"{key_prefix}_result")

        with st.expander("排查用：查看完整Excel原始列", expanded=False):
            st.caption("这里保留所有原始列，例如 soul.md 检索召回、研发说明、备注摘要等，只在排查输入数据时查看。")
            st.dataframe(full_df.head(300), width="stretch", hide_index=True)


def resolve_sheet_name(raw: str) -> str | int | None:
    raw = str(raw or "").strip()
    if not raw:
        return 0
    try:
        return int(raw)
    except ValueError:
        return raw


def resolve_prompt_text(
    source: str,
    saved_file: str,
    local_path: str,
    task_type: str,
) -> tuple[str, str, str]:
    local_path = str(local_path or "").strip().strip('"')
    if source == LOCAL_PROMPT:
        if not local_path:
            return "", "", ""
        templates = load_generation_prompt_templates(local_path)
        return templates["update"], Path(local_path).stem, templates["create"]

    if source == CONFIG_PROMPT:
        configured_task = st.session_state.get("selected_prompt_task_type")
        if configured_task and configured_task != task_type:
            raise ValueError(
                f"配置页当前编辑的是 {TASK_TYPE_LABELS.get(configured_task, configured_task)} 提取提示词，"
                f"与当前提取任务 {TASK_TYPE_LABELS.get(task_type, task_type)} 不一致。"
            )
        text = st.session_state.get("extraction_prompt_text", "")
        version = infer_prompt_version(
            st.session_state.get("selected_extraction_prompt_file", "")
            or get_default_extraction_prompt_file(task_type)
        )
        return text, version, text

    if not saved_file:
        return "", "", ""
    templates = load_extraction_prompt_templates(saved_file)
    return templates["update"], infer_prompt_version(saved_file), templates["create"]


def load_input_excel(uploaded_file, local_path: str) -> str:
    if uploaded_file is not None:
        return save_uploaded_file(uploaded_file, suffix=Path(uploaded_file.name).suffix)
    local_path = str(local_path or "").strip().strip('"')
    if not local_path:
        raise ValueError("请上传原始对话 Excel，或填写本地 Excel 路径。")
    if not Path(local_path).is_file():
        raise FileNotFoundError(f"本地 Excel 文件不存在：{local_path}")
    return local_path


def render_memory_extraction_job_state(job_id: str) -> None:
    state = read_memory_extraction_job_state(job_id)
    if memory_extraction_job_is_stale(state):
        state = mark_memory_extraction_job_interrupted(job_id)
    if not state:
        st.info("暂无这个记忆提取任务的状态。")
        return
    render_state_file_notice(state)

    status = str(state.get("status") or "")
    done = int(state.get("done", 0) or 0)
    total = int(state.get("total", 0) or 0)
    fraction = done / total if total else 0.0

    st.subheader("后台记忆提取进度")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("状态", status or "-")
    c2.metric("阶段", state.get("stage", "-"))
    c3.metric("进度", f"{done}/{total}" if total else "准备中")
    c4.metric("更新时间", str(state.get("updated_at", ""))[:19])
    st.progress(fraction)
    st.write(state.get("message", ""))

    if status == "running":
        if st.button("请求终止记忆提取", type="secondary", width="stretch", key=f"{job_id}_stop"):
            request_memory_extraction_stop(job_id)
            st.warning("已写入终止请求。当前 API 调用返回后会在下一个检查点停止。")
            st.rerun()
    elif status == "interrupted":
        st.warning("任务状态为已中断。通常是程序关闭、后台线程退出或长时间没有心跳导致；可以重新启动任务，或使用已生成文件继续后续流程。")

    output_path = state.get("output_path", "")
    if output_path:
        st.caption(f"输出 Excel：{output_path}")
    journal_path = state.get("journal_path", "")
    if journal_path and Path(journal_path).exists():
        st.caption(f"增量 journal：{journal_path}")
    if state.get("cases_path"):
        st.caption(f"完整 case：{state.get('cases_path')}")
        st.session_state.cases_file = state.get("cases_path", "")
    if state.get("missed_cases_path"):
        st.caption(f"漏抽 case：{state.get('missed_cases_path')}")
        st.session_state.missed_cases_file = state.get("missed_cases_path", "")

    stats = state.get("stats") or {}
    if stats:
        with st.expander("查看提取统计", expanded=False):
            st.json(stats)
    case_stats = state.get("case_stats") or {}
    if case_stats:
        call_counts = case_stats.get("call_status_counts") or {}
        parse_counts = case_stats.get("parse_status_counts") or {}
        case_counts = case_stats.get("case_status_counts") or {}
        st.markdown("**提取运行完整度**")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("可评测 case", case_stats.get("generated_cases", 0))
        s2.metric("需人工确认", case_counts.get("review_required", 0))
        s3.metric("提取正文为空", parse_counts.get("empty", 0))
        s4.metric("接口失败/终止", int(call_counts.get("failed", 0)) + int(call_counts.get("stopped", 0)))
        st.caption(
            "接口失败/终止属于运行问题，不计为模型质量 0 分；接口成功但没有可用正文属于提取质量失败。"
            "raw_fallback 会生成“需人工确认”的 case，而不是静默丢弃。"
        )
        with st.expander("查看 case 生成统计", expanded=False):
            st.json(case_stats)

    preview_rows = state.get("preview_rows") or []
    if preview_rows:
        st.markdown("**提取结果预览（核心列）**")
        st.caption("默认只展示 chunk 结果行和关键列；完整原始列可在下方“详细查看提取结果”中展开。")
        render_extraction_dataframe_preview(pd.DataFrame(preview_rows), max_rows=50)

    if status in {"completed", "stopped", "interrupted", "failed"} and output_path and Path(output_path).exists():
        st.download_button(
            "下载提取结果 Excel",
            data=Path(output_path).read_bytes(),
            file_name=Path(output_path).name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
            key=f"{job_id}_download_output",
        )
        render_extraction_detail_view(output_path, key_prefix=f"{job_id}_detail")

    cases_path = str(state.get("cases_path") or "")
    if status == "completed" and cases_path and Path(cases_path).exists():
        if st.button(
            "加载 case 并进入执行评测",
            type="primary",
            width="stretch",
            key=f"{job_id}_open_eval",
        ):
            loaded_cases = cases_from_jsonl(cases_path)
            st.session_state.cases = loaded_cases
            st.session_state.cases_file = cases_path
            missed_cases_path = str(state.get("missed_cases_path") or "")
            if missed_cases_path and Path(missed_cases_path).exists():
                st.session_state.missed_cases = cases_from_jsonl(missed_cases_path)
                st.session_state.missed_cases_file = missed_cases_path
            if loaded_cases:
                st.session_state.task_type = loaded_cases[0].task_type.value
            st.switch_page("pages/3_执行评测.py")

    if state.get("traceback"):
        with st.expander("错误堆栈", expanded=True):
            st.code(state.get("traceback", ""), language="text")


@st.fragment(run_every="10s")
def render_memory_extraction_job_state_auto(job_id: str) -> None:
    require_page_identity()
    render_memory_extraction_job_state(job_id)


render_page_header(
    "记忆提取",
    "从原始对话 Excel 生成 USER.md 用户画像或 MEMORY.md 长期记忆，并可自动生成评测 case。",
    category="提取工作流",
)

if "ui_config" not in st.session_state:
    st.session_state.ui_config = load_config()
if "cases" not in st.session_state:
    st.session_state.cases = []
if "cases_file" not in st.session_state:
    st.session_state.cases_file = ""

cfg = dict(st.session_state.ui_config)
render_run_preset_selector(cfg, key="memory_extraction")
extraction_task_type = st.selectbox(
    "提取任务",
    [TaskType.USER_MD.value, TaskType.LONG_MEMORY.value],
    format_func=lambda value: TASK_TYPE_LABELS.get(value, value),
    key="standalone_extraction_task_type",
)
document_name = "MEMORY.md" if extraction_task_type == TaskType.LONG_MEMORY.value else "USER.md"

with st.expander("使用说明", expanded=False):
    st.markdown(
        f"""
1. 上传或填写原始对话 Excel，列至少包含：`轮次`、`query`、`answer`、`评测人`。
2. 选择真实的 {document_name} 提取提示词。提取结果只写在每个 chunk 的最后一行，格式兼容后续 case 生成。
3. 同一评测人内部串行继承旧 {document_name}；不同评测人之间可以并发，不会互相继承记忆。
4. 运行完成后会生成 Excel，可直接下载，也可以自动转换成普通“执行评测”所需的 case 文件。
        """.strip()
    )

st.subheader("1. 输入数据")
with st.container(border=True):
    uploaded = st.file_uploader("上传原始对话 Excel", type=["xlsx", "xls"], key="standalone_memory_extract_upload")
    local_excel_path = st.text_input(
        "或填写本地 Excel 路径",
        value="",
        placeholder=r"C:\Users\...\dialogues.xlsx",
        help="单人本地使用时可以直接填写路径，绕过浏览器上传。",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        sheet_name_raw = st.text_input("Sheet 名称或序号", value="", help="留空默认第一个 sheet。")
    with c2:
        chunk_size = st.number_input("chunk_size", min_value=1, max_value=200, value=10, step=1)
    with c3:
        reviewer_filter = st.text_input("评测人筛选", value="", help="可选。多个评测人用逗号分隔。")

st.subheader("2. 提取提示词")
with st.container(border=True):
    extraction_prompt_files = list_extraction_prompt_files()
    default_extraction_prompt = get_default_extraction_prompt_file(extraction_task_type)
    if default_extraction_prompt and default_extraction_prompt not in extraction_prompt_files:
        extraction_prompt_files = [default_extraction_prompt] + extraction_prompt_files

    prompt_source = st.selectbox(
        "提示词来源",
        [SAVED_PROMPT, CONFIG_PROMPT, LOCAL_PROMPT],
        help="推荐使用已保存文件，便于记录版本和复现。",
    )

    selected_prompt_file = ""
    local_prompt_path = ""
    if prompt_source == SAVED_PROMPT:
        if extraction_prompt_files:
            default_index = extraction_prompt_files.index(default_extraction_prompt) if default_extraction_prompt in extraction_prompt_files else 0
            selected_prompt_file = st.selectbox("提取提示词文件", extraction_prompt_files, index=default_index)
        else:
            st.warning("prompts/generation 下暂无提取提示词文件。")
    elif prompt_source == LOCAL_PROMPT:
        local_prompt_path = st.text_input(
            "本地提取 prompt 路径",
            value="",
            placeholder=r"C:\Users\...\user_10.1.2.yaml",
            help="支持 .md/.yaml/.yml。",
        )

    try:
        extraction_prompt_text, extraction_prompt_version, extraction_create_prompt_text = resolve_prompt_text(
            prompt_source,
            selected_prompt_file,
            local_prompt_path,
            extraction_task_type,
        )
    except Exception as exc:
        extraction_prompt_text, extraction_prompt_version, extraction_create_prompt_text = "", "", ""
        st.error(f"提取提示词读取失败：{exc}")

    prompt_hash = prompt_text_hash(extraction_prompt_text)
    c1, c2 = st.columns(2)
    c1.metric("提取提示词版本", extraction_prompt_version or "未识别")
    c2.metric("提取提示词 Hash", prompt_hash[:12] if prompt_hash else "空")

    with st.expander("查看提取提示词全文", expanded=False):
        if extraction_task_type == TaskType.LONG_MEMORY.value:
            prompt_tabs = st.tabs(["更新模板", "新建模板"])
            with prompt_tabs[0]:
                st.text_area("更新模板", value=extraction_prompt_text, height=260, disabled=True)
            with prompt_tabs[1]:
                st.text_area("新建模板", value=extraction_create_prompt_text, height=260, disabled=True)
        else:
            st.text_area("提取提示词", value=extraction_prompt_text, height=260, disabled=True)

st.subheader("3. 运行参数")
with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        extract_model = st.text_input("提取模型", value=cfg.get("judge_model", "") or "AGENT-GLM5-PERF")
        max_tokens = st.number_input("最大输出长度", min_value=1000, max_value=100000, value=50000, step=1000)
        timeout = st.number_input("单次请求超时秒数", min_value=10, max_value=600, value=int(cfg.get("judge_timeout", 120) or 120), step=10)
    with c2:
        request_interval = st.number_input(
            "请求间隔秒数",
            min_value=0.0,
            max_value=300.0,
            value=float(cfg.get("judge_request_interval", 10.0) or 10.0),
            step=0.5,
        )
        max_attempts = st.number_input(
            "最大尝试次数（含首次）",
            min_value=1,
            max_value=11,
            value=max(1, int(cfg.get("judge_max_retries", 3) or 3)),
            step=1,
            help="例如设置为 3 表示最多请求 3 次：首次 1 次，失败后最多再尝试 2 次。",
        )
        retry_sleep = st.number_input(
            "失败后重试等待秒数",
            min_value=0.0,
            max_value=300.0,
            value=float(cfg.get("judge_qps_backoff", 15.0) or 15.0),
            step=1.0,
        )
    with c3:
        extraction_concurrency = st.number_input(
            "提取并发数",
            min_value=1,
            max_value=100,
            value=min(100, max(1, int(cfg.get("judge_concurrency", 1) or 1))),
            step=1,
            help=f"不同评测人之间可以并发；同一评测人内部仍串行，避免 {document_name} 继承错乱。",
        )
        send_enable_thinking = st.checkbox("发送 enable_thinking 字段", value=True)
        enable_thinking = st.checkbox("enable_thinking=true", value=True, disabled=not send_enable_thinking)
        mock = st.checkbox(
            "模拟模式",
            value=bool(cfg.get("mock", False)),
            help="开启后不调用真实模型接口，用于检查页面流程、后台进度和 case 生成。",
        )

    with st.expander("查看复用的接口核心配置", expanded=False):
        st.write({
            "接口地址": cfg.get("api_base", ""),
            "模型": extract_model,
            "模拟模式": mock,
            "温度": cfg.get("judge_temperature", 0),
            "top_p": cfg.get("judge_top_p", 1.0),
            "top_k": cfg.get("judge_top_k", None),
            "请求间隔": request_interval,
            "并发数": extraction_concurrency,
            "重试等待": retry_sleep,
        })

st.subheader("4. 输出设置")
with st.container(border=True):
    auto_make_cases = st.checkbox("提取完成后自动生成评测 case", value=True)
    c1, c2 = st.columns(2)
    with c1:
        model_name_for_case = st.text_input("case 的模型名", value=extract_model)
    with c2:
        prompt_version_for_case = st.text_input("case 的提示词版本", value=extraction_prompt_version or "unknown")

input_ready = uploaded is not None or bool(local_excel_path.strip())

job_ids = list_memory_extraction_job_ids()
last_job_id = st.session_state.get("memory_extraction_job_id", "")
active_running = bool(last_job_id and memory_extraction_job_is_running(last_job_id))

eval_config = build_eval_config({**cfg, "judge_model": extract_model, "mock": mock}, mock=mock)
render_workspace_context(
    task_type=extraction_task_type,
    case_count=None,
    cases_file=(uploaded.name if uploaded is not None else local_excel_path),
    model_name=extract_model,
    extraction_prompt=selected_prompt_file or extraction_prompt_version,
    mock=mock,
)
preflight_checks = build_extraction_preflight(
    uploaded_name=uploaded.name if uploaded is not None else "",
    local_path=local_excel_path,
    prompt_text=extraction_prompt_text,
    eval_config=eval_config,
    model_name=extract_model,
    concurrency=int(extraction_concurrency),
    request_interval=float(request_interval),
    auto_make_cases=bool(auto_make_cases),
    case_model_name=model_name_for_case,
    case_prompt_version=prompt_version_for_case,
)
preflight_ready = render_preflight(preflight_checks)

if st.button(
    "开始记忆提取",
    type="primary",
    width="stretch",
    disabled=not input_ready or active_running or not preflight_ready,
):
    try:
        if not extraction_prompt_text.strip():
            raise ValueError("提取提示词为空，请先选择或填写提取 prompt。")

        input_path = load_input_excel(uploaded, local_excel_path)
        extraction_config = MemoryExtractionConfig.from_eval_config(
            eval_config,
            model=extract_model,
            max_tokens=int(max_tokens),
            request_interval=float(request_interval),
            max_retries=max(0, int(max_attempts) - 1),
            retry_sleep=float(retry_sleep),
            enable_thinking=bool(enable_thinking),
            timeout=int(timeout),
        )
        extraction_config.send_enable_thinking = bool(send_enable_thinking)
        extraction_config.concurrency = int(extraction_concurrency)

        EXTRACTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        task_slug = "long_memory" if extraction_task_type == TaskType.LONG_MEMORY.value else "user_md"
        job_id = f"{task_slug}_extract_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_name = (
            f"{task_slug}_extract_{sanitize_filename(extract_model)}_"
            f"{sanitize_filename(extraction_prompt_version or 'prompt')}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )
        output_path = EXTRACTION_OUTPUT_DIR / output_name

        job_config = MemoryExtractionJobConfig(
            job_id=job_id,
            input_path=input_path,
            output_path=str(output_path),
            prompt_text=extraction_prompt_text,
            prompt_version=extraction_prompt_version or "unknown",
            task_type=extraction_task_type,
            create_prompt_text=extraction_create_prompt_text,
            update_prompt_text=extraction_prompt_text,
            sheet_name=resolve_sheet_name(sheet_name_raw),
            reviewer_filter=reviewer_filter.strip(),
            chunk_size=int(chunk_size),
            auto_make_cases=bool(auto_make_cases),
            case_model_name=model_name_for_case,
            case_prompt_version=prompt_version_for_case,
            extraction_config=extraction_config,
        )
        launch_background_task("memory_extraction", job_config)
        st.session_state.memory_extraction_job_id = job_id
        st.success(f"已启动独立后台记忆提取进程：{job_id}")
        st.rerun()
    except Exception as exc:
        st.error(f"记忆提取失败：{exc}")

st.divider()
st.subheader("后台任务")
job_ids = list_memory_extraction_job_ids()
last_job_id = st.session_state.get("memory_extraction_job_id", "")
if not last_job_id and job_ids:
    last_job_id = job_ids[0]
    st.session_state.memory_extraction_job_id = last_job_id

if job_ids:
    index = job_ids.index(last_job_id) if last_job_id in job_ids else 0
    selected_job_id = st.selectbox("查看记忆提取任务", job_ids, index=index)
    if selected_job_id != last_job_id:
        st.session_state.memory_extraction_job_id = selected_job_id
        last_job_id = selected_job_id
    state = read_memory_extraction_job_state(last_job_id)
    if state.get("status") == "running":
        auto_refresh = st.checkbox(
            "运行中每10秒自动刷新进度区",
            value=True,
            key=f"{last_job_id}_auto_refresh",
            help="只刷新下面的任务状态区域，不刷新整个页面。",
        )
        if auto_refresh:
            render_memory_extraction_job_state_auto(last_job_id)
        else:
            render_memory_extraction_job_state(last_job_id)
    else:
        render_memory_extraction_job_state(last_job_id)
else:
    st.info("暂无后台记忆提取任务。")

st.divider()
st.subheader("历史提取结果")

if EXTRACTION_OUTPUT_DIR.exists():
    files = sorted(EXTRACTION_OUTPUT_DIR.glob("*.xlsx"), key=lambda path: path.stat().st_mtime, reverse=True)
else:
    files = []

if files:
    labels = [path.name for path in files]
    selected_label = st.selectbox("选择历史提取 Excel", labels)
    selected_path = files[labels.index(selected_label)]
    st.caption(str(selected_path))
    if st.button("预览历史提取结果", width="stretch"):
        st.markdown("**历史提取结果预览（核心列）**")
        render_extraction_dataframe_preview(pd.read_excel(selected_path).fillna(""), max_rows=80)
        render_extraction_detail_view(str(selected_path), key_prefix=f"history_{selected_path.stem}")
    st.download_button(
        "下载历史提取 Excel",
        data=selected_path.read_bytes(),
        file_name=selected_path.name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )
else:
    st.info("暂无历史提取结果。")
