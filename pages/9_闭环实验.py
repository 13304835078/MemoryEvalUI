from __future__ import annotations

import json
import sys
import threading
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.extraction.memory_extractor import load_generation_prompt_templates, sanitize_filename
from src.loop import (
    CLOSED_LOOP_DIR,
    ClosedLoopConfig,
    loop_is_running,
    loop_state_is_stale,
    mark_loop_interrupted,
    read_loop_controls,
    read_loop_state,
    request_stop,
    run_closed_loop,
    update_loop_controls,
)
from src.loop.progress import compute_closed_loop_progress
from src.schema import TASK_TYPE_LABELS, TaskType
from src.ui.config_store import build_eval_config, load_config
from src.ui.components import render_state_file_notice
from src.ui.data_service import save_uploaded_file
from src.ui.prompt_editor import (
    get_default_extraction_prompt_file,
    get_default_prompt_file,
    infer_prompt_version,
    list_extraction_prompt_files,
    list_prompt_files,
    load_extraction_prompt_templates,
    load_prompt,
    prompt_text_hash,
    get_extraction_prompt_path,
)


USE_CONFIG_PROMPT = "使用配置页当前编辑文本"


def read_text_file(path_like: str | Path) -> str:
    if not path_like:
        return ""
    path = Path(path_like)
    if not path.is_absolute():
        path = get_extraction_prompt_path(str(path_like))
    if not path.exists() or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def read_json_file(path_like: str | Path) -> dict:
    if not path_like:
        return {}
    path = Path(path_like)
    if not path.exists() or not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def render_stat_metrics(title: str, stats: dict, fields: list[tuple[str, str]]) -> None:
    st.markdown(f"**{title}**")
    if not stats:
        st.caption("暂无")
        return
    cols = st.columns(min(4, max(1, len(fields))))
    for index, (label, key) in enumerate(fields):
        cols[index % len(cols)].metric(label, stats.get(key, "-"))


def render_file_table(paths: dict[str, str]) -> None:
    rows = []
    for label, value in paths.items():
        if not value:
            rows.append({"文件": label, "状态": "未生成", "路径/文件名": ""})
            continue
        path = Path(value)
        exists = path.exists() if path.is_absolute() else get_extraction_prompt_path(value).exists()
        rows.append({"文件": label, "状态": "存在" if exists else "待确认", "路径/文件名": value})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_candidate_prompt(round_item: dict, *, expanded: bool = False, key_prefix: str = "round") -> None:
    prompt_name = str(round_item.get("candidate_prompt_saved") or "")
    if not prompt_name:
        reason = round_item.get("no_candidate_reason") if isinstance(round_item.get("no_candidate_reason"), dict) else {}
        title = reason.get("title") or "本轮暂未生成候选提取提示词"
        message = reason.get("message") or "本轮没有保存新的候选提取提示词。"
        st.info(f"{title}：{message}")
        if reason:
            with st.expander("查看未生成原因", expanded=False):
                st.json(reason)
        return

    prompt_text = read_text_file(prompt_name)
    with st.expander("查看候选提取提示词", expanded=expanded):
        st.caption(f"文件：{prompt_name}；来源：{round_item.get('candidate_prompt_source') or '未记录'}")
        if prompt_text:
            st.text_area(
                "候选提取提示词内容",
                value=prompt_text,
                height=320,
                disabled=True,
                key=f"{key_prefix}_candidate_prompt_{round_item.get('round')}_{Path(prompt_name).name}",
            )
            prompt_path = get_extraction_prompt_path(prompt_name)
            if prompt_path.exists():
                st.download_button(
                    "下载候选提取提示词",
                    data=prompt_path.read_bytes(),
                    file_name=prompt_path.name,
                    mime="text/markdown",
                    use_container_width=True,
                    key=f"{key_prefix}_download_candidate_{round_item.get('round')}_{prompt_path.name}",
                )
        else:
            st.warning("候选提示词文件暂时读取不到，可能仍在生成或路径已移动。")


def render_advisor_details(round_item: dict) -> None:
    advisor = read_json_file(str(round_item.get("advisor_path") or ""))
    result = advisor.get("result") if isinstance(advisor.get("result"), dict) else {}
    if not result:
        return

    patch_result = result.get("extraction_prompt_patch_result") if isinstance(result.get("extraction_prompt_patch_result"), dict) else {}
    if patch_result:
        st.markdown("**提示词增量修改情况**")
        c1, c2, c3 = st.columns(3)
        c1.metric("已应用 patch", len(patch_result.get("applied_edits") or []))
        c2.metric("未应用 patch", len(patch_result.get("skipped_edits") or []))
        c3.metric("修改比例", f"{float(patch_result.get('change_ratio') or 0) * 100:.1f}%")
        applied = patch_result.get("applied_edits") or []
        if applied:
            with st.expander("已应用 patch", expanded=False):
                st.dataframe(pd.DataFrame(applied), use_container_width=True, hide_index=True)
        skipped = patch_result.get("skipped_edits") or []
        if skipped:
            with st.expander("未应用 patch 和原因", expanded=False):
                st.dataframe(pd.DataFrame(skipped), use_container_width=True, hide_index=True)

    diff_text = result.get("extraction_prompt_diff") or patch_result.get("diff") or ""
    if diff_text:
        with st.expander("候选提示词 diff", expanded=False):
            st.caption("说明：diff 中行首的 + 表示新增行，- 表示删除行，不会作为提示词正文写入。候选提示词正文请看本轮候选文件内容。")
            st.code(diff_text, language="diff")

    summary = {
        "can_suggest": result.get("can_suggest"),
        "evidence_summary": result.get("evidence_summary"),
        "candidate_prompt_source": result.get("candidate_prompt_source"),
        "risks": result.get("risks", []),
    }
    with st.expander("提示词建议摘要", expanded=False):
        st.json(summary)


def list_loop_run_ids() -> list[str]:
    if not CLOSED_LOOP_DIR.exists():
        return []
    paths = [p for p in CLOSED_LOOP_DIR.iterdir() if p.is_dir()]
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in paths]


def ensure_thread_registry() -> dict[str, threading.Thread]:
    if "closed_loop_threads" not in st.session_state:
        st.session_state.closed_loop_threads = {}
    return st.session_state.closed_loop_threads


def resolve_sheet_name(raw: str) -> str | int | None:
    raw = str(raw or "").strip()
    if not raw:
        return 0
    try:
        return int(raw)
    except ValueError:
        return raw


def resolve_extraction_prompt(
    source: str,
    local_path: str,
    task_type: str,
) -> tuple[str, str, str]:
    local_path = str(local_path or "").strip().strip('"')
    if local_path:
        templates = load_generation_prompt_templates(local_path)
        return templates["update"], Path(local_path).stem, templates["create"]
    if source == USE_CONFIG_PROMPT:
        configured_task = st.session_state.get("selected_prompt_task_type")
        if configured_task and configured_task != task_type:
            raise ValueError(
                f"配置页当前编辑的是 {TASK_TYPE_LABELS.get(configured_task, configured_task)} 提取提示词，"
                f"与当前闭环任务 {TASK_TYPE_LABELS.get(task_type, task_type)} 不一致。"
            )
        text = st.session_state.get("extraction_prompt_text", "")
        version = infer_prompt_version(
            st.session_state.get("selected_extraction_prompt_file", "")
            or get_default_extraction_prompt_file(task_type)
        )
        return text, version, text
    templates = load_extraction_prompt_templates(source)
    return templates["update"], infer_prompt_version(source), templates["create"]


def render_usage_guide(document_name: str) -> None:
    with st.expander("使用说明和功能介绍", expanded=True):
        st.markdown(
            f"""
**这个页面做什么**

把原本需要人工串起来的几步自动化：记忆提取 → 生成评测 case → 绝对评测 → 生成候选提取提示词 → 用候选提示词进入下一轮。

**推荐使用方式**

1. 先用少量数据试跑：闭环轮次设为 `1-2`，每轮最多评测 case 设为 `10-30`。
2. 确认候选提示词没有明显跑偏后，再增加轮次或放开 case 数量。
3. 每一轮生成的提示词都会另存为新文件，不会覆盖原始提取提示词。
4. 如果发现方向不对，点击“请求终止闭环”。已经发出去的 API 请求会先返回，之后不再继续下一步。

**输入文件要求**

原始对话 Excel 至少需要包含：`轮次`、`query`、`answer`、`评测人`。系统会按 `轮次 == 1` 切 session，再按 `chunk_size` 分块提取 {document_name}。

**结果保存位置**

每次运行的状态和中间产物保存在 `data/closed_loop/<运行编号>/`。生成的候选提取提示词保存在 `prompts/generation/`。
            """
        )


def render_state(run_id: str) -> None:
    state = read_loop_state(run_id)
    if loop_state_is_stale(state):
        state = mark_loop_interrupted(run_id)
    if not state:
        st.info("暂无这个运行编号的状态。")
        return
    render_state_file_notice(state)

    status = state.get("status", "")
    stage = state.get("stage", "")
    rounds = state.get("rounds", [])
    config = state.get("config") or {}
    eval_config = config.get("eval_config") if isinstance(config.get("eval_config"), dict) else {}
    controls = read_loop_controls(run_id) or state.get("controls") or {}
    if controls:
        state["controls"] = controls

    progress = compute_closed_loop_progress(state)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("状态", status or "-")
    c2.metric("当前轮次", f"{progress.get('current_round')}/{progress.get('total_rounds')}")
    c3.metric("当前步骤", progress.get("current_step") or stage or "-")
    c4.metric("更新时间", str(state.get("updated_at", ""))[:19])

    st.markdown("**整体进度**")
    st.progress(float(progress["overall_fraction"]))
    st.caption(progress["label"])
    st.markdown("**当前轮次进度**")
    current_round_fraction = float(progress["current_round_fraction"])
    st.progress(current_round_fraction)
    st.caption(progress["current_label"])
    if progress.get("latest_message"):
        st.info(f"最近进展：{progress.get('latest_message')}")
    elif stage:
        st.info(f"当前阶段：{stage}")

    if status == "running":
        with st.expander("运行中可调整参数", expanded=False):
            st.caption(
                "这些参数只影响后续调度：已发出的 API 请求不会被强制取消；目标轮数缩减后，当前轮会先跑完，再决定是否进入下一轮。"
            )
            max_rounds = max(1, int(config.get("rounds") or 1))
            c1, c2, c3 = st.columns(3)
            with c1:
                target_rounds = st.number_input(
                    "目标总轮数",
                    min_value=1,
                    max_value=max_rounds,
                    value=min(max_rounds, max(1, int(controls.get("target_rounds") or max_rounds))),
                    step=1,
                    key=f"{run_id}_ctl_target_rounds",
                )
                priority = st.number_input(
                    "任务优先级（1低-10高）",
                    min_value=1,
                    max_value=10,
                    value=min(10, max(1, int(controls.get("priority") or 5))),
                    step=1,
                    key=f"{run_id}_ctl_priority",
                )
            with c2:
                extraction_concurrency = st.number_input(
                    "后续提取并发",
                    min_value=1,
                    max_value=100,
                    value=min(100, max(1, int(controls.get("extraction_concurrency") or config.get("extraction_concurrency") or 1))),
                    step=1,
                    key=f"{run_id}_ctl_extraction_concurrency",
                )
                judge_concurrency = st.number_input(
                    "后续评测并发",
                    min_value=1,
                    max_value=100,
                    value=min(100, max(1, int(controls.get("judge_concurrency") or eval_config.get("judge_concurrency") or 1))),
                    step=1,
                    key=f"{run_id}_ctl_judge_concurrency",
                )
            with c3:
                judge_interval = st.number_input(
                    "后续评测请求间隔",
                    min_value=0.0,
                    max_value=300.0,
                    value=float(controls.get("judge_request_interval") if controls.get("judge_request_interval") is not None else eval_config.get("judge_request_interval") or 0.0),
                    step=0.5,
                    key=f"{run_id}_ctl_judge_interval",
                )
            if st.button("应用运行中参数", type="primary", use_container_width=True, key=f"{run_id}_apply_controls"):
                update_loop_controls(run_id, {
                    "target_rounds": int(target_rounds),
                    "priority": int(priority),
                    "extraction_concurrency": int(extraction_concurrency),
                    "judge_concurrency": int(judge_concurrency),
                    "judge_request_interval": float(judge_interval),
                })
                st.success("已保存运行中参数，后台任务会在后续调度点读取。")
                st.rerun()
        st.caption("终止是协作式停止：当前 API 调用会先返回，然后在下一步边界停止。")
        if st.button("请求终止闭环", type="secondary", use_container_width=True):
            request_stop(run_id)
            st.warning("已写入终止请求。后台任务会在当前阶段的下一个检查点停止。")
            st.rerun()
    elif status == "interrupted":
        st.warning("闭环任务可能已中断：通常是程序关闭或后台线程退出导致。已保存的中间产物仍可在下方查看。")

    if rounds:
        rows = []
        for item in rounds:
            case_stats = item.get("case_stats") or {}
            eval_stats = item.get("eval_stats") or {}
            no_candidate_reason = item.get("no_candidate_reason") if isinstance(item.get("no_candidate_reason"), dict) else {}
            rows.append({
                "轮次": item.get("round"),
                "状态": item.get("status", ""),
                "当前步骤": compute_closed_loop_progress({
                    "status": "running",
                    "config": {"rounds": 1},
                    "rounds": [item],
                }).get("current_step", ""),
                "提取进度": item.get("extraction_progress", ""),
                "评测进度": item.get("eval_progress", ""),
                "生成case": case_stats.get("generated_cases", ""),
                "漏抽case": case_stats.get("missed_cases", ""),
                "平均分": eval_stats.get("avg_score_total", ""),
                "最近消息": item.get("latest_message", ""),
                "候选提示词": Path(item.get("candidate_prompt_saved", "")).name if item.get("candidate_prompt_saved") else "",
                "候选来源": item.get("candidate_prompt_source", ""),
                "未生成原因": no_candidate_reason.get("title", ""),
            })
        st.subheader("轮次摘要")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        completed_with_prompt = [item for item in rounds if item.get("candidate_prompt_saved")]
        if completed_with_prompt:
            st.subheader("最新候选提取提示词")
            render_candidate_prompt(completed_with_prompt[-1], expanded=True, key_prefix="latest")
            render_advisor_details(completed_with_prompt[-1])
        else:
            latest_advisor_rounds = [item for item in rounds if item.get("advisor_path") or item.get("no_candidate_reason")]
            if latest_advisor_rounds:
                st.subheader("最新提示词建议结论")
                render_candidate_prompt(latest_advisor_rounds[-1], expanded=True, key_prefix="latest_no_candidate")
                render_advisor_details(latest_advisor_rounds[-1])

        with st.expander("查看每轮中间结果和文件路径", expanded=False):
            st.caption("这里用于排查问题和回溯产物。正常观察进度只看上面的轮次摘要即可。")
            for item in rounds:
                title = f"第 {item.get('round')} 轮 - {item.get('status', 'running')}"
                with st.expander(title, expanded=False):
                    paths = {
                        "记忆提取Excel": item.get("extraction_output", ""),
                        "完整case文件": item.get("cases_path", ""),
                        "漏抽case文件": item.get("missed_cases_path", ""),
                        "评测结果JSONL": item.get("results_path", ""),
                        "提示词建议原始结果": item.get("advisor_path", ""),
                        "候选提取提示词": item.get("candidate_prompt_saved", ""),
                    }
                    st.markdown("**文件路径**")
                    render_file_table(paths)

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        render_stat_metrics(
                            "提取统计",
                            item.get("extraction_stats") or {},
                            [("session", "sessions"), ("chunk", "chunks"), ("API调用", "api_calls"), ("并发", "concurrency")],
                        )
                    with c2:
                        render_stat_metrics(
                            "case 统计",
                            item.get("case_stats") or {},
                            [("完整case", "generated_cases"), ("漏抽case", "missed_cases"), ("总行数", "total_rows")],
                        )
                    with c3:
                        render_stat_metrics(
                            "评测统计",
                            item.get("eval_stats") or {},
                            [("平均分", "avg_score_total"), ("样本数", "total"), ("严重失败", "fatal_count"), ("严重失败率", "fatal_rate")],
                        )

                    with st.expander("查看原始统计 JSON", expanded=False):
                        st.json({
                            "提取统计": item.get("extraction_stats") or {},
                            "case统计": item.get("case_stats") or {},
                            "评测统计": item.get("eval_stats") or {},
                        })

                    if item.get("candidate_prompt_saved") or item.get("advisor_path") or item.get("no_candidate_reason"):
                        render_candidate_prompt(item, expanded=False, key_prefix="round_detail")
                    if item.get("advisor_path"):
                        render_advisor_details(item)

                    preview = item.get("eval_preview") or []
                    if preview:
                        with st.expander("评测结果预览", expanded=False):
                            st.dataframe(pd.DataFrame(preview), use_container_width=True, hide_index=True)

                    if item.get("latest_message"):
                        st.caption(item.get("latest_message"))

    with st.expander("查看本次运行配置", expanded=False):
        st.json({
            "运行编号": run_id,
            "轮次": config.get("rounds"),
            "chunk_size": config.get("chunk_size"),
            "每轮最多评测case": config.get("max_cases_per_round") or "全部",
            "提取模型": config.get("extraction_model"),
            "提取接口": config.get("extraction_api_base") or eval_config.get("judge_api_base_url"),
            "提取并发": config.get("extraction_concurrency"),
            "提取提示词版本": config.get("extraction_prompt_version"),
            "裁判提示词版本": config.get("judge_prompt_version"),
            "裁判模型": eval_config.get("judge_model"),
            "裁判接口": eval_config.get("judge_api_base_url"),
            "提示词改进模型": config.get("advisor_model") or eval_config.get("judge_model"),
            "提示词改进接口": config.get("advisor_api_base") or eval_config.get("judge_api_base_url"),
            "温度": eval_config.get("judge_temperature"),
            "top_p": eval_config.get("judge_top_p"),
            "top_k": eval_config.get("judge_top_k"),
            "发送enable_thinking": eval_config.get("judge_send_enable_thinking"),
            "enable_thinking": eval_config.get("judge_enable_thinking"),
            "评测并发": eval_config.get("judge_concurrency"),
            "评测请求间隔": eval_config.get("judge_request_interval"),
            "评测最大尝试（含首次）": eval_config.get("judge_max_retries"),
        })

    events = state.get("events") or []
    if events:
        with st.expander("查看运行日志", expanded=False):
            st.dataframe(pd.DataFrame(events[-80:]), use_container_width=True, hide_index=True)

    if state.get("traceback"):
        with st.expander("错误堆栈", expanded=True):
            st.code(state.get("traceback", ""), language="text")


@st.fragment(run_every="10s")
def render_state_auto(run_id: str) -> None:
    render_state(run_id)


st.title("闭环实验")
st.caption("自动跑多轮提取提示词改进实验；默认只展示必要信息，细节在展开项里。")

if "ui_config" not in st.session_state:
    st.session_state.ui_config = load_config()

thread_registry = ensure_thread_registry()
cfg = dict(st.session_state.ui_config)

st.warning("实验功能：候选提示词可能沿着当前 Judge 的偏差自我强化。建议先小样本试跑，再决定是否扩大。")

default_run_id = f"closed_loop_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

st.subheader("1. 基础设置")
with st.container(border=True):
    loop_task_type = st.selectbox(
        "闭环任务",
        [TaskType.USER_MD.value, TaskType.LONG_MEMORY.value],
        format_func=lambda value: TASK_TYPE_LABELS.get(value, value),
        key="closed_loop_task_type",
    )
    document_name = "MEMORY.md" if loop_task_type == TaskType.LONG_MEMORY.value else "USER.md"
    render_usage_guide(document_name)

    run_id_raw = st.text_input(
        "运行编号",
        value=st.session_state.get("closed_loop_last_run_id", default_run_id),
        help="用于保存本次闭环状态和中间产物。可以保留默认值，也可以改成便于识别的名字。",
    )
    run_id = sanitize_filename(run_id_raw) or default_run_id
    if run_id != run_id_raw:
        st.caption(f"运行编号会保存为：{run_id}")

    uploaded_excel = st.file_uploader(
        "上传原始对话 Excel",
        type=["xlsx", "xls"],
        key="closed_loop_upload",
        help="推荐直接上传。Excel 需要包含：轮次、query、answer、评测人。",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        rounds = st.number_input(
            "闭环轮次",
            min_value=1,
            max_value=20,
            value=3,
            step=1,
            help="每轮都会生成一个候选提取提示词，并用于下一轮记忆提取。",
        )
    with c2:
        chunk_size = st.number_input(
            "每次提取的对话条数",
            min_value=1,
            max_value=200,
            value=10,
            step=1,
            help="对应原 run_user.py 的 chunk_size。每个 session 内按这个大小分块。",
        )
    with c3:
        max_cases_per_round = st.number_input(
            "每轮最多评测case",
            min_value=0,
            max_value=100000,
            value=0,
            step=1,
            help="0 表示全部。试跑时建议设为 10-30，避免一次跑太久。",
        )

    with st.expander("可选输入设置", expanded=False):
        local_excel_path = st.text_input(
            "本地 Excel 路径",
            value="",
            placeholder=r"C:\Users\...\dialogues.xlsx",
            help="如果浏览器上传不方便，可以填本地路径。上传文件优先级更高。",
        )
        sheet_name_raw = st.text_input(
            "Sheet 名称或序号",
            value="",
            help="留空默认读取第一个 Sheet。可以填 0、1 这样的序号，也可以填 Sheet 名称。",
        )
        reviewer_filter = st.text_input(
            "评测人筛选",
            value="",
            help="可选。多个评测人用逗号分隔；留空表示不筛选。",
        )

st.subheader("2. 提示词设置")
with st.container(border=True):
    extraction_prompt_files = list_extraction_prompt_files()
    default_extraction_prompt = get_default_extraction_prompt_file(loop_task_type)
    if default_extraction_prompt and default_extraction_prompt not in extraction_prompt_files:
        extraction_prompt_files = [default_extraction_prompt] + extraction_prompt_files
    extraction_options = [USE_CONFIG_PROMPT] + extraction_prompt_files
    extraction_source = st.selectbox(
        "初始提取提示词",
        extraction_options,
        index=(
            extraction_options.index(default_extraction_prompt)
            if default_extraction_prompt in extraction_options
            else 0
        ),
        help="第一轮使用这里的提示词；第二轮开始使用上一轮自动生成的新版本。",
    )

    local_prompt_path = ""
    with st.expander("可选：从本地文件读取提取提示词", expanded=False):
        local_prompt_path = st.text_input(
            "本地提取 prompt 路径",
            value="",
            placeholder=r"C:\Users\...\user_10.1.2.yaml",
            help="支持 .md/.yaml/.yml。填写后会覆盖上面的选择。",
        )

    try:
        extraction_prompt_text, extraction_prompt_version, extraction_create_prompt_text = resolve_extraction_prompt(
            extraction_source,
            local_prompt_path,
            loop_task_type,
        )
    except Exception as exc:
        extraction_prompt_text, extraction_prompt_version, extraction_create_prompt_text = "", "", ""
        st.error(f"提取提示词读取失败：{exc}")

    judge_prompt_files = list_prompt_files()
    default_judge_prompt = get_default_prompt_file(loop_task_type)
    if default_judge_prompt and default_judge_prompt not in judge_prompt_files:
        judge_prompt_files = [default_judge_prompt] + judge_prompt_files
    selected_judge_prompt = st.selectbox(
        "裁判提示词",
        judge_prompt_files,
        index=judge_prompt_files.index(default_judge_prompt) if default_judge_prompt in judge_prompt_files else 0,
        help=f"用于给每轮 {document_name} 提取结果打分。建议使用绝对评测稳定版。",
    )
    judge_prompt_text = load_prompt(selected_judge_prompt)

    c1, c2, c3 = st.columns(3)
    c1.metric("提取提示词版本", extraction_prompt_version or "未识别")
    c2.metric("提取提示词Hash", prompt_text_hash(extraction_prompt_text)[:12] if extraction_prompt_text else "空")
    c3.metric("裁判提示词版本", infer_prompt_version(selected_judge_prompt))

    with st.expander("查看提示词全文", expanded=False):
        st.caption("这里只读展示，修改提示词请去配置页保存版本，或使用本地 prompt 路径。")
        if loop_task_type == TaskType.LONG_MEMORY.value:
            prompt_tabs = st.tabs(["更新模板", "新建模板"])
            with prompt_tabs[0]:
                st.text_area("初始更新模板全文", value=extraction_prompt_text, height=260, disabled=True)
            with prompt_tabs[1]:
                st.text_area("初始新建模板全文", value=extraction_create_prompt_text, height=260, disabled=True)
        else:
            st.text_area("初始提取提示词全文", value=extraction_prompt_text, height=260, disabled=True)
        st.text_area("裁判提示词全文", value=judge_prompt_text, height=220, disabled=True)

mock = bool(cfg.get("mock", False))
default_api_base = cfg.get("api_base", "")
default_api_token = cfg.get("api_token", "")
default_model = cfg.get("judge_model", "") or "AGENT-GLM5-PERF"
extraction_model = cfg.get("extraction_model", "") or default_model
extraction_api_base = cfg.get("extraction_api_base", "") or default_api_base
extraction_api_token = cfg.get("extraction_api_token", "") or default_api_token
eval_model = cfg.get("judge_model", "") or default_model
eval_api_base = cfg.get("api_base", "") or default_api_base
eval_api_token = cfg.get("api_token", "") or default_api_token
advisor_model = cfg.get("advisor_model", "") or eval_model
advisor_api_base = cfg.get("advisor_api_base", "") or eval_api_base
advisor_api_token = cfg.get("advisor_api_token", "") or eval_api_token
extraction_max_tokens = 50000
extraction_timeout = int(cfg.get("judge_timeout", 120) or 120)
extraction_request_interval = float(cfg.get("judge_request_interval", 10.0) or 10.0)
extraction_max_attempts = max(1, int(cfg.get("judge_max_retries", 3) or 3))
extraction_retry_sleep = float(cfg.get("judge_qps_backoff", 15.0) or 15.0)
extraction_concurrency = min(100, max(1, int(cfg.get("judge_concurrency", 1) or 1)))
extraction_send_enable_thinking = True
extraction_enable_thinking = True
advisor_max_items = 40

with st.expander("模型与接口配置", expanded=False):
    st.caption("三组默认沿用配置页。只有需要不同模型或不同接口时才修改；token 输入框不会写回配置页。")
    col_extract, col_eval, col_advisor = st.columns(3)
    with col_extract:
        st.markdown("**记忆提取**")
        extraction_model = st.text_input("提取模型名称/型号", value=extraction_model, key="closed_loop_extraction_model")
        extraction_api_base = st.text_input("提取 API 地址", value=extraction_api_base, key="closed_loop_extraction_api")
        extraction_api_token = st.text_input(
            "提取 API Token",
            value=extraction_api_token,
            type="password",
            key="closed_loop_extraction_token",
            help="留空会使用配置页 token。",
        )
    with col_eval:
        st.markdown("**执行评测**")
        eval_model = st.text_input("评测模型名称/型号", value=eval_model, key="closed_loop_eval_model")
        eval_api_base = st.text_input("评测 API 地址", value=eval_api_base, key="closed_loop_eval_api")
        eval_api_token = st.text_input(
            "评测 API Token",
            value=eval_api_token,
            type="password",
            key="closed_loop_eval_token",
            help="留空会使用配置页 token。",
        )
    with col_advisor:
        st.markdown("**改提示词**")
        advisor_model = st.text_input("改提示词模型名称/型号", value=advisor_model, key="closed_loop_advisor_model")
        advisor_api_base = st.text_input("改提示词 API 地址", value=advisor_api_base, key="closed_loop_advisor_api")
        advisor_api_token = st.text_input(
            "改提示词 API Token",
            value=advisor_api_token,
            type="password",
            key="closed_loop_advisor_token",
            help="留空会使用评测 token；如果评测 token 也为空，则使用配置页 token。",
        )

with st.expander("高级运行参数", expanded=False):
    st.markdown(
        "这些参数通常不需要改。模型和接口在上面的“模型与接口配置”里设置；这里控制限流、重试和并发。"
    )
    mock = st.checkbox("模拟模式", value=mock, help="开启后提取、评测、提示词建议都不调用真实接口，适合测试页面流程和进度条。")

    c1, c2, c3 = st.columns(3)
    with c1:
        extraction_max_tokens = st.number_input("提取最大输出长度", min_value=1000, max_value=100000, value=extraction_max_tokens, step=1000)
        extraction_timeout = st.number_input("提取超时秒数", min_value=10, max_value=600, value=extraction_timeout, step=10)
    with c2:
        extraction_request_interval = st.number_input("提取请求间隔", min_value=0.0, max_value=300.0, value=extraction_request_interval, step=0.5)
        extraction_max_attempts = st.number_input(
            "提取最大尝试次数（含首次）",
            min_value=1,
            max_value=11,
            value=extraction_max_attempts,
            step=1,
            help="例如设置为 3 表示最多请求 3 次：首次 1 次，失败后最多再尝试 2 次。",
        )
        extraction_retry_sleep = st.number_input("提取重试等待", min_value=0.0, max_value=300.0, value=extraction_retry_sleep, step=1.0)
    with c3:
        extraction_concurrency = st.number_input(
            "提取并发数",
            min_value=1,
            max_value=100,
            value=int(extraction_concurrency),
            step=1,
            help=f"不同评测人之间可并发；同一评测人内部仍串行，避免 {document_name} 继承关系错乱。",
        )
        extraction_send_enable_thinking = st.checkbox("提取发送enable_thinking字段", value=extraction_send_enable_thinking)
        extraction_enable_thinking = st.checkbox("提取enable_thinking=True", value=extraction_enable_thinking)
        advisor_max_items = st.number_input("建议阶段最多证据条数", min_value=1, max_value=300, value=advisor_max_items, step=1)

    with st.expander("查看当前评测接口参数", expanded=False):
        st.write({
            "提取模型": extraction_model,
            "提取接口": extraction_api_base,
            "评测模型": eval_model,
            "评测接口": eval_api_base,
            "提示词改进模型": advisor_model,
            "提示词改进接口": advisor_api_base,
            "评测并发": min(100, max(1, int(cfg.get("judge_concurrency", 1) or 1))),
            "评测请求间隔": cfg.get("judge_request_interval", 0),
            "评测最大尝试（含首次）": cfg.get("judge_max_retries", 3),
            "评测限流等待": cfg.get("judge_qps_backoff", 12),
            "温度": cfg.get("judge_temperature", 0),
            "top_p": cfg.get("judge_top_p", 1.0),
            "top_k": cfg.get("judge_top_k", None),
            "发送enable_thinking": cfg.get("judge_send_enable_thinking", True),
            "enable_thinking": cfg.get("judge_enable_thinking", False),
        })

st.subheader("3. 启动")
with st.container(border=True):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("计划轮次", int(rounds))
    c2.metric("每轮最多case", "全部" if int(max_cases_per_round) == 0 else int(max_cases_per_round))
    c3.metric("提取并发", int(extraction_concurrency))
    c4.metric("评测并发", min(100, max(1, int(cfg.get("judge_concurrency", 1) or 1))))

    start_disabled = loop_is_running(run_id)
    if start_disabled:
        st.info("这个运行编号正在运行中，不能重复启动。")

    if st.button("启动自动闭环", type="primary", use_container_width=True, disabled=start_disabled):
        if not extraction_prompt_text.strip():
            st.error("初始提取提示词为空。")
            st.stop()
        if not judge_prompt_text.strip():
            st.error("裁判提示词为空。")
            st.stop()

        if uploaded_excel is not None:
            input_excel_path = save_uploaded_file(uploaded_excel, suffix=Path(uploaded_excel.name).suffix)
        else:
            input_excel_path = local_excel_path.strip().strip('"')

        if not input_excel_path:
            st.error("请上传原始对话 Excel，或在可选输入设置里填写本地 Excel 路径。")
            st.stop()
        if not Path(input_excel_path).exists():
            st.error(f"Excel 文件不存在：{input_excel_path}")
            st.stop()

        eval_config = build_eval_config({
            **cfg,
            "mock": mock,
            "judge_model": eval_model,
            "api_base": eval_api_base,
            "api_token": eval_api_token or default_api_token,
        }, mock=mock)
        errors = eval_config.validate()
        if errors:
            st.error("配置错误：\n" + "\n".join([f"- {item}" for item in errors]))
            st.stop()

        loop_config = ClosedLoopConfig(
            run_id=run_id,
            input_excel_path=input_excel_path,
            task_type=loop_task_type,
            sheet_name=resolve_sheet_name(sheet_name_raw),
            reviewer_filter=reviewer_filter,
            rounds=int(rounds),
            chunk_size=int(chunk_size),
            max_cases_per_round=int(max_cases_per_round),
            extraction_model=extraction_model,
            extraction_api_base=extraction_api_base or eval_api_base,
            extraction_api_token=extraction_api_token or eval_api_token or default_api_token,
            extraction_prompt_text=extraction_prompt_text,
            extraction_create_prompt_text=extraction_create_prompt_text,
            extraction_prompt_version=extraction_prompt_version,
            extraction_max_tokens=int(extraction_max_tokens),
            extraction_request_interval=float(extraction_request_interval),
            extraction_max_retries=max(0, int(extraction_max_attempts) - 1),
            extraction_retry_sleep=float(extraction_retry_sleep),
            extraction_timeout=int(extraction_timeout),
            extraction_concurrency=int(extraction_concurrency),
            extraction_send_enable_thinking=bool(extraction_send_enable_thinking),
            extraction_enable_thinking=bool(extraction_enable_thinking),
            judge_prompt_file=selected_judge_prompt,
            judge_prompt_text=judge_prompt_text,
            judge_prompt_version=infer_prompt_version(selected_judge_prompt),
            advisor_max_items=int(advisor_max_items),
            advisor_model=advisor_model or eval_model,
            advisor_api_base=advisor_api_base or eval_api_base,
            advisor_api_token=advisor_api_token or eval_api_token or default_api_token,
            eval_config=eval_config,
        )

        thread = threading.Thread(target=run_closed_loop, args=(loop_config,), daemon=True, name=f"closed-loop-{run_id}")
        thread.start()
        thread_registry[run_id] = thread
        st.session_state.closed_loop_last_run_id = run_id
        st.success(f"已启动闭环实验：{run_id}")
        st.rerun()

st.subheader("4. 运行状态")
run_ids = list_loop_run_ids()
if run_id and run_id not in run_ids:
    run_ids = [run_id] + run_ids

if run_ids:
    default_index = run_ids.index(st.session_state.get("closed_loop_last_run_id", run_id)) if st.session_state.get("closed_loop_last_run_id", run_id) in run_ids else 0
    selected_run_id = st.selectbox("查看运行", run_ids, index=default_index)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("刷新状态", use_container_width=True):
            st.rerun()
    with c2:
        auto_refresh = st.checkbox(
            "运行中每10秒自动刷新进度区",
            value=True,
            help="只刷新下面的运行状态区域，不刷新整个页面。",
        )
    if auto_refresh and read_loop_state(selected_run_id).get("status") == "running":
        render_state_auto(selected_run_id)
    else:
        render_state(selected_run_id)
else:
    st.info("暂无闭环运行记录。")
