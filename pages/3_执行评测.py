from __future__ import annotations

import sys
import threading
from datetime import datetime
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.schema import (
    EVALUATABLE_TASK_TYPES,
    TASK_TYPE_LABELS,
    TaskType,
    cases_from_jsonl,
)
from src.ui.config_store import load_config, build_eval_config
from src.ui.eval_job_runner import (
    EvalJobConfig,
    RESUME_SKIP_ALL,
    RESUME_STRATEGIES,
    eval_job_is_running,
    eval_job_is_stale,
    list_eval_job_ids,
    load_job_results_from_state,
    mark_eval_job_interrupted,
    read_eval_job_state,
    request_eval_stop,
    run_eval_job,
)
from src.ui.prompt_editor import (
    get_default_prompt_file,
    get_default_extraction_prompt_file,
    infer_prompt_version,
    list_extraction_prompt_files,
    list_prompt_files,
    load_prompt,
    prompt_text_hash,
)
from src.ui.data_service import (
    list_case_files,
    list_result_files,
    load_results,
    results_to_dataframe,
    RESULTS_DIR,
)
from src.ui.components import render_state_file_notice


def get_eval_task_choices() -> list[str]:
    return [task.value for task in EVALUATABLE_TASK_TYPES]


NO_EXTRACTION_PROMPT = "不使用提取规则辅助评测"


def find_extraction_prompt_file(version: str = "", prompt_hash: str = "") -> str:
    for filename in list_extraction_prompt_files():
        text = load_prompt(filename, prompt_kind="extraction")
        if prompt_hash and prompt_text_hash(text) == prompt_hash:
            return filename
        if version and infer_prompt_version(filename) == version:
            return filename
    return ""


def ensure_eval_job_threads() -> dict[str, threading.Thread]:
    if "eval_job_threads" not in st.session_state:
        st.session_state.eval_job_threads = {}
    return st.session_state.eval_job_threads


def render_eval_job_state(job_id: str) -> None:
    state = read_eval_job_state(job_id)
    if eval_job_is_stale(state):
        state = mark_eval_job_interrupted(job_id)
    if not state:
        return
    render_state_file_notice(state)

    total = int(state.get("total", 0) or 0)
    done = int(state.get("done", 0) or 0)
    status = str(state.get("status", ""))
    progress_value = done / total if total else 0.0

    st.subheader("后台评测进度")
    st.progress(progress_value)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("状态", status or "-")
    c2.metric("进度", f"{done}/{total}")
    c3.metric("新增评测", int(state.get("evaluated", 0) or 0))
    c4.metric("跳过", int(state.get("skipped", 0) or 0))
    c5.metric("严重失败", int(state.get("fatal_count", 0) or 0))

    st.write(state.get("message", ""))
    if state.get("effective_request_interval") is not None:
        st.caption(
            f"实际请求启动间隔：{float(state.get('effective_request_interval') or 0):.1f}s"
            f"（配置请求间隔：{float(state.get('configured_request_interval') or 0):.1f}s）"
        )
    if state.get("output_path"):
        st.caption(f"结果文件：{state.get('output_path')}")

    if status == "running":
        st.info("任务仍在后台运行。切换页面后再回来，进度会从状态文件恢复。")
        if st.button("请求终止评测任务", type="secondary", use_container_width=True, key=f"{job_id}_stop"):
            request_eval_stop(job_id)
            st.warning("已写入终止请求。已发出的单次 Judge 调用无法立即强制中断，会在下一个检查点停止，未开始的样本不会继续提交。")
            st.rerun()
    elif status == "interrupted":
        st.warning("任务状态为已中断。通常是程序关闭或后台线程退出导致；可以重新开始评测，或选择已有结果文件断点续跑。")
    elif status == "stopped":
        st.warning("任务已按请求终止。已完成结果保留在结果文件中，可以断点续跑。")

    if status in {"interrupted", "stopped"}:
        if st.button("载入该任务并从中断处继续", key=f"{job_id}_resume_interrupted", use_container_width=True):
            job_config = state.get("config") if isinstance(state.get("config"), dict) else {}
            extraction_file = find_extraction_prompt_file(
                str(job_config.get("extraction_prompt_version") or ""),
                str(job_config.get("extraction_prompt_hash") or ""),
            )
            st.session_state.resume_prefill = {
                "resume_enabled": True,
                "resume_strategy": RESUME_SKIP_ALL,
                "resume_result_path": state.get("output_path", ""),
                "task_type": job_config.get("task_type", ""),
                "selected_prompt_file": job_config.get("prompt_file", ""),
                "selected_extraction_prompt_file": extraction_file,
                "notice": (
                    "已载入中断任务：会继续写入原结果文件，并默认跳过已有结果。"
                    "接口密钥和临时编辑过的 prompt 文本不会保存在任务状态里，请确认当前配置后再开始。"
                ),
            }
            st.rerun()

    if status in {"completed", "failed", "interrupted", "stopped"}:
        results = load_job_results_from_state(state)
        if results:
            st.session_state.results = results
            st.session_state.results_file = state.get("output_path", "")
            st.dataframe(results_to_dataframe(results).head(50), use_container_width=True, hide_index=True)

    if state.get("traceback"):
        with st.expander("错误堆栈", expanded=True):
            st.code(state.get("traceback", ""), language="text")


@st.fragment(run_every="10s")
def render_eval_job_state_auto(job_id: str) -> None:
    render_eval_job_state(job_id)


def load_cases_from_job_state(job_id: str) -> bool:
    state = read_eval_job_state(job_id)
    config = state.get("config") if isinstance(state.get("config"), dict) else {}
    cases_file = str(config.get("cases_file") or "")
    if not cases_file:
        return False
    path = Path(cases_file)
    if not path.exists():
        return False
    cases = cases_from_jsonl(path)
    st.session_state.cases = cases
    st.session_state.cases_file = cases_file
    return True


def render_eval_job_panel(job_ids: list[str], last_job_id: str = "") -> str:
    if not last_job_id and job_ids:
        last_job_id = job_ids[0]
        st.session_state.eval_job_id = last_job_id

    if job_ids:
        index = job_ids.index(last_job_id) if last_job_id in job_ids else 0
        selected_job_id = st.selectbox("查看后台评测任务", job_ids, index=index)
        if selected_job_id != last_job_id:
            st.session_state.eval_job_id = selected_job_id
            last_job_id = selected_job_id

    if not last_job_id:
        return ""

    state = read_eval_job_state(last_job_id)
    status = str(state.get("status") or "")
    if status == "running":
        auto_refresh = st.checkbox(
            "运行中每10秒自动刷新进度区",
            value=False,
            key=f"{last_job_id}_auto_refresh",
            help="只刷新下面的进度区域，不刷新整个页面，也不会清空已加载样本。",
        )
        if auto_refresh:
            render_eval_job_state_auto(last_job_id)
        else:
            render_eval_job_state(last_job_id)
    else:
        render_eval_job_state(last_job_id)

    return last_job_id


st.title("执行评测")

if "cases" not in st.session_state:
    st.session_state.cases = []

if "cases_file" not in st.session_state:
    st.session_state.cases_file = ""

if "results" not in st.session_state:
    st.session_state.results = []

if "results_file" not in st.session_state:
    st.session_state.results_file = ""

if "ui_config" not in st.session_state:
    st.session_state.ui_config = load_config()

if "task_type" not in st.session_state:
    st.session_state.task_type = "user_md_update"

resume_prefill = st.session_state.pop("resume_prefill", None)
if isinstance(resume_prefill, dict):
    if resume_prefill.get("task_type"):
        st.session_state.task_type = resume_prefill["task_type"]
    if resume_prefill.get("selected_prompt_file"):
        st.session_state.selected_prompt_file = resume_prefill["selected_prompt_file"]
    if resume_prefill.get("selected_extraction_prompt_file"):
        st.session_state.selected_extraction_prompt_file = resume_prefill["selected_extraction_prompt_file"]
    st.session_state.resume_enabled = bool(resume_prefill.get("resume_enabled", True))
    st.session_state.resume_strategy = resume_prefill.get("resume_strategy") or RESUME_SKIP_ALL
    st.session_state.resume_result_path = resume_prefill.get("resume_result_path", "")
    if resume_prefill.get("notice"):
        st.session_state.resume_notice = resume_prefill["notice"]


job_ids = list_eval_job_ids()
last_job_id = st.session_state.get("eval_job_id", "")
if not last_job_id and job_ids:
    last_job_id = job_ids[0]
    st.session_state.eval_job_id = last_job_id

if not st.session_state.get("cases") and last_job_id:
    load_cases_from_job_state(last_job_id)


st.subheader("样本输入")

cases = st.session_state.get("cases", [])
if cases:
    st.success(f"已从当前页面状态加载 {len(cases)} 条样本")
    st.caption(st.session_state.get("cases_file", ""))
else:
    files = list_case_files()
    if not files:
        st.warning("没有可用样本文件，请先到「数据输入」页生成。")
        if job_ids:
            st.divider()
            st.info("当前没有加载样本，但可以先查看已有后台任务进度。")
            render_eval_job_panel(job_ids, last_job_id)
        st.stop()

    labels = [Path(f).name for f in files]
    selected = st.selectbox("选择样本文件", labels)
    selected_path = files[labels.index(selected)]

    if st.button("加载样本文件", use_container_width=True):
        cases = cases_from_jsonl(selected_path)
        st.session_state.cases = cases
        st.session_state.cases_file = selected_path
        if cases:
            st.session_state.task_type = cases[0].task_type.value
        st.success(f"已加载 {len(cases)} 条样本")
        st.rerun()

    if job_ids:
        st.divider()
        st.info("当前页面还没有加载样本；可以先查看已有后台任务进度，也可以加载上面的样本文件后继续操作。")
        render_eval_job_panel(job_ids, last_job_id)

    st.stop()


st.divider()
st.subheader("评测配置")

case_task_types = {
    case.task_type.value if isinstance(case.task_type, TaskType) else str(case.task_type)
    for case in cases
}
if len(case_task_types) == 1:
    inferred_task_type = next(iter(case_task_types))
    if inferred_task_type in get_eval_task_choices():
        st.session_state.task_type = inferred_task_type
elif len(case_task_types) > 1:
    st.error(f"当前样本文件混合了多个任务类型：{sorted(case_task_types)}。请拆分后分别评测。")
    st.stop()

task_choices = get_eval_task_choices()
task_type = st.selectbox(
    "任务类型",
    task_choices,
    index=task_choices.index(st.session_state.task_type)
    if st.session_state.task_type in task_choices else 0,
    format_func=lambda value: TASK_TYPE_LABELS.get(value, value),
)
st.session_state.task_type = task_type

cfg = dict(st.session_state.ui_config)
mock = st.checkbox("模拟模式", value=bool(cfg.get("mock", True)))
limit = st.number_input("评测条数（0 表示全部）", min_value=0, value=0, step=1)
cfg["judge_concurrency"] = st.number_input(
    "并发数",
    min_value=1,
    max_value=100,
    value=min(100, max(1, int(cfg.get("judge_concurrency", 1) or 1))),
    step=1,
    help="并发不会绕过请求间隔；每次请求启动仍会按请求间隔排队，单条请求内部继续使用重试和限流等待。",
)
request_interval_for_warning = float(cfg.get("judge_request_interval", 0) or 0)
if int(cfg.get("judge_concurrency", 1) or 1) > 1 and request_interval_for_warning < 10:
    st.warning(
        "当前并发数大于 1，但请求间隔小于 10 秒。如果接口返回 `QPS limit exceeded, limit:0.10`，"
        "说明最多约 10 秒 1 次请求；建议把“请求间隔”设为 10.5 秒以上，并把并发降到 1-2。"
    )

if st.session_state.get("selected_prompt_task_type") != task_type:
    st.session_state.selected_prompt_file = get_default_prompt_file(task_type)
    st.session_state.selected_extraction_prompt_file = get_default_extraction_prompt_file(task_type)
    st.session_state.selected_prompt_task_type = task_type

prompt_files = list_prompt_files()
default_prompt_file = get_default_prompt_file(task_type)
selected_prompt_file = st.session_state.get("selected_prompt_file", "") or default_prompt_file
if default_prompt_file and default_prompt_file not in prompt_files:
    prompt_files = [default_prompt_file] + prompt_files
if selected_prompt_file and selected_prompt_file not in prompt_files:
    prompt_files = [selected_prompt_file] + prompt_files

selected_prompt_file = st.selectbox(
    "裁判提示词文件",
    prompt_files,
    index=prompt_files.index(selected_prompt_file)
    if selected_prompt_file in prompt_files else 0,
)
st.session_state.selected_prompt_file = selected_prompt_file

use_edited_prompt = st.checkbox(
    "使用配置页中编辑过的裁判提示词文本覆盖文件内容",
    value=False,
)

extraction_prompt_files = list_extraction_prompt_files()
default_extraction_prompt = get_default_extraction_prompt_file(task_type)
selected_extraction_prompt_file = (
    st.session_state.get("selected_extraction_prompt_file", "") or default_extraction_prompt
)
if default_extraction_prompt and default_extraction_prompt not in extraction_prompt_files:
    extraction_prompt_files = [default_extraction_prompt] + extraction_prompt_files
if selected_extraction_prompt_file and selected_extraction_prompt_file not in extraction_prompt_files:
    extraction_prompt_files = [selected_extraction_prompt_file] + extraction_prompt_files
extraction_prompt_options = [NO_EXTRACTION_PROMPT] + extraction_prompt_files
selected_extraction_prompt_file = st.selectbox(
    "提取提示词文件",
    extraction_prompt_options,
    index=extraction_prompt_options.index(selected_extraction_prompt_file)
    if selected_extraction_prompt_file in extraction_prompt_options else 0,
    help="提取提示词只作为规则来源，不作为用户事实来源。未选择时保持旧评测行为。",
)
if selected_extraction_prompt_file != NO_EXTRACTION_PROMPT:
    st.session_state.selected_extraction_prompt_file = selected_extraction_prompt_file

use_edited_extraction_prompt = st.checkbox(
    "使用配置页中编辑过的提取提示词文本覆盖文件内容",
    value=False,
    disabled=selected_extraction_prompt_file == NO_EXTRACTION_PROMPT,
)

if selected_extraction_prompt_file == NO_EXTRACTION_PROMPT:
    extraction_prompt_preview_text = ""
    extraction_prompt_version_preview = ""
    extraction_prompt_hash_preview = ""
    st.caption("未使用提取规则辅助评测，将保持旧评测行为。")
else:
    extraction_prompt_preview_text = (
        st.session_state.get("extraction_prompt_text", "")
        if use_edited_extraction_prompt
        else load_prompt(selected_extraction_prompt_file, prompt_kind="extraction")
    )
    extraction_prompt_version_preview = infer_prompt_version(selected_extraction_prompt_file)
    extraction_prompt_hash_preview = prompt_text_hash(extraction_prompt_preview_text)
    st.caption(
        f"将使用提取规则辅助评测：{extraction_prompt_version_preview}，"
        f"Hash {extraction_prompt_hash_preview[:12] if extraction_prompt_hash_preview else '空'}"
    )

with st.expander("当前接口配置", expanded=False):
    st.write({
        "模拟模式": mock,
        "接口地址": cfg.get("api_base", ""),
        "裁判模型": cfg.get("judge_model", ""),
        "最大输出长度": cfg.get("judge_max_tokens", 2000),
        "超时秒数": cfg.get("judge_timeout", 120),
        "最大尝试次数（含首次）": cfg.get("judge_max_retries", 3),
        "请求间隔": cfg.get("judge_request_interval", 0),
        "并发数": cfg.get("judge_concurrency", 1),
        "限流等待": cfg.get("judge_qps_backoff", 12),
        "温度": cfg.get("judge_temperature", 0),
        "top_p": cfg.get("judge_top_p", 1.0),
        "top_k": cfg.get("judge_top_k", None),
        "发送enable_thinking": cfg.get("judge_send_enable_thinking", True),
        "思考模式": cfg.get("judge_enable_thinking", False),
        "裁判提示词版本": infer_prompt_version(selected_prompt_file),
        "提取提示词版本": extraction_prompt_version_preview or "未使用",
        "提取提示词Hash": extraction_prompt_hash_preview[:12] if extraction_prompt_hash_preview else "",
    })


st.subheader("输出与断点续跑")

resume_files = [
    path for path in list_result_files()
    if Path(path).suffix.lower() == ".jsonl"
]
if "resume_enabled" not in st.session_state:
    st.session_state.resume_enabled = False
if "resume_strategy" not in st.session_state:
    st.session_state.resume_strategy = RESUME_SKIP_ALL
elif st.session_state.resume_strategy not in RESUME_STRATEGIES:
    st.session_state.resume_strategy = RESUME_SKIP_ALL
if "resume_result_path" not in st.session_state:
    st.session_state.resume_result_path = ""

if st.session_state.get("resume_notice"):
    st.info(st.session_state.pop("resume_notice"))

resume_enabled = st.checkbox("从已有结果文件断点续跑", key="resume_enabled")
resume_strategy = st.selectbox(
    "已有结果处理",
    RESUME_STRATEGIES,
    key="resume_strategy",
)
resume_result_path = st.session_state.get("resume_result_path", "")

if resume_enabled:
    if resume_files:
        resume_labels = [Path(f).name for f in resume_files]
        resume_index = 0
        if resume_result_path in resume_files:
            resume_index = resume_files.index(resume_result_path)
        selected_resume_label = st.selectbox("选择已有结果文件", resume_labels, index=resume_index)
        resume_result_path = resume_files[resume_labels.index(selected_resume_label)]
        st.session_state.resume_result_path = resume_result_path
        st.caption(f"将继续写入：{resume_result_path}")
    else:
        st.warning("data/results 下暂无可续跑的 JSONL 结果文件，将新建结果文件。")


st.divider()

if job_ids or last_job_id:
    last_job_id = render_eval_job_panel(job_ids, last_job_id)

active_running = bool(last_job_id and eval_job_is_running(last_job_id))
if active_running:
    st.info("当前已有后台评测任务运行中。进度不会因为切换页面丢失。")

if st.button("开始评测", type="primary", use_container_width=True, disabled=active_running):
    run_cases = cases[:limit] if limit and limit > 0 else cases
    interval = float(cfg.get("judge_request_interval", 0) or 0) if not mock else 0.0
    concurrency = min(100, max(1, int(cfg.get("judge_concurrency", 1) or 1)))
    if interval > 0:
        estimated = len(run_cases) * interval
        st.info(
            f"当前设置了请求间隔 {interval:.1f}s，并发数 {concurrency}。"
            f"请求启动会按间隔排队，预计至少需要约 {estimated/60:.1f} 分钟。"
        )

    config = build_eval_config(cfg, mock=mock)
    errs = config.validate()
    if errs:
        st.error("配置错误：\n" + "\n".join([f"- {e}" for e in errs]))
        st.stop()

    system_prompt_override = st.session_state.get("judge_prompt_text", "") if use_edited_prompt else ""
    judge_prompt_version = infer_prompt_version(selected_prompt_file)

    if selected_extraction_prompt_file == NO_EXTRACTION_PROMPT:
        extraction_prompt_text = ""
        extraction_prompt_version = ""
        extraction_prompt_hash = ""
    else:
        extraction_prompt_text = (
            st.session_state.get("extraction_prompt_text", "")
            if use_edited_extraction_prompt
            else load_prompt(selected_extraction_prompt_file, prompt_kind="extraction")
        )
        extraction_prompt_version = infer_prompt_version(selected_extraction_prompt_file)
        extraction_prompt_hash = prompt_text_hash(extraction_prompt_text)
        if not extraction_prompt_text:
            st.warning("已选择提取提示词文件，但内容为空，本次将按未使用提取规则辅助评测处理。")
            extraction_prompt_version = ""
            extraction_prompt_hash = ""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if resume_enabled and resume_result_path:
        out_path = Path(resume_result_path)
        existing_results = load_results(out_path)
    else:
        out_path = RESULTS_DIR / f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        existing_results = []

    job_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    job_config = EvalJobConfig(
        job_id=job_id,
        task_type=task_type,
        output_path=str(out_path),
        prompt_file=selected_prompt_file,
        judge_prompt_version=judge_prompt_version,
        cases_file=st.session_state.get("cases_file", ""),
        system_prompt_override=system_prompt_override,
        extraction_prompt_text=extraction_prompt_text,
        extraction_prompt_version=extraction_prompt_version,
        extraction_prompt_hash=extraction_prompt_hash,
        resume_strategy=resume_strategy,
        eval_config=config,
    )

    thread = threading.Thread(
        target=run_eval_job,
        args=(job_config, run_cases, existing_results),
        daemon=True,
        name=f"eval-job-{job_id}",
    )
    thread.start()
    ensure_eval_job_threads()[job_id] = thread
    st.session_state.eval_job_id = job_id
    st.session_state.results_file = str(out_path)
    st.success(f"已启动后台评测任务：{job_id}")
    st.rerun()
