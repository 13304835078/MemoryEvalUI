from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from statistics import mean

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.eval_runner import EvalRunner
from src.schema import Case, EvalResult, TaskType
from src.ui.config_store import build_eval_config, load_config
from src.ui.data_service import dataframe_to_excel_bytes, list_case_files, load_cases
from src.ui.prompt_editor import (
    get_default_extraction_prompt_file,
    get_default_prompt_file,
    infer_prompt_version,
    list_extraction_prompt_files,
    list_prompt_files,
    load_prompt,
    prompt_text_hash,
)


NO_EXTRACTION_PROMPT = "不使用提取规则辅助评测"


st.title("裁判提示词 A/B 对比")
st.caption("单模型绝对评测对比：同一批 case、同一个裁判模型配置，只替换裁判提示词 A/B。")

if "ui_config" not in st.session_state:
    st.session_state.ui_config = load_config()
if "judge_ab_single_result" not in st.session_state:
    st.session_state.judge_ab_single_result = None


def get_task_choices() -> list[str]:
    return [item.value for item in TaskType if item.value != "raw_dialogue"]


def make_rate_limiter(config, concurrency: int):
    configured_interval = float(config.judge_request_interval or 0.0) if not config.mock else 0.0
    interval = configured_interval
    if concurrency > 1 and not config.mock:
        interval = max(interval, float(config.judge_qps_backoff or 0.0))
    lock = threading.Lock()
    next_call_at = {"value": time.monotonic()}

    def wait() -> None:
        if interval <= 0:
            return
        with lock:
            now = time.monotonic()
            wait_seconds = max(0.0, next_call_at["value"] - now)
            next_call_at["value"] = max(now, next_call_at["value"]) + interval
        if wait_seconds > 0:
            time.sleep(wait_seconds)

    return wait, configured_interval, interval


def evaluate_prompt(
    label: str,
    prompt_file: str,
    cases: list[Case],
    task_type: str,
    config,
    extraction_prompt_text: str,
    extraction_prompt_version: str,
    extraction_prompt_hash: str,
    max_workers: int,
) -> tuple[list[EvalResult], dict]:
    prompt_version = infer_prompt_version(prompt_file)
    runner = EvalRunner(
        config=config,
        task_type=TaskType(task_type),
        prompt_file=prompt_file,
        judge_prompt_version=prompt_version,
        extraction_prompt_text=extraction_prompt_text,
        extraction_prompt_version=extraction_prompt_version,
        extraction_prompt_hash=extraction_prompt_hash,
    )
    wait_for_rate_limit, configured_interval, effective_interval = make_rate_limiter(config, max_workers)
    if hasattr(runner.judge_client, "rate_limit_wait_callback"):
        runner.judge_client.rate_limit_wait_callback = wait_for_rate_limit
    rows_by_index: dict[int, EvalResult] = {}
    progress = st.progress(
        0.0,
        text=f"提示词 {label}: 0/{len(cases)}，实际请求启动间隔 {effective_interval:.1f}s",
    )

    def evaluate_one(index: int, case: Case) -> tuple[int, EvalResult]:
        wait_for_rate_limit()
        return index, runner.evaluate_one(case)

    if cases:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(cases))) as executor:
            future_map = {executor.submit(evaluate_one, idx, case): idx for idx, case in enumerate(cases)}
            completed = 0
            for future in as_completed(future_map):
                idx, result = future.result()
                rows_by_index[idx] = result
                completed += 1
                progress.progress(
                    completed / len(cases),
                    text=f"提示词 {label}: {completed}/{len(cases)}，实际请求启动间隔 {effective_interval:.1f}s",
                )

    stats = {
        "prompt_file": prompt_file,
        "prompt_version": prompt_version,
        "configured_request_interval": configured_interval,
        "effective_request_interval": effective_interval,
    }
    return [rows_by_index[i] for i in sorted(rows_by_index)], stats


def summarize_results(results: list[EvalResult]) -> dict:
    if not results:
        return {
            "total": 0,
            "avg_score": 0.0,
            "fatal_count": 0,
            "tagged_count": 0,
            "diagnostics_count": 0,
        }
    return {
        "total": len(results),
        "avg_score": round(mean(float(item.score_total or 0) for item in results), 4),
        "fatal_count": sum(1 for item in results if item.fatal_error),
        "tagged_count": sum(1 for item in results if item.error_tags),
        "diagnostics_count": sum(len(item.diagnostics or []) for item in results),
    }


def avg_dimension_scores(results: list[EvalResult]) -> dict[str, float]:
    dims = sorted({dim for result in results for dim in (result.scores or {})})
    rows: dict[str, float] = {}
    for dim in dims:
        values = [float((result.scores or {}).get(dim, 0.0) or 0.0) for result in results]
        rows[dim] = round(mean(values), 4) if values else 0.0
    return rows


def result_table(results_a: list[EvalResult], results_b: list[EvalResult]) -> pd.DataFrame:
    rows = []
    for a, b in zip(results_a, results_b):
        rows.append({
            "case_id": a.case_id,
            "model_name": a.model_name,
            "candidate_prompt_version": a.prompt_version,
            "score_A": a.score_total,
            "score_B": b.score_total,
            "score_delta_B_minus_A": round(float(b.score_total or 0) - float(a.score_total or 0), 4),
            "fatal_A": a.fatal_error,
            "fatal_B": b.fatal_error,
            "error_tags_A": ", ".join(a.error_tags or []),
            "error_tags_B": ", ".join(b.error_tags or []),
            "diagnostics_A": len(a.diagnostics or []),
            "diagnostics_B": len(b.diagnostics or []),
            "comment_A": a.comment,
            "comment_B": b.comment,
            "rule_refs_A": "; ".join(a.rule_refs or []),
            "rule_refs_B": "; ".join(b.rule_refs or []),
            "evidence_refs_A": "; ".join(a.evidence_refs or []),
            "evidence_refs_B": "; ".join(b.evidence_refs or []),
        })
    return pd.DataFrame(rows)


with st.expander("使用说明", expanded=True):
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
    st.warning("没有可用 case 文件，请先到“数据输入”或“记忆提取”页面生成。")
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
        use_container_width=True,
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
    left, right = st.columns(2)
    with left:
        st.text_area("裁判提示词 A", value=load_prompt(prompt_a), height=300, disabled=True)
    with right:
        st.text_area("裁判提示词 B", value=load_prompt(prompt_b), height=300, disabled=True)
    if extraction_prompt_text:
        st.text_area("共用提取提示词规则", value=extraction_prompt_text, height=220, disabled=True)


st.subheader("3. 运行配置")
cfg = dict(st.session_state.ui_config)
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
        "最大重试": cfg.get("judge_max_retries", 3),
        "提取提示词版本": extraction_prompt_version or "未使用",
        "提取提示词 Hash": extraction_prompt_hash[:12] if extraction_prompt_hash else "",
    })

if st.button("开始 A/B 对比", type="primary", use_container_width=True, disabled=not bool(run_cases)):
    config = build_eval_config(cfg, mock=mock)
    errors = config.validate()
    if errors:
        st.error("配置错误：\n" + "\n".join([f"- {item}" for item in errors]))
        st.stop()

    max_workers = min(100, max(1, int(cfg.get("judge_concurrency", 1) or 1)))
    with st.spinner("正在运行裁判提示词 A..."):
        results_a, stats_a = evaluate_prompt(
            "A",
            prompt_a,
            run_cases,
            task_type,
            config,
            extraction_prompt_text,
            extraction_prompt_version,
            extraction_prompt_hash,
            max_workers,
        )
    with st.spinner("正在运行裁判提示词 B..."):
        results_b, stats_b = evaluate_prompt(
            "B",
            prompt_b,
            run_cases,
            task_type,
            config,
            extraction_prompt_text,
            extraction_prompt_version,
            extraction_prompt_hash,
            max_workers,
        )

    st.session_state.judge_ab_single_result = {
        "results_a": results_a,
        "results_b": results_b,
        "stats_a": stats_a,
        "stats_b": stats_b,
        "case_file": selected_case_path,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


result = st.session_state.judge_ab_single_result
if result:
    st.divider()
    st.subheader("4. 对比结果")
    results_a = result["results_a"]
    results_b = result["results_b"]
    summary_a = summarize_results(results_a)
    summary_b = summarize_results(results_b)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("A 平均分", f"{summary_a['avg_score']:.4f}")
    c2.metric("B 平均分", f"{summary_b['avg_score']:.4f}")
    c3.metric("B-A 平均分差", f"{summary_b['avg_score'] - summary_a['avg_score']:.4f}")
    c4.metric("样本数", summary_a["total"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("A fatal", summary_a["fatal_count"])
    c2.metric("B fatal", summary_b["fatal_count"])
    c3.metric("A 有标签样本", summary_a["tagged_count"])
    c4.metric("B 有标签样本", summary_b["tagged_count"])

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
        st.dataframe(pd.DataFrame(dim_rows), use_container_width=True, hide_index=True)

    table = result_table(results_a, results_b)
    st.markdown("**样本明细**")
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.download_button(
        "下载 A/B 对比结果",
        data=dataframe_to_excel_bytes(table),
        file_name=f"judge_prompt_ab_single_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    with st.expander("运行元信息", expanded=False):
        st.json({
            "case_file": result.get("case_file", ""),
            "created_at": result.get("created_at", ""),
            "prompt_A": result.get("stats_a", {}),
            "prompt_B": result.get("stats_b", {}),
        })
