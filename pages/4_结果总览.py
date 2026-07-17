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

from src.schema import cases_from_jsonl
from src.eval.metrics import compute_aggregations, summarize_by_field, TAG_LABELS, DIM_LABELS
from src.eval.result_status import (
    EVAL_STATUS_SUCCESS,
    STATUS_LABELS,
    result_evaluation_status,
    result_is_score_eligible,
)
from src.eval.run_quality import compute_run_quality
from src.eval.stability import compare_eval_stability
from src.ui.data_service import (
    RESULTS_DIR,
    eval_result_resume_key,
    eval_result_row_key,
    list_case_files,
    list_result_files,
    load_results,
    load_results_bytes,
    merge_cases_results,
    results_to_dataframe,
    dataframe_to_excel_bytes,
)
from src.persistence import atomic_write_bytes
from src.ui.rule_ref_validation import (
    rule_ref_validation_rows,
    summarize_rule_ref_validation,
    validate_result_rule_refs,
)
from src.ui.next_actions import NextAction, render_next_actions
from src.ui.result_triage import result_navigation_key, triage_result_rows
from src.ui.theme import render_page_header
from src.ui.workspace_context import render_workspace_context, summarize_values


render_page_header("结果总览", "汇总评分、错误分布、规则引用质量与跨轮稳定性。", category="评测工作流")


def _safe_upload_name(name: str) -> str:
    stem = Path(name).stem or "uploaded_results"
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)
    return safe[:80] or "uploaded_results"

if "cases" not in st.session_state:
    st.session_state.cases = []

if "results" not in st.session_state:
    st.session_state.results = []


loader_panel = st.sidebar.expander("加载结果与样本", expanded=False)
loader_panel.caption(f"读取目录：{RESULTS_DIR}")
loader_panel.caption("自动识别该目录第一层的 JSONL、CSV 和 Excel 结果。")

result_files = list_result_files()
if result_files:
    result_labels = [Path(f).name for f in result_files]
    selected_result_label = loader_panel.selectbox("结果文件", result_labels)
    selected_result_path = result_files[result_labels.index(selected_result_label)]

    if loader_panel.button("加载结果文件", width="stretch"):
        st.session_state.results = load_results(selected_result_path)
        st.session_state.results_file = selected_result_path
        st.rerun()
else:
    loader_panel.warning(f"{RESULTS_DIR} 下暂无 JSONL、CSV 或 Excel 结果文件。")

uploaded_result = loader_panel.file_uploader(
    "或直接上传结果文件",
    type=["jsonl", "csv", "xlsx"],
    key="overview_result_upload",
    help="支持执行评测的 JSONL，以及结果总览导出的 CSV/Excel。",
)
if uploaded_result is not None and loader_panel.button(
    "加载上传的结果",
    width="stretch",
    key="load_overview_result_upload",
):
    try:
        st.session_state.results = load_results_bytes(uploaded_result.getvalue(), uploaded_result.name)
        st.session_state.results_file = uploaded_result.name
        st.rerun()
    except Exception as exc:
        loader_panel.error(f"结果文件解析失败：{exc}")

case_files = list_case_files()
if case_files:
    case_labels = [Path(f).name for f in case_files]
    selected_case_label = loader_panel.selectbox("样本文件，可选", ["不加载"] + case_labels)

    if selected_case_label != "不加载" and loader_panel.button("加载样本文件", width="stretch"):
        selected_case_path = case_files[case_labels.index(selected_case_label)]
        st.session_state.cases = cases_from_jsonl(selected_case_path)
        st.session_state.cases_file = selected_case_path
        st.rerun()


results = st.session_state.get("results", [])
cases = st.session_state.get("cases", [])
missed_cases = st.session_state.get("missed_cases", [])

if not results:
    st.warning("暂无评测结果，请先到「执行评测」页运行，或在侧边栏加载结果文件。")
    st.stop()


base_df = merge_cases_results(cases, results) if cases else results_to_dataframe(results)
rule_validation_by_key = {
    eval_result_resume_key(result): validate_result_rule_refs(result)
    for result in results
}
result_index_by_key = {
    eval_result_resume_key(result): index
    for index, result in enumerate(results)
}


def _row_rule_report(row) -> dict:
    return rule_validation_by_key.get(eval_result_row_key(row), {})


if not base_df.empty:
    row_rule_reports = base_df.apply(_row_rule_report, axis=1)
    base_df["rule_ref_status"] = row_rule_reports.apply(lambda report: report.get("status_label", ""))
    base_df["rule_ref_invalid_refs"] = row_rule_reports.apply(
        lambda report: "; ".join(report.get("invalid_refs", []) or [])
    )
    base_df["rule_ref_raw_invalid_refs"] = row_rule_reports.apply(
        lambda report: "; ".join(report.get("raw_invalid_refs", []) or [])
    )
    base_df["_result_index"] = base_df.apply(
        lambda row: result_index_by_key.get(eval_result_row_key(row), -1),
        axis=1,
    )

render_workspace_context(
    task_type=summarize_values(result.task_type for result in results),
    case_count=len(results),
    cases_file=st.session_state.get("results_file", ""),
    model_name=summarize_values(result.model_name for result in results),
    judge_prompt=summarize_values(result.judge_prompt_version for result in results),
    extraction_prompt=summarize_values(
        (result.extraction_prompt_version for result in results),
        empty="未使用",
    ),
)

st.subheader("筛选")

col1, col2, col3, col4 = st.columns(4)

df = base_df.copy()

with col1:
    models = ["全部"] + sorted([x for x in df.get("model_name", pd.Series()).dropna().unique()])
    model_filter = st.selectbox("被评测模型", models)
    if model_filter != "全部":
        df = df[df["model_name"] == model_filter]

with col2:
    prompts = ["全部"] + sorted([x for x in df.get("prompt_version", pd.Series()).dropna().unique()])
    prompt_filter = st.selectbox("生成提示词", prompts)
    if prompt_filter != "全部":
        df = df[df["prompt_version"] == prompt_filter]

with col3:
    status_filter = st.selectbox("评测状态", ["全部", "评分成功", "运行失败", "严重质量错误"])
    if status_filter == "评分成功":
        df = df[df["score_eligible"] == True]
    elif status_filter == "运行失败":
        df = df[df["score_eligible"] == False]
    elif status_filter == "严重质量错误":
        df = df[(df["score_eligible"] == True) & (df["fatal_error"] == True)]

with col4:
    min_score = st.slider("最低总分", min_value=0.0, max_value=5.0, value=0.0, step=0.1)
    df = df[(df["score_eligible"] == False) | (df["score_total"] >= min_score)]


st.divider()

filtered_keys = {eval_result_row_key(row) for _, row in df.iterrows()}
filtered_results = [r for r in results if eval_result_resume_key(r) in filtered_keys]
if df.empty:
    st.info("当前筛选条件下没有结果，请调整上方筛选项。")
    st.stop()
if not filtered_results:
    st.error(
        "结果关联异常：筛选表中存在记录，但无法匹配原始评测结果。"
        "请重新加载结果文件；如果仍然出现，请保留该结果文件用于排查。"
    )
    st.stop()

stats = compute_aggregations(filtered_results)
filtered_case_ids = {result.case_id for result in filtered_results}
filtered_cases = [case for case in cases if case.case_id in filtered_case_ids]
run_quality = compute_run_quality(filtered_results, cases=filtered_cases, missed_cases=missed_cases)
filtered_rule_reports = [
    rule_validation_by_key.get(eval_result_resume_key(result), {})
    for result in filtered_results
]
rule_summary = summarize_rule_ref_validation(filtered_rule_reports)

with st.expander("统计口径说明", expanded=False):
    st.markdown(
        """
- **运行失败**：API、网络、超时或 Judge JSON 解析失败。单独计数，不计入平均分，也不当作 0 分。
- **严重质量错误**：Judge 已成功评分，但判定候选存在严重内容问题；仍属于有效质量评分。
- **条件平均分**：只统计 Judge 成功评分的样本。
- **端到端分数**：在条件评分基础上，把“提取调用成功但没有可用正文”作为提取质量失败计入；接口失败仍不计 0 分。
- 只要仍有运行失败，结果就标记为“不完整”，不能据此自动替换提示词。
        """
    )

st.markdown("**运行有效性**")
m1, m2, m3, m4 = st.columns(4)
m1.metric("结果记录", stats.get("total_cases", 0))
m2.metric("成功评分", stats.get("scored_cases", 0))
m3.metric("Judge 运行失败", stats.get("judge_failures", 0))
m4.metric("评分覆盖率", f"{stats.get('score_coverage', 0) * 100:.1f}%")
if stats.get("run_complete"):
    st.success("当前筛选范围内 Judge 结果完整。")
else:
    st.warning("当前结果包含未评分项；平均分是条件平均分，不能代表完整运行。建议先重跑失败项。")

st.markdown("**质量结果（仅成功评分）**")
q1, q2, q3, q4 = st.columns(4)
q1.metric("条件平均分", f"{stats.get('avg_score_total', 0):.2f}/5")
q2.metric("严重质量错误", stats.get("fatal_errors", 0))
q3.metric("严重质量错误率", f"{stats.get('fatal_rate', 0) * 100:.1f}%")
q4.metric("错误标签次数", sum(count for _, count in stats.get("error_tags", [])))

if cases or missed_cases:
    st.markdown("**提取到评测的端到端完整度**")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("提取覆盖率", f"{run_quality.get('extraction_coverage', 0) * 100:.1f}%")
    e2.metric("提取质量失败", run_quality.get("extraction_quality_failures", 0))
    e3.metric("提取接口失败", run_quality.get("extraction_infrastructure_failures", 0))
    e4.metric("端到端分数", f"{run_quality.get('end_to_end_score', 0):.2f}/5")

runtime_failure_results = [item for item in filtered_results if not result_is_score_eligible(item)]
if runtime_failure_results:
    failure_status_counts = {
        status: count
        for status, count in stats.get("evaluation_statuses", [])
        if status != EVAL_STATUS_SUCCESS and count
    }
    if failure_status_counts:
        st.caption(
            "失败构成："
            + "；".join(
                f"{STATUS_LABELS.get(status, status)} {count} 条"
                for status, count in failure_status_counts.items()
            )
        )
    with st.expander(f"查看 Judge 运行失败明细（{len(runtime_failure_results)} 条）", expanded=True):
        st.dataframe(pd.DataFrame([{
            "样本编号": item.case_id,
            "评测状态": STATUS_LABELS.get(result_evaluation_status(item), result_evaluation_status(item)),
            "技术类型": item.failure_type or "-",
            "失败信息": item.failure_message or item.raw_response or item.comment,
        } for item in runtime_failure_results]), width="stretch", hide_index=True)

st.subheader("优先处理")
priority_threshold = st.number_input(
    "低分阈值",
    min_value=0.0,
    max_value=5.0,
    value=4.0,
    step=0.1,
    help="低于该分数的已评分样本会进入优先处理队列。运行失败、严重质量错误和规则引用异常始终进入。",
    key="overview_priority_threshold",
)
filtered_rule_statuses = {
    eval_result_resume_key(result): validate_result_rule_refs(result).get("status", "")
    for result in filtered_results
}
priority_rows = triage_result_rows(
    filtered_results,
    rule_status_by_key=filtered_rule_statuses,
    low_score_threshold=float(priority_threshold),
)
if priority_rows:
    priority_df = pd.DataFrame(priority_rows)
    p0_count = sum("P0" in str(row.get("优先级/原因", "")) for row in priority_rows)
    p1_count = sum("P1" in str(row.get("优先级/原因", "")) for row in priority_rows)
    p2_count = sum("P2" in str(row.get("优先级/原因", "")) for row in priority_rows)
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("待处理样本", len(priority_rows))
    p2.metric("P0 运行失败", p0_count)
    p3.metric("P1 质量/低分/引用", p1_count)
    p4.metric("P2 有诊断", p2_count)
    st.caption("选择一行会直接进入对应样本详情；返回后会保留本页筛选状态。")
    priority_event = st.dataframe(
        priority_df,
        width="stretch",
        hide_index=True,
        column_config={"_result_index": None},
        on_select="rerun",
        selection_mode="single-row",
        key="priority_result_table",
    )
    selected_priority_rows = list(getattr(getattr(priority_event, "selection", None), "rows", []) or [])
    if selected_priority_rows:
        selected_row = priority_df.iloc[selected_priority_rows[0]]
        selected_result = filtered_results[int(selected_row["_result_index"])]
        selection_signature = ("priority", *result_navigation_key(selected_result))
        if st.session_state.get("overview_handled_selection") != selection_signature:
            st.session_state.overview_handled_selection = selection_signature
            st.session_state.detail_open_key = result_navigation_key(selected_result)
            st.session_state.detail_return_page = "结果总览"
            st.switch_page("pages/5_样本详情.py")
else:
    st.success("当前筛选范围内没有需要优先处理的样本。")

st.subheader("规则引用校验")
st.caption("校验 rule_refs、diagnostics.rule_refs、comment 和原始 Judge 输出中的规则引用是否真实存在于当前提取提示词。")
r1, r2, r3, r4, r5 = st.columns(5)
r1.metric("已检查", rule_summary.get("total", 0) - rule_summary.get("not_checked", 0))
r2.metric("通过", rule_summary.get("ok", 0))
r3.metric("疑似幻觉引用", rule_summary.get("invalid", 0))
r4.metric("缺少规则引用", rule_summary.get("missing", 0))
r5.metric("未校验", rule_summary.get("not_checked", 0))

rule_issue_rows = []
for result in filtered_results:
    report = rule_validation_by_key.get(eval_result_resume_key(result), {})
    for row in rule_ref_validation_rows(report):
        rule_issue_rows.append({
            "case_id": result.case_id,
            "model_name": result.model_name,
            "prompt_version": result.prompt_version,
            "judge_prompt_version": result.judge_prompt_version,
            "extraction_prompt_version": result.extraction_prompt_version,
            "rule_ref_status": report.get("status_label", ""),
            **row,
        })

if rule_issue_rows:
    with st.expander("规则引用异常明细", expanded=rule_summary.get("invalid", 0) > 0):
        issue_df = pd.DataFrame(rule_issue_rows)
        st.dataframe(issue_df.head(300), width="stretch", hide_index=True)
        st.download_button(
            "导出规则引用异常 CSV",
            data=issue_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="rule_ref_issues.csv",
            mime="text/csv",
            width="stretch",
        )
else:
    st.info("当前筛选范围内没有规则引用异常明细。")

st.subheader("维度均分")

dim_avgs = stats.get("avg_dimension_scores", {})
if dim_avgs:
    dim_df = pd.DataFrame([
        {"维度": DIM_LABELS.get(k, k), "均分": v}
        for k, v in dim_avgs.items()
    ])
    st.bar_chart(dim_df.set_index("维度"))
else:
    st.info("暂无维度分。")

st.subheader("错误标签分布")

tag_counts = stats.get("error_tags", [])
if tag_counts:
    tag_df = pd.DataFrame([
        {"错误标签": TAG_LABELS.get(tag, tag), "次数": count}
        for tag, count in tag_counts
    ])
    st.bar_chart(tag_df.set_index("错误标签"))
else:
    st.info("暂无错误标签。")

st.subheader("稳定性对比")
st.caption(
    "用于比较两次运行是否真的稳定。总分均值相同不代表具体 case、错误标签、扣分维度和引用完全一致。"
)

with st.expander("选择对照结果并生成稳定性报告", expanded=False):
    st.info("温度为 0 时，建议 top_p 保持 1.0；用 top_p 限制候选集合通常不会比贪心解码更稳定，反而可能改变评分尺度。")

    source = st.radio("对照结果来源", ["历史结果文件", "上传结果文件"], horizontal=True)
    key_label = st.radio(
        "样本匹配方式",
        ["只按 case_id", "按 case_id + 被评测模型 + 生成提示词"],
        horizontal=True,
        help="做同数据重复运行稳定性测试时，通常选“只按 case_id”。如果一个结果文件里混了多个模型或生成提示词，再选更严格的匹配方式。",
    )
    key_mode = "case_model_prompt" if key_label.startswith("按 case_id +") else "case_id"
    tolerance = st.number_input(
        "分数完全一致容差",
        min_value=0.0,
        max_value=1.0,
        value=0.01,
        step=0.01,
        help="总分或维度分差值小于等于该值时，计为分数一致。",
    )

    baseline_results = []
    baseline_name = ""

    if source == "历史结果文件":
        compare_files = list_result_files()
        current_result_file = str(st.session_state.get("results_file", "") or "")
        if current_result_file:
            compare_files = [f for f in compare_files if str(f) != current_result_file] + [
                f for f in compare_files if str(f) == current_result_file
            ]
        if compare_files:
            compare_labels = [Path(f).name for f in compare_files]
            selected_compare_label = st.selectbox("选择对照结果文件", compare_labels)
            selected_compare_path = compare_files[compare_labels.index(selected_compare_label)]
            baseline_results = load_results(selected_compare_path)
            baseline_name = selected_compare_label
        else:
            st.warning("data/results 下暂无可对照的结果文件。")
    else:
        uploaded_result = st.file_uploader(
            "上传对照结果",
            type=["jsonl", "csv", "xlsx"],
            help="支持执行评测 JSONL，以及结果总览导出的 CSV/Excel。",
        )
        if uploaded_result is not None:
            uploaded_bytes = uploaded_result.getvalue()
            try:
                baseline_results = load_results_bytes(uploaded_bytes, uploaded_result.name)
                baseline_name = uploaded_result.name
                st.success(f"已读取 {len(baseline_results)} 条对照结果。")
                if st.button("保存上传结果到 data/results", width="stretch"):
                    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                    uploaded_suffix = Path(uploaded_result.name).suffix.lower()
                    saved_name = (
                        f"{_safe_upload_name(uploaded_result.name)}_"
                        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}{uploaded_suffix}"
                    )
                    saved_path = RESULTS_DIR / saved_name
                    atomic_write_bytes(saved_path, uploaded_bytes)
                    st.success(f"已保存：{saved_path}")
            except Exception as exc:
                st.error(f"上传文件解析失败：{exc}")

    if baseline_results:
        comparison = compare_eval_stability(
            filtered_results,
            baseline_results,
            key_mode=key_mode,
            exact_score_tolerance=float(tolerance),
        )
        summary = comparison["summary"]

        st.markdown(f"**对照文件：{baseline_name}**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("可比共同样本", summary.get("common_count", 0), help="已剔除任一侧运行失败或评分配置明确不一致的匹配对。")
        c2.metric(
            "平均总分",
            f"{summary.get('current_avg_score', 0):.2f} / {summary.get('baseline_avg_score', 0):.2f}",
            help="前者为当前筛选结果，后者为对照结果。",
        )
        c3.metric("总分平均绝对差", f"{summary.get('avg_total_abs_delta', 0):.4f}")
        c4.metric("错误标签完全一致率", f"{summary.get('tag_exact_rate', 0) * 100:.1f}%")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("总分完全一致率", f"{summary.get('total_score_exact_rate', 0) * 100:.1f}%")
        c6.metric("不稳定样本数", summary.get("unstable_case_count", 0))
        c7.metric("最大总分差", f"{summary.get('max_total_abs_delta', 0):.4f}")
        c8.metric("诊断 F1", f"{summary.get('avg_diagnostic_f1', 0) * 100:.1f}%")

        x1, x2, x3 = st.columns(3)
        x1.metric("原始匹配样本", summary.get("matched_count", 0))
        x2.metric("运行失败匹配对", summary.get("execution_failure_pair_count", 0))
        x3.metric("评分配置不一致对", summary.get("config_mismatch_pair_count", 0))
        if comparison.get("execution_failure_rows"):
            with st.expander("运行失败匹配对（不参与稳定性）", expanded=True):
                st.dataframe(pd.DataFrame(comparison["execution_failure_rows"]), width="stretch", hide_index=True)
        if comparison.get("config_mismatch_rows"):
            with st.expander("评分配置不一致对（不直接比较）", expanded=True):
                st.dataframe(pd.DataFrame(comparison["config_mismatch_rows"]), width="stretch", hide_index=True)

        if summary.get("common_count", 0) == 0:
            st.warning("当前结果和对照结果没有匹配到共同样本，请检查匹配方式或 case_id。")
        else:
            st.markdown("**分布散度**")
            st.caption("KL 越接近 0 表示分布越接近；JS 散度更稳定、更适合小样本对比。")
            st.dataframe(pd.DataFrame(comparison["distribution_rows"]), width="stretch", hide_index=True)

            st.markdown("**逐维度分数稳定性**")
            st.dataframe(pd.DataFrame(comparison["dimension_rows"]), width="stretch", hide_index=True)

            if comparison["tag_rows"]:
                st.markdown("**错误标签变化**")
                st.dataframe(pd.DataFrame(comparison["tag_rows"]), width="stretch", hide_index=True)

            type_rows = comparison.get("instability_type_rows", [])
            selected_types = []
            if type_rows:
                st.markdown("**不稳定类型统计**")
                st.caption("同一个样本可能同时属于多个结构化变化类型。评语措辞变化单独展示，但不计入质量不稳定。")
                st.dataframe(pd.DataFrame(type_rows), width="stretch", hide_index=True)
                selected_types = st.multiselect(
                    "只看这些不稳定类型",
                    [row["不稳定类型"] for row in type_rows],
                    default=[],
                    help="不选择时展示全部不稳定样本。",
                )

                with st.expander("不稳定类型说明", expanded=False):
                    st.markdown(
                        """
- **总分变化**：加权总分发生变化。
- **维度分变化**：总分可能相同，但 correctness/coverage 等维度分有变化。
- **错误标签变化**：error_tags 不一致。
- **诊断变化**：按维度、严重程度和规范化引用比较 diagnostics；自由文本理由不参与签名。明细同时提供诊断精确率、召回率和 F1。
- **规则引用变化**：rule_refs 指向的提取规则不一致。
- **证据引用变化**：evidence_refs 指向的事实证据不一致。
- **输出引用变化**：output_refs 指向的候选输出片段不一致，例如新 USER.md 或新 MEMORY.md。
- **评语变化**：comment 文本不一致，仅作为表达波动信息，不单独判为质量不稳定。
                        """.strip()
                    )

            diff_df = pd.DataFrame(comparison["diff_rows"])
            if not diff_df.empty:
                unstable_df = diff_df[diff_df["不稳定类型"] != "完全一致"].copy()
                if selected_types:
                    selected_type_set = set(selected_types)
                    unstable_df = unstable_df[
                        unstable_df["不稳定类型"].apply(
                            lambda value: bool(selected_type_set & set(str(value).split("；")))
                        )
                    ]
                st.markdown("**不稳定样本明细（按类型拆分）**")
                st.caption("评分稳定但结构化证据或引用变化的样本会保留；评语变化作为附加列展示。")
                st.dataframe(unstable_df.head(200), width="stretch", hide_index=True)
                st.download_button(
                    "导出稳定性差异 CSV",
                    data=unstable_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="stability_diff.csv",
                    mime="text/csv",
                    width="stretch",
                )

        if comparison["current_only_keys"] or comparison["baseline_only_keys"]:
            with st.expander("未匹配样本", expanded=False):
                st.write({
                    "当前独有数量": summary.get("current_only_count", 0),
                    "对照独有数量": summary.get("baseline_only_count", 0),
                    "当前独有样本": comparison["current_only_keys"][:200],
                    "对照独有样本": comparison["baseline_only_keys"][:200],
                })

st.subheader("分组对比")

g1, g2 = st.columns(2)
with g1:
    by_model = pd.DataFrame(summarize_by_field(filtered_results, "model_name"))
    st.markdown("**按被评测模型**")
    st.dataframe(by_model, width="stretch", hide_index=True)

with g2:
    by_prompt = pd.DataFrame(summarize_by_field(filtered_results, "prompt_version"))
    st.markdown("**按生成提示词**")
    st.dataframe(by_prompt, width="stretch", hide_index=True)

st.subheader("结果明细")

st.caption("选择一行可直接打开样本详情。")
detail_event = st.dataframe(
    df,
    width="stretch",
    hide_index=True,
    column_config={"_result_index": None},
    on_select="rerun",
    selection_mode="single-row",
    key="overview_result_table",
)
selected_detail_rows = list(getattr(getattr(detail_event, "selection", None), "rows", []) or [])
if selected_detail_rows:
    selected_result_index = int(df.iloc[selected_detail_rows[0]].get("_result_index", -1))
    if 0 <= selected_result_index < len(results):
        selected_result = results[selected_result_index]
        selection_signature = ("detail", *result_navigation_key(selected_result))
        if st.session_state.get("overview_handled_selection") != selection_signature:
            st.session_state.overview_handled_selection = selection_signature
            st.session_state.detail_open_key = result_navigation_key(selected_result)
            st.session_state.detail_return_page = "结果总览"
            st.switch_page("pages/5_样本详情.py")

export_df = df.drop(columns=["_result_index"], errors="ignore")
csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")
excel_bytes = dataframe_to_excel_bytes(export_df)

c1, c2 = st.columns(2)
with c1:
    st.download_button(
        "导出 CSV",
        data=csv_bytes,
        file_name="eval_results_filtered.csv",
        mime="text/csv",
        width="stretch",
    )

with c2:
    st.download_button(
        "导出 Excel",
        data=excel_bytes,
        file_name="eval_results_filtered.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )

render_next_actions([
    NextAction("pages/5_样本详情.py", "查看样本详情", ":material/fact_check:", "复核单条评分与证据"),
    NextAction("pages/7_提示词改进建议.py", "生成提示词建议", ":material/edit_note:", "基于当前评测结果生成候选修改"),
])
