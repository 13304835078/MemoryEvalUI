from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.schema import cases_from_jsonl
from src.eval.metrics import compute_aggregations, summarize_by_field, TAG_LABELS, DIM_LABELS
from src.eval.stability import compare_eval_stability, results_from_jsonl_text
from src.ui.data_service import (
    RESULTS_DIR,
    eval_result_resume_key,
    list_case_files,
    list_result_files,
    load_results,
    merge_cases_results,
    results_to_dataframe,
    dataframe_to_excel_bytes,
)
from src.ui.rule_ref_validation import (
    rule_ref_validation_rows,
    summarize_rule_ref_validation,
    validate_result_rule_refs,
)


st.title("结果总览")


def _safe_upload_name(name: str) -> str:
    stem = Path(name).stem or "uploaded_results"
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)
    return safe[:80] or "uploaded_results"

if "cases" not in st.session_state:
    st.session_state.cases = []

if "results" not in st.session_state:
    st.session_state.results = []


st.sidebar.subheader("加载文件")

result_files = list_result_files()
if result_files:
    result_labels = [Path(f).name for f in result_files]
    selected_result_label = st.sidebar.selectbox("结果文件", result_labels)
    selected_result_path = result_files[result_labels.index(selected_result_label)]

    if st.sidebar.button("加载结果文件", use_container_width=True):
        st.session_state.results = load_results(selected_result_path)
        st.session_state.results_file = selected_result_path
        st.rerun()
else:
    st.sidebar.warning("data/results 下暂无结果文件。")

case_files = list_case_files()
if case_files:
    case_labels = [Path(f).name for f in case_files]
    selected_case_label = st.sidebar.selectbox("样本文件，可选", ["不加载"] + case_labels)

    if selected_case_label != "不加载" and st.sidebar.button("加载样本文件", use_container_width=True):
        selected_case_path = case_files[case_labels.index(selected_case_label)]
        st.session_state.cases = cases_from_jsonl(selected_case_path)
        st.session_state.cases_file = selected_case_path
        st.rerun()


results = st.session_state.get("results", [])
cases = st.session_state.get("cases", [])

if not results:
    st.warning("暂无评测结果，请先到「执行评测」页运行，或在侧边栏加载结果文件。")
    st.stop()


base_df = merge_cases_results(cases, results) if cases else results_to_dataframe(results)
rule_validation_by_key = {
    eval_result_resume_key(result): validate_result_rule_refs(result)
    for result in results
}


def _row_rule_report(row) -> dict:
    key = (
        str(row.get("case_id", "")),
        str(row.get("model_name", "unknown") or "unknown"),
        str(row.get("prompt_version", "unknown") or "unknown"),
        str(row.get("judge_model", "") or ""),
        str(row.get("judge_prompt_version", "") or ""),
        str(row.get("extraction_prompt_hash", "") or ""),
    )
    return rule_validation_by_key.get(key, {})


if not base_df.empty:
    row_rule_reports = base_df.apply(_row_rule_report, axis=1)
    base_df["rule_ref_status"] = row_rule_reports.apply(lambda report: report.get("status_label", ""))
    base_df["rule_ref_invalid_refs"] = row_rule_reports.apply(
        lambda report: "; ".join(report.get("invalid_refs", []) or [])
    )
    base_df["rule_ref_raw_invalid_refs"] = row_rule_reports.apply(
        lambda report: "; ".join(report.get("raw_invalid_refs", []) or [])
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
    fatal_filter = st.selectbox("严重失败", ["全部", "否", "是"])
    if fatal_filter == "是":
        df = df[df["fatal_error"] == True]
    elif fatal_filter == "否":
        df = df[df["fatal_error"] == False]

with col4:
    min_score = st.slider("最低总分", min_value=0.0, max_value=5.0, value=0.0, step=0.1)
    df = df[df["score_total"] >= min_score]


st.divider()

filtered_keys = set()
for _, row in df.iterrows():
    filtered_keys.add((
        str(row.get("case_id", "")),
        str(row.get("model_name", "unknown") or "unknown"),
        str(row.get("prompt_version", "unknown") or "unknown"),
        str(row.get("judge_model", "") or ""),
        str(row.get("judge_prompt_version", "") or ""),
        str(row.get("extraction_prompt_hash", "") or ""),
    ))
filtered_results = [r for r in results if eval_result_resume_key(r) in filtered_keys]

stats = compute_aggregations(filtered_results)
filtered_rule_reports = [
    rule_validation_by_key.get(eval_result_resume_key(result), {})
    for result in filtered_results
]
rule_summary = summarize_rule_ref_validation(filtered_rule_reports)

m1, m2, m3, m4 = st.columns(4)
m1.metric("样本数", stats.get("total_cases", 0))
m2.metric("平均总分", f"{stats.get('avg_score_total', 0):.2f}/5")
m3.metric("严重失败数", stats.get("fatal_errors", 0))
m4.metric("严重失败率", f"{stats.get('fatal_rate', 0) * 100:.1f}%")

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
        st.dataframe(issue_df.head(300), use_container_width=True, hide_index=True)
        st.download_button(
            "导出规则引用异常 CSV",
            data=issue_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="rule_ref_issues.csv",
            mime="text/csv",
            use_container_width=True,
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

    source = st.radio("对照结果来源", ["历史结果文件", "上传 JSONL"], horizontal=True)
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
        uploaded_result = st.file_uploader("上传对照结果 JSONL", type=["jsonl", "txt"])
        if uploaded_result is not None:
            uploaded_bytes = uploaded_result.getvalue()
            try:
                baseline_results = results_from_jsonl_text(uploaded_bytes.decode("utf-8-sig"))
                baseline_name = uploaded_result.name
                st.success(f"已读取 {len(baseline_results)} 条对照结果。")
                if st.button("保存上传结果到 data/results", use_container_width=True):
                    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                    saved_name = f"{_safe_upload_name(uploaded_result.name)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
                    saved_path = RESULTS_DIR / saved_name
                    saved_path.write_bytes(uploaded_bytes)
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
        c1.metric("共同样本数", summary.get("common_count", 0))
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
        c8.metric("诊断数平均差", f"{summary.get('avg_diagnostics_count_abs_delta', 0):.3f}")

        if summary.get("common_count", 0) == 0:
            st.warning("当前结果和对照结果没有匹配到共同样本，请检查匹配方式或 case_id。")
        else:
            st.markdown("**分布散度**")
            st.caption("KL 越接近 0 表示分布越接近；JS 散度更稳定、更适合小样本对比。")
            st.dataframe(pd.DataFrame(comparison["distribution_rows"]), use_container_width=True, hide_index=True)

            st.markdown("**逐维度分数稳定性**")
            st.dataframe(pd.DataFrame(comparison["dimension_rows"]), use_container_width=True, hide_index=True)

            if comparison["tag_rows"]:
                st.markdown("**错误标签变化**")
                st.dataframe(pd.DataFrame(comparison["tag_rows"]), use_container_width=True, hide_index=True)

            type_rows = comparison.get("instability_type_rows", [])
            selected_types = []
            if type_rows:
                st.markdown("**不稳定类型统计**")
                st.caption("同一个样本可能同时属于多个类型，例如“证据引用变化；评语变化”。")
                st.dataframe(pd.DataFrame(type_rows), use_container_width=True, hide_index=True)
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
- **诊断变化**：diagnostics 的数量或内容不一致。
- **规则引用变化**：rule_refs 指向的提取规则不一致。
- **证据引用变化**：evidence_refs 指向的事实证据不一致。
- **输出引用变化**：output_refs 指向的新 USER.md 片段不一致。
- **评语变化**：comment 文本不一致；如果只有这一项变化，通常是低风险表达波动。
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
                st.caption("评分稳定但证据、引用或评语变化的样本会保留，便于定位 Judge 解释是否稳定。")
                st.dataframe(unstable_df.head(200), use_container_width=True, hide_index=True)
                st.download_button(
                    "导出稳定性差异 CSV",
                    data=unstable_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="stability_diff.csv",
                    mime="text/csv",
                    use_container_width=True,
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
    st.dataframe(by_model, use_container_width=True, hide_index=True)

with g2:
    by_prompt = pd.DataFrame(summarize_by_field(filtered_results, "prompt_version"))
    st.markdown("**按生成提示词**")
    st.dataframe(by_prompt, use_container_width=True, hide_index=True)

st.subheader("结果明细")

st.dataframe(df, use_container_width=True, hide_index=True)

csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
excel_bytes = dataframe_to_excel_bytes(df)

c1, c2 = st.columns(2)
with c1:
    st.download_button(
        "导出 CSV",
        data=csv_bytes,
        file_name="eval_results_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )

with c2:
    st.download_button(
        "导出 Excel",
        data=excel_bytes,
        file_name="eval_results_filtered.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

st.info("查看单条详情：复制表格中的样本编号，到「样本详情」页选择或输入。")
