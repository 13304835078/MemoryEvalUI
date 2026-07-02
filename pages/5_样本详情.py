from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.schema import VALID_ERROR_TAGS, cases_from_jsonl
from src.ui.data_service import (
    list_case_files,
    list_result_files,
    load_results,
    find_case_for_result,
)
from src.ui.components import (
    render_case_input,
    render_case_metadata,
    render_case_reasoning,
    render_dialogue,
    render_eval_result,
    render_raw_response,
)
from src.ui.review_store import (
    DEFAULT_REVIEW_PATH,
    load_reviews,
    upsert_review,
    review_key,
)


st.title("样本详情")

if "cases" not in st.session_state:
    st.session_state.cases = []

if "results" not in st.session_state:
    st.session_state.results = []


with st.sidebar:
    st.subheader("加载数据")

    result_files = list_result_files()
    if result_files:
        labels = [Path(f).name for f in result_files]
        selected = st.selectbox("结果文件", labels)
        selected_path = result_files[labels.index(selected)]

        if st.button("加载结果", use_container_width=True):
            st.session_state.results = load_results(selected_path)
            st.session_state.results_file = selected_path
            st.rerun()

    case_files = list_case_files()
    if case_files:
        labels = [Path(f).name for f in case_files]
        selected = st.selectbox("样本文件", labels)
        selected_path = case_files[labels.index(selected)]

        if st.button("加载样本", use_container_width=True):
            st.session_state.cases = cases_from_jsonl(selected_path)
            st.session_state.cases_file = selected_path
            st.rerun()


results = st.session_state.get("results", [])
cases = st.session_state.get("cases", [])

if not results:
    st.warning("暂无结果，请先在「执行评测」页运行，或从侧边栏加载结果文件。")
    st.stop()


option_labels = []
for i, r in enumerate(results):
    option_labels.append(
        f"{i + 1}. {r.case_id} | 模型={r.model_name} | 提示词={r.prompt_version} | 总分={r.score_total:.2f}"
    )

selected_label = st.selectbox("选择样本结果", option_labels)
selected_idx = option_labels.index(selected_label)
result = results[selected_idx]
case = find_case_for_result(cases, result) if cases else None

st.caption(
    f"样本编号={result.case_id} | 被评测模型={result.model_name} | 提示词版本={result.prompt_version} | 裁判模型={result.judge_model}"
)

st.divider()

if case:
    render_case_input(case)
    render_case_reasoning(case)
    render_case_metadata(case)
    render_dialogue(case)
else:
    st.warning("没有加载对应样本文件，因此只能展示评测结果，无法展示旧用户画像、对话和候选用户画像。")

st.divider()
render_eval_result(result)
render_raw_response(result)

st.divider()
st.subheader("人工复核")

reviews = load_reviews(DEFAULT_REVIEW_PATH)
key = review_key(result.case_id, result.model_name, result.prompt_version)
old_review = reviews.get(key, {})

status_options = ["unreviewed", "reviewed", "need_discussion"]
status_labels = {
    "unreviewed": "未复核",
    "reviewed": "已复核",
    "need_discussion": "需要讨论",
}
default_status = old_review.get("review_status", "unreviewed")
if default_status not in status_options:
    default_status = "unreviewed"

human_score = st.number_input(
    "人工评分",
    min_value=0.0,
    max_value=5.0,
    value=float(old_review.get("human_score", result.score_total or 0.0)),
    step=0.1,
)

review_status = st.selectbox(
    "复核状态",
    status_options,
    index=status_options.index(default_status),
    format_func=lambda x: status_labels.get(x, x),
)

human_error_tags = st.multiselect(
    "人工错误标签",
    sorted(list(VALID_ERROR_TAGS)),
    default=old_review.get("human_error_tags", []),
)

human_comment = st.text_area(
    "人工备注",
    value=old_review.get("human_comment", ""),
    height=180,
)

if st.button("保存人工复核", type="primary", use_container_width=True):
    review = {
        "case_id": result.case_id,
        "task_type": result.task_type,
        "model_name": result.model_name,
        "prompt_version": result.prompt_version,
        "judge_model": result.judge_model,
        "judge_prompt_version": result.judge_prompt_version,
        "llm_score_total": result.score_total,
        "human_score": human_score,
        "human_error_tags": human_error_tags,
        "human_comment": human_comment,
        "review_status": review_status,
    }
    upsert_review(review, DEFAULT_REVIEW_PATH)
    st.success(f"已保存到 {DEFAULT_REVIEW_PATH}")
