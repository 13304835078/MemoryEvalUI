from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.user_identity import require_page_identity
require_page_identity()

from src.schema import VALID_ERROR_TAGS, cases_from_jsonl
from src.ui.data_service import (
    list_case_files,
    list_result_files,
    load_results,
    load_results_bytes,
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
from src.ui.next_actions import NextAction, render_next_actions
from src.ui.result_triage import result_matches_filter, result_navigation_key
from src.eval.result_status import result_is_score_eligible
from src.ui.theme import render_page_header
from src.ui.workspace_context import render_workspace_context


render_page_header("样本复核", "核对单条样本的输入、评分、诊断证据与人工复核记录。", category="评测工作流")

if "cases" not in st.session_state:
    st.session_state.cases = []

if "results" not in st.session_state:
    st.session_state.results = []


with st.sidebar.expander("加载结果与样本", expanded=False):
    result_files = list_result_files()
    if result_files:
        labels = [Path(f).name for f in result_files]
        selected = st.selectbox("结果文件", labels)
        selected_path = result_files[labels.index(selected)]

        if st.button("加载结果", width="stretch"):
            st.session_state.results = load_results(selected_path)
            st.session_state.results_file = selected_path
            st.rerun()

    uploaded_result = st.file_uploader(
        "或上传结果文件",
        type=["jsonl", "csv", "xlsx"],
        key="detail_result_upload",
        help="支持执行评测 JSONL，以及结果总览导出的 CSV/Excel。",
    )
    if uploaded_result is not None and st.button(
        "加载上传结果",
        width="stretch",
        key="detail_load_uploaded_result",
    ):
        try:
            st.session_state.results = load_results_bytes(
                uploaded_result.getvalue(),
                uploaded_result.name,
            )
            st.session_state.results_file = uploaded_result.name
            st.rerun()
        except Exception as exc:
            st.error(f"结果文件解析失败：{exc}")

    case_files = list_case_files()
    if case_files:
        labels = [Path(f).name for f in case_files]
        selected = st.selectbox("样本文件", labels)
        selected_path = case_files[labels.index(selected)]

        if st.button("加载样本", width="stretch"):
            st.session_state.cases = cases_from_jsonl(selected_path)
            st.session_state.cases_file = selected_path
            st.rerun()


results = st.session_state.get("results", [])
cases = st.session_state.get("cases", [])

if not results:
    st.warning("暂无结果，请先在「执行评测」页运行，或从侧边栏加载结果文件。")
    st.stop()


filter_col, search_col = st.columns([1, 2])
with filter_col:
    detail_filter = st.selectbox(
        "样本筛选",
        ["全部", "运行失败", "严重失败", "低分", "有错误标签", "有结构化诊断"],
        key="detail_filter_mode",
    )
with search_col:
    detail_search = st.text_input(
        "搜索样本编号或评语",
        value="",
        key="detail_search_text",
        placeholder="输入 case_id 或评语关键词",
    ).strip().lower()

filtered_indices = [
    index
    for index, item in enumerate(results)
    if result_matches_filter(item, detail_filter)
    and (
        not detail_search
        or detail_search in str(item.case_id).lower()
        or detail_search in str(item.comment).lower()
    )
]
if not filtered_indices:
    st.warning("当前筛选条件下没有样本。")
    st.stop()

incoming_key = st.session_state.pop("detail_open_key", None)
incoming_index = next(
    (index for index, item in enumerate(results) if incoming_key and result_navigation_key(item) == tuple(incoming_key)),
    None,
)
if incoming_index is not None and incoming_index not in filtered_indices:
    filtered_indices.insert(0, incoming_index)

selector_key = "detail_result_choice"
if incoming_index is not None or st.session_state.get(selector_key) not in filtered_indices:
    st.session_state[selector_key] = incoming_index if incoming_index is not None else filtered_indices[0]

def _result_option_label(index: int) -> str:
    item = results[index]
    score_text = f"{item.score_total:.2f} 分" if result_is_score_eligible(item) else "未评分"
    return f"{index + 1}. {item.case_id} | {item.model_name} | {score_text}"


selected_idx = st.selectbox(
    "选择样本结果",
    filtered_indices,
    format_func=_result_option_label,
    key=selector_key,
)
result = results[selected_idx]
case = find_case_for_result(cases, result) if cases else None

position = filtered_indices.index(selected_idx)
nav_prev, nav_next, nav_back = st.columns([1, 1, 2])
nav_prev.button(
    "上一条",
    icon=":material/arrow_back:",
    width="stretch",
    disabled=position == 0,
    on_click=lambda: st.session_state.__setitem__(selector_key, filtered_indices[max(0, position - 1)]),
)
nav_next.button(
    "下一条",
    icon=":material/arrow_forward:",
    width="stretch",
    disabled=position >= len(filtered_indices) - 1,
    on_click=lambda: st.session_state.__setitem__(
        selector_key,
        filtered_indices[min(len(filtered_indices) - 1, position + 1)],
    ),
)
with nav_back:
    st.page_link(
        "pages/4_结果总览.py",
        label="返回结果总览",
        icon=":material/analytics:",
        width="stretch",
    )

render_workspace_context(
    task_type=result.task_type,
    case_count=len(results),
    cases_file=st.session_state.get("results_file", ""),
    model_name=result.model_name,
    judge_prompt=result.judge_prompt_version,
    extraction_prompt=result.extraction_prompt_version,
)

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
    document_name = "MEMORY.md" if result.task_type == "long_memory" else "USER.md"
    st.warning(
        f"没有加载对应样本文件，因此只能展示评测结果，"
        f"无法展示旧 {document_name}、对话和候选 {document_name}。"
    )

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

if st.button("保存人工复核", type="primary", width="stretch"):
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

render_next_actions([
    NextAction("pages/4_结果总览.py", "返回结果总览", ":material/analytics:"),
    NextAction("pages/7_提示词改进建议.py", "生成提示词建议", ":material/edit_note:"),
])
