from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.extraction.client import MemoryExtractionConfig
from src.loop.validation_gate import ValidationGateConfig
from src.schema import TaskType
from src.ui.components import render_state_file_notice
from src.ui.config_store import build_eval_config, load_config
from src.ui.data_service import save_uploaded_file
from src.ui.extraction_prompt_ab_job_runner import (
    ExtractionPromptAbJobConfig,
    ensure_extraction_prompt_ab_diff_excel,
    extraction_prompt_ab_job_is_stale,
    list_extraction_prompt_ab_job_ids,
    load_extraction_prompt_ab_report,
    mark_extraction_prompt_ab_job_interrupted,
    read_extraction_prompt_ab_job_state,
    report_excel_path,
    request_extraction_prompt_ab_stop,
    results_path,
)
from src.ui.preflight import ERROR, PASS, WARNING, PreflightCheck, render_preflight
from src.ui.prompt_editor import (
    get_default_extraction_prompt_file,
    get_default_prompt_file,
    infer_prompt_version,
    list_extraction_prompt_files,
    list_prompt_files,
    load_extraction_prompt_templates,
    load_prompt,
    prompt_text_hash,
)
from src.ui.task_worker import launch_background_task
from src.ui.theme import render_page_header
from src.ui.user_identity import require_page_identity


require_page_identity()


TASK_LABELS = {
    TaskType.USER_MD.value: "用户画像 USER.md",
    TaskType.LONG_MEMORY.value: "长期记忆 MEMORY.md",
}
BASELINE_RULE = "使用 A 作为冻结规则（推荐）"
INDEPENDENT_RULE = "选择独立规则版本"


def _combined_prompt(create_text: str, update_text: str) -> str:
    create = str(create_text or "").strip()
    update = str(update_text or "").strip()
    if not create:
        return update
    if not update or create == update:
        return create
    return f"# 首次创建规则\n\n{create}\n\n# 增量更新规则\n\n{update}"


def _load_templates(prompt_file: str) -> tuple[str, str, str]:
    templates = load_extraction_prompt_templates(prompt_file)
    fallback = load_prompt(prompt_file, prompt_kind="extraction")
    create = str(templates.get("create") or fallback)
    update = str(templates.get("update") or fallback)
    return create, update, _combined_prompt(create, update)


def _prompt_files_for_task(prompt_files: list[str], task_type: str) -> list[str]:
    """Exclude files that are explicitly named for the other memory task."""
    selected: list[str] = []
    for prompt_file in prompt_files:
        name = Path(prompt_file).name.lower()
        if task_type == TaskType.USER_MD.value and "long_memory" in name:
            continue
        if task_type == TaskType.LONG_MEMORY.value and ("user_md" in name or "user.md" in name):
            continue
        selected.append(prompt_file)
    return selected or prompt_files


def _resolve_sheet_name(raw: str) -> str | int | None:
    value = str(raw or "").strip()
    if not value:
        return 0
    try:
        return int(value)
    except ValueError:
        return value


def _handoff_to_advisor(job_id: str, label: str, state: dict) -> None:
    config = state.get("config") if isinstance(state.get("config"), dict) else {}
    prompt_file = str(config.get(f"prompt_{label.lower()}_file") or "")
    prompt_version = str(config.get(f"prompt_{label.lower()}_version") or label)
    judge_prompt_file = str(config.get("judge_prompt_file") or "")
    result_file = results_path(job_id, label)
    st.session_state["prompt_advisor_external_result_path"] = str(result_file)
    st.session_state["prompt_advisor_external_result_label"] = f"{job_id} · 提取提示词 {label}"
    st.session_state["abs_result_source"] = "提取 A/B 传入结果"
    st.session_state["selected_prompt_task_type"] = str(config.get("task_type") or TaskType.USER_MD.value)
    st.session_state["selected_extraction_prompt_file"] = prompt_file
    st.session_state["absolute_extraction_prompt_file"] = prompt_file
    if judge_prompt_file:
        st.session_state["absolute_judge_prompt_file"] = judge_prompt_file
    st.session_state["extraction_prompt_text"] = (
        load_prompt(prompt_file, prompt_kind="extraction") if prompt_file else ""
    )
    st.session_state["prompt_advisor_handoff_note"] = f"来自 {prompt_version} 的 A/B 评测结果"
    st.switch_page("pages/7_提示词改进建议.py")


def _render_model_comparison(report: dict) -> None:
    model_result = report.get("model_comparison")
    if not isinstance(model_result, dict) or not model_result:
        return
    status = str(model_result.get("status") or "")
    if status == "skipped":
        st.caption("本次任务未启用独立对比总结模型。")
        return

    st.markdown("**对比总结模型的补充意见**")
    if status == "failed":
        st.warning(
            f"对比模型未生成可用结果：{model_result.get('error') or '调用或 JSON 解析失败'}。"
            "统计结论和 A/B 报告仍然有效。"
        )
        return

    preference_labels = {
        "A": "倾向 A",
        "B": "倾向 B",
        "TIE": "认为持平",
        "INSUFFICIENT": "证据不足",
    }
    confidence_labels = {"low": "低", "medium": "中", "high": "高"}
    c1, c2, c3 = st.columns(3)
    c1.metric("对比模型", str(model_result.get("model") or "-"))
    c2.metric(
        "模型倾向",
        preference_labels.get(str(model_result.get("preferred_version") or ""), "证据不足"),
    )
    c3.metric(
        "模型自报置信度",
        confidence_labels.get(str(model_result.get("confidence") or "low"), "低"),
    )
    st.info(str(model_result.get("summary") or "模型未提供综合说明。"))

    formal_preference = "B" if report.get("recommendation") == "建议选择 B" else (
        "A" if report.get("recommendation") == "建议保留 A" else "INSUFFICIENT"
    )
    model_preference = str(model_result.get("preferred_version") or "INSUFFICIENT")
    if model_preference in {"A", "B"} and model_preference != formal_preference:
        st.warning("对比模型倾向与统计门槛结论不一致；版本选择仍以统计结论为准。")

    with st.expander("查看模型依据、风险与通用修改方向", expanded=False):
        for title, key in (
            ("主要依据", "reasons"),
            ("风险或证据不足", "risks"),
            ("提示词修改方向", "prompt_suggestions"),
        ):
            st.markdown(f"**{title}**")
            items = model_result.get(key) if isinstance(model_result.get(key), list) else []
            if items:
                for item in items:
                    st.write(f"- {item}")
            else:
                st.caption("无")
        if model_result.get("reasoning"):
            st.markdown("**模型 reasoning（仅供排查）**")
            st.text_area(
                "对比模型 reasoning",
                str(model_result.get("reasoning") or ""),
                height=180,
                disabled=True,
                label_visibility="collapsed",
            )


def _render_report(job_id: str, state: dict) -> None:
    report = load_extraction_prompt_ab_report(job_id)
    if not report:
        return

    st.divider()
    st.subheader("比较结论")
    recommendation = str(report.get("recommendation") or "暂不定版")
    reason = str(report.get("recommendation_reason") or "")
    if recommendation == "建议选择 B":
        st.success(f"**{recommendation}**：{reason}")
    elif recommendation == "建议保留 A":
        st.warning(f"**{recommendation}**：{reason}")
    else:
        st.info(f"**{recommendation}**：{reason}")

    model_roles = report.get("model_roles") if isinstance(report.get("model_roles"), dict) else {}
    if model_roles:
        st.caption(
            f"提取模型：{model_roles.get('extraction_model') or '-'} ｜ "
            f"绝对评测模型：{model_roles.get('evaluation_model') or '-'} ｜ "
            f"对比总结模型：{model_roles.get('comparison_model') or '未启用'}"
        )

    quality_a = report.get("quality_a") or {}
    quality_b = report.get("quality_b") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "条件平均分",
        f"A {float(quality_a.get('conditional_avg_score') or 0):.3f}",
        delta=f"B-A {float(quality_b.get('conditional_avg_score') or 0) - float(quality_a.get('conditional_avg_score') or 0):+.3f}",
    )
    c2.metric(
        "端到端分数",
        f"A {float(quality_a.get('end_to_end_score') or 0):.3f}",
        delta=f"B-A {float(quality_b.get('end_to_end_score') or 0) - float(quality_a.get('end_to_end_score') or 0):+.3f}",
    )
    c3.metric(
        "提取覆盖率",
        f"A {float(quality_a.get('extraction_coverage') or 0):.1%}",
        delta=f"B-A {float(quality_b.get('extraction_coverage') or 0) - float(quality_a.get('extraction_coverage') or 0):+.1%}",
    )

    gate = report.get("validation_gate") or {}
    confidence = gate.get("confidence_interval") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("成功配对", int(gate.get("paired_case_count") or 0))
    c2.metric("独立评测人/时序簇", int(gate.get("paired_cluster_count") or 0))
    c3.metric("配对平均差 B-A", f"{float(gate.get('paired_score_delta') or 0):+.3f}")
    lower = confidence.get("lower")
    upper = confidence.get("upper")
    c4.metric(
        f"{float(gate.get('confidence_level') or 0.95):.0%} 置信区间",
        "证据不足" if lower is None or upper is None else f"[{float(lower):+.3f}, {float(upper):+.3f}]",
    )
    identical_count = int(report.get("identical_output_count") or 0)
    judge_disagreement_count = int(report.get("judge_disagreement_on_identical_output_count") or 0)
    if identical_count:
        st.caption(
            f"A/B 提取正文相同 {identical_count} 条，其中 Judge 打分或标签不一致 {judge_disagreement_count} 条；"
            "这些差异按裁判波动处理，不归因于提示词优劣。"
        )
    st.caption("条件平均分只看 Judge 成功样本；端到端分数会把成功调用但未提取出正文的漏抽计入质量损失；API/网络/JSON 解析失败均单独统计，不按 0 分处理。")
    _render_model_comparison(report)

    winner_counts = report.get("winner_counts") or {}
    if winner_counts:
        st.markdown("**配对胜负与覆盖差异**")
        st.dataframe(
            pd.DataFrame([{"结论": key, "样本数": value} for key, value in winner_counts.items()]),
            width="stretch",
            hide_index=True,
        )

    dimension_rows = report.get("dimension_summary") or []
    if dimension_rows:
        st.markdown("**评分维度变化**")
        dimension_df = pd.DataFrame(dimension_rows).rename(
            columns={
                "dimension": "维度",
                "avg_a": "A 平均分",
                "avg_b": "B 平均分",
                "delta_b_minus_a": "B-A",
                "paired_count": "配对数",
            }
        )
        st.dataframe(dimension_df, width="stretch", hide_index=True)

    rows = report.get("rows") or []
    if rows:
        st.markdown("**逐样本对比备注**")
        table = pd.DataFrame(rows).rename(
            columns={
                "reviewer": "评测人",
                "session_id": "session",
                "chunk_index": "chunk",
                "extraction_a": "A 提取状态",
                "extraction_b": "B 提取状态",
                "judge_status_a": "A Judge 状态",
                "judge_status_b": "B Judge 状态",
                "score_a": "A 得分",
                "score_b": "B 得分",
                "score_delta_b_minus_a": "B-A",
                "comparison": "对比结论",
                "comparison_note": "对比备注",
                "error_tags_a": "A 错误标签",
                "error_tags_b": "B 错误标签",
                "comment_a": "A 评语",
                "comment_b": "B 评语",
                "old_memory_a": "A 侧旧记忆",
                "old_memory_b": "B 侧旧记忆",
                "candidate_output_a": "A 提取正文",
                "candidate_output_b": "B 提取正文",
            }
        )
        preview_columns = [
            "评测人", "session", "chunk", "A 提取状态", "B 提取状态",
            "A 得分", "B 得分", "B-A", "对比结论", "对比备注",
        ]
        st.dataframe(table[[column for column in preview_columns if column in table]], width="stretch", hide_index=True)
        with st.expander("查看完整 Judge 评语、错误标签与规则引用", expanded=False):
            st.dataframe(table, width="stretch", hide_index=True)

    duplicate_keys = report.get("duplicate_source_keys") or []
    if duplicate_keys:
        st.error(f"发现 {len(duplicate_keys)} 个重复来源键，这些样本未进入配对结论。")

    with st.expander("查看未通过替换门槛的具体原因", expanded=False):
        reasons = gate.get("reasons") or []
        if reasons:
            for item in reasons:
                st.write(f"- {item}")
        else:
            st.write("B 已通过当前门槛。")

    excel_file = report_excel_path(job_id)
    st.caption(
        "逐行 Diff 保留 session_id、chunk_id、query、answer 和评测人；"
        "两版提取结果、reasoning 与比较结论只写在每个 chunk 末行，另附一行一个 chunk 的对照表。"
    )
    diff_file = None
    try:
        diff_file = ensure_extraction_prompt_ab_diff_excel(job_id)
    except Exception as exc:
        st.warning(f"逐行 Diff Excel 生成失败：{type(exc).__name__}: {exc}")
    c1, c2, c3 = st.columns(3)
    if diff_file and diff_file.exists():
        c1.download_button(
            "下载逐行 Diff Excel",
            data=diff_file.read_bytes(),
            file_name=diff_file.name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
            key=f"{job_id}_download_diff_excel",
        )
    else:
        c1.button(
            "逐行 Diff 尚未生成",
            disabled=True,
            width="stretch",
            key=f"{job_id}_diff_unavailable",
        )
    if excel_file.exists():
        c2.download_button(
            "下载评测汇总 Excel",
            data=excel_file.read_bytes(),
            file_name=excel_file.name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
            key=f"{job_id}_download_excel",
        )
    c3.download_button(
        "下载比较报告 JSON",
        data=json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"{job_id}_comparison.json",
        mime="application/json",
        width="stretch",
        key=f"{job_id}_download_json",
    )

    st.markdown("**继续改进提示词**")
    st.caption("下面只会把对应版本的评测结果送入提示词改进页面生成候选版本，不会覆盖当前提示词文件。通常应选择表现较弱或仍有明确错误的版本。")
    c1, c2 = st.columns(2)
    if c1.button("基于 A 的问题生成改进建议", width="stretch", key=f"{job_id}_advise_a"):
        _handoff_to_advisor(job_id, "A", state)
    if c2.button("基于 B 的问题生成改进建议", width="stretch", key=f"{job_id}_advise_b"):
        _handoff_to_advisor(job_id, "B", state)


def _render_job_state(job_id: str) -> None:
    state = read_extraction_prompt_ab_job_state(job_id)
    if extraction_prompt_ab_job_is_stale(state):
        state = mark_extraction_prompt_ab_job_interrupted(job_id)
    if not state:
        st.info("暂无该任务状态。")
        return
    render_state_file_notice(state)
    status = str(state.get("status") or "")
    done = int(state.get("done", 0) or 0)
    total = int(state.get("total", 0) or 0)
    phase_done = state.get("phase_done")
    phase_total = state.get("phase_total")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("状态", status or "-")
    c2.metric("当前阶段", state.get("stage", "-"))
    c3.metric("总进度", f"{done / total:.1%}" if total else "准备中")
    c4.metric("阶段进度", f"{phase_done}/{phase_total}" if phase_total else "-")
    st.progress(done / total if total else 0.0)
    st.write(state.get("message", ""))
    if status == "running":
        st.info("任务由独立后台进程执行，切换页面不会丢失进度。运行参数可在任务中心调整。")
        if st.button("请求终止提取 A/B 实验", type="secondary", width="stretch", key=f"{job_id}_stop"):
            request_extraction_prompt_ab_stop(job_id)
            st.warning("已提交终止请求。正在执行的单次 API 调用返回后，后续步骤会停止。")
            st.rerun()
    elif status == "failed":
        st.error(state.get("message", "任务失败。"))
    elif status in {"stopped", "interrupted"}:
        st.warning(state.get("message", "任务未完整结束。"))
    if status == "completed":
        _render_report(job_id, state)
    if state.get("traceback"):
        with st.expander("错误堆栈", expanded=True):
            st.code(state.get("traceback", ""), language="text")


@st.fragment(run_every="10s")
def _render_job_state_auto(job_id: str) -> None:
    require_page_identity()
    _render_job_state(job_id)


render_page_header(
    "提取提示词 A/B 对比",
    "在同一原始数据与冻结评测标准下，比较两个提取提示词版本的覆盖率、质量和逐样本差异。",
    category="优化实验",
)

if "ui_config" not in st.session_state:
    st.session_state.ui_config = load_config()

with st.expander("使用说明", expanded=False):
    st.markdown(
        """
1. A 是当前基线版本，B 是候选版本；两侧共用原始 Excel 和同一个提取模型，实验变量仍只有提取提示词。
2. 冻结评测规则不会随 B 改变，避免候选提示词通过改变规则来证明自己更好。
3. 系统按评测人、session、chunk 和原始行范围配对；漏抽会单独显示，不会造成后续 case 错位。
4. 只有覆盖率、退化率、关键错误和统计置信度同时达标时才建议替换 A。
5. 提取模型、绝对评测模型和对比总结模型可以分别指定；对比模型只补充说明，不覆盖统计结论。
6. 比较完成后可以把任一版本的问题样本送入“提示词改进”，但候选版本仍需另存和复验。
        """.strip()
    )

st.subheader("1. 选择原始数据")
c1, c2 = st.columns(2)
with c1:
    uploaded = st.file_uploader("上传原始对话 Excel", type=["xlsx", "xls"], key="extract_ab_upload")
with c2:
    local_excel_path = st.text_input("或填写本地 Excel 路径", key="extract_ab_local_path")
c1, c2, c3 = st.columns(3)
with c1:
    sheet_name_raw = st.text_input("工作表名称或序号", value="0")
with c2:
    reviewer_filter = st.text_input("评测人筛选（可选）", help="多个评测人可用逗号分隔。")
with c3:
    chunk_size = st.number_input("每个 chunk 行数", min_value=1, max_value=200, value=10, step=1)

st.subheader("2. 固定变量并选择 A/B")
task_type = st.selectbox(
    "任务类型",
    list(TASK_LABELS),
    format_func=lambda value: TASK_LABELS[value],
)
extraction_files = _prompt_files_for_task(list_extraction_prompt_files(), task_type)
judge_files = list_prompt_files()
if not extraction_files or not judge_files:
    st.error("缺少提取提示词或裁判提示词文件，请先到配置页保存对应版本。")
    st.stop()

default_a = get_default_extraction_prompt_file(task_type)
prompt_a_file = st.selectbox(
    "提取提示词 A（当前基线）",
    extraction_files,
    index=extraction_files.index(default_a) if default_a in extraction_files else 0,
)
different_candidates = [item for item in extraction_files if item != prompt_a_file]
default_b = different_candidates[0] if different_candidates else prompt_a_file
prompt_b_file = st.selectbox(
    "提取提示词 B（候选版本）",
    extraction_files,
    index=extraction_files.index(default_b),
)
prompt_a_create, prompt_a_update, prompt_a_full = _load_templates(prompt_a_file)
prompt_b_create, prompt_b_update, prompt_b_full = _load_templates(prompt_b_file)

rule_mode = st.radio("冻结评测规则来源", [BASELINE_RULE, INDEPENDENT_RULE], horizontal=True)
if rule_mode == BASELINE_RULE:
    rule_file = prompt_a_file
else:
    rule_candidates = [item for item in extraction_files if item != prompt_b_file] or extraction_files
    rule_file = st.selectbox("独立规则提示词版本", rule_candidates)
rule_create, rule_update, evaluation_rule_text = _load_templates(rule_file)

default_judge = get_default_prompt_file(task_type)
judge_prompt_file = st.selectbox(
    "共同裁判提示词",
    judge_files,
    index=judge_files.index(default_judge) if default_judge in judge_files else 0,
)
judge_prompt_text = load_prompt(judge_prompt_file)

if prompt_a_file == prompt_b_file or prompt_text_hash(prompt_a_full) == prompt_text_hash(prompt_b_full):
    st.warning("A 与 B 内容相同，无法形成有效对比。")
if rule_mode == BASELINE_RULE:
    st.caption("当前按 A 的业务规则共同评测 A/B，适合验证 B 是否在不改变既定口径的前提下改善。")
else:
    st.caption("当前使用独立规则版本共同评测 A/B，适合团队已有稳定规范的场景。")

with st.expander("查看 A、B、冻结规则与裁判提示词全文", expanded=False):
    tab_a, tab_b, tab_rule, tab_judge = st.tabs(["提取 A", "提取 B", "冻结规则", "裁判提示词"])
    tab_a.text_area("提取 A 全文", prompt_a_full, height=320, disabled=True)
    tab_b.text_area("提取 B 全文", prompt_b_full, height=320, disabled=True)
    tab_rule.text_area("冻结规则全文", evaluation_rule_text, height=320, disabled=True)
    tab_judge.text_area("裁判提示词全文", judge_prompt_text, height=320, disabled=True)

st.subheader("3. 运行配置")
cfg = dict(st.session_state.ui_config)
mock = st.checkbox("模拟模式", value=bool(cfg.get("mock", False)))
default_model = str(cfg.get("judge_model") or "") or ("mock-model" if mock else "")
st.markdown("**模型角色**")
c1, c2, c3 = st.columns(3)
with c1:
    extraction_model = st.text_input("提取模型名称", value=default_model)
    st.caption("仅负责使用 A/B 提示词生成 USER.md 或 MEMORY.md。")
with c2:
    judge_model = st.text_input("绝对评测模型名称", value=default_model)
    st.caption("使用冻结规则分别评价 A/B，不直接判定胜负。")
with c3:
    comparison_enabled = st.checkbox("启用独立对比总结模型", value=True)
    comparison_model = st.text_input(
        "对比总结模型名称",
        value=default_model,
        disabled=not comparison_enabled,
    )
    st.caption("综合统计量与代表性差异，生成辅助说明和修改方向。")
st.caption("三个角色默认共用配置页中的 API 地址和令牌，但模型型号与调用参数彼此独立。")

with st.expander("各阶段调用参数", expanded=True):
    extraction_tab, judge_tab, comparison_tab = st.tabs(["提取", "绝对评测", "最终对比"])
    with extraction_tab:
        c1, c2, c3 = st.columns(3)
        extraction_max_tokens = c1.number_input(
            "提取最大 tokens",
            min_value=500,
            max_value=100000,
            value=50000,
            step=500,
        )
        extraction_concurrency = c2.number_input(
            "提取并发",
            min_value=1,
            max_value=100,
            value=min(100, max(1, int(cfg.get("judge_concurrency") or 1))),
            step=1,
        )
        extraction_timeout = c3.number_input(
            "提取超时（秒）",
            min_value=10,
            max_value=1800,
            value=int(cfg.get("judge_timeout") or 120),
            step=10,
        )
        c1, c2, c3 = st.columns(3)
        extraction_attempts = c1.number_input(
            "提取最大尝试次数（含首次）",
            min_value=1,
            max_value=10,
            value=int(cfg.get("judge_max_retries") or 3),
            step=1,
        )
        extraction_interval = c2.number_input(
            "提取请求间隔（秒）",
            min_value=0.0,
            max_value=300.0,
            value=float(cfg.get("judge_request_interval") or 0.0),
            step=0.5,
        )
        extraction_enable_thinking = c3.checkbox(
            "提取启用 thinking",
            value=bool(cfg.get("judge_enable_thinking", False)),
        )
    with judge_tab:
        c1, c2, c3 = st.columns(3)
        judge_max_tokens = c1.number_input(
            "评测最大 tokens",
            min_value=500,
            max_value=100000,
            value=int(cfg.get("judge_max_tokens") or 2000),
            step=500,
        )
        judge_concurrency = c2.number_input(
            "评测并发",
            min_value=1,
            max_value=100,
            value=min(100, max(1, int(cfg.get("judge_concurrency") or 1))),
            step=1,
        )
        judge_timeout = c3.number_input(
            "评测超时（秒）",
            min_value=10,
            max_value=1800,
            value=int(cfg.get("judge_timeout") or 120),
            step=10,
        )
        c1, c2, c3 = st.columns(3)
        judge_attempts = c1.number_input(
            "评测最大尝试次数（含首次）",
            min_value=1,
            max_value=10,
            value=int(cfg.get("judge_max_retries") or 3),
            step=1,
        )
        judge_interval = c2.number_input(
            "评测请求间隔（秒）",
            min_value=0.0,
            max_value=300.0,
            value=float(cfg.get("judge_request_interval") or 0.0),
            step=0.5,
        )
        judge_enable_thinking = c3.checkbox(
            "评测启用 thinking",
            value=bool(cfg.get("judge_enable_thinking", False)),
        )
    with comparison_tab:
        c1, c2, c3 = st.columns(3)
        comparison_max_tokens = c1.number_input(
            "对比最大 tokens",
            min_value=500,
            max_value=20000,
            value=min(20000, max(500, int(cfg.get("judge_max_tokens") or 2000))),
            step=500,
            disabled=not comparison_enabled,
        )
        comparison_timeout = c2.number_input(
            "对比超时（秒）",
            min_value=10,
            max_value=1800,
            value=int(cfg.get("judge_timeout") or 120),
            step=10,
            disabled=not comparison_enabled,
        )
        comparison_attempts = c3.number_input(
            "对比最大尝试次数（含首次）",
            min_value=1,
            max_value=10,
            value=int(cfg.get("judge_max_retries") or 3),
            step=1,
            disabled=not comparison_enabled,
        )
        c1, c2, c3 = st.columns(3)
        comparison_interval = c1.number_input(
            "对比请求间隔（秒）",
            min_value=0.0,
            max_value=300.0,
            value=float(cfg.get("judge_request_interval") or 0.0),
            step=0.5,
            disabled=not comparison_enabled,
        )
        comparison_max_evidence = c2.number_input(
            "代表性差异上限",
            min_value=1,
            max_value=30,
            value=8,
            step=1,
            disabled=not comparison_enabled,
            help="只把最有区分度的样本交给对比模型，控制上下文长度。",
        )
        comparison_enable_thinking = c3.checkbox(
            "对比启用 thinking",
            value=False,
            disabled=not comparison_enabled,
        )

with st.expander("版本选择门槛", expanded=False):
    st.caption("这些门槛只决定是否给出“建议选择 B”，不会修改任何提示词。小样本时系统通常会返回“证据不足”。")
    c1, c2, c3 = st.columns(3)
    min_score_delta = c1.number_input("最小配对平均提升", min_value=0.0, max_value=2.0, value=0.03, step=0.01)
    max_regression_rate = c2.number_input("最大单样本退化率", min_value=0.0, max_value=1.0, value=0.10, step=0.01)
    score_tolerance = c3.number_input("单样本持平容差", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
    c1, c2, c3 = st.columns(3)
    min_paired_cases = c1.number_input("最少配对 case", min_value=1, max_value=10000, value=8, step=1)
    min_paired_clusters = c2.number_input("最少独立评测人/时序簇", min_value=1, max_value=1000, value=2, step=1)
    confidence_level = c3.selectbox("置信水平", [0.90, 0.95, 0.99], index=1, format_func=lambda value: f"{value:.0%}")

judge_cfg = dict(cfg)
judge_cfg["judge_model"] = judge_model.strip()
judge_cfg["judge_max_tokens"] = int(judge_max_tokens)
judge_cfg["judge_timeout"] = int(judge_timeout)
judge_cfg["judge_max_retries"] = int(judge_attempts)
judge_cfg["judge_concurrency"] = int(judge_concurrency)
judge_cfg["judge_request_interval"] = float(judge_interval)
judge_cfg["judge_enable_thinking"] = bool(judge_enable_thinking)
judge_config = build_eval_config(judge_cfg, mock=mock)
extraction_base_config = build_eval_config({**cfg, "judge_model": extraction_model}, mock=mock)
comparison_cfg = dict(cfg)
comparison_cfg["judge_model"] = comparison_model.strip()
comparison_cfg["judge_max_tokens"] = int(comparison_max_tokens)
comparison_cfg["judge_timeout"] = int(comparison_timeout)
comparison_cfg["judge_max_retries"] = int(comparison_attempts)
comparison_cfg["judge_request_interval"] = float(comparison_interval)
comparison_cfg["judge_enable_thinking"] = bool(comparison_enable_thinking)
comparison_cfg["judge_prompt_cache_id"] = ""
comparison_cfg["judge_prompt_cache_location"] = "none"
comparison_config = build_eval_config(comparison_cfg, mock=mock)

local_input = Path(local_excel_path.strip().strip('"')) if local_excel_path.strip() else None
input_ready = uploaded is not None or bool(local_input and local_input.is_file())
checks = [
    PreflightCheck("input", "原始 Excel", PASS if input_ready else ERROR, "已选择输入。" if input_ready else "请上传 Excel 或填写存在的本地路径。"),
    PreflightCheck("prompt_a", "提取提示词 A", PASS if prompt_a_full.strip() else ERROR, f"版本：{infer_prompt_version(prompt_a_file)}"),
    PreflightCheck("prompt_b", "提取提示词 B", PASS if prompt_b_full.strip() else ERROR, f"版本：{infer_prompt_version(prompt_b_file)}"),
    PreflightCheck(
        "prompt_difference",
        "A/B 变量唯一且有差异",
        PASS if prompt_text_hash(prompt_a_full) != prompt_text_hash(prompt_b_full) else ERROR,
        "A/B 只改变提取提示词内容。" if prompt_text_hash(prompt_a_full) != prompt_text_hash(prompt_b_full) else "A/B 提示词内容相同。",
    ),
    PreflightCheck("frozen_rule", "冻结评测规则", PASS if evaluation_rule_text.strip() else ERROR, f"版本：{infer_prompt_version(rule_file)}"),
    PreflightCheck("judge_prompt", "共同裁判提示词", PASS if judge_prompt_text.strip() else ERROR, f"版本：{infer_prompt_version(judge_prompt_file)}"),
    PreflightCheck(
        "model_roles",
        "三个模型角色",
        PASS
        if extraction_model.strip() and judge_model.strip() and (
            comparison_model.strip() or not comparison_enabled
        )
        else ERROR,
        (
            f"提取：{extraction_model.strip()}；评测：{judge_model.strip()}；"
            f"对比：{comparison_model.strip() if comparison_enabled else '未启用'}"
        ),
    ),
]
api_errors = judge_config.validate()
if not mock and not extraction_model.strip():
    api_errors.append("提取模型名称为空")
if comparison_enabled:
    api_errors.extend(f"对比模型：{item}" for item in comparison_config.validate())
checks.append(PreflightCheck("api", "模型与接口", PASS if not api_errors else ERROR, "配置完整。" if not api_errors else "；".join(api_errors)))
qps_risk = not mock and (int(extraction_concurrency) > 1 or int(judge_concurrency) > 1) and min(float(extraction_interval), float(judge_interval)) < 10
checks.append(PreflightCheck(
    "rate_limit",
    "限流设置",
    WARNING if qps_risk else PASS,
    "并发大于 1 且请求间隔较短，低 QPS 接口可能限流。" if qps_risk else "提取、评测与最终对比请求会共用全局限流器。",
))
preflight_ready = render_preflight(checks)

if st.button("开始提取提示词 A/B 实验", type="primary", width="stretch", disabled=not preflight_ready):
    input_path = (
        save_uploaded_file(uploaded, suffix=Path(uploaded.name).suffix)
        if uploaded is not None
        else str(Path(local_excel_path.strip().strip('"')).resolve())
    )
    extraction_config = MemoryExtractionConfig.from_eval_config(
        extraction_base_config,
        model=extraction_model,
        max_tokens=int(extraction_max_tokens),
        request_interval=float(extraction_interval),
        max_retries=max(0, int(extraction_attempts) - 1),
        retry_sleep=float(cfg.get("judge_qps_backoff") or 12.0),
        enable_thinking=bool(extraction_enable_thinking),
        timeout=int(extraction_timeout),
    )
    extraction_config.concurrency = int(extraction_concurrency)
    job_id = f"extract_prompt_ab_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    job_config = ExtractionPromptAbJobConfig(
        job_id=job_id,
        task_type=task_type,
        input_path=input_path,
        prompt_a_text=prompt_a_update,
        prompt_a_create_text=prompt_a_create,
        prompt_a_version=infer_prompt_version(prompt_a_file),
        prompt_a_file=prompt_a_file,
        prompt_a_hash=prompt_text_hash(prompt_a_full),
        prompt_b_text=prompt_b_update,
        prompt_b_create_text=prompt_b_create,
        prompt_b_version=infer_prompt_version(prompt_b_file),
        prompt_b_file=prompt_b_file,
        prompt_b_hash=prompt_text_hash(prompt_b_full),
        judge_prompt_text=judge_prompt_text,
        judge_prompt_version=infer_prompt_version(judge_prompt_file),
        judge_prompt_file=judge_prompt_file,
        evaluation_rule_prompt_text=evaluation_rule_text,
        evaluation_rule_prompt_version=infer_prompt_version(rule_file),
        evaluation_rule_prompt_file=rule_file,
        evaluation_rule_prompt_hash=prompt_text_hash(evaluation_rule_text),
        sheet_name=_resolve_sheet_name(sheet_name_raw),
        reviewer_filter=reviewer_filter.strip(),
        chunk_size=int(chunk_size),
        score_tolerance=float(score_tolerance),
        extraction_config=extraction_config,
        eval_config=judge_config,
        comparison_config=comparison_config,
        enable_model_comparison=bool(comparison_enabled),
        comparison_max_evidence=int(comparison_max_evidence),
        validation_config=ValidationGateConfig(
            min_score_delta=float(min_score_delta),
            max_case_regression_rate=float(max_regression_rate),
            score_regression_tolerance=float(score_tolerance),
            min_paired_cases=int(min_paired_cases),
            min_paired_clusters=int(min_paired_clusters),
            confidence_level=float(confidence_level),
        ),
    )
    launch_background_task("extraction_prompt_ab", job_config)
    st.session_state["extraction_prompt_ab_job_id"] = job_id
    st.success(f"已启动独立后台任务：{job_id}")
    st.rerun()

st.divider()
st.subheader("4. 后台任务与结果")
job_ids = list_extraction_prompt_ab_job_ids()
if not job_ids:
    st.info("尚未运行提取提示词 A/B 实验。")
else:
    current_job = str(st.session_state.get("extraction_prompt_ab_job_id") or job_ids[0])
    index = job_ids.index(current_job) if current_job in job_ids else 0
    selected_job = st.selectbox("查看历史 A/B 任务", job_ids, index=index)
    st.session_state["extraction_prompt_ab_job_id"] = selected_job
    selected_state = read_extraction_prompt_ab_job_state(selected_job)
    if selected_state.get("status") == "running":
        auto_refresh = st.checkbox("运行中每 10 秒自动刷新进度区", value=True, key=f"{selected_job}_auto_refresh")
        if auto_refresh:
            _render_job_state_auto(selected_job)
        else:
            _render_job_state(selected_job)
    else:
        _render_job_state(selected_job)
