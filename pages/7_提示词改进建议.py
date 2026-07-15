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

from src.schema import EVALUATABLE_TASK_TYPES, TASK_TYPE_LABELS, TaskType
from src.ui.config_store import build_eval_config, load_config
from src.ui.components import render_state_file_notice
from src.ui.data_service import list_result_files, load_results, load_results_bytes
from src.ui.next_actions import NextAction, render_next_actions
from src.ui.preflight import build_advisor_preflight, render_preflight
from src.ui.run_presets import render_run_preset_selector
from src.ui.prompt_advisor import (
    collect_absolute_eval_evidence,
)
from src.ui.prompt_advisor_job_runner import (
    PromptAdvisorJobConfig,
    list_prompt_advisor_job_ids,
    load_prompt_advisor_job_result,
    mark_prompt_advisor_job_interrupted,
    prompt_advisor_job_is_stale,
    read_prompt_advisor_job_state,
    request_prompt_advisor_stop,
)
from src.ui.task_worker import launch_background_task
from src.ui.prompt_editor import (
    get_default_extraction_prompt_file,
    get_default_prompt_file,
    infer_prompt_version,
    list_extraction_prompt_files,
    list_prompt_files,
    load_prompt,
    save_prompt_version,
)
from src.ui.theme import render_page_header
from src.ui.workspace_context import render_workspace_context


render_page_header(
    "提示词改进建议",
    "基于绝对评测证据生成受约束的候选修改，不自动覆盖当前提示词。",
    category="优化实验",
)

if "ui_config" not in st.session_state:
    st.session_state.ui_config = load_config()

render_run_preset_selector(st.session_state.ui_config, key="prompt_advisor")


def _prompt_task_slug(task_type: str) -> str:
    return "long_memory" if task_type == TaskType.LONG_MEMORY.value else "user_md"


def infer_single_task_type(results: list) -> str:
    allowed = {task.value for task in EVALUATABLE_TASK_TYPES}
    task_types = {
        str(getattr(result, "task_type", "") or "")
        for result in results
        if str(getattr(result, "task_type", "") or "") in allowed
    }
    if len(task_types) == 1:
        return next(iter(task_types))
    return TaskType.USER_MD.value


def render_advisor_result(result: dict | None, raw: str, key_prefix: str, task_type: str = TaskType.USER_MD.value) -> None:
    if not result:
        return

    st.divider()
    st.subheader("建议结果")
    st.json({
        "can_suggest": result.get("can_suggest"),
        "evidence_summary": result.get("evidence_summary"),
        "error": result.get("error", ""),
        "risks": result.get("risks", []),
        "validation_plan": result.get("validation_plan", []),
    })
    evidence_usage = result.get("evidence_usage") or {}
    if evidence_usage:
        c1, c2, c3 = st.columns(3)
        c1.metric("已选择证据", int(evidence_usage.get("selected_count") or 0))
        c2.metric("首次实际使用", int(evidence_usage.get("initial_used_count") or 0))
        c3.metric("请求次数", int(evidence_usage.get("request_count") or 0))
        if not evidence_usage.get("all_selected_used_initially", False):
            st.warning("首次请求没有使用全部已选择证据；请展开请求明细查看重试压缩情况。")
        usage_requests = evidence_usage.get("request_metrics") or []
        if usage_requests:
            with st.expander("证据实际使用与请求明细", expanded=False):
                st.dataframe(pd.DataFrame(usage_requests), width="stretch", hide_index=True)
    if result.get("error"):
        st.error(f"生成失败原因：{result.get('error')}")

    diagnoses = result.get("diagnoses", [])
    st.markdown("**问题归因**")
    if diagnoses:
        st.dataframe(pd.DataFrame(diagnoses), width="stretch", hide_index=True)
    else:
        st.info("暂无问题归因。")

    changes = result.get("judge_prompt_changes", [])
    st.markdown("**裁判提示词修改点**")
    if changes:
        st.dataframe(pd.DataFrame(changes), width="stretch", hide_index=True)
    else:
        st.info("暂无裁判提示词修改点。")

    candidate_judge_prompt = str(result.get("candidate_judge_prompt") or "")
    if candidate_judge_prompt:
        st.markdown("**候选裁判提示词**")
        st.text_area(
            "候选裁判提示词内容",
            value=candidate_judge_prompt,
            height=360,
            key=f"{key_prefix}_candidate_judge_prompt",
        )
        version_name = st.text_input(
            "保存为新的裁判提示词文件名",
            value=f"judge_{_prompt_task_slug(task_type)}_advised_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            key=f"{key_prefix}_judge_save_name",
        )
        if st.button("保存候选裁判提示词为新版本", width="stretch", key=f"{key_prefix}_save_judge"):
            saved = save_prompt_version(task_type, candidate_judge_prompt, version_name)
            st.success(f"已保存：prompts/judge/{saved}。注意：尚未自动启用，请到对应页面手动选择。")

    st.markdown("**提取提示词建议**")
    st.write(result.get("extraction_prompt_notes") or "无")

    if result.get("advisor_flow") == "two_stage_extraction_prompt_advisor":
        with st.expander("分批定位与段落级提示词改进过程", expanded=True):
            intents = result.get("extraction_prompt_patch_intents") or []
            plan = result.get("extraction_prompt_patch_plan") or []
            stage2 = result.get("extraction_prompt_stage2_summaries") or []
            request_metrics = result.get("extraction_prompt_request_metrics") or []
            conflicts = result.get("extraction_prompt_patch_conflicts") or []
            skipped_before_apply = result.get("extraction_prompt_patch_skipped_before_apply") or []
            stage_errors = result.get("extraction_prompt_stage_errors") or []
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("定位意图", len(intents))
            c2.metric("目标章节", len(plan))
            c3.metric("章节生成", len(stage2))
            c4.metric("冲突/错误", len(conflicts) + len(stage_errors))
            if intents:
                with st.expander("阶段1：定位意图", expanded=False):
                    st.dataframe(pd.DataFrame(intents), width="stretch", hide_index=True)
            if plan:
                with st.expander("本地合并后的章节计划", expanded=False):
                    st.dataframe(pd.DataFrame(plan), width="stretch", hide_index=True)
            if stage2:
                with st.expander("阶段2：段落级 patch 生成结果", expanded=False):
                    st.dataframe(pd.DataFrame(stage2), width="stretch", hide_index=True)
            if request_metrics:
                with st.expander("各批次请求大小与状态", expanded=False):
                    st.dataframe(pd.DataFrame(request_metrics), width="stretch", hide_index=True)
            if conflicts:
                with st.expander("冲突修改（未自动采用）", expanded=True):
                    st.dataframe(pd.DataFrame(conflicts), width="stretch", hide_index=True)
            if skipped_before_apply:
                with st.expander("应用前跳过项", expanded=False):
                    st.dataframe(pd.DataFrame(skipped_before_apply), width="stretch", hide_index=True)
            if stage_errors:
                with st.expander("阶段错误", expanded=True):
                    st.dataframe(pd.DataFrame(stage_errors), width="stretch", hide_index=True)

    patch_result = result.get("extraction_prompt_patch_result") or {}
    if patch_result:
        st.markdown("**提取提示词增量修改 Patch**")
        c1, c2, c3 = st.columns(3)
        c1.metric("已应用", len(patch_result.get("applied_edits") or []))
        c2.metric("未应用", len(patch_result.get("skipped_edits") or []))
        c3.metric("修改比例", f"{float(patch_result.get('change_ratio') or 0) * 100:.1f}%")

        applied_edits = patch_result.get("applied_edits") or []
        if applied_edits:
            with st.expander("查看已应用 patch", expanded=True):
                st.dataframe(pd.DataFrame(applied_edits), width="stretch", hide_index=True)
        skipped_edits = patch_result.get("skipped_edits") or []
        if skipped_edits:
            with st.expander("查看未应用 patch 和原因", expanded=True):
                st.dataframe(pd.DataFrame(skipped_edits), width="stretch", hide_index=True)

        diff_text = result.get("extraction_prompt_diff") or patch_result.get("diff") or ""
        if diff_text:
            with st.expander("查看提取提示词 diff", expanded=True):
                st.caption("说明：diff 中行首的 + 表示新增行，- 表示删除行，不会作为提示词正文写入。下方“应用 patch 后的候选提取提示词内容”才是最终候选正文。")
                st.code(diff_text, language="diff")
        else:
            st.info("增量 patch 未造成文本差异。")

    model_candidate = str(result.get("model_candidate_extraction_prompt") or "")
    if model_candidate and result.get("candidate_prompt_source") != "applied_incremental_patch":
        with st.expander("模型返回的完整候选提取提示词（未自动采用）", expanded=False):
            st.warning("模型返回了完整重写版本，但没有通过增量 patch 校验；系统不会默认采用。")
            st.text_area(
                "模型完整候选原文",
                value=model_candidate,
                height=220,
                key=f"{key_prefix}_model_candidate_extraction_prompt",
            )

    candidate_extraction_prompt = str(result.get("candidate_extraction_prompt") or "")
    if candidate_extraction_prompt:
        st.text_area(
            "应用 patch 后的候选提取提示词内容",
            value=candidate_extraction_prompt,
            height=260,
            key=f"{key_prefix}_candidate_extraction_prompt",
        )
        extraction_version_name = st.text_input(
            "保存为新的提取提示词文件名",
            value=f"extract_{_prompt_task_slug(task_type)}_advised_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            key=f"{key_prefix}_extract_save_name",
        )
        if st.button("保存候选提取提示词为新版本", width="stretch", key=f"{key_prefix}_save_extract"):
            saved = save_prompt_version(
                task_type,
                candidate_extraction_prompt,
                extraction_version_name,
                prompt_kind="extraction",
            )
            st.success(f"已保存：prompts/generation/{saved}。注意：尚未自动启用，请到对应页面手动选择。")

    with st.expander("原始模型输出", expanded=False):
        st.code(raw or "", language="json")

    render_next_actions([
        NextAction("pages/1_配置.py", "比较并管理提示词版本", ":material/settings:"),
        NextAction("pages/10_记忆提取.py", "用新版本重新提取", ":material/memory:"),
        NextAction("pages/9_闭环实验.py", "进入闭环实验", ":material/autorenew:"),
    ])


def load_prompt_selectors(
    key_prefix: str,
    default_judge_prompt: str = "",
    task_type: str = TaskType.USER_MD.value,
) -> tuple[str, str]:
    prompt_files = list_prompt_files()
    default_prompt = default_judge_prompt or get_default_prompt_file(task_type)
    selected_prompt = st.selectbox(
        "裁判提示词文件",
        prompt_files,
        index=prompt_files.index(default_prompt) if default_prompt in prompt_files else 0,
        key=f"{key_prefix}_judge_prompt_file",
    )

    extraction_files = list_extraction_prompt_files()
    configured_task = st.session_state.get("selected_prompt_task_type")
    configured_extraction = st.session_state.get("selected_extraction_prompt_file", "")
    default_extraction = (
        configured_extraction
        if configured_task == task_type and configured_extraction
        else get_default_extraction_prompt_file(task_type)
    )
    extraction_options = ["不提供提取提示词"] + extraction_files
    selected_extraction = st.selectbox(
        "提取提示词文件",
        extraction_options,
        index=extraction_options.index(default_extraction) if default_extraction in extraction_options else 0,
        key=f"{key_prefix}_extraction_prompt_file",
        help="这里放生成 USER.md 或 MEMORY.md 时使用的原版提取提示词。建议模型会把它作为规则来源分析，不会当作用户事实来源。",
    )
    if selected_extraction == "不提供提取提示词":
        extraction_default_text = ""
    elif (
        configured_task == task_type
        and selected_extraction == st.session_state.get("selected_extraction_prompt_file")
        and st.session_state.get("extraction_prompt_text")
    ):
        extraction_default_text = st.session_state.get("extraction_prompt_text", "")
    else:
        extraction_default_text = load_prompt(selected_extraction, prompt_kind="extraction")
    st.caption(
        f"裁判提示词版本：{infer_prompt_version(selected_prompt)}；"
        f"提取提示词版本：{infer_prompt_version(selected_extraction) if selected_extraction != '不提供提取提示词' else '未提供'}"
    )
    with st.expander("查看或临时编辑提示词全文", expanded=False):
        st.caption("这里的修改只影响本次建议任务，不会覆盖磁盘中的提示词文件。")
        judge_prompt = st.text_area(
            "当前裁判提示词内容",
            value=load_prompt(selected_prompt),
            height=300,
            key=f"{key_prefix}_judge_prompt_text::{selected_prompt}",
        )
        extraction_prompt = st.text_area(
            "当前提取提示词内容",
            value=extraction_default_text,
            height=240,
            key=f"{key_prefix}_extraction_prompt_text::{selected_extraction}",
        )
    return judge_prompt, extraction_prompt


def render_generation_controls(key_prefix: str) -> tuple[bool, int, str]:
    cfg = dict(st.session_state.ui_config)
    with st.expander("生成设置", expanded=False):
        mock = st.checkbox("模拟模式（只生成占位建议）", value=bool(cfg.get("mock", False)), key=f"{key_prefix}_mock")
        min_evidence = st.number_input(
            "最少证据条数",
            min_value=1,
            max_value=20,
            value=3,
            step=1,
            key=f"{key_prefix}_min_evidence",
            help="证据太少时不建议生成候选提示词，避免根据个例过拟合。",
        )
        target_options = {
            "只给评测诊断，不生成候选提示词": "analysis_only",
            "只优化裁判提示词": "judge_prompt",
            "只优化提取提示词": "extraction_prompt",
            "两个都给建议": "both",
        }
        target_label = st.selectbox("希望优化的对象", list(target_options.keys()), key=f"{key_prefix}_target")
    return mock, int(min_evidence), target_options[target_label]


def render_prompt_advisor_job_state(job_id: str) -> None:
    state = read_prompt_advisor_job_state(job_id)
    if prompt_advisor_job_is_stale(state):
        state = mark_prompt_advisor_job_interrupted(job_id)
    if not state:
        st.info("暂无这个提示词建议任务的状态。")
        return
    render_state_file_notice(state)

    status = str(state.get("status") or "")
    done = int(state.get("done", 0) or 0)
    total = int(state.get("total", 0) or 0)
    progress = done / total if total else 0.0

    st.subheader("后台提示词建议进度")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("状态", status or "-")
    c2.metric("阶段", state.get("stage", "-"))
    c3.metric("进度", f"{done}/{total}" if total else "准备中")
    c4.metric("更新时间", str(state.get("updated_at", ""))[:19])
    st.progress(progress)
    st.write(state.get("message", ""))
    if state.get("source_name"):
        st.caption(f"证据来源：{state.get('source_name')}")

    if status == "running":
        st.info("任务仍在后台运行。切换页面后再回来，进度会从状态文件恢复。")
        if st.button("请求终止提示词建议", type="secondary", width="stretch", key=f"{job_id}_stop"):
            request_prompt_advisor_stop(job_id)
            st.warning("已写入终止请求。若任务正在等待或下一次调用前，会尽快停止；正在进行的单次模型调用无法强制中断。")
            st.rerun()
    elif status == "interrupted":
        st.warning("任务状态为已中断。通常是程序关闭或后台线程退出导致；可以重新生成建议。")
    elif status == "stopped":
        st.warning("任务已终止。")
    elif status == "corrupt":
        st.error(state.get("message") or "任务状态文件损坏。")

    if status in {"completed", "failed", "interrupted", "stopped"}:
        result, raw = load_prompt_advisor_job_result(job_id, state)
        if result:
            config = state.get("config") if isinstance(state.get("config"), dict) else {}
            render_advisor_result(
                result,
                raw,
                f"advisor_job_{job_id}",
                task_type=str(config.get("task_type") or TaskType.USER_MD.value),
            )

    if state.get("traceback"):
        with st.expander("错误堆栈", expanded=True):
            st.code(state.get("traceback", ""), language="text")


@st.fragment(run_every="10s")
def render_prompt_advisor_job_state_auto(job_id: str) -> None:
    render_prompt_advisor_job_state(job_id)


def render_prompt_advisor_job_panel() -> str:
    job_ids = list_prompt_advisor_job_ids()
    if not job_ids:
        return ""
    last_job_id = st.session_state.get("prompt_advisor_job_id", "") or job_ids[0]
    index = job_ids.index(last_job_id) if last_job_id in job_ids else 0
    selected_job_id = st.selectbox("查看后台提示词建议任务", job_ids, index=index)
    st.session_state.prompt_advisor_job_id = selected_job_id
    state = read_prompt_advisor_job_state(selected_job_id)
    if state.get("status") == "running":
        auto_refresh = st.checkbox(
            "运行中每10秒自动刷新进度区",
            value=False,
            key=f"{selected_job_id}_advisor_auto_refresh",
            help="只刷新下面的进度区域，不刷新整个页面。",
        )
        if auto_refresh:
            render_prompt_advisor_job_state_auto(selected_job_id)
        else:
            render_prompt_advisor_job_state(selected_job_id)
    else:
        render_prompt_advisor_job_state(selected_job_id)
    return selected_job_id


tab_absolute = st.container()

with tab_absolute:
    st.subheader("单模型绝对评测建议")
    st.info(
        "这个入口只分析普通“执行评测”的单模型绝对评测结果。"
        "它会基于低分、错误标签、diagnostics、规则/证据/输出引用来给评测建议。"
        "注意：没有人工复核时，这些建议是待人工确认的诊断，不应直接当成正确修复。"
    )

    result_source = st.radio("结果来源", ["历史结果文件", "上传结果文件"], horizontal=True, key="abs_result_source")
    absolute_results = []
    absolute_source_name = ""
    if result_source == "历史结果文件":
        result_files = list_result_files()
        if result_files:
            labels = [Path(f).name for f in result_files]
            selected_result = st.selectbox("选择普通评测结果文件", labels, key="abs_result_file")
            result_path = result_files[labels.index(selected_result)]
            absolute_results = load_results(result_path)
            absolute_source_name = selected_result
        else:
            st.warning("data/results 下暂无普通评测结果文件。")
    else:
        uploaded_results = st.file_uploader(
            "上传普通评测结果",
            type=["jsonl", "csv", "xlsx"],
            key="abs_upload",
            help="支持执行评测 JSONL，以及结果总览导出的 CSV/Excel。",
        )
        if uploaded_results is not None:
            try:
                absolute_results = load_results_bytes(uploaded_results.getvalue(), uploaded_results.name)
                absolute_source_name = uploaded_results.name
                st.success(f"已读取 {len(absolute_results)} 条结果。")
            except Exception as exc:
                st.error(f"解析失败：{exc}")

    with st.expander("证据筛选", expanded=True):
        no_gate_extraction_loop = st.checkbox(
            "实验：无门槛生成提取提示词候选",
            value=False,
            key="abs_no_gate_extraction_loop",
            help="开启后不要求低分/错误证据，也不要求至少 3 条；会把当前结果作为弱证据，强制生成提取提示词候选。适合探索闭环效果，但风险较高。",
        )
        if no_gate_extraction_loop:
            st.warning(
                "高风险实验模式：候选提取提示词只适合另存为新版本再跑下一轮，"
                "不能直接覆盖线上版本；它可能沿着当前 Judge 的偏差自我强化。"
            )
        score_threshold = st.number_input(
            "低分证据阈值",
            min_value=0.0,
            max_value=5.0,
            value=4.8,
            step=0.1,
            help="低于该总分的样本会进入证据池；fatal、error_tags、diagnostics 样本也会进入。",
            key="abs_score_threshold",
            disabled=no_gate_extraction_loop,
        )
        max_items = st.number_input("最多使用证据条数", min_value=1, max_value=200, value=40, step=1, key="abs_max_items")

    absolute_candidates = collect_absolute_eval_evidence(
        absolute_results,
        max_items=int(max_items),
        score_threshold=float(score_threshold),
        include_all=bool(no_gate_extraction_loop),
        positive_boundary_limit=0 if no_gate_extraction_loop else min(3, max(1, int(max_items) // 10)),
    ) if absolute_results else []
    absolute_evidence = absolute_candidates
    evidence_composition: dict[str, int] = {}
    for item in absolute_evidence:
        mode = str(item.get("evidence_mode") or "unknown")
        evidence_composition[mode] = evidence_composition.get(mode, 0) + 1
    actionable_evidence_count = sum(
        count for mode, count in evidence_composition.items()
        if mode not in {"positive_boundary", "regression_boundary", "weak_context_from_result"}
    )
    absolute_task_type = infer_single_task_type(absolute_results) if absolute_results else TaskType.USER_MD.value

    if absolute_results:
        result_task_types = sorted({str(getattr(item, "task_type", "") or "") for item in absolute_results})
        if len(result_task_types) > 1:
            st.warning(f"当前结果包含多个任务类型：{result_task_types}。提示词选择默认按 USER.md 处理，建议拆分后分别生成建议。")
        st.caption(f"当前普通评测任务：{TASK_TYPE_LABELS.get(absolute_task_type, absolute_task_type)}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("结果总数", len(absolute_results))
        c2.metric("符合条件", len(absolute_candidates))
        c3.metric("本次已选择", len(absolute_evidence))
        c4.metric("设置上限", int(max_items))
        st.caption(f"来源：{absolute_source_name or '未命名'}。证据按问题严重程度排序后取前 {int(max_items)} 条。")
        st.caption(
            f"证据组成：问题证据 {actionable_evidence_count}；正例/回归/弱上下文边界 "
            f"{len(absolute_evidence) - actionable_evidence_count}。运行失败已自动排除。"
        )

    if no_gate_extraction_loop:
        st.caption(f"当前会用于生成提取提示词候选的结果上下文：{len(absolute_evidence)} 条")
    else:
        st.caption(f"当前会用于生成建议的普通评测证据：{len(absolute_evidence)} 条")
    if absolute_evidence:
        preview_cols = [
            "evidence_mode", "case_id", "score_total", "fatal_error", "error_tags", "comment",
            "rule_refs", "evidence_refs", "output_refs",
        ]
        st.dataframe(
            pd.DataFrame(absolute_evidence).head(30)[[c for c in preview_cols if c in absolute_evidence[0]]],
            width="stretch",
            hide_index=True,
        )

    st.divider()
    st.subheader("当前提示词")
    absolute_judge_prompt, absolute_extraction_prompt = load_prompt_selectors(
        "absolute",
        default_judge_prompt=get_default_prompt_file(absolute_task_type),
        task_type=absolute_task_type,
    )
    abs_mock, abs_min_evidence, abs_target = render_generation_controls("absolute")
    if no_gate_extraction_loop:
        abs_min_evidence = 0
        abs_target = "extraction_prompt"
        st.caption("实验模式已开启：本次会跳过证据条数门槛，并强制目标为“只优化提取提示词”。")

    cfg = dict(st.session_state.ui_config)
    config = build_eval_config(cfg, mock=abs_mock)
    render_workspace_context(
        task_type=absolute_task_type,
        case_count=len(absolute_results),
        cases_file=absolute_source_name,
        model_name=cfg.get("judge_model", ""),
        judge_prompt=st.session_state.get("absolute_judge_prompt_file", ""),
        extraction_prompt=st.session_state.get("absolute_extraction_prompt_file", ""),
        mock=abs_mock,
        title="本次提示词建议上下文",
    )
    advisor_checks = build_advisor_preflight(
        results_count=len(absolute_results),
        evidence_count=actionable_evidence_count if not no_gate_extraction_loop else len(absolute_evidence),
        min_evidence=abs_min_evidence,
        target=abs_target,
        judge_prompt_text=absolute_judge_prompt,
        extraction_prompt_text=absolute_extraction_prompt,
        eval_config=config,
    )
    advisor_ready = render_preflight(advisor_checks)

    if st.button(
        "生成单模型评测建议",
        type="primary",
        width="stretch",
        key="abs_generate",
        disabled=not advisor_ready,
    ):
        if not absolute_results:
            st.error("请先加载普通评测结果文件。无门槛模式也需要基于一批结果生成候选提取提示词。")
            st.stop()
        if no_gate_extraction_loop and not absolute_extraction_prompt.strip():
            st.error("无门槛生成提取提示词候选需要提供当前提取提示词，否则模型会编造完整 prompt。请先选择或粘贴提取提示词。")
            st.stop()
        if actionable_evidence_count < abs_min_evidence:
            st.error(f"问题证据少于 {abs_min_evidence} 条，拒绝生成候选提示词。正例边界不计入门槛。")
            st.stop()

        job_id = f"advisor_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        job_config = PromptAdvisorJobConfig(
            job_id=job_id,
            task_type=absolute_task_type,
            evidence=absolute_evidence,
            current_judge_prompt=absolute_judge_prompt,
            extraction_prompt=absolute_extraction_prompt,
            target=abs_target,
            advisor_mode="absolute_eval",
            min_evidence=abs_min_evidence,
            source_name=absolute_source_name,
            eval_config=config,
        )
        launch_background_task("prompt_advisor", job_config)
        st.session_state.prompt_advisor_job_id = job_id
        st.success(f"已启动独立后台提示词建议进程：{job_id}")
        st.rerun()

    st.divider()
    render_prompt_advisor_job_panel()
