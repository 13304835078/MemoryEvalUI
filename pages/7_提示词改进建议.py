from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.config_store import build_eval_config, load_config
from src.ui.data_service import list_result_files, load_results, load_results_bytes, save_uploaded_file
from src.ui.prompt_advisor import (
    call_prompt_advisor,
    collect_absolute_eval_evidence,
    collect_gsb_evidence,
    collect_review_evidence,
    load_prompt_advisor_table,
)
from src.ui.prompt_patch import apply_prompt_patch
from src.ui.prompt_editor import (
    get_default_extraction_prompt_file,
    get_default_prompt_file,
    infer_prompt_version,
    list_extraction_prompt_files,
    list_prompt_files,
    load_prompt,
    save_prompt_version,
)
from src.ui.review_store import DEFAULT_REVIEW_PATH, load_reviews, reviews_to_dataframe


st.title("提示词改进建议")
st.caption("普通绝对评测建议和人工 GSB 对齐建议已分离；只生成候选版本，不会自动覆盖当前提示词。")

if "ui_config" not in st.session_state:
    st.session_state.ui_config = load_config()
if "absolute_advisor_result" not in st.session_state:
    st.session_state.absolute_advisor_result = None
if "gsb_advisor_result" not in st.session_state:
    st.session_state.gsb_advisor_result = None


def render_advisor_result(result: dict | None, raw: str, key_prefix: str) -> None:
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
                st.dataframe(pd.DataFrame(usage_requests), use_container_width=True, hide_index=True)
    if result.get("error"):
        st.error(f"生成失败原因：{result.get('error')}")

    diagnoses = result.get("diagnoses", [])
    st.markdown("**问题归因**")
    if diagnoses:
        st.dataframe(pd.DataFrame(diagnoses), use_container_width=True, hide_index=True)
    else:
        st.info("暂无问题归因。")

    changes = result.get("judge_prompt_changes", [])
    st.markdown("**裁判提示词修改点**")
    if changes:
        st.dataframe(pd.DataFrame(changes), use_container_width=True, hide_index=True)
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
            value=f"judge_user_md_advised_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            key=f"{key_prefix}_judge_save_name",
        )
        if st.button("保存候选裁判提示词为新版本", use_container_width=True, key=f"{key_prefix}_save_judge"):
            saved = save_prompt_version("user_md_update", candidate_judge_prompt, version_name)
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
                    st.dataframe(pd.DataFrame(intents), use_container_width=True, hide_index=True)
            if plan:
                with st.expander("本地合并后的章节计划", expanded=False):
                    st.dataframe(pd.DataFrame(plan), use_container_width=True, hide_index=True)
            if stage2:
                with st.expander("阶段2：段落级 patch 生成结果", expanded=False):
                    st.dataframe(pd.DataFrame(stage2), use_container_width=True, hide_index=True)
            if request_metrics:
                with st.expander("各批次请求大小与状态", expanded=False):
                    st.dataframe(pd.DataFrame(request_metrics), use_container_width=True, hide_index=True)
            if conflicts:
                with st.expander("冲突修改（未自动采用）", expanded=True):
                    st.dataframe(pd.DataFrame(conflicts), use_container_width=True, hide_index=True)
            if skipped_before_apply:
                with st.expander("应用前跳过项", expanded=False):
                    st.dataframe(pd.DataFrame(skipped_before_apply), use_container_width=True, hide_index=True)
            if stage_errors:
                with st.expander("阶段错误", expanded=True):
                    st.dataframe(pd.DataFrame(stage_errors), use_container_width=True, hide_index=True)

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
                st.dataframe(pd.DataFrame(applied_edits), use_container_width=True, hide_index=True)
        skipped_edits = patch_result.get("skipped_edits") or []
        if skipped_edits:
            with st.expander("查看未应用 patch 和原因", expanded=True):
                st.dataframe(pd.DataFrame(skipped_edits), use_container_width=True, hide_index=True)

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
            value=f"extract_user_md_advised_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            key=f"{key_prefix}_extract_save_name",
        )
        if st.button("保存候选提取提示词为新版本", use_container_width=True, key=f"{key_prefix}_save_extract"):
            saved = save_prompt_version(
                "user_md_update",
                candidate_extraction_prompt,
                extraction_version_name,
                prompt_kind="extraction",
            )
            st.success(f"已保存：prompts/generation/{saved}。注意：尚未自动启用，请到对应页面手动选择。")

    with st.expander("原始模型输出", expanded=False):
        st.code(raw or "", language="json")


def load_prompt_selectors(key_prefix: str, default_judge_prompt: str = "") -> tuple[str, str]:
    prompt_files = list_prompt_files()
    default_prompt = default_judge_prompt or get_default_prompt_file("user_md_update")
    selected_prompt = st.selectbox(
        "裁判提示词文件",
        prompt_files,
        index=prompt_files.index(default_prompt) if default_prompt in prompt_files else 0,
        key=f"{key_prefix}_judge_prompt_file",
    )
    judge_prompt = st.text_area(
        "当前裁判提示词内容",
        value=load_prompt(selected_prompt),
        height=300,
        key=f"{key_prefix}_judge_prompt_text",
    )

    extraction_files = list_extraction_prompt_files()
    default_extraction = st.session_state.get("selected_extraction_prompt_file", "") or get_default_extraction_prompt_file("user_md_update")
    extraction_options = ["不提供提取提示词"] + extraction_files
    selected_extraction = st.selectbox(
        "提取提示词文件",
        extraction_options,
        index=extraction_options.index(default_extraction) if default_extraction in extraction_options else 0,
        key=f"{key_prefix}_extraction_prompt_file",
        help="这里放生成 USER.md 时使用的原版提取提示词。建议模型会把它作为规则来源分析，不会当作用户事实来源。",
    )
    if selected_extraction == "不提供提取提示词":
        extraction_default_text = ""
    elif (
        selected_extraction == st.session_state.get("selected_extraction_prompt_file")
        and st.session_state.get("extraction_prompt_text")
    ):
        extraction_default_text = st.session_state.get("extraction_prompt_text", "")
    else:
        extraction_default_text = load_prompt(selected_extraction, prompt_kind="extraction")
    extraction_prompt = st.text_area(
        "当前提取提示词内容",
        value=extraction_default_text,
        height=240,
        key=f"{key_prefix}_extraction_prompt_text",
    )
    st.caption(
        f"裁判提示词版本：{infer_prompt_version(selected_prompt)}；"
        f"提取提示词版本：{infer_prompt_version(selected_extraction) if selected_extraction != '不提供提取提示词' else '未提供'}"
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


tab_absolute, tab_gsb = st.tabs(["单模型绝对评测建议", "GSB对齐建议"])

with tab_absolute:
    st.subheader("单模型绝对评测建议")
    st.info(
        "这个入口只分析普通“执行评测”的单模型结果，不使用 GSB。"
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
        max_items=max(1, len(absolute_results)),
        score_threshold=float(score_threshold),
        include_all=bool(no_gate_extraction_loop),
    ) if absolute_results else []
    absolute_evidence = absolute_candidates[:int(max_items)]

    if absolute_results:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("结果总数", len(absolute_results))
        c2.metric("符合条件", len(absolute_candidates))
        c3.metric("本次已选择", len(absolute_evidence))
        c4.metric("设置上限", int(max_items))
        st.caption(f"来源：{absolute_source_name or '未命名'}。证据按问题严重程度排序后取前 {int(max_items)} 条。")

    if no_gate_extraction_loop:
        st.caption(f"当前会用于生成提取提示词候选的结果上下文：{len(absolute_evidence)} 条")
    else:
        st.caption(f"当前会用于生成建议的普通评测证据：{len(absolute_evidence)} 条")
    if absolute_evidence:
        preview_cols = [
            "case_id", "score_total", "fatal_error", "error_tags", "comment",
            "rule_refs", "evidence_refs", "output_refs",
        ]
        st.dataframe(
            pd.DataFrame(absolute_evidence).head(30)[[c for c in preview_cols if c in absolute_evidence[0]]],
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.subheader("当前提示词")
    absolute_judge_prompt, absolute_extraction_prompt = load_prompt_selectors(
        "absolute",
        default_judge_prompt="judge_user_md_absolute_stable_with_rules_v1.md",
    )
    abs_mock, abs_min_evidence, abs_target = render_generation_controls("absolute")
    if no_gate_extraction_loop:
        abs_min_evidence = 0
        abs_target = "extraction_prompt"
        st.caption("实验模式已开启：本次会跳过证据条数门槛，并强制目标为“只优化提取提示词”。")

    if st.button("生成单模型评测建议", type="primary", use_container_width=True, key="abs_generate"):
        if not absolute_results:
            st.error("请先加载普通评测结果文件。无门槛模式也需要基于一批结果生成候选提取提示词。")
            st.stop()
        if no_gate_extraction_loop and not absolute_extraction_prompt.strip():
            st.error("无门槛生成提取提示词候选需要提供当前提取提示词，否则模型会编造完整 prompt。请先选择或粘贴提取提示词。")
            st.stop()
        if len(absolute_evidence) < abs_min_evidence:
            st.error(f"证据少于 {abs_min_evidence} 条，拒绝生成候选提示词。可以降低阈值或积累更多评测结果。")
            st.stop()

        if abs_mock:
            result = {
                "can_suggest": True,
                "evidence_summary": f"Mock：已收集 {len(absolute_evidence)} 条普通评测证据。",
                "diagnoses": [
                    {
                        "problem": "示例：某类低分样本的规则引用或证据引用不够稳定。",
                        "evidence_refs": [str(absolute_evidence[0].get("case_id"))],
                        "problem_type": "judge_prompt_issue",
                        "why_it_matters": "会影响评测解释一致性。",
                        "confidence": "low",
                    }
                ],
                "judge_prompt_changes": [],
                "candidate_judge_prompt": absolute_judge_prompt if abs_target != "analysis_only" else "",
                "extraction_prompt_notes": "模拟模式不生成真实建议。",
                "extraction_prompt_patch": {
                    "mode": "incremental_patch",
                    "edits": [],
                },
                "candidate_extraction_prompt": "",
                "risks": ["Mock 结果不能用于真实调参。", "无门槛闭环可能沿着 Judge 偏差自我强化。"],
                "validation_plan": ["用同一批结果重新跑稳定性对比，并抽样人工复核。"],
                "evidence_usage": {
                    "selected_count": len(absolute_evidence),
                    "initial_used_count": len(absolute_evidence),
                    "all_selected_used_initially": True,
                    "request_count": 0,
                    "request_metrics": [],
                },
            }
            if no_gate_extraction_loop and absolute_extraction_prompt.strip():
                result["extraction_prompt_patch"] = {
                    "mode": "incremental_patch",
                    "edits": [
                        {
                            "op": "append_to_section",
                            "target_id": "S001",
                            "text": "<!-- MOCK: 示例增量 patch，真实运行不会插入这段。 -->",
                            "reason": "模拟模式用于验证 patch 展示。",
                            "evidence_refs": [str(absolute_evidence[0].get("case_id") if absolute_evidence else "mock_case")],
                        }
                    ],
                }
                patch_result = apply_prompt_patch(absolute_extraction_prompt, result["extraction_prompt_patch"])
                result["extraction_prompt_patch_result"] = patch_result
                result["extraction_prompt_diff"] = patch_result.get("diff", "")
                result["candidate_extraction_prompt"] = patch_result.get("candidate_prompt", "") if patch_result.get("applied_edits") else ""
                result["candidate_prompt_source"] = "applied_incremental_patch" if result["candidate_extraction_prompt"] else "no_valid_incremental_patch"
            raw = ""
        else:
            cfg = dict(st.session_state.ui_config)
            config = build_eval_config(cfg, mock=False)
            errs = config.validate()
            if errs:
                st.error("配置错误：\n" + "\n".join([f"- {e}" for e in errs]))
                st.stop()
            with st.spinner("正在生成单模型评测建议..."):
                result, raw = call_prompt_advisor(
                    config,
                    evidence=absolute_evidence,
                    current_judge_prompt=absolute_judge_prompt,
                    extraction_prompt=absolute_extraction_prompt,
                    target=abs_target,
                    advisor_mode="absolute_eval",
                    min_evidence=abs_min_evidence,
                )

        st.session_state.absolute_advisor_result = result
        st.session_state.absolute_advisor_raw = raw

    render_advisor_result(
        st.session_state.get("absolute_advisor_result"),
        st.session_state.get("absolute_advisor_raw", "") or "",
        "absolute",
    )

with tab_gsb:
    st.subheader("GSB 对齐建议")
    st.warning(
        "这个入口只用于人工审核/GSB 闭环。它会基于人工 GSB 与自动 GSB 不一致样本，"
        "或样本详情页保存的人工复核记录生成建议。普通单模型评测请使用左侧标签页。"
    )

    source_mode = st.radio(
        "选择证据来源",
        [
            "上传人工审核评估结果表",
            "加载普通评测人工复核记录",
            "加载已有结果文件（需要合并人工复核记录）",
        ],
        key="gsb_source_mode",
    )

    df = pd.DataFrame()
    if source_mode == "上传人工审核评估结果表":
        uploaded = st.file_uploader("上传人工审核评估结果（Excel/CSV/JSONL）", type=["xlsx", "xls", "csv", "jsonl"], key="gsb_upload")
        local_path = st.text_input("或输入本地结果文件路径", value="", placeholder=r"C:\Users\...\human_review_eval.xlsx", key="gsb_local_path")
        if uploaded is not None or local_path.strip():
            try:
                if uploaded is not None:
                    path = save_uploaded_file(uploaded, suffix=Path(uploaded.name).suffix)
                else:
                    path = local_path.strip().strip('"')
                df = load_prompt_advisor_table(path)
                st.success(f"已加载 {len(df)} 行：{path}")
            except Exception as e:
                st.error(f"加载失败：{e}")
    elif source_mode == "加载普通评测人工复核记录":
        reviews = load_reviews(DEFAULT_REVIEW_PATH)
        df = reviews_to_dataframe(reviews)
        if df.empty:
            st.info(f"暂无人工复核记录：{DEFAULT_REVIEW_PATH}")
        else:
            st.success(f"已加载 {len(df)} 条人工复核记录：{DEFAULT_REVIEW_PATH}")
    else:
        result_files = list_result_files()
        if result_files:
            labels = [Path(f).name for f in result_files]
            selected = st.selectbox("选择结果文件", labels, key="gsb_result_reference")
            st.info(f"结果文件仅用于记录来源：{selected}；实际证据来自 data/results/human_reviews.jsonl。")
            reviews = load_reviews(DEFAULT_REVIEW_PATH)
            df = reviews_to_dataframe(reviews)
            if df.empty:
                st.warning("没有人工复核记录，无法生成建议。")
            else:
                st.success(f"已加载 {len(df)} 条人工复核记录。")
        else:
            st.info("data/results 下暂无结果文件。")

    evidence_kind = st.selectbox(
        "证据类型",
        ["自动识别", "人工 GSB 不一致", "普通评测人工复核"],
        help="上传人工审核评估结果时通常选“人工 GSB 不一致”；样本详情页保存的人工复核记录选“普通评测人工复核”。",
        key="gsb_evidence_kind",
    )
    max_items_gsb = st.number_input("最多使用证据条数", min_value=3, max_value=100, value=30, step=1, key="gsb_max_items")

    gsb_evidence = []
    gsb_evidence_count = 0
    review_evidence_count = 0
    if not df.empty:
        gsb_evidence_count = len(collect_gsb_evidence(df, max_items=100000))
        review_evidence_count = len(collect_review_evidence(df, max_items=100000))
        if evidence_kind in {"自动识别", "人工 GSB 不一致"}:
            gsb_evidence = collect_gsb_evidence(df, max_items=int(max_items_gsb))
        if not gsb_evidence and evidence_kind in {"自动识别", "普通评测人工复核"}:
            gsb_evidence = collect_review_evidence(df, max_items=int(max_items_gsb))

    if not df.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("已加载行数", len(df))
        c2.metric("GSB不一致证据", gsb_evidence_count)
        c3.metric("人工复核证据", review_evidence_count)

    st.caption(f"当前会用于生成建议的 GSB/人工复核证据：{len(gsb_evidence)} 条")
    if gsb_evidence:
        evidence_columns = {
            "case_id": "样本编号",
            "row_id": "行编号",
            "issue_type": "问题类型",
            "human_gsb": "人工GSB",
            "auto_gsb": "自动GSB",
            "review_decision": "复核结论",
            "review_note": "复核备注",
            "query": "用户问题",
            "answer": "助手回答",
        }
        st.dataframe(pd.DataFrame(gsb_evidence).head(30).rename(columns=evidence_columns), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("当前提示词")
    gsb_judge_prompt, gsb_extraction_prompt = load_prompt_selectors(
        "gsb",
        default_judge_prompt="judge_user_md_human_aligned_v1.md",
    )
    gsb_mock, gsb_min_evidence, gsb_target = render_generation_controls("gsb")

    if st.button("生成 GSB 对齐建议", type="primary", use_container_width=True, key="gsb_generate"):
        if len(gsb_evidence) < gsb_min_evidence:
            st.error(
                f"人工证据少于 {gsb_min_evidence} 条，拒绝生成候选提示词。"
                "这里需要人工 GSB 与自动 GSB 不一致样本，或样本详情页保存的人工复核记录。"
            )
            st.stop()

        if gsb_mock:
            result = {
                "can_suggest": True,
                "evidence_summary": f"Mock：已收集 {len(gsb_evidence)} 条 GSB/人工复核证据。",
                "diagnoses": [
                    {
                        "problem": "示例：裁判模型对某类错误的扣分力度可能与人工不一致。",
                        "evidence_refs": [str(gsb_evidence[0].get("row_id") or gsb_evidence[0].get("case_id"))],
                        "problem_type": "judge_prompt_issue",
                        "why_it_matters": "会影响人工 GSB 对齐。",
                        "confidence": "low",
                    }
                ],
                "judge_prompt_changes": [],
                "candidate_judge_prompt": gsb_judge_prompt if gsb_target != "analysis_only" else "",
                "extraction_prompt_notes": "模拟模式不生成真实建议。",
                "candidate_extraction_prompt": "",
                "risks": ["Mock 结果不能用于真实调参。"],
                "validation_plan": ["用留出集重新跑 GSB 一致率。"],
            }
            raw = ""
        else:
            cfg = dict(st.session_state.ui_config)
            config = build_eval_config(cfg, mock=False)
            errs = config.validate()
            if errs:
                st.error("配置错误：\n" + "\n".join([f"- {e}" for e in errs]))
                st.stop()
            with st.spinner("正在生成 GSB 对齐建议..."):
                result, raw = call_prompt_advisor(
                    config,
                    evidence=gsb_evidence,
                    current_judge_prompt=gsb_judge_prompt,
                    extraction_prompt=gsb_extraction_prompt,
                    target=gsb_target,
                    advisor_mode="gsb_alignment",
                    min_evidence=gsb_min_evidence,
                )

        st.session_state.gsb_advisor_result = result
        st.session_state.gsb_advisor_raw = raw

    render_advisor_result(
        st.session_state.get("gsb_advisor_result"),
        st.session_state.get("gsb_advisor_raw", "") or "",
        "gsb",
    )
