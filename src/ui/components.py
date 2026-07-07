from __future__ import annotations

import difflib
import json

import pandas as pd
import streamlit as st

from src.schema import Case, EvalResult, TaskType
from src.eval.metrics import DIM_LABELS, TAG_LABELS
from src.ui.rule_ref_validation import rule_ref_validation_rows, validate_result_rule_refs


def render_state_file_notice(state: dict | None) -> None:
    if not isinstance(state, dict) or not state.get("_state_error"):
        return
    st.error(state.get("message") or f"状态文件损坏：{state.get('_state_error')}")
    if state.get("_state_corrupt_path"):
        st.caption(f"损坏文件备份：{state.get('_state_corrupt_path')}")


def make_text_diff(old: str | None, new: str | None) -> str:
    old_lines = (old or "").splitlines()
    new_lines = (new or "").splitlines()

    return "\n".join(difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile="old_memory",
        tofile="candidate_output",
        lineterm="",
    ))


def render_score_cards(result: EvalResult) -> None:
    dims = ["correctness", "coverage", "update_logic", "memory_boundary", "conciseness", "format"]
    cols = st.columns(len(dims))

    for col, dim in zip(cols, dims):
        score = result.scores.get(dim, 0)
        label = DIM_LABELS.get(dim, dim)
        col.metric(label, f"{score:.1f}/5")

    st.metric("加权总分", f"{result.score_total:.2f}/5")


def render_rule_ref_validation(result: EvalResult) -> None:
    report = validate_result_rule_refs(result)
    status = report.get("status")

    with st.expander("规则引用校验", expanded=status in {"invalid", "missing", "hash_mismatch"}):
        if status == "ok":
            st.success("规则引用校验通过：解析后的 rule_refs 能在当前提取提示词中找到。")
        elif status == "invalid":
            st.error("发现疑似幻觉规则引用：部分 rule_refs、comment 或原始 Judge 输出中的规则编号/标题不在当前提取提示词中。")
        elif status == "missing":
            st.warning("缺少规则引用：本结果使用了提取规则辅助评测，但解析后的 rule_refs 为空。")
        elif status == "hash_mismatch":
            st.warning("找到了同名提取提示词，但 hash 与结果记录不一致，校验仅供参考。")
        elif status == "no_prompt_found":
            st.info("结果记录了提取提示词版本或 hash，但当前 prompts/generation/ 下未找到对应文件，暂时无法校验。")
        else:
            st.info("本结果未记录提取提示词，不需要做规则引用校验。")

        st.caption(
            f"状态：{report.get('status_label', status)}；提取提示词文件：{report.get('prompt_source') or '未找到'}；"
            f"hash 匹配：{report.get('prompt_hash_match')}"
        )

        rows = rule_ref_validation_rows(report)
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        elif report.get("valid_refs"):
            st.dataframe(
                pd.DataFrame([{"有效规则引用": ref} for ref in report.get("valid_refs", [])]),
                use_container_width=True,
                hide_index=True,
            )


def render_eval_result(result: EvalResult) -> None:
    st.subheader("评测结果")
    document_name = "MEMORY.md" if result.task_type == TaskType.LONG_MEMORY.value else "USER.md"

    if result.fatal_error:
        st.error("严重失败：是")
    else:
        st.success("严重失败：否")

    render_score_cards(result)

    dim_scores = "; ".join(
        f"{DIM_LABELS.get(dim, dim)}={score:.1f}"
        for dim, score in result.scores.items()
    )
    key_rows = [
        {"字段": "加权总分", "内容": f"{result.score_total:.2f}/5"},
        {"字段": "维度得分", "内容": dim_scores or "（无）"},
        {"字段": "comment", "内容": result.comment or "（无）"},
        {"字段": "error_tags", "内容": "; ".join(result.error_tags or []) or "（无）"},
        {"字段": "rule_refs", "内容": "\n".join(result.rule_refs or []) or "（无）"},
        {"字段": "evidence_refs", "内容": "\n".join(result.evidence_refs or []) or "（无）"},
        {"字段": "output_refs / out_refs", "内容": "\n".join(result.output_refs or []) or "（无）"},
        {"字段": "diagnostics_count", "内容": str(len(result.diagnostics or []))},
    ]
    st.markdown("**核心结果字段**")
    st.dataframe(pd.DataFrame(key_rows), use_container_width=True, hide_index=True)
    st.caption(
        f"output_refs / out_refs 指裁判引用的新 {document_name} 候选输出片段；"
        "它不是事实证据，事实证据在 evidence_refs。"
    )

    render_rule_ref_validation(result)

    if result.error_tags:
        tag_text = ", ".join([TAG_LABELS.get(t, t) for t in result.error_tags])
        st.warning(f"错误标签：{tag_text}")
    else:
        st.info("错误标签：无")

    st.markdown("**评语**")
    st.write(result.comment or "（无）")

    meta_cols = st.columns(4)
    meta_cols[0].caption(f"被评测模型：{result.model_name}")
    meta_cols[1].caption(f"生成提示词：{result.prompt_version}")
    meta_cols[2].caption(f"裁判模型：{result.judge_model}")
    meta_cols[3].caption(f"裁判提示词：{result.judge_prompt_version}")

    if result.extraction_prompt_version or result.extraction_prompt_hash:
        extraction_cols = st.columns(2)
        extraction_cols[0].caption(f"提取提示词：{result.extraction_prompt_version or '未记录'}")
        extraction_cols[1].caption(
            f"提取提示词Hash：{result.extraction_prompt_hash[:12] if result.extraction_prompt_hash else '未记录'}"
        )

    if result.diagnostics or result.rule_refs or result.evidence_refs or result.output_refs:
        with st.expander("规则与证据引用", expanded=True):
            if result.diagnostics:
                rows = []
                for item in result.diagnostics:
                    rows.append({
                        "维度": item.get("dimension", ""),
                        "严重程度": item.get("severity", ""),
                        "规则引用": "; ".join(item.get("rule_refs") or []),
                        "证据引用": "; ".join(item.get("evidence_refs") or []),
                        "输出引用": "; ".join(item.get("output_refs") or []),
                        "原因": item.get("reason", ""),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            refs = []
            for label, values in [
                ("规则引用", result.rule_refs),
                ("证据引用", result.evidence_refs),
                ("输出引用（output_refs / out_refs）", result.output_refs),
            ]:
                if values:
                    refs.append({"类型": label, "内容": "\n".join(values)})
            if refs:
                st.dataframe(pd.DataFrame(refs), use_container_width=True, hide_index=True)


def _render_dialogue_turn(turn, i: int) -> None:
    role = turn.role.lower()
    icon = "👤" if role in {"user", "query"} else "🤖"
    name = "用户" if role in {"user", "query"} else "助手"

    with st.chat_message("user" if role in {"user", "query"} else "assistant"):
        st.markdown(f"**{icon} {name} #{i}**")
        st.write(turn.content)

        if getattr(turn, "metadata", None):
            with st.expander("metadata", expanded=False):
                st.json(turn.metadata)


def render_dialogue(case: Case) -> None:
    st.subheader("对话记录")

    if not case.dialogue:
        st.info("无对话记录")
        return

    total = len(case.dialogue)
    preview_count = min(6, total)
    st.caption(f"默认仅展示前 {preview_count} 条，共 {total} 条。完整对话可在下方展开。")

    for i, turn in enumerate(case.dialogue[:preview_count], 1):
        _render_dialogue_turn(turn, i)

    if total > preview_count:
        with st.expander(f"展开完整对话记录（共 {total} 条）", expanded=False):
            for i, turn in enumerate(case.dialogue, 1):
                _render_dialogue_turn(turn, i)


def render_case_input(case: Case) -> None:
    st.subheader("输入与候选输出")
    try:
        case_task_type = TaskType(case.task_type)
    except ValueError:
        case_task_type = case.task_type
    is_long_memory = case_task_type == TaskType.LONG_MEMORY
    document_label = "长期记忆 MEMORY.md" if is_long_memory else "用户画像 USER.md"

    col_old, col_new = st.columns(2)

    with col_old:
        st.markdown(f"**旧{document_label}**")
        st.text_area(
            "old_memory",
            value=case.old_memory or "",
            height=320,
            disabled=True,
            label_visibility="collapsed",
        )

    with col_new:
        st.markdown(f"**候选{document_label}**")
        st.text_area(
            "candidate_output",
            value=case.candidate_output or "",
            height=320,
            disabled=True,
            label_visibility="collapsed",
        )

    with st.expander(f"旧{document_label}和候选{document_label}差异", expanded=False):
        diff_text = make_text_diff(case.old_memory, case.candidate_output)
        st.code(diff_text or "无差异", language="diff")


def render_case_reasoning(case: Case) -> None:
    reasoning = ""
    if isinstance(case.metadata, dict):
        reasoning = str(case.metadata.get("reasoning") or "").strip()

    st.subheader("模型推理过程")
    if reasoning:
        st.text_area(
            "reasoning",
            value=reasoning,
            height=220,
            disabled=True,
            label_visibility="collapsed",
        )
    else:
        st.info("无推理过程")


def render_case_metadata(case: Case) -> None:
    metadata = case.metadata if isinstance(case.metadata, dict) else {}
    if not metadata:
        return

    st.subheader("样本元信息")

    fields = [
        ("source_file", "source_file"),
        ("source_session_id", "source_session_id"),
        ("reviewer", "reviewer"),
        ("row_start", "row_start"),
        ("row_end", "row_end"),
        ("boundary_row", "boundary_row"),
        ("chunk_size", "chunk_size"),
        ("chunk_index_in_session", "chunk_index_in_session"),
        ("extraction_status", "status"),
        ("extraction_error", "error"),
    ]
    rows = [
        {"字段": label, "值": metadata.get(key)}
        for key, label in fields
        if metadata.get(key) not in (None, "")
    ]
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    raw_result = str(metadata.get("raw_result") or "").strip()
    if raw_result:
        with st.expander("源 Excel raw_result", expanded=False):
            st.code(raw_result, language="text")

    with st.expander("完整元信息", expanded=False):
        st.json(metadata)


def render_raw_response(result: EvalResult) -> None:
    with st.expander("裁判模型原始响应", expanded=False):
        raw = result.raw_response or ""
        try:
            parsed = json.loads(raw)
            st.json(parsed)
        except Exception:
            st.code(raw or "（无）", language="text")
