from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Callable

import requests

from src.eval.judge_client import RealJudgeClient
from src.schema import EvalConfig
from src.ui.global_rate_limiter import api_rate_scope, wait_for_global_rate_slot
from src.ui.prompt_advisor_evidence import collect_absolute_eval_evidence, collect_review_evidence
from src.ui.prompt_advisor_limits import (
    ADVISOR_MAX_EDITABLE_BLOCKS,
    ADVISOR_PATCH_MAX_CHANGE_RATIO,
    ADVISOR_PATCH_MIN_CHANGE_CHARS,
    ADVISOR_SECTION_BLOCK_CHARS,
    ADVISOR_SECTION_BLOCK_PREVIEW_CHARS,
    ADVISOR_STAGE1_BATCH_SIZE,
    ADVISOR_STAGE1_MAX_TOKENS,
    ADVISOR_STAGE2_MAX_TOKENS,
    MAX_ADVISOR_APPEND_TEXT_CHARS,
    MAX_ADVISOR_EDIT_TEXT_CHARS,
    MAX_ADVISOR_PATCH_EDITS,
    MAX_ADVISOR_REPLACE_TEXT_CHARS,
    MAX_ADVISOR_TOTAL_PATCH_TEXT_CHARS,
)
from src.ui.prompt_advisor_model_call import (
    _advisor_attempt_profile,
    _advisor_max_tokens,
    _advisor_retry_wait_seconds,
    _call_advisor_json,
    _compact_advisor_evidence,
    _compact_extraction_advisor_evidence,
    _evidence_usage,
)
from src.ui.prompt_advisor_prompts import (
    ABSOLUTE_ADVISOR_SYSTEM_PROMPT,
    EXTRACTION_INTENT_SYSTEM_PROMPT,
    EXTRACTION_PATCH_SYSTEM_PROMPT,
)
from src.ui.prompt_patch import PromptSection, apply_prompt_patch, prompt_sections_for_model, split_prompt_sections
from src.ui.prompt_rule_similarity import (
    _is_similar_rule_key,
    _normalize_rule_similarity_key,
    _normalize_text_key,
    _prompt_rule_units,
    _prune_duplicate_insert_text,
    _prune_duplicate_replacement_text,
)


AdvisorProgressCallback = Callable[[int, int, str, str], None]


def _emit_advisor_progress(
    callback: AdvisorProgressCallback | None,
    done: int,
    total: int,
    stage: str,
    message: str,
) -> None:
    if callback is None:
        return
    callback(max(0, int(done)), max(1, int(total)), stage, message)


def build_advisor_user_message(
    evidence: list[dict[str, Any]],
    current_judge_prompt: str,
    extraction_prompt: str = "",
    target: str = "judge_prompt",
    advisor_mode: str = "absolute_eval",
    judge_prompt_limit: int = 6000,
    extraction_prompt_limit: int = 0,
    extraction_section_limit: int = 80,
    extraction_section_preview_chars: int = 320,
    retry_note: str = "",
) -> str:
    extraction_prompt = extraction_prompt or ""
    extraction_hash = hashlib.sha256(extraction_prompt.encode("utf-8")).hexdigest()[:12] if extraction_prompt else ""
    if not extraction_prompt.strip():
        extraction_note = "（未提供原始提取 prompt；禁止编造完整提取 prompt，只能给修改方向或片段。）"
    elif extraction_prompt_limit > 0:
        extraction_note = extraction_prompt.strip()[:extraction_prompt_limit]
    else:
        extraction_note = "（为降低请求体积，未发送提取 prompt 全文；请只依据 extraction_prompt_sections 的 section_id/title/preview 生成增量 patch。）"
    task = "根据单模型绝对评测结果诊断评测链路并给出受约束的改进建议"
    evidence_name = "absolute_eval_result_evidence"
    warning = "这些证据来自 Judge 结果，不是人工真值；候选 prompt 必须标注为待人工确认，不能直接视为正确修复。"
    weak_context_count = sum(1 for item in evidence if item.get("evidence_mode") == "weak_context_from_result")
    positive_boundary_count = sum(1 for item in evidence if item.get("evidence_mode") == "positive_boundary")
    regression_boundary_count = sum(1 for item in evidence if item.get("evidence_mode") == "regression_boundary")
    loop_constraints = []
    if advisor_mode == "absolute_eval" and target == "extraction_prompt":
        loop_constraints = [
            "本次目标是生成下一轮提取实验使用的候选提取 prompt。",
            "如果 evidence 包含 weak_context_from_result，说明这些样本不是错误证据，只能作为结果分布上下文。",
            "positive_boundary 和 regression_boundary 是防回归边界，只能用于约束改法，不能据此新增问题规则。",
            "不要声称候选 prompt 已经被人工确认；必须在 risks 中说明可能沿着 Judge 偏差自我强化。",
            "默认不要完整重写提取 prompt；请输出 extraction_prompt_patch，由系统应用 patch 得到候选全文。",
            "patch 只能引用 extraction_prompt_sections 中真实存在的 section_id。",
            "每个 edit 必须包含 evidence_refs，引用 evidence 中的 case_id/row_id。",
            "优先使用 replace_within_section 合并到已有规则；冗余或冲突规则可用 delete_within_section 删除；append_to_section 只能作为最后手段。",
            "append_to_section 必须匹配目标章节原格式；如果目标章节是列表，新增内容必须是同级列表项。",
            "候选提取 prompt 应保留原 prompt 的核心约束，只做可解释、可回滚的增量澄清。",
            "修改必须是跨样本可复用的通用规则，禁止写入具体 case 的人名、地点、作品名、原句或一次性事实。",
            "优先合并、替换或删除已有重复规则；除非现有章节完全无法承载，否则禁止追加新规则。",
            "控制提示词规模：候选全文长度和规则数量不得无理由增长，多个同义问题必须合并为一次修改。",
            "validation_plan 必须包含：另存新版本、重新提取、重新评测、稳定性对比、抽样人工复核、必要时回滚。",
        ]
    extraction_sections = prompt_sections_for_model(
        extraction_prompt,
        max_sections=extraction_section_limit,
        preview_chars=extraction_section_preview_chars,
    ) if extraction_prompt.strip() else []
    output_schema: dict[str, Any] = {
        "can_suggest": True,
        "evidence_summary": "不超过300字，说明证据总体问题和置信度",
        "diagnoses": [
            {
                "problem": "当前 prompt 或结果中的问题",
                "evidence_refs": ["row_id 或 case_id"],
                "problem_type": "model_output_issue/judge_prompt_issue/extraction_prompt_issue/data_issue/uncertain",
                "why_it_matters": "为什么影响稳定性、准确性或人工对齐",
                "confidence": "low/medium/high",
            }
        ],
        "judge_prompt_changes": [],
        "candidate_judge_prompt": "",
        "extraction_prompt_notes": "如果和提取 prompt 有关，给简短修改方向",
        "extraction_prompt_patch": {
            "mode": "incremental_patch",
            "edits": [
                {
                    "op": "replace_within_section/delete_within_section/append_to_section",
                    "target_id": "必须来自 extraction_prompt_sections",
                    "old_text": "replace/delete 需要，必须是 section preview 或目标章节中能确认存在的原文",
                    "new_text": "仅 replace_within_section 需要",
                    "text": "仅 append 需要；必须是通用规则，且匹配目标章节格式",
                    "reason": "修改原因",
                    "evidence_refs": ["case_id 或 row_id"],
                }
            ],
        },
        "candidate_extraction_prompt": "",
        "risks": ["不超过3条"],
        "validation_plan": ["不超过5条"],
    }
    if target == "analysis_only":
        output_schema["extraction_prompt_patch"] = {"mode": "incremental_patch", "edits": []}
    if target in {"judge_prompt", "both"}:
        output_schema["judge_prompt_changes"] = [
            {
                "change": "建议修改点",
                "evidence_refs": ["row_id 或 case_id"],
                "expected_effect": "预期影响",
                "risk": "可能副作用",
            }
        ]
        output_schema["candidate_judge_prompt"] = "只有目标包含 judge_prompt 且确有必要时才输出候选 Judge Prompt；否则留空。"
    return json.dumps({
        "task": task,
        "advisor_mode": advisor_mode,
        "target": target,
        "evidence_count": len(evidence),
        "weak_context_count": weak_context_count,
        "positive_boundary_count": positive_boundary_count,
        "regression_boundary_count": regression_boundary_count,
        "evidence_name": evidence_name,
        "evidence_warning": warning,
        "retry_note": retry_note,
        "loop_constraints": loop_constraints,
        "request_contract": [
            "只输出一个 JSON object，不要 Markdown，不要解释性前后缀。",
            "所有建议都必须引用 evidence 中存在的 case_id/row_id/pair_id。",
            "修改提取 prompt 时只输出 extraction_prompt_patch；candidate_extraction_prompt 必须留空，由系统应用 patch 后生成。",
            "不要输出完整提取 prompt；不要删除原 prompt 的核心章节。",
            "如果证据只能支持诊断、不能支持修改，can_suggest=true 也可以只给 diagnoses 和 risks，patch edits 为空。",
        ],
        "evidence": evidence,
        "extraction_prompt_sections": extraction_sections,
        "extraction_prompt_hash": extraction_hash,
        "extraction_prompt_section_count_sent": len(extraction_sections),
        "current_judge_prompt": current_judge_prompt[:judge_prompt_limit],
        "original_extraction_prompt": extraction_note[:extraction_prompt_limit] if extraction_prompt_limit > 0 else extraction_note,
        "output_schema": output_schema,
    }, ensure_ascii=False, indent=2)


def _section_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:12]


def _section_outline(sections: list[PromptSection], *, max_sections: int = 120, preview_chars: int = 180) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for section in sections[:max_sections]:
        rows.append({
            "section_id": section.section_id,
            "title": section.title,
            "level": section.level,
            "hash": _section_hash(section.text),
            "preview": _truncate(section.text.strip(), preview_chars),
        })
    return rows


def _split_complete_blocks(text: str, *, max_chars: int = ADVISOR_SECTION_BLOCK_CHARS) -> list[dict[str, Any]]:
    source = str(text or "").strip()
    if not source:
        return []
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", source) if part.strip()]
    atoms: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            atoms.append(paragraph)
            continue
        lines = [line.rstrip() for line in paragraph.splitlines() if line.strip()]
        if len(lines) <= 1:
            atoms.append(paragraph)
            continue
        current_lines: list[str] = []
        current_size = 0
        for line in lines:
            added = len(line) + (1 if current_lines else 0)
            if current_lines and current_size + added > max_chars:
                atoms.append("\n".join(current_lines))
                current_lines = []
                current_size = 0
            current_lines.append(line)
            current_size += len(line) + (1 if len(current_lines) > 1 else 0)
        if current_lines:
            atoms.append("\n".join(current_lines))

    blocks: list[dict[str, Any]] = []
    current: list[str] = []
    current_size = 0
    for atom in atoms:
        added = len(atom) + (2 if current else 0)
        if current and current_size + added > max_chars:
            text_value = "\n\n".join(current)
            blocks.append({"text": text_value, "editable": len(text_value) <= max_chars})
            current = []
            current_size = 0
        if len(atom) > max_chars:
            if current:
                text_value = "\n\n".join(current)
                blocks.append({"text": text_value, "editable": True})
                current = []
                current_size = 0
            blocks.append({"text": atom, "editable": False})
            continue
        current.append(atom)
        current_size += added
    if current:
        blocks.append({"text": "\n\n".join(current), "editable": True})

    for index, block in enumerate(blocks, 1):
        block["block_id"] = f"B{index:03d}"
        block["chars"] = len(block["text"])
    return blocks


def _match_terms(values: list[Any]) -> set[str]:
    terms: set[str] = set()
    for value in values:
        text = _clean(value)
        for token in re.findall(r"[\u4e00-\u9fff]{2,8}|[A-Za-z0-9_]{3,}", text):
            terms.add(token.lower())
    return terms


def _section_block_context(section: PromptSection, group: dict[str, Any]) -> dict[str, Any]:
    blocks = _split_complete_blocks(section.text)
    intent_values: list[Any] = [section.title]
    for intent in group.get("intents") or []:
        intent_values.extend([
            intent.get("issue_summary"),
            intent.get("proposed_direction"),
            intent.get("problem_type"),
        ])
    for evidence in group.get("evidence") or []:
        intent_values.extend(evidence.get("rule_refs") or [])
        intent_values.extend(evidence.get("error_tags") or [])
        intent_values.append(evidence.get("comment"))
    terms = _match_terms(intent_values)

    ranked: list[tuple[int, int, dict[str, Any]]] = []
    for index, block in enumerate(blocks):
        lowered = str(block["text"]).lower()
        score = sum(1 for term in terms if term in lowered)
        ranked.append((score, -index, block))
    ranked.sort(reverse=True, key=lambda item: (item[0], item[1]))
    editable = [
        block for _, _, block in ranked
        if block.get("editable")
    ][:ADVISOR_MAX_EDITABLE_BLOCKS]

    return {
        "section_id": section.section_id,
        "title": section.title,
        "level": section.level,
        "hash": _section_hash(section.text),
        "section_chars": len(section.text),
        "block_count": len(blocks),
        "block_outline": [
            {
                "block_id": block["block_id"],
                "chars": block["chars"],
                "editable": bool(block["editable"]),
                "preview": _truncate(block["text"], ADVISOR_SECTION_BLOCK_PREVIEW_CHARS),
            }
            for block in blocks[:30]
        ],
        "editable_blocks": [
            {
                "block_id": block["block_id"],
                "full_text": block["text"],
            }
            for block in editable
        ],
        "has_oversized_uneditable_block": any(not block.get("editable") for block in blocks),
    }


def _evidence_ref_id(item: dict[str, Any]) -> str:
    for key in ("case_id", "row_id", "pair_id"):
        value = _clean(item.get(key))
        if value:
            return value
    return ""


def _allowed_evidence_refs(evidence: list[dict[str, Any]]) -> set[str]:
    return {ref for ref in (_evidence_ref_id(item) for item in evidence) if ref}


def _normalize_string_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    stripped = str(value).strip()
    return [stripped] if stripped else []


def _normalize_patch_intents(
    value: Any,
    *,
    valid_section_ids: set[str],
    allowed_refs: set[str],
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for idx, raw in enumerate(value, 1):
        item = raw if isinstance(raw, dict) else {}
        section_id = _clean(item.get("section_id") or item.get("target_id") or item.get("target_section_id"))
        if section_id not in valid_section_ids:
            continue
        refs = [ref for ref in _normalize_string_values(item.get("evidence_refs") or item.get("case_refs")) if not allowed_refs or ref in allowed_refs]
        if not refs:
            continue
        issue_summary = _truncate(item.get("issue_summary") or item.get("problem") or item.get("reason"), 500)
        direction = _truncate(item.get("proposed_direction") or item.get("direction") or item.get("suggestion"), 500)
        problem_type = _clean(item.get("problem_type")) or "uncertain"
        key = (section_id, _normalize_text_key(issue_summary + direction), tuple(sorted(set(refs))))
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "intent_id": _clean(item.get("intent_id")) or f"I{idx:03d}",
            "section_id": section_id,
            "problem_type": problem_type,
            "issue_summary": issue_summary,
            "proposed_direction": direction,
            "confidence": _clean(item.get("confidence")) or "medium",
            "evidence_refs": _dedupe_preserve_order(refs),
        })
    return rows


def _normalize_for_match(value: str) -> str:
    text = _clean(value).lower()
    return "".join(ch for ch in text if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")


def _local_patch_intents_from_evidence(
    evidence: list[dict[str, Any]],
    sections: list[PromptSection],
    *,
    max_intents: int = 10,
) -> list[dict[str, Any]]:
    section_titles = [(section, _normalize_for_match(section.title)) for section in sections]
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in evidence:
        ref_id = _evidence_ref_id(item)
        if not ref_id:
            continue
        refs = []
        refs.extend(_normalize_string_values(item.get("rule_refs")))
        for diag in item.get("diagnostics") or []:
            if isinstance(diag, dict):
                refs.extend(_normalize_string_values(diag.get("rule_refs")))
        for ref in refs:
            ref_key = _normalize_for_match(ref)
            if not ref_key:
                continue
            for section, title_key in section_titles:
                if not title_key:
                    continue
                if title_key in ref_key or ref_key in title_key:
                    key = (section.section_id, ref_id)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append({
                        "intent_id": f"L{len(rows) + 1:03d}",
                        "section_id": section.section_id,
                        "problem_type": "local_rule_ref_match",
                        "issue_summary": _truncate(item.get("comment") or "rule_refs 命中该章节，需要判断是否澄清规则边界。", 500),
                        "proposed_direction": "结合该组证据判断是否需要在本章节补充边界说明；没有足够证据则不要修改。",
                        "confidence": "medium",
                        "evidence_refs": [ref_id],
                    })
                    if len(rows) >= max_intents:
                        return rows
    return rows


def _evidence_by_refs(evidence: list[dict[str, Any]], refs: list[str], *, max_items: int = 8) -> list[dict[str, Any]]:
    ref_set = set(refs)
    rows = [item for item in evidence if _evidence_ref_id(item) in ref_set]
    if not rows:
        rows = evidence[:max_items]
    return _compact_extraction_advisor_evidence(rows, max_items=min(max_items, 4), text_limit=240)


def _build_patch_plan(intents: list[dict[str, Any]], sections: list[PromptSection], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    section_map = {section.section_id: section for section in sections}
    groups: dict[str, dict[str, Any]] = {}
    skipped: list[dict[str, Any]] = []
    for intent in intents:
        section_id = _clean(intent.get("section_id"))
        if section_id not in section_map:
            skipped.append({**intent, "message": "目标章节不存在"})
            continue
        group = groups.setdefault(section_id, {
            "section_id": section_id,
            "section_title": section_map[section_id].title,
            "section_hash": _section_hash(section_map[section_id].text),
            "intents": [],
            "evidence_refs": [],
            "problem_types": [],
        })
        group["intents"].append(intent)
        group["evidence_refs"].extend(intent.get("evidence_refs") or [])
        group["problem_types"].append(intent.get("problem_type") or "uncertain")

    planned = []
    for group in groups.values():
        group["evidence_refs"] = _dedupe_preserve_order(group["evidence_refs"])
        group["problem_types"] = _dedupe_preserve_order(group["problem_types"])
        group["evidence"] = _evidence_by_refs(evidence, group["evidence_refs"], max_items=8)
        planned.append(group)

    planned.sort(key=lambda item: (-len(item.get("evidence_refs") or []), item.get("section_id") or ""))
    return {"groups": planned, "skipped_intents": skipped}


def _build_extraction_intent_message(
    *,
    evidence: list[dict[str, Any]],
    sections: list[PromptSection],
    current_judge_prompt: str,
    advisor_mode: str,
    target: str,
    retry_note: str,
    max_sections: int = 100,
    section_preview_chars: int = 220,
    judge_prompt_limit: int = 2500,
) -> str:
    return json.dumps({
        "stage": "1_intent_localization",
        "task": "先定位需要澄清的提取 Prompt 章节，输出 patch_intents；不要生成最终 patch。",
        "advisor_mode": advisor_mode,
        "target": target,
        "retry_note": retry_note,
        "evidence": evidence,
        "prompt_global_outline": _section_outline(sections, max_sections=max_sections, preview_chars=section_preview_chars),
        "current_judge_prompt_excerpt": current_judge_prompt[:judge_prompt_limit],
        "output_schema": {
            "can_suggest": True,
            "evidence_summary": "证据总体说明，不超过300字",
            "diagnoses": [
                {
                    "problem": "问题描述",
                    "evidence_refs": ["case_id 或 row_id"],
                    "problem_type": "model_output_issue/judge_prompt_issue/extraction_prompt_issue/data_issue/uncertain",
                    "why_it_matters": "影响",
                    "confidence": "low/medium/high",
                }
            ],
            "judge_prompt_changes": [],
            "candidate_judge_prompt": "",
            "extraction_prompt_notes": "提取规则层面的简短方向",
            "patch_intents": [
                {
                    "intent_id": "I001",
                    "section_id": "必须来自 prompt_global_outline",
                    "problem_type": "missing_boundary/over_extraction/under_extraction/format/uncertain",
                    "issue_summary": "该章节需要澄清什么边界",
                    "proposed_direction": "建议澄清方向，不写最终规则全文",
                    "confidence": "low/medium/high",
                    "evidence_refs": ["case_id 或 row_id"],
                }
            ],
            "risks": ["不超过3条"],
            "validation_plan": ["不超过5条"],
        },
    }, ensure_ascii=False, indent=2)


def _build_section_patch_message(
    *,
    group: dict[str, Any],
    sections: list[PromptSection],
    advisor_mode: str,
    target: str,
    retry_note: str,
) -> str:
    section_map = {section.section_id: section for section in sections}
    target_section = section_map.get(str(group.get("section_id") or ""))
    block_context = _section_block_context(target_section, group) if target_section else {}
    section_index = next(
        (index for index, section in enumerate(sections) if section.section_id == group.get("section_id")),
        -1,
    )
    neighbor_outline = []
    if section_index >= 0:
        for index in range(max(0, section_index - 1), min(len(sections), section_index + 2)):
            section = sections[index]
            if section.section_id == group.get("section_id"):
                continue
            neighbor_outline.append({
                "section_id": section.section_id,
                "title": section.title,
                "level": section.level,
                "preview": _truncate(section.text, 260),
            })
    return json.dumps({
        "stage": "2_section_patch",
        "task": "基于目标章节的逻辑块和同组证据，生成一个章节级增量 patch。不要输出完整 prompt。",
        "advisor_mode": advisor_mode,
        "target": target,
        "retry_note": retry_note,
        "section_group": {
            "section_id": group.get("section_id"),
            "section_title": group.get("section_title"),
            "section_hash": group.get("section_hash"),
            "problem_types": group.get("problem_types") or [],
            "evidence_refs": group.get("evidence_refs") or [],
            "patch_intents": group.get("intents") or [],
        },
        "target_section_blocks": block_context,
        "neighbor_section_outline": neighbor_outline,
        "evidence": group.get("evidence") or [],
        "request_contract": [
            "只能修改 target_section_blocks 对应的目标章节；邻近章节只用于避免冲突。",
            "replace_within_section 的 old_text 必须完整出现在 editable_blocks.full_text 中；不得根据截断预览改写原文。",
            "delete_within_section 的 old_text 也必须完整出现在 editable_blocks.full_text 中；只能删除冗余、冲突或格式错误的规则，不能删除核心约束。",
            "block_outline 只用于判断是否重复，不能从预览中复制 old_text。",
            "如 has_oversized_uneditable_block=true，不得改写该块，只能在确有必要时向章节末尾追加一条通用规则。",
            "同一章节的相似规则必须合并，不能按 case 重复追加。",
            "只写通用规则，不要把证据中的具体实体、人名、作品名、地点写进 prompt。",
            "优先用 replace_within_section 澄清/合并已有规则；已有规则冗余或冲突时用 delete_within_section；只有没有可承载规则时才 append_to_section 追加 1 条短规则。",
            "append_to_section 必须延续目标章节格式；如果目标章节正文是列表，text 必须以同级列表符号开头，例如 '- '。",
            "不要为了单个样本补丁式追加；如果只是已有规则换一种说法，输出空 edits。",
            f"每条 edit 文本不超过 {MAX_ADVISOR_EDIT_TEXT_CHARS} 字；本章节 patch 总文本不超过 {MAX_ADVISOR_TOTAL_PATCH_TEXT_CHARS} 字。",
            "如果现有章节已经覆盖该规则，输出空 edits 并说明原因。",
            "candidate_extraction_prompt 必须为空。",
        ],
        "output_schema": {
            "can_suggest": True,
            "section_id": group.get("section_id"),
            "section_hash": group.get("section_hash"),
            "extraction_prompt_patch": {
                "mode": "incremental_patch",
                "edits": [
                    {
                        "op": "replace_within_section/delete_within_section/append_to_section",
                        "target_id": group.get("section_id"),
                        "old_text": "replace/delete 需要，必须从目标章节全文精确复制",
                        "new_text": "仅 replace_within_section 需要",
                        "text": "仅 append_to_section 需要；必须是通用规则，不要针对具体 case，不要冗余，并匹配目标章节格式",
                        "reason": "修改原因",
                        "evidence_refs": ["case_id 或 row_id"],
                    }
                ],
            },
            "section_notes": "本章节为什么这样改；如果不改，说明原因",
            "risks": ["不超过3条"],
        },
    }, ensure_ascii=False, indent=2)


def _dedupe_preserve_order(values: list[Any]) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean(value)
        if text and text not in seen:
            seen.add(text)
            rows.append(text)
    return rows


def _invalid_patch_edit(index: int, message: str, raw: Any = None) -> dict[str, Any]:
    return {
        "_invalid": True,
        "index": index,
        "message": message,
        "raw": raw if isinstance(raw, dict) else {},
    }


def _line_count(text: str) -> int:
    return len([line for line in str(text or "").splitlines() if line.strip()])


def _contains_evidence_ref(text: str, refs: list[str]) -> bool:
    body = str(text or "")
    return any(ref and ref in body for ref in refs)


def _looks_case_specific(text: str, refs: list[str]) -> bool:
    body = str(text or "")
    lowered = body.lower()
    if _contains_evidence_ref(body, refs):
        return True
    case_markers = ("case_", "row_", "样本", "本case", "该case", "这条case", "针对本例", "针对该样本")
    return any(marker in lowered for marker in case_markers)


def _normalize_patch_edit(raw: Any, index: int, *, valid_section_ids: set[str], allowed_refs: set[str]) -> dict[str, Any]:
    item = raw if isinstance(raw, dict) else {}
    op = _clean(item.get("op") or item.get("operation"))
    target_id = _clean(item.get("target_id") or item.get("section_id") or item.get("target"))
    if target_id not in valid_section_ids:
        return _invalid_patch_edit(index, "目标章节不存在，已跳过。", raw)
    refs = [ref for ref in _normalize_string_values(item.get("evidence_refs") or item.get("case_refs")) if not allowed_refs or ref in allowed_refs]
    if not refs:
        return _invalid_patch_edit(index, "缺少有效 evidence_refs，已跳过。", raw)
    if op in {"delete", "remove", "remove_within_section"}:
        op = "delete_within_section"
    text = _clean(item.get("text") or item.get("insert_text"))
    old_text = _clean(item.get("old_text") or item.get("target_text"))
    new_text = _clean(item.get("new_text") or item.get("replacement_text"))
    if op not in {"replace_within_section", "delete_within_section", "append_to_section"}:
        return _invalid_patch_edit(index, f"不支持的操作 {op or '<empty>'}；只允许 replace/delete/append，已跳过。", raw)
    if op == "replace_within_section" and (not old_text or not new_text):
        return _invalid_patch_edit(index, "replace 缺少 old_text 或 new_text，已跳过。", raw)
    if op == "replace_within_section" and (len(old_text) > MAX_ADVISOR_REPLACE_TEXT_CHARS or len(new_text) > MAX_ADVISOR_REPLACE_TEXT_CHARS):
        return _invalid_patch_edit(index, f"replace 文本过长，超过 {MAX_ADVISOR_REPLACE_TEXT_CHARS} 字，已跳过。", raw)
    if op == "replace_within_section" and _looks_case_specific(new_text, refs):
        return _invalid_patch_edit(index, "new_text 看起来针对具体 case/样本，已跳过。", raw)
    if op == "delete_within_section":
        if not old_text:
            return _invalid_patch_edit(index, "delete 缺少 old_text，已跳过。", raw)
        if len(old_text) > MAX_ADVISOR_REPLACE_TEXT_CHARS:
            return _invalid_patch_edit(index, f"delete 文本过长，超过 {MAX_ADVISOR_REPLACE_TEXT_CHARS} 字，已跳过。", raw)
    if op == "append_to_section":
        if not text:
            text = new_text
        if not text:
            return _invalid_patch_edit(index, "插入类操作缺少 text，已跳过。", raw)
        if len(text) > MAX_ADVISOR_EDIT_TEXT_CHARS:
            return _invalid_patch_edit(index, f"新增规则过长，超过 {MAX_ADVISOR_EDIT_TEXT_CHARS} 字，已跳过。", raw)
        if _line_count(text) > 4:
            return _invalid_patch_edit(index, "新增规则行数过多，可能导致 prompt 冗余膨胀，已跳过。", raw)
        if _looks_case_specific(text, refs):
            return _invalid_patch_edit(index, "新增规则看起来针对具体 case/样本，已跳过。", raw)
    return {
        "edit_id": _clean(item.get("edit_id")) or f"E{index:03d}",
        "op": op,
        "target_id": target_id,
        "old_text": old_text,
        "new_text": new_text,
        "text": text,
        "reason": _clean(item.get("reason")),
        "evidence_refs": _dedupe_preserve_order(refs),
    }


def _merge_section_patch_edits(
    raw_edits: list[dict[str, Any]],
    *,
    sections: list[PromptSection],
    allowed_refs: set[str],
) -> dict[str, Any]:
    section_map = {section.section_id: section for section in sections}
    valid_ids = set(section_map)
    append_groups: dict[tuple[str, str], dict[str, Any]] = {}
    replace_groups: dict[tuple[str, str], dict[str, Any]] = {}
    delete_groups: dict[tuple[str, str], dict[str, Any]] = {}
    conflicts: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for idx, raw in enumerate(raw_edits, 1):
        edit = _normalize_patch_edit(raw, idx, valid_section_ids=valid_ids, allowed_refs=allowed_refs)
        if edit.get("_invalid"):
            skipped.append(edit)
            continue

        if edit["op"] == "replace_within_section":
            section_text = section_map[edit["target_id"]].text
            start = section_text.find(edit["old_text"])
            if start < 0:
                skipped.append({**edit, "message": "old_text 未在目标章节全文中命中，已跳过。"})
                continue
            key = (edit["target_id"], edit["old_text"])
            existing = replace_groups.get(key)
            if existing and existing.get("new_text") != edit["new_text"]:
                conflicts.append({
                    "target_id": edit["target_id"],
                    "old_text": edit["old_text"],
                    "message": "同一 old_text 出现不同替换结果，已保留第一条并跳过后续冲突项。",
                    "kept_new_text": existing.get("new_text"),
                    "skipped_new_text": edit["new_text"],
                    "evidence_refs": _dedupe_preserve_order((existing.get("evidence_refs") or []) + edit["evidence_refs"]),
                })
                continue
            if existing:
                existing["evidence_refs"] = _dedupe_preserve_order((existing.get("evidence_refs") or []) + edit["evidence_refs"])
                if edit.get("reason") and edit["reason"] not in existing.get("reason", ""):
                    existing["reason"] = (existing.get("reason") + "；" + edit["reason"]).strip("；")
            else:
                replace_groups[key] = edit
            continue

        if edit["op"] == "delete_within_section":
            section_text = section_map[edit["target_id"]].text
            start = section_text.find(edit["old_text"])
            if start < 0:
                skipped.append({**edit, "message": "old_text 未在目标章节全文中命中，已跳过。"})
                continue
            key = (edit["target_id"], edit["old_text"])
            existing = delete_groups.get(key)
            if existing:
                existing["evidence_refs"] = _dedupe_preserve_order((existing.get("evidence_refs") or []) + edit["evidence_refs"])
                if edit.get("reason") and edit["reason"] not in existing.get("reason", ""):
                    existing["reason"] = (existing.get("reason") + "；" + edit["reason"]).strip("；")
            else:
                delete_groups[key] = edit
            continue

        normalized_text = _normalize_text_key(edit["text"])
        key = (edit["target_id"], edit["op"])
        existing = append_groups.get(key)
        if not existing:
            append_groups[key] = {**edit, "_text_keys": {normalized_text}}
            continue
        existing_keys = existing.setdefault("_text_keys", set())
        if normalized_text not in existing_keys:
            existing["text"] = "\n".join([existing.get("text", "").strip(), edit["text"].strip()]).strip()
            existing_keys.add(normalized_text)
        existing["evidence_refs"] = _dedupe_preserve_order((existing.get("evidence_refs") or []) + edit["evidence_refs"])
        if edit.get("reason") and edit["reason"] not in existing.get("reason", ""):
            existing["reason"] = (existing.get("reason") + "；" + edit["reason"]).strip("；")

    edits: list[dict[str, Any]] = []
    for item in replace_groups.values():
        edits.append({k: v for k, v in item.items() if not k.startswith("_")})
    for item in delete_groups.values():
        edits.append({k: v for k, v in item.items() if not k.startswith("_")})
    for item in append_groups.values():
        edits.append({k: v for k, v in item.items() if not k.startswith("_")})
    edits.sort(key=lambda item: (item.get("target_id") or "", item.get("op") or "", item.get("edit_id") or ""))
    limited_edits: list[dict[str, Any]] = []
    existing_rule_units = _prompt_rule_units("\n\n".join(section.text for section in sections))
    total_patch_chars = 0
    total_append_chars = 0
    for edit in edits:
        op = edit.get("op")
        new_units: list[dict[str, str]] = []
        if op == "append_to_section":
            unique_text, new_units, duplicate_rows = _prune_duplicate_insert_text(str(edit.get("text") or ""), existing_rule_units)
            for duplicate in duplicate_rows:
                skipped.append({
                    **edit,
                    "text": duplicate["text"],
                    "message": f"新增规则与整篇提取 prompt 已有规则高度相似，已跳过。已有规则：{_truncate(duplicate['existing_text'], 120)}",
                })
            if not unique_text:
                continue
            edit = {**edit, "text": unique_text}
        elif op == "replace_within_section":
            unique_text, new_units, duplicate_rows = _prune_duplicate_replacement_text(
                str(edit.get("old_text") or ""),
                str(edit.get("new_text") or ""),
                existing_rule_units,
            )
            for duplicate in duplicate_rows:
                skipped.append({
                    **edit,
                    "text": duplicate["text"],
                    "message": f"replace 新增行与已有规则高度相似，已从替换文本中移除。已有规则：{_truncate(duplicate['existing_text'], 120)}",
                })
            if unique_text == str(edit.get("old_text") or "").strip():
                skipped.append({**edit, "message": "replace 去重后没有新增有效内容，已跳过。"})
                continue
            edit = {**edit, "new_text": unique_text}
        elif op == "delete_within_section":
            pass
        else:
            skipped.append({**edit, "message": f"不支持的操作 {op}，已跳过。"})
            continue

        change_text = str(
            edit.get("text")
            if op == "append_to_section"
            else edit.get("new_text")
            if op == "replace_within_section"
            else edit.get("old_text") or ""
        )
        if op == "append_to_section" and total_append_chars + len(change_text) > MAX_ADVISOR_APPEND_TEXT_CHARS:
            remaining_chars = MAX_ADVISOR_APPEND_TEXT_CHARS - total_append_chars
            kept_lines: list[str] = []
            kept_size = 0
            for line in change_text.splitlines():
                added = len(line) + (1 if kept_lines else 0)
                if remaining_chars > 0 and kept_size + added <= remaining_chars:
                    kept_lines.append(line)
                    kept_size += added
                else:
                    skipped.append({**edit, "text": line, "message": f"追加文本累计超过 {MAX_ADVISOR_APPEND_TEXT_CHARS} 字，为避免提示词持续膨胀，已跳过该行。"})
            change_text = "\n".join(kept_lines).strip()
            if not change_text:
                continue
            edit = {**edit, "text": change_text}
            new_units = _prompt_rule_units(change_text)
        change_size = len(change_text)
        if len(limited_edits) >= MAX_ADVISOR_PATCH_EDITS:
            skipped.append({**edit, "message": f"patch edit 数超过 {MAX_ADVISOR_PATCH_EDITS} 条，为避免 prompt 暴增，已跳过。"})
            continue
        if total_patch_chars + change_size > MAX_ADVISOR_TOTAL_PATCH_TEXT_CHARS:
            skipped.append({**edit, "message": f"patch 总新增文本超过 {MAX_ADVISOR_TOTAL_PATCH_TEXT_CHARS} 字，为避免 prompt 暴增，已跳过。"})
            continue
        limited_edits.append(edit)
        total_patch_chars += change_size
        if op == "append_to_section":
            total_append_chars += change_size
            existing_rule_units.extend(new_units)
        elif op == "replace_within_section":
            existing_rule_units.extend(new_units or _prompt_rule_units(str(edit.get("new_text") or "")))
    edits = limited_edits
    return {"edits": edits, "conflicts": conflicts, "skipped": skipped}


def _finalize_advisor_result(
    result: dict[str, Any],
    *,
    extraction_prompt: str,
    target: str,
) -> dict[str, Any]:
    if not isinstance(result, dict):
        return result

    risks = list(result.get("risks") or [])
    model_candidate = str(result.get("candidate_extraction_prompt") or "")
    if model_candidate:
        result["model_candidate_extraction_prompt"] = model_candidate

    needs_extraction_patch = target in {"extraction_prompt", "both"} and bool(extraction_prompt.strip())
    if not needs_extraction_patch:
        return result

    patch = result.get("extraction_prompt_patch") or result.get("extraction_prompt_edits") or {}
    patch_result = apply_prompt_patch(
        extraction_prompt,
        patch,
        max_change_ratio=ADVISOR_PATCH_MAX_CHANGE_RATIO,
        min_change_chars=ADVISOR_PATCH_MIN_CHANGE_CHARS,
    )
    result["extraction_prompt_patch_result"] = patch_result
    result["extraction_prompt_diff"] = patch_result.get("diff", "")

    if patch_result.get("applied_edits"):
        result["candidate_extraction_prompt"] = patch_result.get("candidate_prompt", "")
        result["candidate_prompt_source"] = "applied_incremental_patch"
        if patch_result.get("skipped_edits"):
            risks.append("部分提取 prompt patch 未通过校验，已跳过；请查看 patch 校验结果。")
    else:
        result["candidate_extraction_prompt"] = ""
        result["candidate_prompt_source"] = "no_valid_incremental_patch"
        if model_candidate:
            risks.append("模型返回了完整候选提取 prompt，但未提供可验证的增量 patch；系统未自动采用完整重写。")
        else:
            risks.append("未生成可应用的增量 patch，因此没有候选提取 prompt。")

    result["risks"] = risks
    return result


def _merge_stage1_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {}

    def collect_list(key: str, limit: int) -> list[Any]:
        rows: list[Any] = []
        seen: set[str] = set()
        for result in results:
            for item in result.get(key) or []:
                marker = json.dumps(item, ensure_ascii=False, sort_keys=True) if isinstance(item, (dict, list)) else str(item)
                if marker in seen:
                    continue
                seen.add(marker)
                rows.append(item)
                if len(rows) >= limit:
                    return rows
        return rows

    summaries = _dedupe_preserve_order([result.get("evidence_summary") for result in results])
    notes = _dedupe_preserve_order([result.get("extraction_prompt_notes") for result in results])
    return {
        "can_suggest": any(bool(result.get("can_suggest", True)) for result in results),
        "evidence_summary": _truncate("；".join(summaries), 600),
        "diagnoses": collect_list("diagnoses", 12),
        "judge_prompt_changes": collect_list("judge_prompt_changes", 8),
        "candidate_judge_prompt": next(
            (str(result.get("candidate_judge_prompt") or "") for result in results if result.get("candidate_judge_prompt")),
            "",
        ),
        "candidate_extraction_prompt": next(
            (str(result.get("candidate_extraction_prompt") or "") for result in results if result.get("candidate_extraction_prompt")),
            "",
        ),
        "extraction_prompt_patch": next(
            (
                result.get("extraction_prompt_patch")
                for result in results
                if isinstance(result.get("extraction_prompt_patch"), dict)
                and (result.get("extraction_prompt_patch") or {}).get("edits")
            ),
            {"mode": "incremental_patch", "edits": []},
        ),
        "extraction_prompt_notes": _truncate("；".join(notes), 600),
        "patch_intents": collect_list("patch_intents", 24),
        "risks": collect_list("risks", 8),
        "validation_plan": collect_list("validation_plan", 8),
    }


def _call_two_stage_extraction_advisor(
    *,
    config: EvalConfig,
    evidence: list[dict[str, Any]],
    current_judge_prompt: str,
    extraction_prompt: str,
    target: str,
    advisor_mode: str,
    min_evidence: int,
    url: str,
    headers: dict[str, str],
    client: RealJudgeClient,
    progress_callback: AdvisorProgressCallback | None = None,
) -> tuple[dict[str, Any], str]:
    sections = split_prompt_sections(extraction_prompt)
    if not sections:
        return {
            "can_suggest": False,
            "evidence_summary": "未能把提取提示词切分成可编辑章节，无法生成章节级 patch。",
            "diagnoses": [],
            "judge_prompt_changes": [],
            "candidate_judge_prompt": "",
            "extraction_prompt_notes": "",
            "candidate_extraction_prompt": "",
            "risks": ["提取提示词为空或无法切分。"],
            "validation_plan": ["检查提取提示词内容，至少保留清晰段落或 Markdown 标题后重试。"],
            "error": "提取提示词无法切分。",
        }, ""

    _emit_advisor_progress(progress_callback, 0, 3, "准备章节定位", "正在切分提取提示词并准备证据。")
    allowed_refs = _allowed_evidence_refs(evidence)
    stage1_profile = _advisor_attempt_profile(
        attempt=1,
        target=target,
        evidence_count=len(evidence),
        min_evidence=min_evidence,
    )
    stage1_evidence = _compact_extraction_advisor_evidence(
        evidence,
        max_items=stage1_profile["max_items"],
        text_limit=min(stage1_profile["text_limit"], 260),
    )
    raw_payload: dict[str, Any] = {
        "mode": "two_stage_extraction_prompt_advisor",
        "strategy": "batched_localization_and_paragraph_blocks",
        "stage1_raw": [],
        "stage2_raw": [],
    }
    request_metrics: list[dict[str, Any]] = []
    stage1_results: list[dict[str, Any]] = []
    stage1_errors: list[dict[str, Any]] = []
    prelocalized_intents = _local_patch_intents_from_evidence(stage1_evidence, sections, max_intents=12)
    prelocalized_refs = {
        ref
        for intent in prelocalized_intents
        for ref in (intent.get("evidence_refs") or [])
    }
    unresolved_evidence = [
        item for item in stage1_evidence
        if _evidence_ref_id(item) not in prelocalized_refs
    ]
    if prelocalized_intents:
        stage1_results.append({
            "can_suggest": True,
            "evidence_summary": f"系统根据 rule_refs 在本地定位了 {len(prelocalized_refs)} 条证据，无需让模型重复定位这些章节。",
            "diagnoses": [],
            "judge_prompt_changes": [],
            "candidate_judge_prompt": "",
            "candidate_extraction_prompt": "",
            "extraction_prompt_notes": "已优先使用评测结果中的规则引用做确定性章节定位。",
            "patch_intents": prelocalized_intents,
            "risks": [],
            "validation_plan": [],
        })
        raw_payload["stage1_raw"].append({
            "mode": "local_rule_ref_localization",
            "evidence_refs": sorted(prelocalized_refs),
            "intent_count": len(prelocalized_intents),
        })
        request_metrics.append({
            "stage": "1_本地规则定位",
            "unit": "local",
            "request_chars": 0,
            "evidence_count": len(prelocalized_refs),
            "success": True,
            "error": "",
        })
    stage1_batches = [
        unresolved_evidence[index:index + ADVISOR_STAGE1_BATCH_SIZE]
        for index in range(0, len(unresolved_evidence), ADVISOR_STAGE1_BATCH_SIZE)
    ]
    stage_total = 1 + len(stage1_batches) + 1
    _emit_advisor_progress(
        progress_callback,
        1,
        stage_total,
        "定位章节",
        f"本地已定位 {len(prelocalized_refs)} 条证据，剩余 {len(unresolved_evidence)} 条证据需要分批定位。",
    )
    for batch_index, batch in enumerate(stage1_batches, 1):
        if batch_index > 1:
            interval = float(getattr(config, "judge_request_interval", 0.0) or 0.0)
            if interval > 0:
                time.sleep(interval)
        stage1_message = _build_extraction_intent_message(
            evidence=batch,
            sections=sections,
            current_judge_prompt=current_judge_prompt,
            advisor_mode=advisor_mode,
            target=target,
            retry_note=f"分批定位第 {batch_index}/{len(stage1_batches)} 批：只定位问题簇和目标章节，不生成最终 patch。",
            max_sections=min(120, max(len(sections), 1)),
            section_preview_chars=60,
            judge_prompt_limit=0 if target == "extraction_prompt" else 1200,
        )
        batch_result, batch_raw, batch_error = _call_advisor_json(
            config=config,
            url=url,
            headers=headers,
            client=client,
            system_prompt=EXTRACTION_INTENT_SYSTEM_PROMPT,
            user_message=stage1_message,
            max_tokens=min(_advisor_max_tokens(config, stage1_profile), ADVISOR_STAGE1_MAX_TOKENS),
        )
        raw_payload["stage1_raw"].append({
            "batch": batch_index,
            "evidence_refs": [_evidence_ref_id(item) for item in batch],
            "raw": batch_raw,
            "error": batch_error,
        })
        request_metrics.append({
            "stage": "1_分批定位",
            "unit": f"{batch_index}/{len(stage1_batches)}",
            "request_chars": len(stage1_message) + len(EXTRACTION_INTENT_SYSTEM_PROMPT),
            "evidence_count": len(batch),
            "success": isinstance(batch_result, dict),
            "error": batch_error,
        })
        if isinstance(batch_result, dict):
            stage1_results.append(batch_result)
        else:
            stage1_errors.append({
                "stage": "1_分批定位",
                "batch": batch_index,
                "evidence_refs": [_evidence_ref_id(item) for item in batch],
                "error": batch_error or "第1阶段输出不可解析。",
            })
        _emit_advisor_progress(
            progress_callback,
            1 + batch_index,
            stage_total,
            "定位章节",
            f"章节定位第 {batch_index}/{len(stage1_batches)} 批完成。",
        )

    stage1_result = _merge_stage1_results(stage1_results)
    if not stage1_result:
        return {
            "can_suggest": False,
            "evidence_summary": "提示词改进建议第1阶段定位失败，未进入 patch 生成。",
            "diagnoses": [],
            "judge_prompt_changes": [],
            "candidate_judge_prompt": "",
            "extraction_prompt_notes": "",
            "candidate_extraction_prompt": "",
            "risks": ["第1阶段没有可解析输出，不能据此修改 prompt。"],
            "validation_plan": [
                "减少证据条数后重试。",
                "如果仍是 websocket/Connection Idle Timeout，换更快的建议模型或只做人工定位。",
            ],
            "error": "；".join(item.get("error", "") for item in stage1_errors) or "第1阶段定位失败。",
            "extraction_prompt_request_metrics": request_metrics,
            "extraction_prompt_stage_errors": stage1_errors,
            "evidence_usage": _evidence_usage(
                len(evidence),
                len(stage1_evidence),
                request_metrics=request_metrics,
            ),
        }, json.dumps(raw_payload, ensure_ascii=False, indent=2)

    base_result: dict[str, Any] = {
        "can_suggest": bool(stage1_result.get("can_suggest", True)),
        "evidence_summary": stage1_result.get("evidence_summary") or "",
        "diagnoses": stage1_result.get("diagnoses") or [],
        "judge_prompt_changes": stage1_result.get("judge_prompt_changes") or [],
        "candidate_judge_prompt": stage1_result.get("candidate_judge_prompt") or "",
        "extraction_prompt_notes": stage1_result.get("extraction_prompt_notes") or "",
        "candidate_extraction_prompt": "",
        "risks": list(stage1_result.get("risks") or []),
        "validation_plan": list(stage1_result.get("validation_plan") or [
            "另存候选提取提示词为新版本。",
            "用同一批数据重新提取和评测。",
            "对比稳定性报告并抽样人工复核。",
        ]),
        "advisor_flow": "two_stage_extraction_prompt_advisor",
    }
    if stage1_result.get("candidate_extraction_prompt"):
        base_result["model_candidate_extraction_prompt"] = str(stage1_result.get("candidate_extraction_prompt") or "")

    stage1_direct_patch = stage1_result.get("extraction_prompt_patch") or {}
    direct_edits = stage1_direct_patch.get("edits") if isinstance(stage1_direct_patch, dict) else []
    intents = _normalize_patch_intents(
        stage1_result.get("patch_intents") or stage1_result.get("extraction_prompt_patch_intents"),
        valid_section_ids={section.section_id for section in sections},
        allowed_refs=allowed_refs,
    )
    local_intents = []
    if not intents:
        local_intents = _local_patch_intents_from_evidence(stage1_evidence, sections, max_intents=8)
        intents = local_intents
        if local_intents:
            base_result["risks"].append("第1阶段未返回可用 patch_intents；系统改用 rule_refs 本地命中的章节作为保守定位。")

    raw_edits: list[dict[str, Any]] = list(direct_edits or [])
    plan = _build_patch_plan(intents, sections, stage1_evidence)
    stage_errors: list[dict[str, Any]] = list(stage1_errors)
    stage2_summaries: list[dict[str, Any]] = []

    if not raw_edits:
        groups = plan.get("groups") or []
        stage2_total = min(len(groups), MAX_ADVISOR_PATCH_EDITS)
        completed_before_stage2 = 1 + len(stage1_batches)
        stage_total = completed_before_stage2 + stage2_total + 1
        _emit_advisor_progress(
            progress_callback,
            completed_before_stage2,
            stage_total,
            "生成章节修改",
            f"已合并为 {len(groups)} 个目标章节，准备生成段落级修改。",
        )
        for group_index, group in enumerate(groups[:MAX_ADVISOR_PATCH_EDITS], 1):
            interval = float(getattr(config, "judge_request_interval", 0.0) or 0.0)
            if interval > 0:
                time.sleep(interval)
            stage2_message = _build_section_patch_message(
                group=group,
                sections=sections,
                advisor_mode=advisor_mode,
                target=target,
                retry_note="两阶段流程第2步：只基于该章节全文和同组证据生成章节级 patch。",
            )
            section_result, section_raw, section_error = _call_advisor_json(
                config=config,
                url=url,
                headers=headers,
                client=client,
                system_prompt=EXTRACTION_PATCH_SYSTEM_PROMPT,
                user_message=stage2_message,
                max_tokens=ADVISOR_STAGE2_MAX_TOKENS,
            )
            raw_payload["stage2_raw"].append({
                "section_id": group.get("section_id"),
                "raw": section_raw,
                "error": section_error,
            })
            request_metrics.append({
                "stage": "2_段落级编辑",
                "unit": f"{group_index}/{min(len(groups), MAX_ADVISOR_PATCH_EDITS)}",
                "section_id": group.get("section_id"),
                "request_chars": len(stage2_message) + len(EXTRACTION_PATCH_SYSTEM_PROMPT),
                "evidence_count": len(group.get("evidence") or []),
                "success": isinstance(section_result, dict),
                "error": section_error,
            })
            if not isinstance(section_result, dict):
                stage_errors.append({
                    "section_id": group.get("section_id"),
                    "section_title": group.get("section_title"),
                    "error": section_error or "第2阶段输出不可解析。",
                })
                continue
            returned_hash = _clean(section_result.get("section_hash"))
            if returned_hash and returned_hash != group.get("section_hash"):
                stage_errors.append({
                    "section_id": group.get("section_id"),
                    "section_title": group.get("section_title"),
                    "error": "模型返回的 section_hash 与当前章节不一致，已跳过该章节 patch。",
                })
                continue
            patch = section_result.get("extraction_prompt_patch") or {}
            edits = patch.get("edits") if isinstance(patch, dict) else []
            accepted = 0
            for edit in edits or []:
                edit_target = _clean((edit or {}).get("target_id") or (edit or {}).get("section_id"))
                if edit_target and edit_target != group.get("section_id"):
                    stage_errors.append({
                        "section_id": group.get("section_id"),
                        "section_title": group.get("section_title"),
                        "error": f"第2阶段试图修改非目标章节 {edit_target}，已跳过该 edit。",
                    })
                    continue
                raw_edits.append(edit)
                accepted += 1
            stage2_summaries.append({
                "section_id": group.get("section_id"),
                "section_title": group.get("section_title"),
                "intent_count": len(group.get("intents") or []),
                "evidence_count": len(group.get("evidence_refs") or []),
                "accepted_edit_count": accepted,
                "section_notes": section_result.get("section_notes") or "",
            })
            _emit_advisor_progress(
                progress_callback,
                completed_before_stage2 + group_index,
                stage_total,
                "生成章节修改",
                f"段落级修改第 {group_index}/{stage2_total} 个章节完成。",
            )
    else:
        completed_before_stage2 = 1 + len(stage1_batches)
        stage_total = completed_before_stage2 + 1

    merged = _merge_section_patch_edits(raw_edits, sections=sections, allowed_refs=allowed_refs)
    base_result["extraction_prompt_patch_intents"] = intents
    base_result["extraction_prompt_patch_plan"] = [
        {
            "section_id": group.get("section_id"),
            "section_title": group.get("section_title"),
            "section_hash": group.get("section_hash"),
            "intent_count": len(group.get("intents") or []),
            "evidence_count": len(group.get("evidence_refs") or []),
            "problem_types": group.get("problem_types") or [],
        }
        for group in plan.get("groups") or []
    ]
    base_result["extraction_prompt_stage2_summaries"] = stage2_summaries
    base_result["extraction_prompt_request_metrics"] = request_metrics
    base_result["extraction_prompt_patch_conflicts"] = merged.get("conflicts") or []
    base_result["extraction_prompt_patch_skipped_before_apply"] = (merged.get("skipped") or []) + (plan.get("skipped_intents") or [])
    base_result["extraction_prompt_stage_errors"] = stage_errors
    base_result["evidence_usage"] = _evidence_usage(
        len(evidence),
        len(stage1_evidence),
        request_metrics=request_metrics,
    )
    base_result["extraction_prompt_patch"] = {
        "mode": "incremental_patch",
        "edits": merged.get("edits") or [],
    }
    if local_intents:
        base_result["local_fallback_intents"] = local_intents
    if stage_errors:
        base_result["risks"].append("部分章节 patch 生成失败或被安全规则跳过；请查看阶段错误。")
    if merged.get("conflicts"):
        base_result["risks"].append("检测到同一章节内冲突修改，冲突项未自动应用。")

    _emit_advisor_progress(progress_callback, max(stage_total - 1, 0), stage_total, "应用候选修改", "正在校验并应用候选提示词修改。")
    finalized = _finalize_advisor_result(base_result, extraction_prompt=extraction_prompt, target=target)
    _emit_advisor_progress(progress_callback, stage_total, stage_total, "完成", "提示词建议生成完成。")
    return finalized, json.dumps(raw_payload, ensure_ascii=False, indent=2)


def call_prompt_advisor(
    config: EvalConfig,
    evidence: list[dict[str, Any]],
    current_judge_prompt: str,
    extraction_prompt: str = "",
    target: str = "judge_prompt",
    advisor_mode: str = "absolute_eval",
    min_evidence: int = 3,
    progress_callback: AdvisorProgressCallback | None = None,
) -> tuple[dict[str, Any] | None, str]:
    if advisor_mode != "absolute_eval":
        _emit_advisor_progress(progress_callback, 1, 1, "不支持", "当前提示词建议模式不支持。")
        return {
            "can_suggest": False,
            "evidence_summary": f"不支持的提示词建议模式：{advisor_mode}",
            "diagnoses": [],
            "judge_prompt_changes": [],
            "candidate_judge_prompt": "",
            "extraction_prompt_notes": "",
            "candidate_extraction_prompt": "",
            "risks": ["该版本只保留单模型绝对评测建议。"],
            "validation_plan": ["请使用普通执行评测结果生成建议。"],
            "evidence_usage": _evidence_usage(len(evidence), 0),
        }, ""

    boundary_modes = {"positive_boundary", "regression_boundary", "weak_context_from_result"}
    actionable_evidence_count = sum(
        1 for item in evidence if str(item.get("evidence_mode") or "") not in boundary_modes
    )
    if actionable_evidence_count < min_evidence:
        _emit_advisor_progress(progress_callback, 1, 1, "证据不足", "证据条数不足，未生成候选提示词。")
        summary = f"可用于修改的评测结果证据少于 {min_evidence} 条，拒绝生成候选 prompt，避免根据正例边界或个例过拟合。"
        plan = [f"至少收集 {min_evidence} 条低分、带错误标签、带 diagnostics 或 fatal 的普通评测结果。"]
        return {
            "can_suggest": False,
            "evidence_summary": summary,
            "diagnoses": [],
            "judge_prompt_changes": [],
            "candidate_judge_prompt": "",
            "extraction_prompt_notes": "",
            "candidate_extraction_prompt": "",
            "risks": ["证据不足"],
            "validation_plan": plan,
            "evidence_usage": _evidence_usage(len(evidence), 0),
            "evidence_composition": {
                "actionable": actionable_evidence_count,
                "boundary": len(evidence) - actionable_evidence_count,
            },
        }, ""

    if config.mock:
        _emit_advisor_progress(progress_callback, 0, 2, "模拟模式", "正在生成模拟提示词建议。")
        extraction_patch: dict[str, Any] = {"mode": "incremental_patch", "edits": []}
        if target in {"extraction_prompt", "both"} and extraction_prompt.strip():
            sections = prompt_sections_for_model(extraction_prompt, max_sections=1, preview_chars=120)
            if sections:
                first_evidence = evidence[0] if evidence else {}
                extraction_patch = {
                    "mode": "incremental_patch",
                    "edits": [
                        {
                            "op": "append_to_section",
                            "target_id": sections[0]["section_id"],
                            "text": "- [MOCK] 这里示例展示增量修改机制，真实运行时不会插入这段。",
                            "reason": "模拟模式用于验证 patch 展示和保存流程。",
                            "evidence_refs": [str(first_evidence.get("case_id") or first_evidence.get("row_id") or "mock_case")],
                        }
                    ],
                }
        mock_result = {
            "can_suggest": True,
            "evidence_summary": f"[MOCK] 已接收 {len(evidence)} 条证据。",
            "diagnoses": [],
            "judge_prompt_changes": [],
            "candidate_judge_prompt": current_judge_prompt if target in {"judge_prompt", "both"} else "",
            "extraction_prompt_notes": "[MOCK] 模拟模式未生成真实修改建议。",
            "extraction_prompt_patch": extraction_patch,
            "candidate_extraction_prompt": "",
            "risks": ["模拟模式结果不可用于真实闭环"],
            "validation_plan": ["关闭模拟模式后重新生成建议"],
            "evidence_usage": _evidence_usage(len(evidence), len(evidence)),
        }
        mock_result = _finalize_advisor_result(mock_result, extraction_prompt=extraction_prompt, target=target)
        _emit_advisor_progress(progress_callback, 2, 2, "完成", "模拟提示词建议生成完成。")
        return mock_result, json.dumps(mock_result, ensure_ascii=False)

    url = RealJudgeClient._normalize_chat_completions_url(config.judge_api_base_url)
    system_prompt = ABSOLUTE_ADVISOR_SYSTEM_PROMPT
    client = RealJudgeClient(config)
    headers = client._build_headers()
    if target in {"extraction_prompt", "both"} and extraction_prompt.strip():
        return _call_two_stage_extraction_advisor(
            config=config,
            evidence=evidence,
            current_judge_prompt=current_judge_prompt,
            extraction_prompt=extraction_prompt,
            target=target,
            advisor_mode=advisor_mode,
            min_evidence=min_evidence,
            url=url,
            headers=headers,
            client=client,
            progress_callback=progress_callback,
        )

    last_error = ""
    used_compact_retry = False
    attempt_metrics: list[dict[str, Any]] = []

    max_attempts = max(1, int(config.judge_max_retries or 1))
    for attempt in range(1, max_attempts + 1):
        _emit_advisor_progress(
            progress_callback,
            attempt - 1,
            max_attempts,
            "调用模型",
            f"正在调用提示词建议模型，第 {attempt}/{max_attempts} 次尝试。",
        )
        raw_text = ""
        profile = _advisor_attempt_profile(
            attempt=attempt,
            target=target,
            evidence_count=len(evidence),
            min_evidence=min_evidence,
        )
        attempt_evidence = _compact_advisor_evidence(
            evidence,
            max_items=profile["max_items"],
            text_limit=profile["text_limit"],
            diagnostics_limit=profile["diagnostics_limit"],
            refs_limit=profile["refs_limit"],
        )
        if attempt > 1:
            used_compact_retry = True
            retry_note = (
                "上一次提示词建议调用遇到超时/连接空闲或瞬时服务错误；"
                "本次已进一步压缩 evidence、prompt 分节和输出长度。请只输出高置信、可回滚的修改建议。"
            )
        else:
            retry_note = (
                "本请求已使用轻量模式：证据和 prompt 均已压缩；修改提取 prompt 时只输出增量 patch，不输出完整 prompt。"
            )

        user_message = build_advisor_user_message(
            attempt_evidence,
            current_judge_prompt,
            extraction_prompt,
            target,
            advisor_mode=advisor_mode,
            judge_prompt_limit=profile["judge_prompt_limit"],
            extraction_prompt_limit=profile["extraction_prompt_limit"],
            extraction_section_limit=profile["extraction_section_limit"],
            extraction_section_preview_chars=profile["extraction_section_preview_chars"],
            retry_note=retry_note,
        )
        payload = {
            "model": config.judge_model,
            "max_tokens": _advisor_max_tokens(config, profile),
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "extra_body": {
                "enable_thinking": False,
                "skip_special_tokens": False,
            },
        }
        attempt_metric = {
            "attempt": attempt,
            "evidence_count": len(attempt_evidence),
            "request_chars": len(user_message) + len(system_prompt),
            "success": False,
            "error": "",
        }
        try:
            wait_for_global_rate_slot(
                api_rate_scope(config.judge_api_base_url, config.judge_api_bearer_token),
                float(getattr(config, "judge_request_interval", 0.0) or 0.0),
                disabled=bool(config.mock),
            )
            response = requests.post(
                url,
                headers=headers,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                timeout=config.judge_timeout,
            )
            raw_text = response.text
            try:
                data = response.json()
            except Exception:
                response.raise_for_status()
                last_error = f"响应不是 JSON: {raw_text[:1000]}"
                data = None

            if isinstance(data, dict):
                is_err, err_msg = RealJudgeClient._is_api_error(data)
                if is_err:
                    last_error = f"API error: {err_msg}. raw={raw_text[:1000]}"
                else:
                    response.raise_for_status()
                    content = client._extract_content(data)
                    parsed = RealJudgeClient._parse_json_response(content)
                    if isinstance(parsed, dict):
                        parsed = _finalize_advisor_result(parsed, extraction_prompt=extraction_prompt, target=target)
                        attempt_metric["success"] = True
                        attempt_metrics.append(attempt_metric)
                        parsed["evidence_usage"] = _evidence_usage(
                            len(evidence),
                            attempt_metrics[0]["evidence_count"],
                            attempts=attempt_metrics,
                        )
                        _emit_advisor_progress(progress_callback, max_attempts, max_attempts, "完成", "提示词建议生成完成。")
                        return parsed, content or raw_text
                    last_error = f"提示词建议输出不是可解析 JSON: {content[:1000]}"

        except requests.exceptions.Timeout:
            last_error = f"请求超时 ({attempt}/{max_attempts})"
        except requests.exceptions.RequestException as exc:
            last_error = f"请求异常 ({attempt}/{max_attempts}): {exc}"
        except Exception as exc:
            last_error = f"未知错误 ({attempt}/{max_attempts}): {exc}"

        attempt_metric["error"] = last_error
        attempt_metrics.append(attempt_metric)
        if attempt < max_attempts:
            time.sleep(_advisor_retry_wait_seconds(config, last_error, attempt))

    _emit_advisor_progress(progress_callback, max_attempts, max_attempts, "失败", "提示词建议生成失败。")
    return {
        "can_suggest": False,
        "evidence_summary": "提示词改进建议生成失败，未拿到可解析的模型输出。",
        "diagnoses": [],
        "judge_prompt_changes": [],
        "candidate_judge_prompt": "",
        "extraction_prompt_notes": "",
        "candidate_extraction_prompt": "",
        "risks": ["建议生成调用失败，不能据此修改 prompt。"],
        "validation_plan": [
            "检查下方原始错误；如果是 QPS limit，请降低并发或把请求间隔设置到接口限制以上。",
            "如果是 websocket/Connection Idle Timeout，说明服务端在生成期间断开连接；系统已自动压缩请求，仍失败时请减少证据条数或换更快的建议模型。",
            "如果是 JSON 解析失败，请优先减少目标为“只优化提取提示词”，避免让模型输出完整 prompt；必要时再提高最大输出长度。",
        ],
        "error": last_error,
        "used_compact_retry": used_compact_retry,
        "evidence_usage": _evidence_usage(
            len(evidence),
            attempt_metrics[0]["evidence_count"] if attempt_metrics else 0,
            attempts=attempt_metrics,
        ),
    }, last_error


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _truncate(value: Any, max_len: int) -> str:
    text = _clean(value)
    return text[:max_len] + ("..." if len(text) > max_len else "")
