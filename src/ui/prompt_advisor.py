from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.eval.judge_client import RealJudgeClient
from src.schema import EvalConfig, EvalResult
from src.ui.prompt_patch import PromptSection, apply_prompt_patch, prompt_sections_for_model, split_prompt_sections


MAX_ADVISOR_PATCH_EDITS = 5
MAX_ADVISOR_EDIT_TEXT_CHARS = 420
MAX_ADVISOR_REPLACE_TEXT_CHARS = 700
MAX_ADVISOR_TOTAL_PATCH_TEXT_CHARS = 900
ADVISOR_PATCH_MAX_CHANGE_RATIO = 0.06
ADVISOR_PATCH_MIN_CHANGE_CHARS = 400
ADVISOR_STAGE1_BATCH_SIZE = 4
ADVISOR_STAGE1_MAX_TOKENS = 1200
ADVISOR_STAGE2_MAX_TOKENS = 1000
ADVISOR_SECTION_BLOCK_CHARS = 2400
ADVISOR_SECTION_BLOCK_PREVIEW_CHARS = 180
ADVISOR_MAX_EDITABLE_BLOCKS = 2


ABSOLUTE_ADVISOR_SYSTEM_PROMPT = """你是一个 USER.md 绝对评测诊断助手。你的任务是根据单模型评测结果，诊断当前评测链路、Judge Prompt 和提取 Prompt 中可能需要澄清的部分。

硬性约束：
1. 只能基于用户提供的评测结果证据提出建议，不允许凭空猜测。
2. 每条建议必须引用 evidence 中的 case_id。
3. 不要把 Judge 的结论当作人工真值；没有人工复核时，必须把建议标注为“待人工确认”。
4. 必须区分三类问题：模型输出本身的问题、Judge Prompt 口径不清的问题、提取 Prompt 规则边界不清的问题。
5. 不要为了提高分数而放宽质量标准；建议应服务于稳定性、可解释性和规则一致性。
6. 不要自动覆盖原 prompt，只输出候选文本和修改理由。
7. 修改提取 Prompt 时默认输出 extraction_prompt_patch，不要完整重写；patch 必须引用提供的 section_id 和 evidence_refs。
8. 如果没有原始提取 prompt，只能给 extraction_prompt_notes 或片段建议，不能编造完整提取 prompt。
9. 如果用户开启了无门槛实验模式，必须在 risks 中明确说明：这不是人工确认的改进，可能沿着 Judge 偏差自我强化；候选提取 prompt 只能作为下一轮实验版本。
10. 提取 Prompt 修改必须是通用规则澄清，不要针对某个具体 case 写专门补丁；不要重复、冗余、堆砌示例。

严格输出 JSON，不要输出 Markdown 代码块。
输出尽量短：修改提取 Prompt 时只输出 extraction_prompt_patch，不要输出完整提取 Prompt。"""


EXTRACTION_INTENT_SYSTEM_PROMPT = """你是提示词改进的第一阶段定位器。你的任务不是改写 prompt，而是根据评测证据定位可能需要澄清的提取 Prompt 章节。

硬性约束：
1. 只输出 JSON，不要 Markdown。
2. 不生成最终 patch，不输出完整 prompt。
3. patch_intents 中的 section_id 必须来自 prompt_global_outline。
4. 每个 intent 必须引用 evidence 中真实存在的 case_id/row_id。
5. 没有足够证据时 patch_intents 输出空数组，并在 risks 中说明原因。
6. 如果只是 Judge 误判或样本证据不足，不要强行要求修改提取 Prompt。
7. intent 必须抽象成问题类型和规则边界，不要按单个 case 生成细碎修改。"""


EXTRACTION_PATCH_SYSTEM_PROMPT = """你是提示词改进的第二阶段章节编辑器。你的任务是基于目标章节全文和同组证据，生成一个小而精确的增量 patch。

硬性约束：
1. 只输出 JSON，不要 Markdown。
2. 只能修改 target_section_blocks 指定的章节；替换原文时只能使用 editable_blocks 中提供的完整逻辑块。
3. 不输出完整 prompt；candidate_extraction_prompt 必须留空。
4. 优先 append_to_section；只有 old_text 能从章节全文中精确复制时才用 replace_within_section。
5. 同一章节的相似修改必须合并成一条规则，不能按 case 重复追加。
6. 每条 edit 必须包含 evidence_refs，且引用本请求 evidence 中真实存在的 case_id/row_id。
7. 不删除原有核心约束；不为了提高分数放宽质量标准。
8. 生成通用规则，不要写“针对 case_xxx”这种专门补丁；不要把证据里的具体人名、剧名、地点照搬进新规则。
9. 每条新增规则尽量 1-3 行，总字数保持精简；如果现有规则已经能覆盖，输出空 edits 并说明无需修改。"""


GSB_ADVISOR_SYSTEM_PROMPT = """你是一个评测 Prompt 诊断助手。你的任务是根据人工 GSB 标注/人工复核证据，提出如何修改 Judge Prompt 或提取 Prompt。

硬性约束：
1. 只能基于用户提供的人工证据提出建议，不允许凭空猜测。
2. 每条建议必须引用 evidence 中的 row_id/case_id/pair_id。
3. 如果证据不足，请明确输出 can_suggest=false，不要生成候选 prompt。
4. 不要为了提高一致率而迎合明显错误的人工标签；如果人工标注可能有歧义，要在 risks 中说明。
5. 不要自动覆盖原 prompt，只输出候选文本和修改理由。
6. 修改提取 Prompt 时默认输出 extraction_prompt_patch，不要完整重写；patch 必须引用提供的 section_id 和 evidence_refs。
7. 如果没有原始提取 prompt，只能给 extraction_prompt_notes 或 extraction_prompt_patch，不能编造完整提取 prompt。
8. 修改必须是通用口径澄清，不要针对单个 case 写过细补丁；不要让 prompt 体积明显膨胀。

严格输出 JSON，不要输出 Markdown 代码块。
输出尽量短：修改提取 Prompt 时只输出 extraction_prompt_patch，不要输出完整提取 Prompt。"""


def load_prompt_advisor_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path).fillna("")
    if suffix == ".csv":
        return pd.read_csv(path).fillna("")
    if suffix == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows).fillna("")
    raise ValueError(f"不支持的文件格式：{suffix}")


def collect_gsb_evidence(df: pd.DataFrame, max_items: int = 30) -> list[dict[str, Any]]:
    required = {"人工GSB", "自动GSB"}
    if not required.issubset(set(df.columns)):
        return []

    rows = []
    for idx, row in df.iterrows():
        human = _clean(row.get("人工GSB"))
        auto = _clean(row.get("自动GSB"))
        if not human or not auto:
            continue
        agree = _parse_bool(row.get("是否一致"))
        if agree is True:
            continue
        rows.append({
            "row_id": _clean(row.get("row_number")) or str(idx + 2),
            "pair_id": _clean(row.get("pair_id")),
            "issue_type": _clean(row.get("问题类型")),
            "human_gsb": human,
            "auto_gsb": auto,
            "score_diff": _clean(row.get("score_diff_model1_minus_model2")),
            "model1_score": _first_existing(row, "_score", prefer_first=True),
            "model2_score": _first_existing(row, "_score", prefer_first=False),
            "model1_judge_comment": _first_existing(row, "_judge备注", prefer_first=True),
            "model2_judge_comment": _first_existing(row, "_judge备注", prefer_first=False),
            "auto_reason": _clean(row.get("自动判断备注")),
            "human_remark": _clean(row.get("备注")),
            "query": _truncate(row.get("query"), 800),
            "answer": _truncate(row.get("answer"), 800),
        })
        if len(rows) >= max_items:
            break
    return rows


def collect_review_evidence(df: pd.DataFrame, max_items: int = 30) -> list[dict[str, Any]]:
    if "human_score" not in df.columns and "human_comment" not in df.columns:
        return []

    rows = []
    for idx, row in df.iterrows():
        human_comment = _clean(row.get("human_comment"))
        human_tags = row.get("human_error_tags", "")
        human_score = row.get("human_score", "")
        llm_score = row.get("llm_score_total", row.get("score_total", ""))
        if not human_comment and not _clean(human_tags) and not _clean(human_score):
            continue
        rows.append({
            "row_id": str(idx + 1),
            "case_id": _clean(row.get("case_id")),
            "model_name": _clean(row.get("model_name")),
            "prompt_version": _clean(row.get("prompt_version")),
            "llm_score": _clean(llm_score),
            "human_score": _clean(human_score),
            "llm_error_tags": _clean(row.get("error_tags")),
            "human_error_tags": _clean(human_tags),
            "llm_comment": _clean(row.get("comment")),
            "human_comment": human_comment,
        })
        if len(rows) >= max_items:
            break
    return rows


def collect_absolute_eval_evidence(
    results: list[EvalResult],
    max_items: int = 30,
    score_threshold: float = 4.8,
    include_high_score_with_diagnostics: bool = True,
    include_all: bool = False,
) -> list[dict[str, Any]]:
    """从普通单模型评测结果中抽取可用于诊断的证据。"""
    rows: list[dict[str, Any]] = []
    for result in results:
        diagnostics = result.diagnostics or []
        error_tags = result.error_tags or []
        score_total = float(result.score_total or 0.0)
        has_issue = (
            bool(result.fatal_error)
            or score_total < score_threshold
            or bool(error_tags)
            or (include_high_score_with_diagnostics and bool(diagnostics))
        )
        if not include_all and not has_issue:
            continue

        severity = 0
        if result.fatal_error:
            severity += 100
        severity += max(0, int(round((5.0 - score_total) * 10)))
        severity += len(error_tags) * 5
        severity += len(diagnostics) * 3

        rows.append({
            "_severity": severity,
            "evidence_mode": "issue_or_low_score" if has_issue else "weak_context_from_result",
            "case_id": result.case_id,
            "model_name": result.model_name,
            "prompt_version": result.prompt_version,
            "score_total": round(score_total, 4),
            "scores": result.scores or {},
            "fatal_error": bool(result.fatal_error),
            "error_tags": error_tags,
            "comment": _truncate(result.comment, 1000),
            "diagnostics": diagnostics[:5],
            "rule_refs": (result.rule_refs or [])[:10],
            "evidence_refs": (result.evidence_refs or [])[:10],
            "output_refs": (result.output_refs or [])[:10],
            "judge_model": result.judge_model,
            "judge_prompt_version": result.judge_prompt_version,
            "extraction_prompt_version": result.extraction_prompt_version,
            "extraction_prompt_hash": result.extraction_prompt_hash,
        })

    rows.sort(key=lambda item: item.get("_severity", 0), reverse=True)
    for row in rows:
        row.pop("_severity", None)
    return rows[:max_items]


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
    if advisor_mode == "gsb_alignment":
        task = "根据人工 GSB/人工复核证据诊断 prompt 问题并给出受约束的候选修改"
        evidence_name = "human_gsb_or_review_evidence"
        warning = "这些证据来自人工标注或人工复核，可用于对齐人工口径，但仍要避免过拟合少量样本。"
    else:
        task = "根据单模型绝对评测结果诊断评测链路并给出受约束的改进建议"
        evidence_name = "absolute_eval_result_evidence"
        warning = "这些证据来自 Judge 结果，不是人工真值；候选 prompt 必须标注为待人工确认，不能直接视为正确修复。"
    weak_context_count = sum(1 for item in evidence if item.get("evidence_mode") == "weak_context_from_result")
    loop_constraints = []
    if advisor_mode == "absolute_eval" and target == "extraction_prompt":
        loop_constraints = [
            "本次目标是生成下一轮提取实验使用的候选提取 prompt。",
            "如果 evidence 包含 weak_context_from_result，说明这些样本不是错误证据，只能作为结果分布上下文。",
            "不要声称候选 prompt 已经被人工确认；必须在 risks 中说明可能沿着 Judge 偏差自我强化。",
            "默认不要完整重写提取 prompt；请输出 extraction_prompt_patch，由系统应用 patch 得到候选全文。",
            "patch 只能引用 extraction_prompt_sections 中真实存在的 section_id。",
            "每个 edit 必须包含 evidence_refs，引用 evidence 中的 case_id/row_id。",
            "优先使用 append_to_section 或 replace_within_section；不要删除核心章节。",
            "候选提取 prompt 应保留原 prompt 的核心约束，只做可解释、可回滚的增量澄清。",
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
                    "op": "append_to_section/insert_before_section/insert_after_section/replace_within_section",
                    "target_id": "必须来自 extraction_prompt_sections",
                    "old_text": "仅 replace_within_section 需要，必须是 section preview 中能确认存在的原文",
                    "new_text": "仅 replace_within_section 需要",
                    "text": "仅 append/insert 需要，直接写要追加或插入的提示词条款",
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


def _compact_json_value(value: Any, text_limit: int = 500, list_limit: int = 5) -> Any:
    if isinstance(value, str):
        return _truncate(value, text_limit)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _compact_json_value(v, text_limit=text_limit, list_limit=list_limit) for k, v in value.items()}
    if isinstance(value, list):
        return [_compact_json_value(item, text_limit=text_limit, list_limit=list_limit) for item in value[:list_limit]]
    return _truncate(value, text_limit)


def _compact_advisor_evidence(
    evidence: list[dict[str, Any]],
    *,
    max_items: int,
    text_limit: int,
    diagnostics_limit: int = 2,
    refs_limit: int = 5,
) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for item in evidence[:max_items]:
        if not isinstance(item, dict):
            continue
        row: dict[str, Any] = {}
        for key, value in item.items():
            if key == "diagnostics" and isinstance(value, list):
                row[key] = [_compact_json_value(v, text_limit=text_limit, list_limit=refs_limit) for v in value[:diagnostics_limit]]
            elif key in {"rule_refs", "evidence_refs", "output_refs", "error_tags"} and isinstance(value, list):
                row[key] = [_truncate(v, min(300, text_limit)) for v in value[:refs_limit]]
            else:
                row[key] = _compact_json_value(value, text_limit=text_limit, list_limit=refs_limit)
        compacted.append(row)
    return compacted


def _compact_extraction_advisor_evidence(
    evidence: list[dict[str, Any]],
    *,
    max_items: int,
    text_limit: int = 260,
) -> list[dict[str, Any]]:
    scalar_keys = (
        "case_id", "row_id", "pair_id", "evidence_mode", "score_total",
        "fatal_error", "comment", "human_comment", "human_score",
        "llm_comment", "llm_error_tags", "human_error_tags",
    )
    list_keys = ("error_tags", "rule_refs", "evidence_refs", "output_refs")
    rows: list[dict[str, Any]] = []
    for item in evidence[:max_items]:
        if not isinstance(item, dict):
            continue
        row: dict[str, Any] = {}
        for key in scalar_keys:
            if key in item and item.get(key) not in (None, "", [], {}):
                row[key] = _compact_json_value(item.get(key), text_limit=text_limit, list_limit=3)
        for key in list_keys:
            values = item.get(key)
            if isinstance(values, list) and values:
                row[key] = [_truncate(value, min(text_limit, 180)) for value in values[:3]]
        diagnostics = []
        for diagnostic in (item.get("diagnostics") or [])[:1]:
            if not isinstance(diagnostic, dict):
                continue
            diagnostics.append({
                key: _compact_json_value(diagnostic.get(key), text_limit=text_limit, list_limit=3)
                for key in ("dimension", "severity", "reason", "rule_refs", "evidence_refs", "output_refs")
                if diagnostic.get(key) not in (None, "", [], {})
            })
        if diagnostics:
            row["diagnostics"] = diagnostics
        if _evidence_ref_id(row):
            rows.append(row)
    return rows


def _advisor_retry_wait_seconds(config: EvalConfig, last_error: str, attempt: int) -> float:
    client = RealJudgeClient(config)
    if RealJudgeClient._is_rate_limit_error(last_error):
        return client._get_rate_limit_backoff(last_error)
    if RealJudgeClient._is_retryable_transient_error(last_error):
        return max(float(getattr(config, "judge_qps_backoff", 12.0) or 12.0), float(2 ** attempt))
    return float(2 ** attempt)


def _advisor_attempt_profile(
    *,
    attempt: int,
    target: str,
    evidence_count: int,
    min_evidence: int,
) -> dict[str, int]:
    is_extraction_target = target in {"extraction_prompt", "both"}
    floor = max(1, int(min_evidence or 0))

    if attempt <= 1:
        cap = 12 if is_extraction_target else 20
        profile = {
            "max_items": cap,
            "text_limit": 420,
            "diagnostics_limit": 2,
            "refs_limit": 4,
            "judge_prompt_limit": 3500 if is_extraction_target else 6000,
            "extraction_prompt_limit": 0 if is_extraction_target else 2500,
            "extraction_section_limit": 70,
            "extraction_section_preview_chars": 280,
            "max_tokens_cap": 3200 if is_extraction_target else 5000,
        }
    elif attempt == 2:
        cap = 8 if is_extraction_target else 12
        profile = {
            "max_items": cap,
            "text_limit": 280,
            "diagnostics_limit": 2,
            "refs_limit": 3,
            "judge_prompt_limit": 2500 if is_extraction_target else 4500,
            "extraction_prompt_limit": 0 if is_extraction_target else 1500,
            "extraction_section_limit": 45,
            "extraction_section_preview_chars": 220,
            "max_tokens_cap": 2400 if is_extraction_target else 4000,
        }
    else:
        cap = 5 if is_extraction_target else 8
        profile = {
            "max_items": cap,
            "text_limit": 180,
            "diagnostics_limit": 1,
            "refs_limit": 2,
            "judge_prompt_limit": 1600 if is_extraction_target else 3000,
            "extraction_prompt_limit": 0 if is_extraction_target else 900,
            "extraction_section_limit": 28,
            "extraction_section_preview_chars": 160,
            "max_tokens_cap": 1600 if is_extraction_target else 3000,
        }

    profile["max_items"] = min(evidence_count, max(floor, profile["max_items"]))
    return profile


def _advisor_max_tokens(config: EvalConfig, profile: dict[str, int]) -> int:
    configured = int(getattr(config, "judge_max_tokens", 2000) or 2000)
    cap = int(profile.get("max_tokens_cap", 3000) or 3000)
    if configured <= 0:
        configured = 2000
    return max(800, min(configured, cap))


def _shrink_advisor_message_for_retry(user_message: str, attempt: int) -> str:
    if attempt <= 1:
        return user_message
    try:
        payload = json.loads(user_message)
    except Exception:
        return user_message
    if not isinstance(payload, dict):
        return user_message

    item_limit = 2 if attempt == 2 else 1
    evidence = payload.get("evidence")
    if isinstance(evidence, list):
        payload["evidence"] = evidence[:item_limit]

    outline = payload.get("prompt_global_outline")
    if isinstance(outline, list):
        outline_limit = 80 if attempt == 2 else 40
        payload["prompt_global_outline"] = [
            {
                key: value
                for key, value in item.items()
                if key != "preview" or attempt == 2
            }
            for item in outline[:outline_limit]
            if isinstance(item, dict)
        ]
        if attempt == 2:
            for item in payload["prompt_global_outline"]:
                if "preview" in item:
                    item["preview"] = _truncate(item["preview"], 60)
    if "current_judge_prompt_excerpt" in payload:
        payload["current_judge_prompt_excerpt"] = ""

    section_group = payload.get("section_group")
    if isinstance(section_group, dict):
        section_group["patch_intents"] = list(section_group.get("patch_intents") or [])[:item_limit]
        section_group["evidence_refs"] = list(section_group.get("evidence_refs") or [])[:item_limit]

    block_context = payload.get("target_section_blocks")
    if isinstance(block_context, dict):
        block_context["editable_blocks"] = list(block_context.get("editable_blocks") or [])[:1]
        outlines = list(block_context.get("block_outline") or [])
        block_context["block_outline"] = [
            {
                **item,
                "preview": _truncate(item.get("preview"), 80 if attempt == 2 else 40),
            }
            for item in outlines[:(20 if attempt == 2 else 10)]
            if isinstance(item, dict)
        ]
    if attempt >= 2 and "neighbor_section_outline" in payload:
        payload["neighbor_section_outline"] = []

    payload["retry_compaction_level"] = attempt - 1
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _call_advisor_json(
    *,
    config: EvalConfig,
    url: str,
    headers: dict[str, str],
    client: RealJudgeClient,
    system_prompt: str,
    user_message: str,
    max_tokens: int,
) -> tuple[dict[str, Any] | None, str, str]:
    max_attempts = max(1, int(config.judge_max_retries or 1))
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        raw_text = ""
        attempt_message = _shrink_advisor_message_for_retry(user_message, attempt)
        attempt_max_tokens = max(800, int(max_tokens or 2000) - ((attempt - 1) * 400))
        payload = {
            "model": config.judge_model,
            "max_tokens": attempt_max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": attempt_message},
            ],
            "extra_body": {
                "enable_thinking": False,
                "skip_special_tokens": False,
            },
        }
        try:
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
                        return parsed, content or raw_text, ""
                    last_error = f"提示词建议输出不是可解析 JSON: {content[:1000]}"
        except requests.exceptions.Timeout:
            last_error = f"请求超时 ({attempt}/{max_attempts})"
        except requests.exceptions.RequestException as exc:
            last_error = f"请求异常 ({attempt}/{max_attempts}): {exc}"
        except Exception as exc:
            last_error = f"未知错误 ({attempt}/{max_attempts}): {exc}"

        if attempt < max_attempts:
            time.sleep(_advisor_retry_wait_seconds(config, last_error, attempt))

    return None, raw_text or last_error, last_error


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


def _section_context(
    sections: list[PromptSection],
    target_id: str,
    *,
    neighbor_count: int = 1,
    text_limit: int = 6000,
) -> list[dict[str, Any]]:
    index_by_id = {section.section_id: idx for idx, section in enumerate(sections)}
    if target_id not in index_by_id:
        return []
    target_index = index_by_id[target_id]
    start = max(0, target_index - neighbor_count)
    end = min(len(sections), target_index + neighbor_count + 1)
    rows: list[dict[str, Any]] = []
    for idx in range(start, end):
        section = sections[idx]
        text = section.text.strip()
        rows.append({
            "section_id": section.section_id,
            "title": section.title,
            "level": section.level,
            "hash": _section_hash(section.text),
            "role": "target" if section.section_id == target_id else "neighbor",
            "is_truncated": len(text) > text_limit,
            "full_text": text[:text_limit],
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
            "block_outline 只用于判断是否重复，不能从预览中复制 old_text。",
            "如 has_oversized_uneditable_block=true，不得改写该块，只能在确有必要时向章节末尾追加一条通用规则。",
            "同一章节的相似规则必须合并，不能按 case 重复追加。",
            "只写通用规则，不要把证据中的具体实体、人名、作品名、地点写进 prompt。",
            "每条新增规则保持短小，优先 1 条合并规则；不要添加多条细碎规则。",
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
                        "op": "append_to_section/replace_within_section",
                        "target_id": group.get("section_id"),
                        "old_text": "仅 replace_within_section 需要，必须从目标章节全文精确复制",
                        "new_text": "仅 replace_within_section 需要",
                        "text": "仅 append_to_section 需要；必须是通用规则，不要针对具体 case，不要冗余",
                        "reason": "修改原因",
                        "evidence_refs": ["case_id 或 row_id"],
                    }
                ],
            },
            "section_notes": "本章节为什么这样改；如果不改，说明原因",
            "risks": ["不超过3条"],
        },
    }, ensure_ascii=False, indent=2)


def _normalize_text_key(value: str) -> str:
    return "".join(str(value or "").lower().split())


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
    text = _clean(item.get("text") or item.get("insert_text"))
    old_text = _clean(item.get("old_text") or item.get("target_text"))
    new_text = _clean(item.get("new_text") or item.get("replacement_text"))
    if op == "replace_within_section" and (not old_text or not new_text):
        return _invalid_patch_edit(index, "replace 缺少 old_text 或 new_text，已跳过。", raw)
    if op == "replace_within_section" and (len(old_text) > MAX_ADVISOR_REPLACE_TEXT_CHARS or len(new_text) > MAX_ADVISOR_REPLACE_TEXT_CHARS):
        return _invalid_patch_edit(index, f"replace 文本过长，超过 {MAX_ADVISOR_REPLACE_TEXT_CHARS} 字，已跳过。", raw)
    if op == "replace_within_section" and _looks_case_specific(new_text, refs):
        return _invalid_patch_edit(index, "new_text 看起来针对具体 case/样本，已跳过。", raw)
    if op != "replace_within_section":
        if op not in {"append_to_section", "insert_before_section", "insert_after_section"}:
            op = "append_to_section"
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
    for item in append_groups.values():
        edits.append({k: v for k, v in item.items() if not k.startswith("_")})
    edits.sort(key=lambda item: (item.get("target_id") or "", item.get("op") or "", item.get("edit_id") or ""))
    limited_edits: list[dict[str, Any]] = []
    total_patch_chars = 0
    for edit in edits:
        change_text = str(edit.get("text") or edit.get("new_text") or "")
        change_size = len(change_text)
        if len(limited_edits) >= MAX_ADVISOR_PATCH_EDITS:
            skipped.append({**edit, "message": f"patch edit 数超过 {MAX_ADVISOR_PATCH_EDITS} 条，为避免 prompt 暴增，已跳过。"})
            continue
        if total_patch_chars + change_size > MAX_ADVISOR_TOTAL_PATCH_TEXT_CHARS:
            skipped.append({**edit, "message": f"patch 总新增文本超过 {MAX_ADVISOR_TOTAL_PATCH_TEXT_CHARS} 字，为避免 prompt 暴增，已跳过。"})
            continue
        limited_edits.append(edit)
        total_patch_chars += change_size
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
    stage1_batches = [
        stage1_evidence[index:index + ADVISOR_STAGE1_BATCH_SIZE]
        for index in range(0, len(stage1_evidence), ADVISOR_STAGE1_BATCH_SIZE)
    ] or [[]]
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
            section_preview_chars=100,
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

    finalized = _finalize_advisor_result(base_result, extraction_prompt=extraction_prompt, target=target)
    return finalized, json.dumps(raw_payload, ensure_ascii=False, indent=2)


def call_prompt_advisor(
    config: EvalConfig,
    evidence: list[dict[str, Any]],
    current_judge_prompt: str,
    extraction_prompt: str = "",
    target: str = "judge_prompt",
    advisor_mode: str = "absolute_eval",
    min_evidence: int = 3,
) -> tuple[dict[str, Any] | None, str]:
    if len(evidence) < min_evidence:
        if advisor_mode == "gsb_alignment":
            summary = f"人工证据少于 {min_evidence} 条，拒绝生成候选 prompt，避免模型凭空调参。"
            plan = [f"至少收集 {min_evidence} 条有人工标签且自动结果不一致/人工复核指出问题的样本。"]
        else:
            summary = f"评测结果证据少于 {min_evidence} 条，拒绝生成候选 prompt，避免根据个例过拟合。"
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
        }, ""

    if config.mock:
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
                            "text": "<!-- MOCK: 这里示例展示增量 patch 机制，真实运行时不会插入这段。 -->",
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
        }
        mock_result = _finalize_advisor_result(mock_result, extraction_prompt=extraction_prompt, target=target)
        return mock_result, json.dumps(mock_result, ensure_ascii=False)

    url = RealJudgeClient._normalize_chat_completions_url(config.judge_api_base_url)
    system_prompt = GSB_ADVISOR_SYSTEM_PROMPT if advisor_mode == "gsb_alignment" else ABSOLUTE_ADVISOR_SYSTEM_PROMPT
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
        )

    last_error = ""
    used_compact_retry = False

    max_attempts = max(1, int(config.judge_max_retries or 1))
    for attempt in range(1, max_attempts + 1):
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
        try:
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
                        return parsed, content or raw_text
                    last_error = f"提示词建议输出不是可解析 JSON: {content[:1000]}"

        except requests.exceptions.Timeout:
            last_error = f"请求超时 ({attempt}/{max_attempts})"
        except requests.exceptions.RequestException as exc:
            last_error = f"请求异常 ({attempt}/{max_attempts}): {exc}"
        except Exception as exc:
            last_error = f"未知错误 ({attempt}/{max_attempts}): {exc}"

        if attempt < max_attempts:
            time.sleep(_advisor_retry_wait_seconds(config, last_error, attempt))

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


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = _clean(value).lower()
    if text in {"true", "1", "yes", "y", "是"}:
        return True
    if text in {"false", "0", "no", "n", "否"}:
        return False
    return None


def _first_existing(row: pd.Series, suffix: str, prefer_first: bool) -> str:
    cols = [c for c in row.index if str(c).endswith(suffix)]
    if not cols:
        return ""
    col = cols[0] if prefer_first else cols[-1]
    return _clean(row.get(col))
