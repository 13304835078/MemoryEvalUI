from __future__ import annotations

from typing import Any

import pandas as pd

from src.schema import EvalResult
from src.eval.result_status import result_is_score_eligible


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
    positive_boundary_limit: int = 0,
    regression_results: list[EvalResult] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    positive_rows: list[dict[str, Any]] = []
    for result in results:
        # API/network/JSON failures describe runtime health, not prompt quality.
        if not result_is_score_eligible(result):
            continue
        diagnostics = result.diagnostics or []
        error_tags = result.error_tags or []
        score_total = float(result.score_total or 0.0)
        has_issue = (
            bool(result.fatal_error)
            or score_total < score_threshold
            or bool(error_tags)
            or (include_high_score_with_diagnostics and bool(diagnostics))
        )
        evidence_mode = "issue_or_low_score" if has_issue else "weak_context_from_result"
        if not has_issue and positive_boundary_limit > 0:
            evidence_mode = "positive_boundary"
        elif not include_all and not has_issue:
            continue

        severity = 0
        if result.fatal_error:
            severity += 100
        severity += max(0, int(round((5.0 - score_total) * 10)))
        severity += len(error_tags) * 5
        severity += len(diagnostics) * 3

        row = {
            "_severity": severity,
            "evidence_mode": evidence_mode,
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
        }
        if evidence_mode == "positive_boundary":
            positive_rows.append(row)
        else:
            rows.append(row)

    for result in regression_results or []:
        if not result_is_score_eligible(result):
            continue
        if (
            float(result.score_total or 0.0) < score_threshold
            or result.fatal_error
            or result.error_tags
        ):
            continue
        rows.append({
            "_severity": 1,
            "evidence_mode": "regression_boundary",
            "case_id": result.case_id,
            "model_name": result.model_name,
            "prompt_version": result.prompt_version,
            "score_total": round(float(result.score_total or 0.0), 4),
            "scores": result.scores or {},
            "fatal_error": bool(result.fatal_error),
            "error_tags": result.error_tags or [],
            "comment": _truncate(result.comment, 1000),
            "diagnostics": (result.diagnostics or [])[:5],
            "rule_refs": (result.rule_refs or [])[:10],
            "evidence_refs": (result.evidence_refs or [])[:10],
            "output_refs": (result.output_refs or [])[:10],
            "judge_model": result.judge_model,
            "judge_prompt_version": result.judge_prompt_version,
            "extraction_prompt_version": result.extraction_prompt_version,
            "extraction_prompt_hash": result.extraction_prompt_hash,
        })

    rows.sort(key=lambda item: item.get("_severity", 0), reverse=True)
    positive_rows.sort(key=lambda item: float(item.get("score_total") or 0.0), reverse=True)
    selected_positive = positive_rows[:min(max(0, int(positive_boundary_limit)), max_items)]
    rows = rows[:max(0, max_items - len(selected_positive))] + selected_positive
    for row in rows:
        row.pop("_severity", None)
    return rows


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
