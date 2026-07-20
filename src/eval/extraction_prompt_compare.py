from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass, replace
from statistics import mean
from typing import Any

from src.eval.result_status import (
    STATUS_LABELS,
    result_evaluation_status,
    result_is_score_eligible,
)
from src.eval.run_quality import compute_run_quality
from src.loop.validation_gate import ValidationGateConfig, evaluate_candidate_gate
from src.schema import Case, EvalResult


_OUTPUT_COLUMNS = (
    "effective_document",
    "user.md",
    "USER.md",
    "MEMORY.md",
    "parsed_document",
)


@dataclass(frozen=True)
class ExtractionPair:
    source_key: str
    case_a: Case | None = None
    case_b: Case | None = None
    missed_a: Case | None = None
    missed_b: Case | None = None


def source_case_key(case: Case) -> str:
    """Return a prompt-independent key for the same source chunk."""
    metadata = case.metadata if isinstance(case.metadata, dict) else {}
    reviewer = str(metadata.get("reviewer") or "unknown_reviewer").strip()
    session_id = str(
        metadata.get("source_session_id") or case.session_id or "unknown_session"
    ).strip()
    chunk_index = metadata.get("chunk_index_in_session")
    row_start = metadata.get("row_start")
    row_end = metadata.get("row_end")
    return "|".join(
        (
            str(case.task_type.value),
            reviewer,
            session_id,
            str(chunk_index if chunk_index is not None else ""),
            str(row_start if row_start is not None else ""),
            str(row_end if row_end is not None else ""),
        )
    )


def _unique_case_map(cases: list[Case]) -> tuple[dict[str, Case], list[str]]:
    mapping: dict[str, Case] = {}
    duplicates: list[str] = []
    for case in cases:
        key = source_case_key(case)
        if key in mapping:
            duplicates.append(key)
            continue
        mapping[key] = case
    return mapping, sorted(set(duplicates))


def _result_map(cases: list[Case], results: list[EvalResult]) -> dict[str, EvalResult]:
    case_by_id = {case.case_id: case for case in cases}
    mapped: dict[str, EvalResult] = {}
    for result in results:
        case = case_by_id.get(result.case_id)
        if case is not None:
            mapped[source_case_key(case)] = result
    return mapped


def _status_label(result: EvalResult | None) -> str:
    if result is None:
        return "未评测"
    status = result_evaluation_status(result)
    return STATUS_LABELS.get(status, status)


def _tag_text(tags: list[str] | None) -> str:
    return "、".join(sorted(set(tags or [])))


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if value != value:
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _canonical_id(value: Any) -> str:
    text = _cell_text(value)
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else text


def _source_row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        _cell_text(row.get("评测人")),
        _canonical_id(row.get("session_id")),
        _canonical_id(row.get("chunk_id")),
    )


def _comparison_row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    chunk_id = row.get("chunk_id")
    if _cell_text(chunk_id) == "":
        chunk_index = _canonical_id(row.get("chunk_index"))
        chunk_id = int(chunk_index) + 1 if chunk_index.lstrip("-").isdigit() else chunk_index
    return (
        _cell_text(row.get("reviewer")),
        _canonical_id(row.get("session_id")),
        _canonical_id(chunk_id),
    )


def _ordered_row_groups(
    rows: list[dict[str, Any]],
) -> tuple[list[tuple[str, str, str]], dict[tuple[str, str, str], list[dict[str, Any]]]]:
    order: list[tuple[str, str, str]] = []
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = _source_row_key(row)
        if key not in groups:
            order.append(key)
            groups[key] = []
        groups[key].append(row)
    return order, groups


def _last_nonempty(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> str:
    for row in reversed(rows):
        for column in columns:
            value = _cell_text(row.get(column))
            if value:
                return value
    return ""


def _dimension_score_text(value: Any) -> str:
    if not isinstance(value, dict):
        return _cell_text(value)
    return "；".join(f"{key}={score}" for key, score in sorted(value.items()))


def _diff_comparison_fields(comparison: dict[str, Any] | None) -> dict[str, Any]:
    comparison = comparison or {}
    return {
        "A提取状态": comparison.get("extraction_a", ""),
        "B提取状态": comparison.get("extraction_b", ""),
        "A Judge状态": comparison.get("judge_status_a", ""),
        "B Judge状态": comparison.get("judge_status_b", ""),
        "A总分": comparison.get("score_a"),
        "B总分": comparison.get("score_b"),
        "B-A": comparison.get("score_delta_b_minus_a"),
        "A维度得分": _dimension_score_text(comparison.get("scores_a")),
        "B维度得分": _dimension_score_text(comparison.get("scores_b")),
        "对比结论": comparison.get("comparison", ""),
        "对比备注": comparison.get("comparison_note", ""),
        "A错误标签": comparison.get("error_tags_a", ""),
        "B错误标签": comparison.get("error_tags_b", ""),
        "A评语": comparison.get("comment_a", ""),
        "B评语": comparison.get("comment_b", ""),
        "A规则引用": comparison.get("rule_refs_a", ""),
        "B规则引用": comparison.get("rule_refs_b", ""),
        "对比调用状态": comparison.get("pairwise_status", ""),
        "对比模型": comparison.get("pairwise_model", ""),
        "对比置信度": comparison.get("pairwise_confidence", ""),
        "判定依据类型": comparison.get("decision_basis", ""),
        "规则引用": comparison.get("rule_refs", ""),
        "策略差异": comparison.get("policy_differences", ""),
        "证据引用": comparison.get("evidence_refs", ""),
        "A相对问题": comparison.get("issues_a", ""),
        "B相对问题": comparison.get("issues_b", ""),
        "A相对优点": comparison.get("strengths_a", ""),
        "B相对优点": comparison.get("strengths_b", ""),
        "对比调用错误": comparison.get("comparison_error", ""),
    }


def build_extraction_prompt_diff(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    """Build row-level and chunk-level A/B views from two extraction workbooks."""
    order_a, groups_a = _ordered_row_groups(rows_a)
    order_b, groups_b = _ordered_row_groups(rows_b)
    ordered_keys = order_a + [key for key in order_b if key not in groups_a]
    comparisons = {_comparison_row_key(row): row for row in comparison_rows}
    include_reasoning = any(
        _last_nonempty(group, ("reasoning",))
        for group in list(groups_a.values()) + list(groups_b.values())
    )

    row_diff: list[dict[str, Any]] = []
    chunk_diff: list[dict[str, Any]] = []
    for key in ordered_keys:
        group_a = groups_a.get(key, [])
        group_b = groups_b.get(key, [])
        row_count = max(len(group_a), len(group_b))
        if row_count == 0:
            continue

        output_a = _last_nonempty(group_a, _OUTPUT_COLUMNS)
        output_b = _last_nonempty(group_b, _OUTPUT_COLUMNS)
        reasoning_a = _last_nonempty(group_a, ("reasoning",))
        reasoning_b = _last_nonempty(group_b, ("reasoning",))
        comparison_fields = _diff_comparison_fields(comparisons.get(key))
        source_pairs: list[tuple[str, str]] = []
        source_consistent = len(group_a) == len(group_b)

        for index in range(row_count):
            row_a = group_a[index] if index < len(group_a) else {}
            row_b = group_b[index] if index < len(group_b) else {}
            query_a, query_b = _cell_text(row_a.get("query")), _cell_text(row_b.get("query"))
            answer_a, answer_b = _cell_text(row_a.get("answer")), _cell_text(row_b.get("answer"))
            if row_a and row_b and (query_a != query_b or answer_a != answer_b):
                source_consistent = False
            query = query_a or query_b
            answer = answer_a or answer_b
            source_pairs.append((query, answer))
            is_boundary = index == row_count - 1
            diff_row: dict[str, Any] = {
                "session_id": key[1],
                "chunk_id": key[2],
                "query": query,
                "answer": answer,
                "评测人": key[0],
                "A提取结果": output_a if is_boundary else "",
                "B提取结果": output_b if is_boundary else "",
            }
            if include_reasoning:
                diff_row["A_reasoning"] = reasoning_a if is_boundary else ""
                diff_row["B_reasoning"] = reasoning_b if is_boundary else ""
            diff_row["源数据一致性"] = (
                ("一致" if source_consistent else "A/B源数据不同") if is_boundary else ""
            )
            diff_row.update(
                comparison_fields
                if is_boundary
                else {column: "" for column in comparison_fields}
            )
            row_diff.append(diff_row)

        chunk_row: dict[str, Any] = {
            "session_id": key[1],
            "chunk_id": key[2],
            "query": "\n\n".join(query for query, _ in source_pairs if query),
            "answer": "\n\n".join(answer for _, answer in source_pairs if answer),
            "评测人": key[0],
            "A提取结果": output_a,
            "B提取结果": output_b,
        }
        if include_reasoning:
            chunk_row["A_reasoning"] = reasoning_a
            chunk_row["B_reasoning"] = reasoning_b
        chunk_row["源数据一致性"] = "一致" if source_consistent else "A/B源数据不同"
        chunk_row.update(comparison_fields)
        chunk_diff.append(chunk_row)

    return row_diff, chunk_diff, include_reasoning


def _normalized_output(case: Case | None) -> str:
    if case is None:
        return ""
    lines = [line.rstrip() for line in str(case.candidate_output or "").replace("\r\n", "\n").split("\n")]
    return "\n".join(lines).strip()


def _judge_outputs_disagree(result_a: EvalResult, result_b: EvalResult) -> bool:
    return bool(
        abs(float(result_a.score_total) - float(result_b.score_total)) > 1e-12
        or (result_a.scores or {}) != (result_b.scores or {})
        or set(result_a.error_tags or []) != set(result_b.error_tags or [])
        or bool(result_a.fatal_error) != bool(result_b.fatal_error)
    )


def _comparison_note(
    case_a: Case | None,
    case_b: Case | None,
    result_a: EvalResult | None,
    result_b: EvalResult | None,
    *,
    tolerance: float,
) -> tuple[str, str]:
    if case_a is None:
        return "B独有", "A 未生成可评测 case；请先核对 A 的提取状态和漏抽原因。"
    if case_b is None:
        return "A独有", "B 未生成可评测 case；这是覆盖率退化，不应由其余高分抵消。"
    if result_a is None or result_b is None:
        missing = "A" if result_a is None else "B"
        return "不可比较", f"{missing} 尚无 Judge 结果，当前样本不进入配对分数统计。"
    if not result_is_score_eligible(result_a) or not result_is_score_eligible(result_b):
        failed = []
        if not result_is_score_eligible(result_a):
            failed.append(f"A：{_status_label(result_a)}")
        if not result_is_score_eligible(result_b):
            failed.append(f"B：{_status_label(result_b)}")
        return "不可比较", "；".join(failed) + "。运行异常不按 0 分计入质量统计。"

    if _normalized_output(case_a) == _normalized_output(case_b):
        if _judge_outputs_disagree(result_a, result_b):
            return "输出相同", "A/B 提取正文相同，质量按持平处理；两次 Judge 结果不同，已单独记为裁判波动。"
        return "输出相同", "A/B 提取正文与 Judge 结果均相同，提示词在该样本上没有可观察差异。"

    delta = float(result_b.score_total) - float(result_a.score_total)
    if delta > tolerance:
        winner = "B较优"
    elif delta < -tolerance:
        winner = "A较优"
    else:
        winner = "基本持平"

    added = sorted(set(result_b.error_tags or []) - set(result_a.error_tags or []))
    removed = sorted(set(result_a.error_tags or []) - set(result_b.error_tags or []))
    details = [f"B-A 得分差 {delta:+.2f}"]
    if removed:
        details.append(f"B 消除了错误标签：{_tag_text(removed)}")
    if added:
        details.append(f"B 新增错误标签：{_tag_text(added)}")
    if bool(result_a.fatal_error) != bool(result_b.fatal_error):
        details.append("B 出现致命错误" if result_b.fatal_error else "B 消除了致命错误")
    if not added and not removed and abs(delta) <= tolerance:
        details.append("得分与错误标签均无实质变化")
    return winner, "；".join(details) + "。"


def _aligned_for_gate(
    cases: list[Case],
    missed_cases: list[Case],
    results: list[EvalResult],
    excluded_keys: set[str],
) -> tuple[list[Case], list[Case], list[EvalResult]]:
    case_by_id = {case.case_id: case for case in cases}
    aligned_cases = [
        replace(case, case_id=source_case_key(case))
        for case in cases
        if source_case_key(case) not in excluded_keys
    ]
    aligned_missed = [
        replace(case, case_id=source_case_key(case))
        for case in missed_cases
        if source_case_key(case) not in excluded_keys
    ]
    aligned_results: list[EvalResult] = []
    for result in results:
        case = case_by_id.get(result.case_id)
        if case is None:
            continue
        key = source_case_key(case)
        if key not in excluded_keys:
            aligned_results.append(replace(result, case_id=key))
    return aligned_cases, aligned_missed, aligned_results


def _dimension_summary(
    result_a_by_key: dict[str, EvalResult],
    result_b_by_key: dict[str, EvalResult],
    identical_output_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    identical_output_keys = identical_output_keys or set()
    paired_keys = [
        key
        for key in sorted(set(result_a_by_key) & set(result_b_by_key))
        if result_is_score_eligible(result_a_by_key[key])
        and result_is_score_eligible(result_b_by_key[key])
    ]
    dimensions = sorted(
        {
            dimension
            for key in paired_keys
            for result in (result_a_by_key[key], result_b_by_key[key])
            for dimension in (result.scores or {})
        }
    )
    rows: list[dict[str, Any]] = []
    for dimension in dimensions:
        values_a = [float((result_a_by_key[key].scores or {}).get(dimension, 0.0)) for key in paired_keys]
        values_b = [
            float(
                (
                    (result_a_by_key[key].scores or {})
                    if key in identical_output_keys
                    else (result_b_by_key[key].scores or {})
                ).get(dimension, 0.0)
            )
            for key in paired_keys
        ]
        avg_a = mean(values_a) if values_a else 0.0
        avg_b = mean(values_b) if values_b else 0.0
        rows.append(
            {
                "dimension": dimension,
                "avg_a": round(avg_a, 4),
                "avg_b": round(avg_b, 4),
                "delta_b_minus_a": round(avg_b - avg_a, 4),
                "paired_count": len(paired_keys),
            }
        )
    return rows


def compare_extraction_prompt_runs(
    *,
    cases_a: list[Case],
    cases_b: list[Case],
    missed_cases_a: list[Case],
    missed_cases_b: list[Case],
    results_a: list[EvalResult],
    results_b: list[EvalResult],
    prompt_a: str,
    prompt_b: str,
    validation_config: ValidationGateConfig | None = None,
    score_tolerance: float = 0.05,
) -> dict[str, Any]:
    """Compare two extraction prompts while keeping Judge inputs fixed."""
    cases_a_by_key, duplicates_a = _unique_case_map(cases_a)
    cases_b_by_key, duplicates_b = _unique_case_map(cases_b)
    missed_a_by_key, missed_duplicates_a = _unique_case_map(missed_cases_a)
    missed_b_by_key, missed_duplicates_b = _unique_case_map(missed_cases_b)
    duplicate_keys = set(duplicates_a + duplicates_b + missed_duplicates_a + missed_duplicates_b)
    results_a_by_key = _result_map(cases_a, results_a)
    results_b_by_key = _result_map(cases_b, results_b)

    all_keys = sorted(
        (set(cases_a_by_key) | set(cases_b_by_key) | set(missed_a_by_key) | set(missed_b_by_key))
        - duplicate_keys
    )
    identical_output_keys = {
        key
        for key in set(cases_a_by_key) & set(cases_b_by_key)
        if _normalized_output(cases_a_by_key[key]) == _normalized_output(cases_b_by_key[key])
    }
    rows: list[dict[str, Any]] = []
    winner_counts = {"A较优": 0, "B较优": 0, "基本持平": 0, "输出相同": 0, "A独有": 0, "B独有": 0, "不可比较": 0}
    for key in all_keys:
        case_a = cases_a_by_key.get(key)
        case_b = cases_b_by_key.get(key)
        result_a = results_a_by_key.get(key)
        result_b = results_b_by_key.get(key)
        winner, note = _comparison_note(
            case_a,
            case_b,
            result_a,
            result_b,
            tolerance=max(0.0, float(score_tolerance)),
        )
        winner_counts[winner] = winner_counts.get(winner, 0) + 1
        source = case_a or case_b or missed_a_by_key.get(key) or missed_b_by_key.get(key)
        metadata = source.metadata if source and isinstance(source.metadata, dict) else {}
        pair_eligible = bool(
            result_a
            and result_b
            and result_is_score_eligible(result_a)
            and result_is_score_eligible(result_b)
        )
        rows.append(
            {
                "source_key": key,
                "reviewer": metadata.get("reviewer", ""),
                "session_id": metadata.get("source_session_id", getattr(source, "session_id", "")),
                "chunk_index": metadata.get("chunk_index_in_session", ""),
                "chunk_id": (
                    int(metadata.get("chunk_index_in_session")) + 1
                    if str(metadata.get("chunk_index_in_session", "")).lstrip("-").isdigit()
                    else metadata.get("chunk_index_in_session", "")
                ),
                "row_start": metadata.get("row_start", ""),
                "row_end": metadata.get("row_end", ""),
                "case_id_a": case_a.case_id if case_a else "",
                "case_id_b": case_b.case_id if case_b else "",
                "extraction_a": "可评测" if case_a else ("漏抽/不可评测" if key in missed_a_by_key else "缺失"),
                "extraction_b": "可评测" if case_b else ("漏抽/不可评测" if key in missed_b_by_key else "缺失"),
                "judge_status_a": _status_label(result_a),
                "judge_status_b": _status_label(result_b),
                "score_a": float(result_a.score_total) if result_a and result_is_score_eligible(result_a) else None,
                "score_b": float(result_b.score_total) if result_b and result_is_score_eligible(result_b) else None,
                "score_delta_b_minus_a": (
                    round(float(result_b.score_total) - float(result_a.score_total), 4)
                    if pair_eligible
                    else None
                ),
                "scores_a": dict(result_a.scores or {}) if result_a else {},
                "scores_b": dict(result_b.scores or {}) if result_b else {},
                "comparison": winner,
                "comparison_note": note,
                "error_tags_a": _tag_text(result_a.error_tags if result_a else []),
                "error_tags_b": _tag_text(result_b.error_tags if result_b else []),
                "comment_a": result_a.comment if result_a else "",
                "comment_b": result_b.comment if result_b else "",
                "rule_refs_a": "；".join(result_a.rule_refs or []) if result_a else "",
                "rule_refs_b": "；".join(result_b.rule_refs or []) if result_b else "",
                "old_memory_a": case_a.old_memory if case_a else "",
                "old_memory_b": case_b.old_memory if case_b else "",
                "candidate_output_a": case_a.candidate_output if case_a else "",
                "candidate_output_b": case_b.candidate_output if case_b else "",
            }
        )

    aligned_a_cases, aligned_a_missed, aligned_a_results = _aligned_for_gate(
        cases_a, missed_cases_a, results_a, duplicate_keys
    )
    aligned_b_cases, aligned_b_missed, aligned_b_results = _aligned_for_gate(
        cases_b, missed_cases_b, results_b, duplicate_keys
    )
    aligned_a_result_map = {result.case_id: result for result in aligned_a_results}
    adjusted_b_results: list[EvalResult] = []
    for result in aligned_b_results:
        baseline = aligned_a_result_map.get(result.case_id)
        if (
            result.case_id in identical_output_keys
            and baseline is not None
            and result_is_score_eligible(baseline)
            and result_is_score_eligible(result)
        ):
            adjusted_b_results.append(
                replace(
                    result,
                    score_total=baseline.score_total,
                    scores=dict(baseline.scores or {}),
                    error_tags=list(baseline.error_tags or []),
                    fatal_error=bool(baseline.fatal_error),
                )
            )
        else:
            adjusted_b_results.append(result)
    gate = evaluate_candidate_gate(
        aligned_a_results,
        adjusted_b_results,
        champion_cases=aligned_a_cases,
        candidate_cases=aligned_b_cases,
        champion_missed_cases=aligned_a_missed,
        candidate_missed_cases=aligned_b_missed,
        champion_prompt=prompt_a,
        candidate_prompt=prompt_b,
        config=validation_config or ValidationGateConfig(),
    )
    quality_a = compute_run_quality(results_a, cases=cases_a, missed_cases=missed_cases_a)
    quality_b = compute_run_quality(results_b, cases=cases_b, missed_cases=missed_cases_b)
    judge_disagreement_keys = sorted(
        key
        for key in identical_output_keys
        if key in results_a_by_key
        and key in results_b_by_key
        and result_is_score_eligible(results_a_by_key[key])
        and result_is_score_eligible(results_b_by_key[key])
        and _judge_outputs_disagree(results_a_by_key[key], results_b_by_key[key])
    )

    if duplicate_keys:
        recommendation = "暂不定版"
        recommendation_reason = "来源键存在重复，需先修复数据边界后再比较。"
    elif gate.get("accepted"):
        recommendation = "建议选择 B"
        recommendation_reason = "B 通过覆盖率、退化率、关键错误和统计置信度门槛。"
    elif not quality_a["run_complete"] or not quality_b["run_complete"]:
        recommendation = "暂不定版"
        recommendation_reason = "存在提取或 Judge 运行异常，先补跑失败样本再做版本选择。"
    elif float(gate.get("paired_score_delta") or 0.0) < -max(0.0, float(score_tolerance)):
        recommendation = "建议保留 A"
        recommendation_reason = "B 在同源配对样本上的平均得分出现实质退化。"
    elif float(gate.get("extraction_coverage_drop") or 0.0) > 0:
        recommendation = "建议保留 A"
        recommendation_reason = "B 的提取覆盖率下降，漏抽不能由其余样本高分抵消。"
    else:
        recommendation = "证据不足，暂时保留 A"
        recommendation_reason = "B 尚未达到替换门槛；可增加独立评测人/时序簇后复验。"

    return {
        "recommendation": recommendation,
        "recommendation_reason": recommendation_reason,
        "quality_a": quality_a,
        "quality_b": quality_b,
        "validation_gate": gate,
        "winner_counts": winner_counts,
        "dimension_summary": _dimension_summary(
            results_a_by_key,
            results_b_by_key,
            identical_output_keys,
        ),
        "identical_output_count": len(identical_output_keys),
        "judge_disagreement_on_identical_output_count": len(judge_disagreement_keys),
        "judge_disagreement_on_identical_output_keys": judge_disagreement_keys,
        "duplicate_source_keys": sorted(duplicate_keys),
        "rows": rows,
    }


def build_extraction_pairs(
    *,
    cases_a: list[Case],
    cases_b: list[Case],
    missed_cases_a: list[Case],
    missed_cases_b: list[Case],
) -> tuple[list[ExtractionPair], list[str]]:
    """Align two extraction outputs without relying on prompt-specific case IDs."""
    cases_a_by_key, duplicates_a = _unique_case_map(cases_a)
    cases_b_by_key, duplicates_b = _unique_case_map(cases_b)
    missed_a_by_key, missed_duplicates_a = _unique_case_map(missed_cases_a)
    missed_b_by_key, missed_duplicates_b = _unique_case_map(missed_cases_b)
    duplicate_keys = sorted(
        set(duplicates_a + duplicates_b + missed_duplicates_a + missed_duplicates_b)
    )
    all_keys = sorted(
        (
            set(cases_a_by_key)
            | set(cases_b_by_key)
            | set(missed_a_by_key)
            | set(missed_b_by_key)
        )
        - set(duplicate_keys)
    )
    return [
        ExtractionPair(
            source_key=key,
            case_a=cases_a_by_key.get(key),
            case_b=cases_b_by_key.get(key),
            missed_a=missed_a_by_key.get(key),
            missed_b=missed_b_by_key.get(key),
        )
        for key in all_keys
    ], duplicate_keys


def _missed_kind(case: Case | None) -> str:
    if case is None:
        return "missing"
    metadata = case.metadata if isinstance(case.metadata, dict) else {}
    call_status = str(metadata.get("call_status") or "").strip().lower()
    parse_status = str(metadata.get("parse_status") or "").strip().lower()
    if call_status in {"failed", "stopped"}:
        return "infrastructure_failure"
    if call_status == "success" and parse_status in {"empty", "unknown", "not_attempted", ""}:
        return "quality_miss"
    return "excluded_or_unknown"


def deterministic_pairwise_result(pair: ExtractionPair) -> dict[str, Any] | None:
    """Resolve comparisons that do not need an LLM call."""
    if pair.case_a is not None and pair.case_b is not None:
        if _normalized_output(pair.case_a) == _normalized_output(pair.case_b):
            return {
                "source_key": pair.source_key,
                "status": "deterministic",
                "model": "rule",
                "winner": "TIE",
                "confidence": "high",
                "reason": "A/B 提取正文相同，无需调用对比模型。",
                "comparison_kind": "identical_output",
                "rule_refs": [],
                "evidence_refs": [],
                "issues_a": [],
                "issues_b": [],
                "error_tags_a": [],
                "error_tags_b": [],
                "strengths_a": [],
                "strengths_b": [],
                "error": "",
            }
        return None

    kind_a = _missed_kind(pair.missed_a) if pair.case_a is None else "ready"
    kind_b = _missed_kind(pair.missed_b) if pair.case_b is None else "ready"
    if "infrastructure_failure" in {kind_a, kind_b}:
        failed_sides = [label for label, kind in (("A", kind_a), ("B", kind_b)) if kind == "infrastructure_failure"]
        return {
            "source_key": pair.source_key,
            "status": "infrastructure_failure",
            "model": "rule",
            "winner": "INSUFFICIENT",
            "confidence": "low",
            "reason": f"{'、'.join(failed_sides)} 侧提取接口失败，本条不计入版本胜负。",
            "comparison_kind": "runtime_failure",
            "rule_refs": [],
            "evidence_refs": [],
            "issues_a": [],
            "issues_b": [],
            "error_tags_a": [],
            "error_tags_b": [],
            "strengths_a": [],
            "strengths_b": [],
            "error": "提取接口失败",
        }
    if (pair.case_a is not None and kind_b == "quality_miss") or (
        pair.case_b is not None and kind_a == "quality_miss"
    ):
        # 单边空输出既可能是真实漏抽，也可能来自双方准入策略不同，不能按覆盖率直接定胜负。
        return None
    if kind_a == "quality_miss" and kind_b == "quality_miss":
        return {
            "source_key": pair.source_key,
            "status": "deterministic",
            "model": "rule",
            "winner": "TIE",
            "confidence": "high",
            "reason": "A/B 均调用成功但未生成可用正文，按共同漏抽处理。",
            "comparison_kind": "both_missed",
            "rule_refs": [],
            "evidence_refs": [],
            "issues_a": ["调用成功但未生成可用正文"],
            "issues_b": ["调用成功但未生成可用正文"],
            "error_tags_a": ["missing_key_info"],
            "error_tags_b": ["missing_key_info"],
            "strengths_a": [],
            "strengths_b": [],
            "error": "",
        }
    return {
        "source_key": pair.source_key,
        "status": "source_mismatch",
        "model": "rule",
        "winner": "INSUFFICIENT",
        "confidence": "low",
        "reason": "A/B 源 chunk 无法完整对齐，本条不进入胜负统计。",
        "comparison_kind": "source_mismatch",
        "rule_refs": [],
        "evidence_refs": [],
        "issues_a": [],
        "issues_b": [],
        "error_tags_a": [],
        "error_tags_b": [],
        "strengths_a": [],
        "strengths_b": [],
        "error": "源数据未对齐",
    }


def _percentile(sorted_values: list[float], probability: float) -> float | None:
    if not sorted_values:
        return None
    probability = min(1.0, max(0.0, probability))
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(len(sorted_values) - 1, lower + 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _pairwise_cluster_interval(
    outcomes_by_cluster: dict[str, list[float]],
    *,
    confidence_level: float,
    samples: int,
    seed_material: str,
) -> tuple[float | None, float | None, float | None]:
    cluster_means = [mean(values) for _key, values in sorted(outcomes_by_cluster.items()) if values]
    if not cluster_means:
        return None, None, None
    point = mean(cluster_means)
    if samples <= 0:
        return point, None, None
    seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    simulated = sorted(
        mean(rng.choice(cluster_means) for _ in cluster_means)
        for _ in range(max(100, int(samples)))
    )
    alpha = 1.0 - min(0.999, max(0.5, float(confidence_level)))
    return point, _percentile(simulated, alpha / 2.0), _percentile(simulated, 1.0 - alpha / 2.0)


def _direct_quality(cases: list[Case], missed_cases: list[Case]) -> dict[str, Any]:
    quality = compute_run_quality([], cases=cases, missed_cases=missed_cases)
    return {
        key: quality[key]
        for key in (
            "ready_cases",
            "extraction_quality_failures",
            "extraction_infrastructure_failures",
            "excluded_or_unknown_chunks",
            "extraction_coverage",
            "infrastructure_success_rate",
            "unresolved_execution_failures",
            "run_complete",
            "status",
        )
    }


def compare_extraction_prompt_pairs(
    *,
    cases_a: list[Case],
    cases_b: list[Case],
    missed_cases_a: list[Case],
    missed_cases_b: list[Case],
    pairwise_results: list[dict[str, Any]],
    prompt_a: str,
    prompt_b: str,
    validation_config: ValidationGateConfig | None = None,
    evaluation_protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate direct A/B decisions without producing separate absolute scores."""
    gate_config = validation_config or ValidationGateConfig()
    pairs, duplicate_keys = build_extraction_pairs(
        cases_a=cases_a,
        cases_b=cases_b,
        missed_cases_a=missed_cases_a,
        missed_cases_b=missed_cases_b,
    )
    result_by_key = {
        str(item.get("source_key") or ""): item
        for item in pairwise_results
        if str(item.get("source_key") or "")
    }
    quality_a = _direct_quality(cases_a, missed_cases_a)
    quality_b = _direct_quality(cases_b, missed_cases_b)
    winner_counts = {
        "A较优": 0,
        "B较优": 0,
        "基本持平": 0,
        "输出相同": 0,
        "双方均漏抽": 0,
        "策略差异": 0,
        "不可比较": 0,
    }
    rows: list[dict[str, Any]] = []
    outcomes_by_cluster: dict[str, list[float]] = {}
    outcome_keys: list[str] = []
    comparison_failures = 0
    infrastructure_failures = 0
    source_mismatches = 0
    insufficient_comparisons = 0

    for pair in pairs:
        result = result_by_key.get(pair.source_key) or deterministic_pairwise_result(pair) or {
            "source_key": pair.source_key,
            "status": "failed",
            "winner": "INSUFFICIENT",
            "confidence": "low",
            "reason": "缺少成对比较结果。",
            "error": "缺少成对比较结果",
        }
        winner = str(result.get("winner") or "INSUFFICIENT").upper()
        kind = str(result.get("comparison_kind") or "model")
        if winner == "A":
            comparison = "A较优"
            outcome = -1.0
        elif winner == "B":
            comparison = "B较优"
            outcome = 1.0
        elif winner == "TIE":
            comparison = (
                "输出相同"
                if kind == "identical_output"
                else "双方均漏抽" if kind == "both_missed" else "基本持平"
            )
            outcome = 0.0
        elif winner == "POLICY_DIFFERENCE":
            comparison = "策略差异"
            outcome = None
        else:
            comparison = "不可比较"
            outcome = None
        winner_counts[comparison] = winner_counts.get(comparison, 0) + 1

        status = str(result.get("status") or "failed")
        if status == "failed":
            comparison_failures += 1
        if status == "infrastructure_failure":
            infrastructure_failures += 1
        if status == "source_mismatch":
            source_mismatches += 1
        if winner == "INSUFFICIENT" and status in {"success", "mock"}:
            insufficient_comparisons += 1

        source = pair.case_a or pair.case_b or pair.missed_a or pair.missed_b
        metadata = source.metadata if source and isinstance(source.metadata, dict) else {}
        if outcome is not None and status in {"success", "mock", "deterministic"}:
            reviewer = str(metadata.get("reviewer") or "").strip()
            session = str(metadata.get("source_session_id") or getattr(source, "session_id", "") or "")
            cluster = f"reviewer:{reviewer}" if reviewer else f"session:{session}"
            outcomes_by_cluster.setdefault(cluster, []).append(outcome)
            outcome_keys.append(pair.source_key)

        rows.append(
            {
                "source_key": pair.source_key,
                "reviewer": metadata.get("reviewer", ""),
                "session_id": metadata.get("source_session_id", getattr(source, "session_id", "")),
                "chunk_index": metadata.get("chunk_index_in_session", ""),
                "chunk_id": (
                    int(metadata.get("chunk_index_in_session")) + 1
                    if str(metadata.get("chunk_index_in_session", "")).lstrip("-").isdigit()
                    else metadata.get("chunk_index_in_session", "")
                ),
                "row_start": metadata.get("row_start", ""),
                "row_end": metadata.get("row_end", ""),
                "case_id_a": pair.case_a.case_id if pair.case_a else "",
                "case_id_b": pair.case_b.case_id if pair.case_b else "",
                "extraction_a": "可比较" if pair.case_a else "漏抽/不可用",
                "extraction_b": "可比较" if pair.case_b else "漏抽/不可用",
                "pairwise_status": status,
                "pairwise_model": result.get("model", ""),
                "pairwise_confidence": result.get("confidence", ""),
                "decision_basis": result.get("decision_basis", ""),
                "comparison": comparison,
                "comparison_note": result.get("reason", ""),
                "rule_refs": "；".join(result.get("rule_refs") or []),
                "policy_differences": "；".join(result.get("policy_differences") or []),
                "evidence_refs": "；".join(result.get("evidence_refs") or []),
                "issues_a": "；".join(result.get("issues_a") or []),
                "issues_b": "；".join(result.get("issues_b") or []),
                "error_tags_a": _tag_text(result.get("error_tags_a") or []),
                "error_tags_b": _tag_text(result.get("error_tags_b") or []),
                "strengths_a": "；".join(result.get("strengths_a") or []),
                "strengths_b": "；".join(result.get("strengths_b") or []),
                "comparison_error": result.get("error", ""),
                "old_memory_a": pair.case_a.old_memory if pair.case_a else "",
                "old_memory_b": pair.case_b.old_memory if pair.case_b else "",
                "candidate_output_a": pair.case_a.candidate_output if pair.case_a else "",
                "candidate_output_b": pair.case_b.candidate_output if pair.case_b else "",
            }
        )

    eligible_count = sum(len(values) for values in outcomes_by_cluster.values())
    b_wins = winner_counts.get("B较优", 0)
    a_wins = winner_counts.get("A较优", 0)
    ties = (
        winner_counts.get("基本持平", 0)
        + winner_counts.get("输出相同", 0)
        + winner_counts.get("双方均漏抽", 0)
    )
    cluster_point, confidence_lower, confidence_upper = _pairwise_cluster_interval(
        outcomes_by_cluster,
        confidence_level=gate_config.confidence_level,
        samples=gate_config.bootstrap_samples,
        seed_material="|".join(sorted(outcome_keys)),
    )
    preference_delta = (
        sum(sum(values) for values in outcomes_by_cluster.values()) / eligible_count
        if eligible_count
        else 0.0
    )
    regression_rate = a_wins / eligible_count if eligible_count else 0.0
    coverage_drop = float(quality_a["extraction_coverage"]) - float(quality_b["extraction_coverage"])
    prompt_growth_ratio = (len(prompt_b) - len(prompt_a)) / max(1, len(prompt_a))
    confidence_ready = (
        eligible_count >= int(gate_config.min_paired_cases)
        and len(outcomes_by_cluster) >= int(gate_config.min_paired_clusters)
    )
    reasons: list[str] = []
    if duplicate_keys:
        reasons.append("来源键存在重复，重复 chunk 已排除。")
    if quality_a["unresolved_execution_failures"] or quality_b["unresolved_execution_failures"]:
        reasons.append("存在提取接口失败，失败 chunk 未按质量问题计入胜负。")
    if comparison_failures:
        reasons.append(f"有 {comparison_failures} 个差异 chunk 对比调用或 JSON 解析失败。")
    if infrastructure_failures:
        reasons.append(f"有 {infrastructure_failures} 个 chunk 因提取运行失败不可比较。")
    if source_mismatches:
        reasons.append(f"有 {source_mismatches} 个 chunk 的 A/B 源数据无法对齐。")
    if not eligible_count:
        reasons.append("没有可进入胜负统计的同源 chunk。")
    if gate_config.require_statistical_confidence:
        if not confidence_ready:
            reasons.append(
                f"统计证据不足：需要至少 {gate_config.min_paired_cases} 个可比较 chunk、"
                f"{gate_config.min_paired_clusters} 个独立评测人/时序簇，当前为 "
                f"{eligible_count} 个、{len(outcomes_by_cluster)} 个。"
            )
    protocol = evaluation_protocol if isinstance(evaluation_protocol, dict) else {}
    prompt_quality_a = protocol.get("prompt_quality_a") if isinstance(protocol.get("prompt_quality_a"), dict) else {}
    prompt_quality_b = protocol.get("prompt_quality_b") if isinstance(protocol.get("prompt_quality_b"), dict) else {}
    try:
        prompt_quality_delta = float(prompt_quality_b.get("overall")) - float(prompt_quality_a.get("overall"))
    except (TypeError, ValueError):
        prompt_quality_delta = None

    operational_incomplete = bool(
        duplicate_keys or comparison_failures or infrastructure_failures or source_mismatches
    )
    statistics_incomplete = bool(gate_config.require_statistical_confidence and not confidence_ready)
    min_delta = max(1e-9, float(gate_config.min_score_delta))
    lower_bound = max(0.0, float(gate_config.min_confidence_lower_bound))
    b_statistically_better = bool(
        eligible_count
        and preference_delta >= min_delta
        and (
            not gate_config.require_statistical_confidence
            or (confidence_lower is not None and confidence_lower > lower_bound)
        )
    )
    a_statistically_better = bool(
        eligible_count
        and preference_delta <= -min_delta
        and (
            not gate_config.require_statistical_confidence
            or (confidence_upper is not None and confidence_upper < -lower_bound)
        )
    )

    if operational_incomplete or statistics_incomplete or not eligible_count:
        recommendation = "暂不定版"
        recommendation_reason = "比较数据或统计证据尚不完整，失败和策略差异样本均未强行计入胜负。"
    elif b_statistically_better:
        recommendation = "建议选择 B"
        recommendation_reason = "B 在候选无关的共同质量比较中显著更优。"
    elif a_statistically_better:
        recommendation = "建议选择 A"
        recommendation_reason = "A 在候选无关的共同质量比较中显著更优。"
    elif prompt_quality_delta is not None and abs(prompt_quality_delta) >= 0.5:
        recommendation = "建议选择 B" if prompt_quality_delta > 0 else "建议选择 A"
        recommendation_reason = (
            "两版输出效果未形成显著差异；"
            f"{'B' if prompt_quality_delta > 0 else 'A'} 的提示词设计质量更高，作为次级依据推荐。"
        )
    else:
        recommendation = "暂不定版"
        recommendation_reason = "共同质量效果和提示词设计质量均未形成足以定版的差异。"

    accepted = recommendation == "建议选择 B"

    return {
        "comparison_mode": "candidate_neutral_pairwise_v2",
        "recommendation": recommendation,
        "recommendation_reason": recommendation_reason,
        "quality_a": quality_a,
        "quality_b": quality_b,
        "validation_gate": {
            "accepted": accepted,
            "decision": (
                "B_better" if recommendation == "建议选择 B"
                else "A_better" if recommendation == "建议选择 A"
                else "inconclusive"
            ),
            "reasons": reasons,
            "config": asdict(gate_config),
            "paired_case_count": eligible_count,
            "paired_cluster_count": len(outcomes_by_cluster),
            "paired_preference_delta": round(preference_delta, 4),
            "paired_score_delta": round(preference_delta, 4),
            "b_win_rate": round(b_wins / eligible_count, 4) if eligible_count else 0.0,
            "a_win_rate": round(a_wins / eligible_count, 4) if eligible_count else 0.0,
            "tie_rate": round(ties / eligible_count, 4) if eligible_count else 0.0,
            "confidence_ready": confidence_ready,
            "confidence_level": gate_config.confidence_level,
            "cluster_mean_delta": round(cluster_point, 4) if cluster_point is not None else None,
            "confidence_interval": {
                "lower": round(confidence_lower, 4) if confidence_lower is not None else None,
                "upper": round(confidence_upper, 4) if confidence_upper is not None else None,
            },
            "extraction_coverage_drop": round(coverage_drop, 4),
            "case_regression_rate": round(regression_rate, 4),
            "prompt_growth_ratio": round(prompt_growth_ratio, 4),
            "prompt_quality_delta_b_minus_a": (
                round(prompt_quality_delta, 4) if prompt_quality_delta is not None else None
            ),
            "comparison_failures": comparison_failures,
            "infrastructure_failures": infrastructure_failures,
            "source_mismatches": source_mismatches,
            "insufficient_comparisons": insufficient_comparisons,
        },
        "winner_counts": winner_counts,
        "evaluation_protocol": protocol,
        "prompt_quality": {"A": prompt_quality_a, "B": prompt_quality_b},
        "dimension_summary": [],
        "identical_output_count": winner_counts.get("输出相同", 0),
        "duplicate_source_keys": duplicate_keys,
        "rows": rows,
    }
