from __future__ import annotations

from dataclasses import replace
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
