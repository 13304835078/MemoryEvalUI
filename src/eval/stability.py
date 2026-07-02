from __future__ import annotations

import json
import math
from collections import Counter
from typing import Callable

from ..schema import EvalResult
from .metrics import DIM_LABELS, TAG_LABELS


ResultKey = tuple[str, ...]


def result_key(result: EvalResult, mode: str = "case_id") -> ResultKey:
    if mode == "case_model_prompt":
        return (
            result.case_id,
            result.model_name or "unknown",
            result.prompt_version or "unknown",
        )
    return (result.case_id,)


def results_from_jsonl_text(text: str) -> list[EvalResult]:
    results: list[EvalResult] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        results.append(EvalResult.from_dict(json.loads(line)))
    return results


def _latest_by_key(results: list[EvalResult], key_func: Callable[[EvalResult], ResultKey]) -> dict[ResultKey, EvalResult]:
    keyed: dict[ResultKey, EvalResult] = {}
    for result in results:
        keyed[key_func(result)] = result
    return keyed


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _score_values(results: list[EvalResult], field: str) -> list[float]:
    values: list[float] = []
    for result in results:
        if field == "score_total":
            values.append(float(result.score_total or 0.0))
        elif field in (result.scores or {}):
            values.append(float(result.scores.get(field, 0.0)))
    return values


def _distribution(values: list[float], bucket_size: float = 0.5) -> list[float]:
    bucket_count = int(5 / bucket_size) + 1
    counts = [0] * bucket_count
    for value in values:
        clipped = min(5.0, max(0.0, float(value)))
        index = int(round(clipped / bucket_size))
        index = max(0, min(bucket_count - 1, index))
        counts[index] += 1
    total = sum(counts)
    if total == 0:
        return [0.0] * bucket_count
    return [count / total for count in counts]


def kl_divergence(p: list[float], q: list[float], eps: float = 1e-9) -> float:
    return sum(pi * math.log((pi + eps) / (qi + eps)) for pi, qi in zip(p, q) if pi > 0)


def js_divergence(p: list[float], q: list[float], eps: float = 1e-9) -> float:
    m = [(pi + qi) / 2 for pi, qi in zip(p, q)]
    return (kl_divergence(p, m, eps) + kl_divergence(q, m, eps)) / 2


def _set_similarity(left: list[str], right: list[str]) -> float:
    a = set(left or [])
    b = set(right or [])
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _format_tags(tags: list[str]) -> str:
    return "; ".join(TAG_LABELS.get(tag, tag) for tag in tags or [])


def _format_bool(value: bool) -> str:
    return "是" if value else "否"


def _normalize_text(value: str | None) -> str:
    return " ".join(str(value or "").split())


def _diagnostic_signatures(diagnostics: list[dict]) -> list[str]:
    signatures: list[str] = []
    for item in diagnostics or []:
        if not isinstance(item, dict):
            continue
        signatures.append("|".join([
            _normalize_text(item.get("dimension")),
            _normalize_text(item.get("severity")),
            ";".join(item.get("rule_refs") or []),
            ";".join(item.get("evidence_refs") or []),
            ";".join(item.get("output_refs") or []),
            _normalize_text(item.get("reason")),
        ]))
    return signatures


def compare_eval_stability(
    current_results: list[EvalResult],
    baseline_results: list[EvalResult],
    key_mode: str = "case_id",
    exact_score_tolerance: float = 0.01,
) -> dict:
    key_func = lambda result: result_key(result, key_mode)
    current_by_key = _latest_by_key(current_results, key_func)
    baseline_by_key = _latest_by_key(baseline_results, key_func)

    common_keys = sorted(set(current_by_key) & set(baseline_by_key))
    current_only = sorted(set(current_by_key) - set(baseline_by_key))
    baseline_only = sorted(set(baseline_by_key) - set(current_by_key))

    paired_current = [current_by_key[key] for key in common_keys]
    paired_baseline = [baseline_by_key[key] for key in common_keys]

    diff_rows: list[dict] = []
    total_abs_deltas: list[float] = []
    total_exact = 0
    tag_exact = 0
    tag_jaccards: list[float] = []
    diag_abs_deltas: list[float] = []
    ref_jaccards: dict[str, list[float]] = {"rule_refs": [], "evidence_refs": [], "output_refs": []}
    added_tags = Counter()
    removed_tags = Counter()
    instability_type_counter = Counter()

    dim_names = sorted({
        dim
        for result in paired_current + paired_baseline
        for dim in (result.scores or {}).keys()
    })
    dim_deltas: dict[str, list[float]] = {dim: [] for dim in dim_names}
    dim_exact: dict[str, int] = {dim: 0 for dim in dim_names}

    for key in common_keys:
        current = current_by_key[key]
        baseline = baseline_by_key[key]
        total_delta = float(current.score_total or 0.0) - float(baseline.score_total or 0.0)
        total_abs = abs(total_delta)
        total_abs_deltas.append(total_abs)
        if total_abs <= exact_score_tolerance:
            total_exact += 1

        baseline_tags = set(baseline.error_tags or [])
        current_tags = set(current.error_tags or [])
        if baseline_tags == current_tags:
            tag_exact += 1
        tag_jaccards.append(_set_similarity(list(current_tags), list(baseline_tags)))
        added_tags.update(current_tags - baseline_tags)
        removed_tags.update(baseline_tags - current_tags)

        diag_delta = len(current.diagnostics or []) - len(baseline.diagnostics or [])
        diag_abs_deltas.append(abs(diag_delta))
        diagnostic_jaccard = _set_similarity(
            _diagnostic_signatures(current.diagnostics or []),
            _diagnostic_signatures(baseline.diagnostics or []),
        )

        for ref_field in ref_jaccards:
            ref_jaccards[ref_field].append(
                _set_similarity(getattr(current, ref_field) or [], getattr(baseline, ref_field) or [])
            )

        changed_dims = 0
        for dim in dim_names:
            current_score = float((current.scores or {}).get(dim, 0.0))
            baseline_score = float((baseline.scores or {}).get(dim, 0.0))
            delta = current_score - baseline_score
            abs_delta = abs(delta)
            dim_deltas[dim].append(abs_delta)
            if abs_delta <= exact_score_tolerance:
                dim_exact[dim] += 1
            else:
                changed_dims += 1

        score_changed = total_abs > exact_score_tolerance
        dimension_score_changed = changed_dims > 0
        tag_changed = baseline_tags != current_tags
        diagnostic_changed = diag_delta != 0 or diagnostic_jaccard < 1.0
        rule_refs_changed = ref_jaccards["rule_refs"][-1] < 1.0
        evidence_refs_changed = ref_jaccards["evidence_refs"][-1] < 1.0
        output_refs_changed = ref_jaccards["output_refs"][-1] < 1.0
        comment_changed = _normalize_text(current.comment) != _normalize_text(baseline.comment)

        instability_types = []
        if score_changed:
            instability_types.append("总分变化")
        if dimension_score_changed:
            instability_types.append("维度分变化")
        if tag_changed:
            instability_types.append("错误标签变化")
        if diagnostic_changed:
            instability_types.append("诊断变化")
        if rule_refs_changed:
            instability_types.append("规则引用变化")
        if evidence_refs_changed:
            instability_types.append("证据引用变化")
        if output_refs_changed:
            instability_types.append("输出引用变化")
        if comment_changed:
            instability_types.append("评语变化")
        if not instability_types:
            instability_types.append("完全一致")
        instability_type_counter.update(t for t in instability_types if t != "完全一致")

        priority = (
            int(score_changed) * 100
            + int(dimension_score_changed) * 80
            + int(tag_changed) * 60
            + int(diagnostic_changed) * 40
            + int(rule_refs_changed or evidence_refs_changed or output_refs_changed) * 20
            + int(comment_changed) * 10
        )

        diff_rows.append({
            "_sort_priority": priority,
            "匹配键": " | ".join(key),
            "case_id": current.case_id,
            "不稳定类型": "；".join(instability_types),
            "总分变化": _format_bool(score_changed),
            "维度分变化": _format_bool(dimension_score_changed),
            "错误标签变化": _format_bool(tag_changed),
            "诊断变化": _format_bool(diagnostic_changed),
            "规则引用变化": _format_bool(rule_refs_changed),
            "证据引用变化": _format_bool(evidence_refs_changed),
            "输出引用变化": _format_bool(output_refs_changed),
            "评语变化": _format_bool(comment_changed),
            "当前总分": round(float(current.score_total or 0.0), 4),
            "对照总分": round(float(baseline.score_total or 0.0), 4),
            "总分差值": round(total_delta, 4),
            "总分绝对差": round(total_abs, 4),
            "变化维度数": changed_dims,
            "错误标签一致": baseline_tags == current_tags,
            "错误标签Jaccard": round(tag_jaccards[-1], 4),
            "当前错误标签": _format_tags(current.error_tags or []),
            "对照错误标签": _format_tags(baseline.error_tags or []),
            "当前诊断数": len(current.diagnostics or []),
            "对照诊断数": len(baseline.diagnostics or []),
            "诊断数差值": diag_delta,
            "诊断Jaccard": round(diagnostic_jaccard, 4),
            "规则引用Jaccard": round(ref_jaccards["rule_refs"][-1], 4),
            "证据引用Jaccard": round(ref_jaccards["evidence_refs"][-1], 4),
            "输出引用Jaccard": round(ref_jaccards["output_refs"][-1], 4),
            "当前评语": current.comment,
            "对照评语": baseline.comment,
        })

    common_count = len(common_keys)
    summary = {
        "current_total": len(current_results),
        "baseline_total": len(baseline_results),
        "common_count": common_count,
        "current_only_count": len(current_only),
        "baseline_only_count": len(baseline_only),
        "current_avg_score": round(_mean([float(r.score_total or 0.0) for r in paired_current]), 4),
        "baseline_avg_score": round(_mean([float(r.score_total or 0.0) for r in paired_baseline]), 4),
        "avg_total_abs_delta": round(_mean(total_abs_deltas), 4),
        "max_total_abs_delta": round(max(total_abs_deltas) if total_abs_deltas else 0.0, 4),
        "total_score_exact_rate": round(total_exact / common_count, 4) if common_count else 0.0,
        "tag_exact_rate": round(tag_exact / common_count, 4) if common_count else 0.0,
        "avg_tag_jaccard": round(_mean(tag_jaccards), 4),
        "avg_diagnostics_count_abs_delta": round(_mean(diag_abs_deltas), 4),
        "avg_rule_refs_jaccard": round(_mean(ref_jaccards["rule_refs"]), 4),
        "avg_evidence_refs_jaccard": round(_mean(ref_jaccards["evidence_refs"]), 4),
        "avg_output_refs_jaccard": round(_mean(ref_jaccards["output_refs"]), 4),
    }
    summary["unstable_case_count"] = sum(1 for row in diff_rows if row["不稳定类型"] != "完全一致")
    summary["stable_case_count"] = common_count - summary["unstable_case_count"]
    summary["unstable_case_rate"] = round(summary["unstable_case_count"] / common_count, 4) if common_count else 0.0

    distribution_rows = []
    for field in ["score_total"] + dim_names:
        current_values = _score_values(paired_current, field)
        baseline_values = _score_values(paired_baseline, field)
        current_dist = _distribution(current_values)
        baseline_dist = _distribution(baseline_values)
        distribution_rows.append({
            "字段": "总分" if field == "score_total" else DIM_LABELS.get(field, field),
            "KL(当前||对照)": round(kl_divergence(current_dist, baseline_dist), 6),
            "KL(对照||当前)": round(kl_divergence(baseline_dist, current_dist), 6),
            "JS散度": round(js_divergence(current_dist, baseline_dist), 6),
            "当前均值": round(_mean(current_values), 4),
            "对照均值": round(_mean(baseline_values), 4),
        })

    dimension_rows = []
    for dim in dim_names:
        values = dim_deltas.get(dim, [])
        dimension_rows.append({
            "维度": DIM_LABELS.get(dim, dim),
            "平均绝对差": round(_mean(values), 4),
            "最大绝对差": round(max(values) if values else 0.0, 4),
            "完全一致率": round(dim_exact.get(dim, 0) / common_count, 4) if common_count else 0.0,
        })

    tag_rows = []
    for tag in sorted(set(added_tags) | set(removed_tags)):
        tag_rows.append({
            "错误标签": TAG_LABELS.get(tag, tag),
            "当前新增次数": added_tags.get(tag, 0),
            "对照独有次数": removed_tags.get(tag, 0),
        })

    instability_type_rows = [
        {"不稳定类型": type_name, "样本数": count, "占共同样本比例": round(count / common_count, 4) if common_count else 0.0}
        for type_name, count in instability_type_counter.most_common()
    ]

    diff_rows.sort(
        key=lambda row: (
            row["_sort_priority"],
            row["总分绝对差"],
            1 - row["错误标签Jaccard"],
            row["变化维度数"],
        ),
        reverse=True,
    )
    for row in diff_rows:
        row.pop("_sort_priority", None)

    return {
        "summary": summary,
        "distribution_rows": distribution_rows,
        "dimension_rows": dimension_rows,
        "tag_rows": tag_rows,
        "instability_type_rows": instability_type_rows,
        "diff_rows": diff_rows,
        "current_only_keys": [" | ".join(key) for key in current_only],
        "baseline_only_keys": [" | ".join(key) for key in baseline_only],
    }
