import json
from collections import Counter
from ..schema import EvalResult, SCORING_DIMENSIONS

DIM_LABELS = {
    "correctness": "正确性",
    "coverage": "完整性",
    "update_logic": "更新合理性",
    "memory_boundary": "记忆边界",
    "conciseness": "去重凝练",
    "format": "格式合规",
}

TAG_LABELS = {
    "hallucination": "幻觉新增",
    "wrong_fact": "事实错误",
    "missing_key_info": "遗漏关键信息",
    "over_memory": "过度记忆",
    "short_term_pollution": "短期信息污染",
    "conflict_not_resolved": "冲突未解决",
    "duplicate_memory": "重复记忆",
    "verbose_or_noisy": "冗长噪声",
    "format_error": "格式错误",
    "privacy_sensitive": "敏感信息",
    "unclear_update": "更新意图不清",
}


def compute_aggregations(results: list[EvalResult]) -> dict:
    if not results:
        return {"total_cases": 0}

    total = len(results)
    fatal_count = sum(1 for r in results if r.fatal_error)
    avg_total = round(sum(r.score_total for r in results) / total, 2)

    dim_scores: dict[str, list[float]] = {}
    for r in results:
        for dim, score in r.scores.items():
            dim_scores.setdefault(dim, []).append(score)
    dim_avgs = {dim: round(sum(vals) / len(vals), 2) for dim, vals in dim_scores.items()}

    tag_counter = Counter()
    for r in results:
        for tag in r.error_tags:
            tag_counter[tag] += 1

    return {
        "total_cases": total,
        "fatal_errors": fatal_count,
        "fatal_rate": round(fatal_count / total, 3) if total > 0 else 0.0,
        "avg_score_total": avg_total,
        "avg_dimension_scores": dim_avgs,
        "error_tags": tag_counter.most_common(),
    }


def group_by(results: list[EvalResult], field: str) -> dict[str, list[EvalResult]]:
    groups: dict[str, list[EvalResult]] = {}
    for r in results:
        key = str(getattr(r, field, "unknown"))
        groups.setdefault(key, []).append(r)
    return dict(sorted(groups.items()))


def print_summary(stats: dict, title: str = "USER.md 更新评测统计") -> None:
    header = "=" * 60

    def _dim_label(key: str) -> str:
        return DIM_LABELS.get(key, key)

    def _tag_label(key: str) -> str:
        return TAG_LABELS.get(key, key)

    print()
    print(header)
    print(f"  {title}")
    print(header)

    total = stats.get("total_cases", 0)
    if total == 0:
        print("  （无数据）")
        print(header)
        return

    fatal = stats.get("fatal_errors", 0)
    fatal_rate = stats.get("fatal_rate", 0) * 100

    print(f"  Case 总数          : {total}")
    print(f"  严重错误 (fatal)   : {fatal} ({fatal_rate:.1f}%)")
    print(f"  加权总分           : {stats.get('avg_score_total', 0):.2f} / 5")
    print()
    print(f"  {'─' * 56}")

    print("  【各维度平均得分】")
    dim_avgs = stats.get("avg_dimension_scores", {})
    bar_max = 35
    for dim, score in dim_avgs.items():
        label = _dim_label(dim)
        bar_len = int(score / 5 * bar_max)
        bar = "█" * bar_len + "░" * (bar_max - bar_len)
        print(f"    {label:<10} {score:.1f}  {bar}")

    print(f"  {'─' * 56}")
    print("  【错误标签分布】")
    tags = stats.get("error_tags", [])
    if tags:
        for tag, count in tags[:10]:
            label = _tag_label(tag)
            print(f"    {label:<12} ({tag:<22}) : {count} 次")
    else:
        print("    （无错误标签）")
    print(header)
    print()

def summarize_by_field(results: list[EvalResult], field: str) -> list[dict]:
    groups = group_by(results, field)
    rows = []
    for key, items in groups.items():
        stats = compute_aggregations(items)
        rows.append({
            field: key,
            "total_cases": stats.get("total_cases", 0),
            "avg_score_total": stats.get("avg_score_total", 0),
            "fatal_errors": stats.get("fatal_errors", 0),
            "fatal_rate": stats.get("fatal_rate", 0),
        })
    return rows

def flatten_results(results: list[EvalResult]) -> list[dict]:
    rows = []
    for r in results:
        row = {
            "case_id": r.case_id,
            "task_type": r.task_type,
            "model_name": r.model_name,
            "prompt_version": r.prompt_version,
            "score_total": r.score_total,
            "fatal_error": r.fatal_error,
            "comment": r.comment,
            "error_tags": ",".join(r.error_tags),
            "judge_model": r.judge_model,
            "judge_prompt_version": r.judge_prompt_version,
            "extraction_prompt_version": r.extraction_prompt_version,
            "extraction_prompt_hash": r.extraction_prompt_hash,
            "diagnostics_count": len(r.diagnostics or []),
            "rule_refs": "; ".join(r.rule_refs or []),
            "evidence_refs": "; ".join(r.evidence_refs or []),
            "output_refs": "; ".join(r.output_refs or []),
            "diagnostics": json.dumps(r.diagnostics or [], ensure_ascii=False),
            "timestamp": r.timestamp,
        }
        for dim, score in r.scores.items():
            row[f"score_{dim}"] = score
        rows.append(row)
    return rows
