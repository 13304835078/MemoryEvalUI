from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

from src.eval.result_status import result_is_score_eligible
from src.eval.run_quality import compute_run_quality


@dataclass(frozen=True)
class ValidationGateConfig:
    min_score_delta: float = 0.03
    min_end_to_end_delta: float = 0.0
    max_extraction_coverage_drop: float = 0.005
    max_case_regression_rate: float = 0.1
    score_regression_tolerance: float = 0.05
    max_prompt_growth_ratio: float = 0.1
    min_paired_cases: int = 8
    min_paired_clusters: int = 2
    confidence_level: float = 0.95
    bootstrap_samples: int = 2000
    min_confidence_lower_bound: float = 0.0
    require_statistical_confidence: bool = True
    critical_error_tags: tuple[str, ...] = ("privacy_sensitive", "hallucination", "wrong_fact")


def _case_cluster_map(cases: list[Any] | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for case in cases or []:
        metadata = getattr(case, "metadata", {}) or {}
        reviewer = str(metadata.get("reviewer") or "").strip()
        session = str(
            metadata.get("source_session_id")
            or getattr(case, "session_id", "")
            or "unknown_session"
        ).strip()
        # USER.md can inherit across sessions for one reviewer. Treat the full
        # reviewer history as one statistical cluster when that identity exists.
        cluster = f"reviewer:{reviewer}" if reviewer else f"session:{session}"
        mapping[str(getattr(case, "case_id", ""))] = cluster
    return mapping


def _percentile(sorted_values: list[float], probability: float) -> float | None:
    if not sorted_values:
        return None
    probability = min(1.0, max(0.0, probability))
    index = probability * (len(sorted_values) - 1)
    lower = int(index)
    upper = min(len(sorted_values) - 1, lower + 1)
    fraction = index - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _cluster_bootstrap_interval(
    deltas_by_cluster: dict[str, list[float]],
    *,
    confidence_level: float,
    samples: int,
    seed_material: str,
) -> tuple[float | None, float | None, float | None]:
    cluster_means = [mean(values) for _key, values in sorted(deltas_by_cluster.items()) if values]
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


def evaluate_candidate_gate(
    champion_results: list[Any],
    candidate_results: list[Any],
    *,
    champion_cases: list[Any] | None = None,
    candidate_cases: list[Any] | None = None,
    champion_missed_cases: list[Any] | None = None,
    candidate_missed_cases: list[Any] | None = None,
    champion_prompt: str = "",
    candidate_prompt: str = "",
    config: ValidationGateConfig | None = None,
) -> dict[str, Any]:
    gate = config or ValidationGateConfig()
    champion_quality = compute_run_quality(
        champion_results, cases=champion_cases, missed_cases=champion_missed_cases
    )
    candidate_quality = compute_run_quality(
        candidate_results, cases=candidate_cases, missed_cases=candidate_missed_cases
    )
    reasons: list[str] = []

    if not champion_quality["run_complete"] or not candidate_quality["run_complete"]:
        reasons.append("Validation 存在提取接口或 Judge 运行失败，比较不完整。")

    score_delta = candidate_quality["conditional_avg_score"] - champion_quality["conditional_avg_score"]
    e2e_delta = candidate_quality["end_to_end_score"] - champion_quality["end_to_end_score"]
    coverage_drop = champion_quality["extraction_coverage"] - candidate_quality["extraction_coverage"]
    if e2e_delta < gate.min_end_to_end_delta:
        reasons.append(f"端到端分数变化 {e2e_delta:.4f}，低于门槛 {gate.min_end_to_end_delta:.4f}。")
    if coverage_drop > gate.max_extraction_coverage_drop:
        reasons.append(f"提取覆盖率下降 {coverage_drop:.4f}，超过允许值 {gate.max_extraction_coverage_drop:.4f}。")

    champion_by_case = {item.case_id: item for item in champion_results if result_is_score_eligible(item)}
    candidate_by_case = {item.case_id: item for item in candidate_results if result_is_score_eligible(item)}
    paired_ids = sorted(set(champion_by_case) & set(candidate_by_case))
    paired_deltas = {
        case_id: float(candidate_by_case[case_id].score_total) - float(champion_by_case[case_id].score_total)
        for case_id in paired_ids
    }
    paired_score_delta = mean(paired_deltas.values()) if paired_deltas else 0.0
    if not paired_ids:
        reasons.append("Validation 没有可配对的成功评分样本，不能判断候选是否提升。")
    elif paired_score_delta + 1e-12 < gate.min_score_delta:
        reasons.append(f"配对样本平均分提升 {paired_score_delta:.4f}，低于门槛 {gate.min_score_delta:.4f}。")

    cluster_map = _case_cluster_map(champion_cases or candidate_cases)
    deltas_by_cluster: dict[str, list[float]] = {}
    for case_id, delta in paired_deltas.items():
        cluster = cluster_map.get(case_id, f"case:{case_id}")
        deltas_by_cluster.setdefault(cluster, []).append(delta)
    cluster_point, confidence_lower, confidence_upper = _cluster_bootstrap_interval(
        deltas_by_cluster,
        confidence_level=gate.confidence_level,
        samples=gate.bootstrap_samples,
        seed_material="|".join(paired_ids),
    )
    confidence_ready = (
        len(paired_ids) >= int(gate.min_paired_cases)
        and len(deltas_by_cluster) >= int(gate.min_paired_clusters)
    )
    if gate.require_statistical_confidence:
        if not confidence_ready:
            reasons.append(
                "Validation 统计证据不足："
                f"需要至少 {gate.min_paired_cases} 个配对 case、{gate.min_paired_clusters} 个独立评测人/时序簇，"
                f"当前为 {len(paired_ids)} 个 case、{len(deltas_by_cluster)} 个簇。"
            )
        elif confidence_lower is None or confidence_lower <= gate.min_confidence_lower_bound:
            lower_text = "不可计算" if confidence_lower is None else f"{confidence_lower:.4f}"
            reasons.append(
                f"配对提升的 {gate.confidence_level:.0%} 置信区间下界为 {lower_text}，"
                f"未高于 {gate.min_confidence_lower_bound:.4f}；当前提升可能来自样本波动。"
            )

    regressions = [
        case_id for case_id in paired_ids
        if float(candidate_by_case[case_id].score_total) + gate.score_regression_tolerance
        < float(champion_by_case[case_id].score_total)
    ]
    regression_rate = len(regressions) / len(paired_ids) if paired_ids else 0.0
    if regression_rate > gate.max_case_regression_rate:
        reasons.append(f"单样本退化率 {regression_rate:.2%}，超过允许值 {gate.max_case_regression_rate:.2%}。")

    critical_tags = set(gate.critical_error_tags)
    new_critical: dict[str, list[str]] = {}
    for case_id in paired_ids:
        added = (
            set(candidate_by_case[case_id].error_tags or [])
            - set(champion_by_case[case_id].error_tags or [])
        ) & critical_tags
        if added:
            new_critical[case_id] = sorted(added)
    if new_critical:
        reasons.append(f"候选新增关键错误标签，共 {len(new_critical)} 个样本。")

    base_length = max(1, len(champion_prompt))
    prompt_growth_ratio = (len(candidate_prompt) - len(champion_prompt)) / base_length
    if prompt_growth_ratio > gate.max_prompt_growth_ratio:
        reasons.append(f"提示词增长 {prompt_growth_ratio:.2%}，超过允许值 {gate.max_prompt_growth_ratio:.2%}。")

    return {
        "accepted": not reasons,
        "decision": "accepted" if not reasons else "rejected",
        "reasons": reasons,
        "config": asdict(gate),
        "champion_quality": champion_quality,
        "candidate_quality": candidate_quality,
        "score_delta": round(score_delta, 4),
        "paired_score_delta": round(paired_score_delta, 4),
        "end_to_end_delta": round(e2e_delta, 4),
        "extraction_coverage_drop": round(coverage_drop, 4),
        "paired_case_count": len(paired_ids),
        "paired_cluster_count": len(deltas_by_cluster),
        "confidence_ready": confidence_ready,
        "confidence_level": gate.confidence_level,
        "cluster_mean_delta": round(cluster_point, 4) if cluster_point is not None else None,
        "confidence_interval": {
            "lower": round(confidence_lower, 4) if confidence_lower is not None else None,
            "upper": round(confidence_upper, 4) if confidence_upper is not None else None,
        },
        "regression_case_ids": regressions,
        "case_regression_rate": round(regression_rate, 4),
        "new_critical_errors": new_critical,
        "prompt_growth_ratio": round(prompt_growth_ratio, 4),
    }
