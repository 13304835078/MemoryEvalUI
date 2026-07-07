from __future__ import annotations

import re
from typing import Any


PROGRESS_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")


def clamp_fraction(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def parse_progress_fraction(value: Any) -> float | None:
    if value is None:
        return None
    match = PROGRESS_RE.match(str(value))
    if not match:
        return None
    done = int(match.group(1))
    total = int(match.group(2))
    if total <= 0:
        return 1.0
    return clamp_fraction(done / total)


def round_progress_fraction(round_record: dict[str, Any]) -> float:
    if not round_record:
        return 0.0
    if round_record.get("status") in {"completed", "completed_no_change"}:
        return 1.0
    if round_record.get("candidate_prompt_saved"):
        return 0.98
    if round_record.get("advisor_path"):
        return 0.95
    if "advisor_evidence_count" in round_record:
        return 0.9
    if round_record.get("eval_stats") or round_record.get("results_path"):
        return 0.85

    eval_fraction = parse_progress_fraction(round_record.get("eval_progress"))
    if eval_fraction is not None:
        return clamp_fraction(0.45 + 0.4 * eval_fraction)

    if round_record.get("case_stats") or round_record.get("cases_path"):
        return 0.4
    if round_record.get("extraction_stats") or round_record.get("extraction_output"):
        return 0.35

    extraction_fraction = parse_progress_fraction(round_record.get("extraction_progress"))
    if extraction_fraction is not None:
        return clamp_fraction(0.05 + 0.3 * extraction_fraction)

    return 0.02


def describe_round_step(round_record: dict[str, Any]) -> str:
    if not round_record:
        return "等待开始"
    if round_record.get("status") == "completed_no_change":
        return "无需修改提示词"
    if round_record.get("status") == "completed":
        return "本轮完成"
    if round_record.get("candidate_prompt_saved"):
        return "保存候选提取提示词"
    if round_record.get("advisor_path"):
        return "提示词建议完成"
    if "advisor_evidence_count" in round_record:
        return f"生成提示词建议（证据 {round_record.get('advisor_evidence_count')} 条）"
    if round_record.get("eval_stats") or round_record.get("results_path"):
        return "评测完成，汇总结果"
    if round_record.get("eval_progress"):
        return f"执行评测 {round_record.get('eval_progress')}"
    if round_record.get("case_stats") or round_record.get("cases_path"):
        stats = round_record.get("case_stats") if isinstance(round_record.get("case_stats"), dict) else {}
        generated = stats.get("generated_cases", "")
        missed = stats.get("missed_cases", "")
        suffix = f"（完整 {generated}，漏抽 {missed}）" if generated != "" else ""
        return f"生成评测 case{suffix}"
    if round_record.get("extraction_stats") or round_record.get("extraction_output"):
        return "记忆提取完成，准备生成 case"
    if round_record.get("extraction_progress"):
        return f"记忆提取 {round_record.get('extraction_progress')}"
    return "本轮初始化"


def compute_closed_loop_progress(state: dict[str, Any]) -> dict[str, Any]:
    config = state.get("config") if isinstance(state.get("config"), dict) else {}
    controls = state.get("controls") if isinstance(state.get("controls"), dict) else {}
    configured_rounds = int(controls.get("target_rounds") or config.get("rounds") or 0)
    rounds = state.get("rounds") if isinstance(state.get("rounds"), list) else []
    total_rounds = max(configured_rounds, len(rounds), 1)

    if state.get("status") in {"completed", "completed_no_change"}:
        current_step = "无需修改提示词，闭环结束" if state.get("status") == "completed_no_change" else "全部完成"
        return {
            "overall_fraction": 1.0,
            "current_round_fraction": 1.0,
            "current_round": total_rounds,
            "total_rounds": total_rounds,
            "label": f"整体进度：100.0%（{total_rounds}/{total_rounds} 轮）",
            "current_label": "当前轮次：已完成",
            "current_step": current_step,
            "latest_message": "",
        }

    round_fractions = [round_progress_fraction(item if isinstance(item, dict) else {}) for item in rounds]
    padded = round_fractions + [0.0] * max(0, total_rounds - len(round_fractions))
    overall_fraction = clamp_fraction(sum(padded[:total_rounds]) / total_rounds)

    current_round = 1
    current_fraction = 0.0
    for index, fraction in enumerate(padded[:total_rounds], 1):
        if fraction < 1.0:
            current_round = index
            current_fraction = fraction
            break
    else:
        current_round = total_rounds
        current_fraction = padded[total_rounds - 1] if padded else 0.0

    current_record = rounds[current_round - 1] if 0 <= current_round - 1 < len(rounds) else {}
    if not isinstance(current_record, dict):
        current_record = {}
    current_step = describe_round_step(current_record)
    latest_message = str(current_record.get("latest_message") or "")

    return {
        "overall_fraction": overall_fraction,
        "current_round_fraction": clamp_fraction(current_fraction),
        "current_round": current_round,
        "total_rounds": total_rounds,
        "label": f"整体进度：{overall_fraction * 100:.1f}%（第 {current_round}/{total_rounds} 轮）",
        "current_label": f"第 {current_round}/{total_rounds} 轮：{current_step}，本轮进度 {current_fraction * 100:.1f}%",
        "current_step": current_step,
        "latest_message": latest_message,
    }
