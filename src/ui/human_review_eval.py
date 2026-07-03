from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.runtime_paths import APP_HOME, DATA_DIR
from src.schema import Case, DialogueTurn, EvalResult, TaskType


PROJECT_ROOT = APP_HOME
HUMAN_REVIEW_CACHE_PATH = DATA_DIR / "results" / "human_review_judge_cache.json"
HUMAN_REVIEW_RUNS_DIR = DATA_DIR / "results" / "human_review_runs"


@dataclass
class HumanReviewPair:
    pair_id: str
    row_number: int
    round_value: str
    reviewer: str
    query: str
    answer: str
    model1_name: str
    model2_name: str
    model1_output: str
    model2_output: str
    human_gsb: str
    issue_type: str
    remark: str
    case_model1: Case
    case_model2: Case


REQUIRED_COLUMNS = [
    "轮次",
    "query",
    "answer",
    "评测人",
    "GSB",
    "问题类型",
    "备注",
]


def read_human_review_excel(path: str | Path, sheet_name: str | int | None = 0) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name).fillna("")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def validate_human_review_columns(
    df: pd.DataFrame,
    model1_column: str,
    model2_column: str,
) -> list[str]:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    for c in [model1_column, model2_column]:
        if c and c not in df.columns:
            missing.append(c)
    return missing


def make_human_review_pairs(
    df: pd.DataFrame,
    model1_column: str = "user.md-glm5-think",
    model2_column: str = "user.md-ds-10.1.2",
    model1_name: str = "glm5-think",
    model2_name: str = "ds-10.1.2",
    prompt_version: str = "human_review",
) -> list[HumanReviewPair]:
    pairs, _ = make_human_review_pairs_with_stats(
        df,
        model1_column=model1_column,
        model2_column=model2_column,
        model1_name=model1_name,
        model2_name=model2_name,
        prompt_version=prompt_version,
        require_gsb=True,
    )
    return pairs


def make_human_review_pairs_with_stats(
    df: pd.DataFrame,
    model1_column: str = "user.md-glm5-think",
    model2_column: str = "user.md-ds-10.1.2",
    model1_name: str = "glm5-think",
    model2_name: str = "ds-10.1.2",
    prompt_version: str = "human_review",
    require_gsb: bool = True,
) -> tuple[list[HumanReviewPair], list[dict[str, Any]]]:
    missing = validate_human_review_columns(df, model1_column, model2_column)
    if missing:
        raise ValueError("缺少必要列：" + ", ".join(missing))

    previous_memory: dict[tuple[str, str], str] = {}
    pairs: list[HumanReviewPair] = []
    skipped_rows: list[dict[str, Any]] = []

    for idx, row in df.iterrows():
        row_number = int(idx) + 2
        round_value = _clean(row.get("轮次"))
        reviewer = _clean(row.get("评测人")) or "unknown_reviewer"
        query = _clean(row.get("query"))
        answer = _clean(row.get("answer"))
        model1_output = _clean(row.get(model1_column))
        model2_output = _clean(row.get(model2_column))
        human_gsb = normalize_gsb(row.get("GSB"))

        if not query and not answer and not model1_output and not model2_output:
            skipped_rows.append({
                "row_number": row_number,
                "round": round_value,
                "reviewer": reviewer,
                "skip_reason": "empty_row",
                "raw_gsb": _clean(row.get("GSB")),
            })
            continue

        if require_gsb and human_gsb not in {"G", "S", "B"}:
            skipped_rows.append({
                "row_number": row_number,
                "round": round_value,
                "reviewer": reviewer,
                "skip_reason": "missing_or_invalid_gsb",
                "raw_gsb": _clean(row.get("GSB")),
                "has_model1_output": bool(model1_output),
                "has_model2_output": bool(model2_output),
            })
            continue

        pair_id = f"row_{row_number}_{reviewer}"
        dialogue = []
        if query:
            dialogue.append(DialogueTurn(role="user", content=query, metadata={"row_number": row_number, "source_column": "query"}))
        if answer:
            dialogue.append(DialogueTurn(role="assistant", content=answer, metadata={"row_number": row_number, "source_column": "answer"}))

        old1 = previous_memory.get((reviewer, model1_name), "")
        old2 = previous_memory.get((reviewer, model2_name), "")

        case1 = Case(
            case_id=f"{pair_id}_{model1_name}",
            task_type=TaskType.USER_MD,
            session_id=f"human_review_row_{row_number}",
            old_memory=old1 or None,
            dialogue=list(dialogue),
            candidate_output=model1_output or None,
            model_name=model1_name,
            prompt_version=prompt_version,
            metadata={
                "source": "human_review_excel",
                "pair_id": pair_id,
                "row_number": row_number,
                "round": round_value,
                "reviewer": reviewer,
                "model_column": model1_column,
            },
        )
        case2 = Case(
            case_id=f"{pair_id}_{model2_name}",
            task_type=TaskType.USER_MD,
            session_id=f"human_review_row_{row_number}",
            old_memory=old2 or None,
            dialogue=list(dialogue),
            candidate_output=model2_output or None,
            model_name=model2_name,
            prompt_version=prompt_version,
            metadata={
                "source": "human_review_excel",
                "pair_id": pair_id,
                "row_number": row_number,
                "round": round_value,
                "reviewer": reviewer,
                "model_column": model2_column,
            },
        )

        pairs.append(HumanReviewPair(
            pair_id=pair_id,
            row_number=row_number,
            round_value=round_value,
            reviewer=reviewer,
            query=query,
            answer=answer,
            model1_name=model1_name,
            model2_name=model2_name,
            model1_output=model1_output,
            model2_output=model2_output,
            human_gsb=human_gsb,
            issue_type=_clean(row.get("问题类型")),
            remark=_clean(row.get("备注")),
            case_model1=case1,
            case_model2=case2,
        ))

        if model1_output:
            previous_memory[(reviewer, model1_name)] = model1_output
        if model2_output:
            previous_memory[(reviewer, model2_name)] = model2_output

    return pairs, skipped_rows


def normalize_gsb(value: Any) -> str:
    text = _clean(value).upper()
    if not text:
        return ""
    if text in {"G", "GOOD", "模型1", "MODEL1", "A", "左", "左胜", "前者", "前者好", "GLM5", "GLM"}:
        return "G"
    if text in {"S", "SAME", "TIE", "平", "平局", "一致", "一样", "相同", "无差异"}:
        return "S"
    if text in {"B", "BAD", "模型2", "MODEL2", "右", "右胜", "后者", "后者好", "DS", "DS-10.1.2"}:
        return "B"
    if "平" in text or "一样" in text or "相同" in text:
        return "S"
    if "GLM" in text or "模型1" in text or "前者" in text or "左" in text:
        return "G"
    if "DS" in text or "模型2" in text or "后者" in text or "右" in text:
        return "B"
    return text[:1] if text[:1] in {"G", "S", "B"} else text


def decide_gsb(score1: float, score2: float, margin: float = 0.25) -> str:
    diff = float(score1) - float(score2)
    if diff > margin:
        return "G"
    if diff < -margin:
        return "B"
    return "S"


def stable_hash(value: Any) -> str:
    if isinstance(value, str):
        payload = value
    else:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def eval_config_fingerprint(config: Any) -> str:
    payload = {
        "judge_model": getattr(config, "judge_model", ""),
        "judge_max_tokens": getattr(config, "judge_max_tokens", ""),
        "judge_temperature": getattr(config, "judge_temperature", 0.0),
        "judge_enable_thinking": getattr(config, "judge_enable_thinking", False),
        "judge_timeout": getattr(config, "judge_timeout", ""),
        "judge_max_retries": getattr(config, "judge_max_retries", ""),
        "judge_request_interval": getattr(config, "judge_request_interval", 0.0),
        "judge_qps_backoff": getattr(config, "judge_qps_backoff", 12.0),
    }
    return stable_hash(payload)


def pair_cache_key(
    pair: HumanReviewPair,
    judge_model: str,
    judge_prompt_version: str,
    repeat_count: int = 1,
    judge_prompt_hash: str = "",
    config_hash: str = "",
    scoring_version: str = "user_md_weighted_v3_tolerant_json",
) -> str:
    payload = {
        "pair_id": pair.pair_id,
        "model1_output": pair.model1_output,
        "model2_output": pair.model2_output,
        "query": pair.query,
        "answer": pair.answer,
        "judge_model": judge_model,
        "judge_prompt_version": judge_prompt_version,
        "repeat_count": int(repeat_count),
        "judge_prompt_hash": judge_prompt_hash,
        "config_hash": config_hash,
        "scoring_version": scoring_version,
    }
    return hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def load_human_review_cache(path: str | Path = HUMAN_REVIEW_CACHE_PATH) -> dict[str, tuple[EvalResult, EvalResult]]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    cache: dict[str, tuple[EvalResult, EvalResult]] = {}
    if not isinstance(data, dict):
        return cache
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        result1 = value.get("result1")
        result2 = value.get("result2")
        if isinstance(result1, dict) and isinstance(result2, dict):
            try:
                cache[str(key)] = (EvalResult.from_dict(result1), EvalResult.from_dict(result2))
            except Exception:
                continue
    return cache


def save_human_review_cache(
    cache: dict[str, tuple[EvalResult, EvalResult]],
    path: str | Path = HUMAN_REVIEW_CACHE_PATH,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        key: {
            "result1": result1.to_dict(),
            "result2": result2.to_dict(),
        }
        for key, (result1, result2) in cache.items()
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    tmp_path.replace(path)


def build_pair_row(
    pair: HumanReviewPair,
    result1: EvalResult,
    result2: EvalResult,
    margin: float,
    *,
    from_cache: bool = False,
    cache_key: str = "",
    run_id: str = "",
    judge_prompt_hash: str = "",
    config_hash: str = "",
    judge_prompt_version: str = "",
) -> dict[str, Any]:
    auto_gsb = decide_gsb(result1.score_total, result2.score_total, margin)
    agree = bool(pair.human_gsb) and auto_gsb == pair.human_gsb
    diff = round(result1.score_total - result2.score_total, 3)
    return {
        "pair_id": pair.pair_id,
        "row_number": pair.row_number,
        "轮次": pair.round_value,
        "评测人": pair.reviewer,
        "人工GSB": pair.human_gsb,
        "自动GSB": auto_gsb,
        "是否一致": agree,
        "问题类型": pair.issue_type,
        "备注": pair.remark,
        f"{pair.model1_name}_score": result1.score_total,
        f"{pair.model2_name}_score": result2.score_total,
        "score_diff_model1_minus_model2": diff,
        "from_cache": from_cache,
        "cache_key": cache_key,
        "run_id": run_id,
        "judge_prompt_hash": judge_prompt_hash,
        "config_hash": config_hash,
        "judge_prompt_version": judge_prompt_version,
        f"{pair.model1_name}_judge备注": result1.comment,
        f"{pair.model2_name}_judge备注": result2.comment,
        f"{pair.model1_name}_raw_response": result1.raw_response or "",
        f"{pair.model2_name}_raw_response": result2.raw_response or "",
        "自动判断备注": make_auto_reason(pair.model1_name, pair.model2_name, result1, result2, margin),
        "query": pair.query,
        "answer": pair.answer,
        f"{pair.model1_name}_user.md": pair.model1_output,
        f"{pair.model2_name}_user.md": pair.model2_output,
    }


def make_human_review_run_id(prefix: str = "human_review_eval") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def get_human_review_run_path(run_id: str) -> Path:
    HUMAN_REVIEW_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return HUMAN_REVIEW_RUNS_DIR / f"{run_id}.jsonl"


def append_human_review_result_row(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def load_human_review_result_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def list_human_review_run_files() -> list[str]:
    HUMAN_REVIEW_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(str(p) for p in HUMAN_REVIEW_RUNS_DIR.glob("*.jsonl") if p.is_file())


def is_low_confidence_row(row: dict[str, Any], margin: float, band: float = 0.15) -> bool:
    if row.get("是否一致") is False:
        return True
    try:
        diff = abs(float(row.get("score_diff_model1_minus_model2", 0)))
    except (TypeError, ValueError):
        diff = 0.0
    if diff <= float(margin) + float(band):
        return True
    text = " ".join(str(v) for k, v in row.items() if "judge" in str(k).lower() or "备注" in str(k))
    return any(word in text for word in ["不确定", "可能", "无法判断", "unclear", "maybe"])


def low_confidence_rows(rows: list[dict[str, Any]], margin: float, band: float = 0.15) -> list[dict[str, Any]]:
    return [row for row in rows if is_low_confidence_row(row, margin, band)]


def make_auto_reason(
    model1_name: str,
    model2_name: str,
    result1: EvalResult,
    result2: EvalResult,
    margin: float,
) -> str:
    auto = decide_gsb(result1.score_total, result2.score_total, margin)
    diff = result1.score_total - result2.score_total
    if auto == "G":
        verdict = f"{model1_name} 更好"
    elif auto == "B":
        verdict = f"{model2_name} 更好"
    else:
        verdict = "两者基本持平"
    return (
        f"{verdict}。分差 {diff:.2f}，阈值 {margin:.2f}。"
        f"{model1_name}: {result1.comment or '无备注'}；"
        f"{model2_name}: {result2.comment or '无备注'}"
    )


def summarize_pair_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    comparable = [r for r in rows if r.get("人工GSB")]
    matched = [r for r in comparable if r.get("是否一致")]
    return {
        "total": total,
        "comparable": len(comparable),
        "matched": len(matched),
        "mismatched": len(comparable) - len(matched),
        "agreement_rate": round(len(matched) / len(comparable), 4) if comparable else 0.0,
    }


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text
