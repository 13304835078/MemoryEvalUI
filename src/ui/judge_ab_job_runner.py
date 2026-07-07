from __future__ import annotations

import json
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from src.eval.eval_runner import EvalRunner
from src.runtime_paths import DATA_DIR
from src.schema import Case, EvalConfig, EvalResult, TaskType, results_from_jsonl, results_to_jsonl
from src.ui.data_service import dataframe_to_excel_bytes
from src.ui.global_rate_limiter import api_rate_scope, wait_for_global_rate_slot
from src.ui.prompt_editor import infer_prompt_version
from src.ui.state_io import atomic_write_json


JUDGE_AB_JOBS_DIR = DATA_DIR / "judge_ab_jobs"


@dataclass
class JudgeAbJobConfig:
    job_id: str
    task_type: str
    prompt_a: str
    prompt_b: str
    cases_file: str = ""
    extraction_prompt_text: str = ""
    extraction_prompt_version: str = ""
    extraction_prompt_hash: str = ""
    eval_config: EvalConfig = field(default_factory=EvalConfig)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_time(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def job_dir(job_id: str) -> Path:
    return JUDGE_AB_JOBS_DIR / job_id


def state_path(job_id: str) -> Path:
    return job_dir(job_id) / "state.json"


def stop_path(job_id: str) -> Path:
    return job_dir(job_id) / "STOP"


def results_a_path(job_id: str) -> Path:
    return job_dir(job_id) / "results_a.jsonl"


def results_b_path(job_id: str) -> Path:
    return job_dir(job_id) / "results_b.jsonl"


def table_path(job_id: str) -> Path:
    return job_dir(job_id) / "judge_ab_result.xlsx"


def read_judge_ab_job_state(job_id: str) -> dict[str, Any]:
    path = state_path(job_id)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_judge_ab_job_state(job_id: str, state: dict[str, Any]) -> None:
    state["heartbeat_at"] = utc_now()
    atomic_write_json(state_path(job_id), state)


def list_judge_ab_job_ids() -> list[str]:
    if not JUDGE_AB_JOBS_DIR.exists():
        return []
    paths = [path for path in JUDGE_AB_JOBS_DIR.iterdir() if path.is_dir()]
    paths.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return [path.name for path in paths]


def request_judge_ab_stop(job_id: str) -> None:
    path = stop_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(utc_now(), encoding="utf-8")


def judge_ab_stop_requested(job_id: str) -> bool:
    return stop_path(job_id).exists()


def judge_ab_job_stale_after_seconds(state: dict[str, Any]) -> float:
    config = state.get("config") or {}
    eval_config = config.get("eval_config") if isinstance(config.get("eval_config"), dict) else {}
    timeout = float(eval_config.get("judge_timeout") or 120)
    retries = float(eval_config.get("judge_max_retries") or 3)
    backoff = float(eval_config.get("judge_qps_backoff") or 12)
    interval = float(eval_config.get("judge_request_interval") or 0)
    return max(300.0, timeout * 2 + retries * max(backoff, 5.0) + interval + 120.0)


def judge_ab_job_is_stale(state: dict[str, Any]) -> bool:
    if state.get("status") != "running":
        return False
    heartbeat = _parse_time(str(state.get("heartbeat_at") or state.get("updated_at") or ""))
    if heartbeat is None:
        return False
    if heartbeat.tzinfo is None:
        heartbeat = heartbeat.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - heartbeat).total_seconds() > judge_ab_job_stale_after_seconds(state)


def mark_judge_ab_job_interrupted(job_id: str) -> dict[str, Any]:
    state = read_judge_ab_job_state(job_id)
    if not state or state.get("status") != "running":
        return state
    state["status"] = "interrupted"
    state["stage"] = "已中断"
    state["message"] = "后台 A/B 对比任务可能已中断：长时间没有心跳。可以重新启动任务。"
    state["finished_at"] = utc_now()
    state["updated_at"] = utc_now()
    write_judge_ab_job_state(job_id, state)
    return state


def judge_ab_job_is_running(job_id: str) -> bool:
    state = read_judge_ab_job_state(job_id)
    if judge_ab_job_is_stale(state):
        mark_judge_ab_job_interrupted(job_id)
        return False
    return state.get("status") == "running"


def _safe_config(config: JudgeAbJobConfig) -> dict[str, Any]:
    value = asdict(config)
    value.pop("extraction_prompt_text", None)
    eval_config = value.get("eval_config")
    if isinstance(eval_config, dict):
        eval_config.pop("judge_api_bearer_token", None)
    return value


def summarize_results(results: list[EvalResult]) -> dict[str, Any]:
    if not results:
        return {
            "total": 0,
            "avg_score": 0.0,
            "fatal_count": 0,
            "tagged_count": 0,
            "diagnostics_count": 0,
        }
    return {
        "total": len(results),
        "avg_score": round(mean(float(item.score_total or 0) for item in results), 4),
        "fatal_count": sum(1 for item in results if item.fatal_error),
        "tagged_count": sum(1 for item in results if item.error_tags),
        "diagnostics_count": sum(len(item.diagnostics or []) for item in results),
    }


def avg_dimension_scores(results: list[EvalResult]) -> dict[str, float]:
    dims = sorted({dim for result in results for dim in (result.scores or {})})
    rows: dict[str, float] = {}
    for dim in dims:
        values = [float((result.scores or {}).get(dim, 0.0) or 0.0) for result in results]
        rows[dim] = round(mean(values), 4) if values else 0.0
    return rows


def result_table(results_a: list[EvalResult], results_b: list[EvalResult]) -> pd.DataFrame:
    rows = []
    for a, b in zip(results_a, results_b):
        rows.append({
            "case_id": a.case_id,
            "model_name": a.model_name,
            "candidate_prompt_version": a.prompt_version,
            "score_A": a.score_total,
            "score_B": b.score_total,
            "score_delta_B_minus_A": round(float(b.score_total or 0) - float(a.score_total or 0), 4),
            "fatal_A": a.fatal_error,
            "fatal_B": b.fatal_error,
            "error_tags_A": ", ".join(a.error_tags or []),
            "error_tags_B": ", ".join(b.error_tags or []),
            "diagnostics_A": len(a.diagnostics or []),
            "diagnostics_B": len(b.diagnostics or []),
            "comment_A": a.comment,
            "comment_B": b.comment,
            "rule_refs_A": "; ".join(a.rule_refs or []),
            "rule_refs_B": "; ".join(b.rule_refs or []),
            "evidence_refs_A": "; ".join(a.evidence_refs or []),
            "evidence_refs_B": "; ".join(b.evidence_refs or []),
        })
    return pd.DataFrame(rows)


def load_judge_ab_results(job_id: str) -> tuple[list[EvalResult], list[EvalResult]]:
    a = results_from_jsonl(str(results_a_path(job_id))) if results_a_path(job_id).exists() else []
    b = results_from_jsonl(str(results_b_path(job_id))) if results_b_path(job_id).exists() else []
    return a, b


def _write_state(
    config: JudgeAbJobConfig,
    *,
    status: str = "running",
    stage: str,
    done: int,
    total: int,
    message: str,
    started_at: str,
    extra: dict[str, Any] | None = None,
) -> None:
    state = {
        "job_id": config.job_id,
        "status": status,
        "stage": stage,
        "done": int(done),
        "total": int(total),
        "message": message,
        "started_at": started_at,
        "updated_at": utc_now(),
        "config": _safe_config(config),
        "results_a_path": str(results_a_path(config.job_id)),
        "results_b_path": str(results_b_path(config.job_id)),
        "table_path": str(table_path(config.job_id)),
    }
    if extra:
        state.update(extra)
    write_judge_ab_job_state(config.job_id, state)


def _evaluate_prompt(
    *,
    label: str,
    prompt_file: str,
    cases: list[Case],
    config: JudgeAbJobConfig,
    completed_offset: int,
    total: int,
    started_at: str,
) -> tuple[list[EvalResult], dict[str, Any]]:
    prompt_version = infer_prompt_version(prompt_file)
    runner = EvalRunner(
        config=config.eval_config,
        task_type=TaskType(config.task_type),
        prompt_file=prompt_file,
        judge_prompt_version=prompt_version,
        extraction_prompt_text=config.extraction_prompt_text,
        extraction_prompt_version=config.extraction_prompt_version,
        extraction_prompt_hash=config.extraction_prompt_hash,
    )

    concurrency = min(100, max(1, int(config.eval_config.judge_concurrency or 1)))
    concurrency = min(concurrency, max(1, len(cases)))
    configured_interval = float(config.eval_config.judge_request_interval or 0.0) if not config.eval_config.mock else 0.0
    effective_interval = configured_interval
    if concurrency > 1 and not config.eval_config.mock:
        effective_interval = max(effective_interval, float(config.eval_config.judge_qps_backoff or 0.0))
    rate_scope = api_rate_scope(config.eval_config.judge_api_base_url, config.eval_config.judge_api_bearer_token)

    def wait_for_rate_slot() -> None:
        wait_for_global_rate_slot(
            rate_scope,
            effective_interval,
            disabled=bool(config.eval_config.mock),
            should_stop=lambda: judge_ab_stop_requested(config.job_id),
        )

    if hasattr(runner.judge_client, "rate_limit_wait_callback"):
        runner.judge_client.rate_limit_wait_callback = wait_for_rate_slot

    results_by_index: dict[int, EvalResult] = {}
    case_iter = iter(enumerate(cases))
    futures = {}
    completed = 0

    def evaluate_one(index: int, case: Case) -> tuple[int, EvalResult]:
        if judge_ab_stop_requested(config.job_id):
            raise StopIteration("收到终止请求")
        wait_for_rate_slot()
        if judge_ab_stop_requested(config.job_id):
            raise StopIteration("收到终止请求")
        return index, runner.evaluate_one(case)

    def submit_next(executor: ThreadPoolExecutor) -> bool:
        if judge_ab_stop_requested(config.job_id):
            return False
        try:
            idx, case = next(case_iter)
        except StopIteration:
            return False
        futures[executor.submit(evaluate_one, idx, case)] = idx
        return True

    if cases:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            for _ in range(concurrency):
                if not submit_next(executor):
                    break
            while futures:
                done_set, _ = wait(set(futures), return_when=FIRST_COMPLETED)
                for future in done_set:
                    futures.pop(future)
                    idx, result = future.result()
                    results_by_index[idx] = result
                    completed += 1
                    global_done = completed_offset + completed
                    _write_state(
                        config,
                        stage=f"评估提示词 {label}",
                        done=global_done,
                        total=total,
                        message=f"提示词 {label}: {completed}/{len(cases)}",
                        started_at=started_at,
                        extra={
                            "current_label": label,
                            "configured_request_interval": configured_interval,
                            "effective_request_interval": effective_interval,
                        },
                    )
                    submit_next(executor)

    return [results_by_index[i] for i in sorted(results_by_index)], {
        "prompt_file": prompt_file,
        "prompt_version": prompt_version,
        "configured_request_interval": configured_interval,
        "effective_request_interval": effective_interval,
    }


def run_judge_ab_job(config: JudgeAbJobConfig, cases: list[Case]) -> None:
    started_at = utc_now()
    if stop_path(config.job_id).exists():
        stop_path(config.job_id).unlink()
    job_dir(config.job_id).mkdir(parents=True, exist_ok=True)
    total = len(cases) * 2

    _write_state(
        config,
        stage="准备",
        done=0,
        total=total,
        message=f"A/B 对比任务已启动，共 {len(cases)} 个 case。",
        started_at=started_at,
    )

    try:
        results_a, stats_a = _evaluate_prompt(
            label="A",
            prompt_file=config.prompt_a,
            cases=cases,
            config=config,
            completed_offset=0,
            total=total,
            started_at=started_at,
        )
        results_to_jsonl(results_a, str(results_a_path(config.job_id)))
        if judge_ab_stop_requested(config.job_id):
            _write_state(
                config,
                status="stopped",
                stage="已终止",
                done=len(results_a),
                total=total,
                message="A/B 对比已在提示词 A 后终止。",
                started_at=started_at,
                extra={"stats_a": stats_a, "finished_at": utc_now()},
            )
            return

        results_b, stats_b = _evaluate_prompt(
            label="B",
            prompt_file=config.prompt_b,
            cases=cases,
            config=config,
            completed_offset=len(cases),
            total=total,
            started_at=started_at,
        )
        results_to_jsonl(results_b, str(results_b_path(config.job_id)))
        table = result_table(results_a, results_b)
        table_path(config.job_id).write_bytes(dataframe_to_excel_bytes(table))
        summary_a = summarize_results(results_a)
        summary_b = summarize_results(results_b)
        _write_state(
            config,
            status="completed",
            stage="完成",
            done=total,
            total=total,
            message="A/B 对比完成。",
            started_at=started_at,
            extra={
                "stats_a": stats_a,
                "stats_b": stats_b,
                "summary_a": summary_a,
                "summary_b": summary_b,
                "table_preview": table.head(100).to_dict("records"),
                "finished_at": utc_now(),
            },
        )
    except StopIteration as exc:
        state = read_judge_ab_job_state(config.job_id)
        _write_state(
            config,
            status="stopped",
            stage="已终止",
            done=int(state.get("done", 0) or 0),
            total=total,
            message=str(exc) or "A/B 对比已终止。",
            started_at=started_at,
            extra={"finished_at": utc_now()},
        )
    except Exception:
        state = read_judge_ab_job_state(config.job_id)
        _write_state(
            config,
            status="failed",
            stage="失败",
            done=int(state.get("done", 0) or 0),
            total=total,
            message="A/B 对比任务失败。",
            started_at=started_at,
            extra={"traceback": traceback.format_exc(), "finished_at": utc_now()},
        )
