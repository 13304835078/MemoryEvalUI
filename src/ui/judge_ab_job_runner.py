from __future__ import annotations

import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from threading import Lock
from typing import Any, Callable

import pandas as pd

from src.eval.eval_runner import EvalRunner
from src.eval.result_status import STATUS_LABELS, result_evaluation_status, result_is_score_eligible
from src.runtime_paths import DATA_DIR
from src.schema import Case, EvalConfig, EvalResult, TaskType, append_result_to_jsonl, results_from_jsonl, results_to_jsonl
from src.ui.background_tasks import (
    list_task_job_ids,
    parse_time as _parse_time,
    read_json_state,
    request_stop_file,
    stop_file_exists,
    task_job_dir,
    task_state_path,
    task_stop_path,
    utc_datetime,
    utc_now,
)
from src.ui.data_service import dataframe_to_excel_bytes
from src.ui.global_rate_limiter import api_rate_scope, wait_for_global_rate_slot
from src.ui.prompt_editor import infer_prompt_version
from src.ui.state_io import atomic_write_json
from src.ui.task_controls import (
    DEFAULT_PRIORITY,
    control_float,
    control_int,
    control_priority,
    init_task_controls,
    merge_task_controls,
    read_task_controls,
)
from src.persistence import atomic_write_bytes


JUDGE_AB_JOBS_DIR = DATA_DIR / "judge_ab_jobs"


class JudgeAbJobStopped(Exception):
    pass


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
    eval_config_a: EvalConfig | None = None
    eval_config_b: EvalConfig | None = None
    parallel_different_models: bool = True


def _side_eval_config(config: JudgeAbJobConfig, label: str) -> EvalConfig:
    selected = config.eval_config_a if label.upper() == "A" else config.eval_config_b
    return selected or config.eval_config


def should_parallelize_judge_ab(config: JudgeAbJobConfig) -> bool:
    """Run both sides together only when the judge model is an intentional variable."""
    model_a = str(_side_eval_config(config, "A").judge_model or "").strip()
    model_b = str(_side_eval_config(config, "B").judge_model or "").strip()
    return bool(config.parallel_different_models and model_a and model_b and model_a != model_b)


def job_dir(job_id: str) -> Path:
    return task_job_dir(JUDGE_AB_JOBS_DIR, job_id)


def state_path(job_id: str) -> Path:
    return task_state_path(JUDGE_AB_JOBS_DIR, job_id)


def stop_path(job_id: str) -> Path:
    return task_stop_path(JUDGE_AB_JOBS_DIR, job_id)


def results_a_path(job_id: str) -> Path:
    return job_dir(job_id) / "results_a.jsonl"


def results_b_path(job_id: str) -> Path:
    return job_dir(job_id) / "results_b.jsonl"


def table_path(job_id: str) -> Path:
    return job_dir(job_id) / "judge_ab_result.xlsx"


def controls_path(job_id: str) -> Path:
    return job_dir(job_id) / "controls.json"


def read_judge_ab_job_state(job_id: str) -> dict[str, Any]:
    return read_json_state(state_path(job_id))


def read_judge_ab_job_controls(job_id: str) -> dict[str, Any]:
    return read_task_controls(controls_path(job_id))


def update_judge_ab_job_controls(job_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    return merge_task_controls(controls_path(job_id), updates)


def write_judge_ab_job_state(job_id: str, state: dict[str, Any]) -> None:
    state["heartbeat_at"] = utc_now()
    atomic_write_json(state_path(job_id), state)


def list_judge_ab_job_ids() -> list[str]:
    return list_task_job_ids(JUDGE_AB_JOBS_DIR)


def request_judge_ab_stop(job_id: str) -> None:
    request_stop_file(stop_path(job_id))


def judge_ab_stop_requested(job_id: str) -> bool:
    return stop_file_exists(stop_path(job_id))


def judge_ab_job_stale_after_seconds(state: dict[str, Any]) -> float:
    config = state.get("config") or {}
    eval_configs = [
        item
        for item in (
            config.get("eval_config_a"),
            config.get("eval_config_b"),
            config.get("eval_config"),
        )
        if isinstance(item, dict)
    ] or [{}]
    timeout = max(float(item.get("judge_timeout") or 120) for item in eval_configs)
    retries = max(float(item.get("judge_max_retries") or 3) for item in eval_configs)
    backoff = max(float(item.get("judge_qps_backoff") or 12) for item in eval_configs)
    interval = max(float(item.get("judge_request_interval") or 0) for item in eval_configs)
    return max(300.0, timeout * 2 + retries * max(backoff, 5.0) + interval + 120.0)


def judge_ab_job_is_stale(state: dict[str, Any]) -> bool:
    if state.get("status") != "running":
        return False
    heartbeat = _parse_time(str(state.get("heartbeat_at") or state.get("updated_at") or ""))
    if heartbeat is None:
        return False
    if heartbeat.tzinfo is None:
        heartbeat = heartbeat.replace(tzinfo=utc_datetime().tzinfo)
    return (utc_datetime() - heartbeat).total_seconds() > judge_ab_job_stale_after_seconds(state)


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
    for key in ("eval_config", "eval_config_a", "eval_config_b"):
        eval_config = value.get(key)
        if isinstance(eval_config, dict):
            eval_config.pop("judge_api_bearer_token", None)
            eval_config["judge_max_attempts"] = int(eval_config.get("judge_max_retries") or 1)
    return value


def summarize_results(results: list[EvalResult]) -> dict[str, Any]:
    if not results:
        return {
            "total": 0,
            "avg_score": 0.0,
            "fatal_count": 0,
            "tagged_count": 0,
            "diagnostics_count": 0,
            "scored_count": 0,
            "judge_failure_count": 0,
        }
    scored = [item for item in results if result_is_score_eligible(item)]
    return {
        "total": len(results),
        "scored_count": len(scored),
        "judge_failure_count": len(results) - len(scored),
        "avg_score": round(mean(float(item.score_total or 0) for item in scored), 4) if scored else 0.0,
        "fatal_count": sum(1 for item in scored if item.fatal_error),
        "tagged_count": sum(1 for item in scored if item.error_tags),
        "diagnostics_count": sum(len(item.diagnostics or []) for item in scored),
    }


def avg_dimension_scores(results: list[EvalResult]) -> dict[str, float]:
    results = [result for result in results if result_is_score_eligible(result)]
    dims = sorted({dim for result in results for dim in (result.scores or {})})
    rows: dict[str, float] = {}
    for dim in dims:
        values = [float((result.scores or {}).get(dim, 0.0) or 0.0) for result in results]
        rows[dim] = round(mean(values), 4) if values else 0.0
    return rows


def result_table(results_a: list[EvalResult], results_b: list[EvalResult]) -> pd.DataFrame:
    rows = []
    for a, b in zip(results_a, results_b):
        pair_eligible = result_is_score_eligible(a) and result_is_score_eligible(b)
        rows.append({
            "case_id": a.case_id,
            "model_name": a.model_name,
            "candidate_prompt_version": a.prompt_version,
            "status_A": STATUS_LABELS.get(result_evaluation_status(a), result_evaluation_status(a)),
            "status_B": STATUS_LABELS.get(result_evaluation_status(b), result_evaluation_status(b)),
            "score_A": a.score_total if result_is_score_eligible(a) else None,
            "score_B": b.score_total if result_is_score_eligible(b) else None,
            "score_delta_B_minus_A": round(float(b.score_total) - float(a.score_total), 4) if pair_eligible else None,
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
        "controls": read_judge_ab_job_controls(config.job_id),
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
    result_path: Path,
    eval_config: EvalConfig | None = None,
    progress_callback: Callable[[str, int, int, dict[str, Any]], None] | None = None,
) -> tuple[list[EvalResult], dict[str, Any]]:
    side_config = eval_config or _side_eval_config(config, label)
    prompt_version = infer_prompt_version(prompt_file)
    runner = EvalRunner(
        config=side_config,
        task_type=TaskType(config.task_type),
        prompt_file=prompt_file,
        judge_prompt_version=prompt_version,
        extraction_prompt_text=config.extraction_prompt_text,
        extraction_prompt_version=config.extraction_prompt_version,
        extraction_prompt_hash=config.extraction_prompt_hash,
    )

    configured_concurrency = min(100, max(1, int(side_config.judge_concurrency or 1)))
    configured_concurrency = min(configured_concurrency, max(1, len(cases)))
    configured_interval = float(side_config.judge_request_interval or 0.0) if not side_config.mock else 0.0
    backoff_interval = float(side_config.judge_qps_backoff or 0.0)
    rate_scope = api_rate_scope(side_config.judge_api_base_url, side_config.judge_api_bearer_token)

    def current_controls() -> dict[str, Any]:
        return read_judge_ab_job_controls(config.job_id)

    def current_concurrency() -> int:
        if not cases:
            return 1
        return min(len(cases), control_int(
            current_controls(),
            "judge_concurrency",
            configured_concurrency,
            min_value=1,
            max_value=100,
        ))

    def current_interval() -> float:
        value = control_float(
            current_controls(),
            "judge_request_interval",
            configured_interval,
            min_value=0.0,
            max_value=300.0,
        ) if not side_config.mock else 0.0
        if current_concurrency() > 1 and not side_config.mock:
            value = max(value, backoff_interval)
        return value

    def current_priority() -> int:
        return control_priority(current_controls())

    def wait_for_rate_slot() -> None:
        wait_for_global_rate_slot(
            rate_scope,
            current_interval(),
            disabled=bool(side_config.mock),
            should_stop=lambda: judge_ab_stop_requested(config.job_id),
            priority=current_priority(),
        )

    if hasattr(runner.judge_client, "rate_limit_wait_callback"):
        runner.judge_client.rate_limit_wait_callback = wait_for_rate_slot

    results_by_index: dict[int, EvalResult] = {}
    case_iter = iter(enumerate(cases))
    futures = {}
    completed = 0
    stopped = False

    def evaluate_one(index: int, case: Case) -> tuple[int, EvalResult]:
        if judge_ab_stop_requested(config.job_id):
            raise JudgeAbJobStopped()
        wait_for_rate_slot()
        if judge_ab_stop_requested(config.job_id):
            raise JudgeAbJobStopped()
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
        with ThreadPoolExecutor(max_workers=min(100, max(1, len(cases)))) as executor:
            for _ in range(current_concurrency()):
                if not submit_next(executor):
                    break
            while futures:
                done_set, _ = wait(set(futures), return_when=FIRST_COMPLETED)
                for future in done_set:
                    idx = futures.pop(future)
                    try:
                        idx, result = future.result()
                    except JudgeAbJobStopped:
                        stopped = True
                        continue
                    results_by_index[idx] = result
                    append_result_to_jsonl(result, str(result_path))
                    completed += 1
                    progress_meta = {
                        "judge_model": side_config.judge_model,
                        "configured_request_interval": configured_interval,
                        "effective_request_interval": current_interval(),
                        "effective_concurrency": current_concurrency(),
                        "priority": current_priority(),
                    }
                    if progress_callback is not None:
                        progress_callback(label, completed, len(cases), progress_meta)
                    else:
                        _write_state(
                            config,
                            stage=f"评估提示词 {label}",
                            done=completed_offset + completed,
                            total=total,
                            message=f"提示词 {label}: {completed}/{len(cases)}",
                            started_at=started_at,
                            extra={"current_label": label, **progress_meta},
                        )
                if stopped or judge_ab_stop_requested(config.job_id):
                    stopped = True
                    for future in list(futures):
                        if future.cancel():
                            futures.pop(future, None)
                    if not futures:
                        break
                    continue

                while len(futures) < current_concurrency() and submit_next(executor):
                    pass

    return [results_by_index[i] for i in sorted(results_by_index)], {
        "prompt_file": prompt_file,
        "prompt_version": prompt_version,
        "judge_model": side_config.judge_model,
        "configured_request_interval": configured_interval,
        "effective_request_interval": current_interval(),
        "effective_concurrency": current_concurrency(),
        "priority": current_priority(),
        "stopped": stopped or judge_ab_stop_requested(config.job_id),
    }


def run_judge_ab_job(config: JudgeAbJobConfig, cases: list[Case]) -> None:
    started_at = utc_now()
    if stop_path(config.job_id).exists():
        stop_path(config.job_id).unlink()
    job_dir(config.job_id).mkdir(parents=True, exist_ok=True)
    side_config_a = _side_eval_config(config, "A")
    side_config_b = _side_eval_config(config, "B")
    parallel_sides = should_parallelize_judge_ab(config)
    init_task_controls(controls_path(config.job_id), {
        "priority": DEFAULT_PRIORITY,
        "judge_concurrency": min(100, max(
            1,
            int(side_config_a.judge_concurrency or 1),
            int(side_config_b.judge_concurrency or 1),
        )),
        "judge_request_interval": max(
            float(side_config_a.judge_request_interval or 0.0),
            float(side_config_b.judge_request_interval or 0.0),
        ),
    })
    total = len(cases) * 2
    results_to_jsonl([], str(results_a_path(config.job_id)))
    results_to_jsonl([], str(results_b_path(config.job_id)))

    _write_state(
        config,
        stage="准备",
        done=0,
        total=total,
        message=(
            f"A/B 对比任务已启动，共 {len(cases)} 个 case；不同裁判模型并行评测。"
            if parallel_sides
            else f"A/B 对比任务已启动，共 {len(cases)} 个 case；相同裁判模型顺序评测。"
        ),
        started_at=started_at,
        extra={
            "parallel_sides": parallel_sides,
            "judge_model_a": side_config_a.judge_model,
            "judge_model_b": side_config_b.judge_model,
        },
    )

    try:
        if parallel_sides:
            progress_lock = Lock()
            side_completed = {"A": 0, "B": 0}

            def on_parallel_progress(
                label: str,
                completed: int,
                side_total: int,
                meta: dict[str, Any],
            ) -> None:
                with progress_lock:
                    side_completed[label] = max(side_completed[label], int(completed))
                    _write_state(
                        config,
                        stage="A/B 不同裁判模型并行评测",
                        done=sum(side_completed.values()),
                        total=total,
                        message=(
                            f"A: {side_completed['A']}/{side_total}；"
                            f"B: {side_completed['B']}/{side_total}"
                        ),
                        started_at=started_at,
                        extra={
                            "parallel_sides": True,
                            "side_progress": dict(side_completed),
                            "current_label": label,
                            **meta,
                        },
                    )

            with ThreadPoolExecutor(max_workers=2) as executor:
                future_a = executor.submit(
                    _evaluate_prompt,
                    label="A",
                    prompt_file=config.prompt_a,
                    cases=cases,
                    config=config,
                    completed_offset=0,
                    total=total,
                    started_at=started_at,
                    result_path=results_a_path(config.job_id),
                    eval_config=side_config_a,
                    progress_callback=on_parallel_progress,
                )
                future_b = executor.submit(
                    _evaluate_prompt,
                    label="B",
                    prompt_file=config.prompt_b,
                    cases=cases,
                    config=config,
                    completed_offset=0,
                    total=total,
                    started_at=started_at,
                    result_path=results_b_path(config.job_id),
                    eval_config=side_config_b,
                    progress_callback=on_parallel_progress,
                )
                results_a, stats_a = future_a.result()
                results_b, stats_b = future_b.result()
        else:
            results_a, stats_a = _evaluate_prompt(
                label="A",
                prompt_file=config.prompt_a,
                cases=cases,
                config=config,
                completed_offset=0,
                total=total,
                started_at=started_at,
                result_path=results_a_path(config.job_id),
                eval_config=side_config_a,
            )
            results_to_jsonl(results_a, str(results_a_path(config.job_id)))
            if stats_a.get("stopped") or judge_ab_stop_requested(config.job_id):
                _write_state(
                    config,
                    status="stopped",
                    stage="已终止",
                    done=len(results_a),
                    total=total,
                    message="A/B 对比已在提示词 A 后终止。",
                    started_at=started_at,
                    extra={
                        "stats_a": stats_a,
                        "summary_a": summarize_results(results_a),
                        "finished_at": utc_now(),
                    },
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
                result_path=results_b_path(config.job_id),
                eval_config=side_config_b,
            )
        results_to_jsonl(results_a, str(results_a_path(config.job_id)))
        results_to_jsonl(results_b, str(results_b_path(config.job_id)))
        table = result_table(results_a, results_b)
        atomic_write_bytes(table_path(config.job_id), dataframe_to_excel_bytes(table))
        summary_a = summarize_results(results_a)
        summary_b = summarize_results(results_b)
        if stats_a.get("stopped") or stats_b.get("stopped") or judge_ab_stop_requested(config.job_id):
            _write_state(
                config,
                status="stopped",
                stage="已终止",
                done=len(results_a) + len(results_b),
                total=total,
                message="A/B 对比已终止，已保留两侧完成的部分结果。",
                started_at=started_at,
                extra={
                    "stats_a": stats_a,
                    "stats_b": stats_b,
                    "summary_a": summary_a,
                    "summary_b": summary_b,
                    "parallel_sides": parallel_sides,
                    "table_preview": table.head(100).to_dict("records"),
                    "finished_at": utc_now(),
                },
            )
            return
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
                "parallel_sides": parallel_sides,
                "table_preview": table.head(100).to_dict("records"),
                "finished_at": utc_now(),
            },
        )
    except JudgeAbJobStopped as exc:
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
