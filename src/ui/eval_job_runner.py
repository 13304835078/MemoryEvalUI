from __future__ import annotations

import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.eval.eval_runner import EvalRunner
from src.eval.result_status import result_is_score_eligible
from src.schema import Case, EvalConfig, EvalResult, TaskType, append_result_to_jsonl, results_to_jsonl
from src.ui.data_service import (
    RESULTS_DIR,
    case_resume_key,
    eval_result_resume_key,
    load_results,
)
from src.ui.global_rate_limiter import api_rate_scope, wait_for_global_rate_slot
from src.ui.task_controls import (
    DEFAULT_PRIORITY,
    control_float,
    control_int,
    control_priority,
    init_task_controls,
    merge_task_controls,
    read_task_controls,
)
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
from src.ui.state_io import atomic_write_json


EVAL_JOBS_DIR = RESULTS_DIR.parent / "eval_jobs"
RESUME_SKIP_ALL = "跳过所有已有结果"
RESUME_SKIP_NON_FATAL = "只跳过评分成功结果（重跑运行失败）"
LEGACY_RESUME_SKIP_NON_FATAL = "只跳过非严重失败结果"
RESUME_RERUN_ALL = "全部重跑"
RESUME_STRATEGIES = [RESUME_SKIP_ALL, RESUME_SKIP_NON_FATAL, RESUME_RERUN_ALL]


class EvalJobStopped(Exception):
    pass


@dataclass
class EvalJobConfig:
    job_id: str
    task_type: str
    output_path: str
    prompt_file: str
    judge_prompt_version: str
    cases_file: str = ""
    system_prompt_override: str = ""
    extraction_prompt_text: str = ""
    extraction_prompt_version: str = ""
    extraction_prompt_hash: str = ""
    resume_strategy: str = RESUME_SKIP_ALL
    eval_config: EvalConfig = field(default_factory=EvalConfig)


def job_dir(job_id: str) -> Path:
    return task_job_dir(EVAL_JOBS_DIR, job_id)


def state_path(job_id: str) -> Path:
    return task_state_path(EVAL_JOBS_DIR, job_id)


def stop_path(job_id: str) -> Path:
    return task_stop_path(EVAL_JOBS_DIR, job_id)


def controls_path(job_id: str) -> Path:
    return job_dir(job_id) / "controls.json"


def request_eval_stop(job_id: str) -> None:
    request_stop_file(stop_path(job_id))


def eval_job_stop_requested(job_id: str) -> bool:
    return stop_file_exists(stop_path(job_id))


def read_eval_job_state(job_id: str) -> dict[str, Any]:
    return read_json_state(state_path(job_id))


def read_eval_job_controls(job_id: str) -> dict[str, Any]:
    return read_task_controls(controls_path(job_id))


def update_eval_job_controls(job_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    return merge_task_controls(controls_path(job_id), updates)


def write_eval_job_state(job_id: str, state: dict[str, Any]) -> None:
    path = state_path(job_id)
    state["heartbeat_at"] = utc_now()
    atomic_write_json(path, state)


def eval_job_stale_after_seconds(state: dict[str, Any]) -> float:
    config = state.get("config") or {}
    eval_config = config.get("eval_config") if isinstance(config.get("eval_config"), dict) else {}
    timeout = float(eval_config.get("judge_timeout") or 120)
    retries = float(eval_config.get("judge_max_retries") or 3)
    backoff = float(eval_config.get("judge_qps_backoff") or 12)
    interval = float(eval_config.get("judge_request_interval") or 0)
    return max(300.0, timeout * 2 + retries * max(backoff, 5.0) + interval + 60.0)


def eval_job_is_stale(state: dict[str, Any]) -> bool:
    if state.get("status") != "running":
        return False
    heartbeat = _parse_time(str(state.get("heartbeat_at") or state.get("updated_at") or ""))
    if heartbeat is None:
        return False
    if heartbeat.tzinfo is None:
        heartbeat = heartbeat.replace(tzinfo=utc_datetime().tzinfo)
    elapsed = (utc_datetime() - heartbeat).total_seconds()
    return elapsed > eval_job_stale_after_seconds(state)


def mark_eval_job_interrupted(job_id: str) -> dict[str, Any]:
    state = read_eval_job_state(job_id)
    if not state or state.get("status") != "running":
        return state
    state["status"] = "interrupted"
    state["message"] = "后台任务可能已中断：长时间没有心跳。可以重新启动评测或断点续跑。"
    state["finished_at"] = utc_now()
    state["updated_at"] = utc_now()
    write_eval_job_state(job_id, state)
    return state


def eval_job_is_running(job_id: str) -> bool:
    state = read_eval_job_state(job_id)
    if eval_job_is_stale(state):
        mark_eval_job_interrupted(job_id)
        return False
    return state.get("status") == "running"


def list_eval_job_ids() -> list[str]:
    return list_task_job_ids(EVAL_JOBS_DIR)


def _safe_config(config: EvalJobConfig) -> dict[str, Any]:
    value = asdict(config)
    eval_config = value.get("eval_config")
    if isinstance(eval_config, dict):
        eval_config.pop("judge_api_bearer_token", None)
        eval_config["judge_max_attempts"] = int(eval_config.get("judge_max_retries") or 1)
    value.pop("system_prompt_override", None)
    value.pop("extraction_prompt_text", None)
    return value


def _write_progress(
    config: EvalJobConfig,
    *,
    status: str = "running",
    done: int,
    total: int,
    skipped: int,
    evaluated: int,
    fatal_count: int,
    message: str,
    started_at: str,
    results: list[EvalResult] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    current_results = list(results or [])
    scored_count = sum(1 for item in current_results if result_is_score_eligible(item))
    judge_failure_count = len(current_results) - scored_count
    state = {
        "job_id": config.job_id,
        "status": status,
        "updated_at": utc_now(),
        "started_at": started_at,
        "done": int(done),
        "total": int(total),
        "skipped": int(skipped),
        "evaluated": int(evaluated),
        "fatal_count": int(fatal_count),
        "scored_count": scored_count,
        "judge_failure_count": judge_failure_count,
        "score_coverage": round(scored_count / len(current_results), 4) if current_results else 0.0,
        "message": message,
        "output_path": config.output_path,
        "config": _safe_config(config),
        "controls": read_eval_job_controls(config.job_id),
    }
    if extra:
        state.update(extra)
    write_eval_job_state(config.job_id, state)


def run_eval_job(config: EvalJobConfig, cases: list[Case], existing_results: list[EvalResult] | None = None) -> None:
    started_at = utc_now()
    if stop_path(config.job_id).exists():
        stop_path(config.job_id).unlink()
    init_task_controls(controls_path(config.job_id), {
        "priority": DEFAULT_PRIORITY,
        "judge_concurrency": min(100, max(1, int(config.eval_config.judge_concurrency or 1))),
        "judge_request_interval": float(config.eval_config.judge_request_interval or 0.0),
    })
    existing_results = list(existing_results or [])
    total = len(cases)
    skipped_count = 0
    evaluated_count = 0
    results_by_key = {eval_result_resume_key(result): result for result in existing_results}

    _write_progress(
        config,
        done=0,
        total=total,
        skipped=0,
        evaluated=0,
        fatal_count=sum(1 for result in existing_results if result.fatal_error),
        results=existing_results,
        message="评测任务启动",
        started_at=started_at,
    )

    try:
        runner = EvalRunner(
            config=config.eval_config,
            task_type=TaskType(config.task_type),
            prompt_file=config.prompt_file,
            judge_prompt_version=config.judge_prompt_version,
            system_prompt_override=config.system_prompt_override,
            extraction_prompt_text=config.extraction_prompt_text,
            extraction_prompt_version=config.extraction_prompt_version,
            extraction_prompt_hash=config.extraction_prompt_hash,
        )

        output_path = Path(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        judge_model_key = config.eval_config.judge_model or "mock"
        existing_by_key = dict(results_by_key)

        tasks: list[tuple[int, Case]] = []

        def should_skip_result(existing_result: EvalResult | None) -> bool:
            if config.resume_strategy == RESUME_SKIP_ALL:
                return existing_result is not None
            if config.resume_strategy in {RESUME_SKIP_NON_FATAL, LEGACY_RESUME_SKIP_NON_FATAL}:
                return existing_result is not None and result_is_score_eligible(existing_result)
            return False

        for index, case in enumerate(cases):
            fingerprint_builder = getattr(runner, "evaluation_fingerprint", None)
            expected_fingerprint = fingerprint_builder(case) if callable(fingerprint_builder) else ""
            current_key = case_resume_key(
                case,
                judge_model_key,
                getattr(runner, "resolved_judge_prompt_version", config.judge_prompt_version),
                getattr(runner, "extraction_prompt_hash", config.extraction_prompt_hash),
                expected_fingerprint,
            )
            existing_result = existing_by_key.get(current_key)
            if should_skip_result(existing_result):
                skipped_count += 1
                _write_progress(
                    config,
                    done=skipped_count,
                    total=total,
                    skipped=skipped_count,
                    evaluated=evaluated_count,
                    fatal_count=sum(1 for item in results_by_key.values() if item.fatal_error),
                    results=list(results_by_key.values()),
                    message=f"跳过已有结果：{index + 1}/{total}，样本：{case.case_id}",
                    started_at=started_at,
                )
                continue
            tasks.append((index, case))

        configured_concurrency = min(100, max(1, int(config.eval_config.judge_concurrency or 1)))
        configured_interval = float(config.eval_config.judge_request_interval or 0.0) if not config.eval_config.mock else 0.0
        backoff_interval = float(config.eval_config.judge_qps_backoff or 0.0)
        rate_scope = api_rate_scope(
            config.eval_config.judge_api_base_url,
            config.eval_config.judge_api_bearer_token,
        )

        def current_controls() -> dict[str, Any]:
            return read_eval_job_controls(config.job_id)

        def current_concurrency() -> int:
            if not tasks:
                return 1
            return min(len(tasks), control_int(
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
            ) if not config.eval_config.mock else 0.0
            if current_concurrency() > 1 and not config.eval_config.mock:
                value = max(value, backoff_interval)
            return value

        def current_priority() -> int:
            return control_priority(current_controls())

        def should_stop() -> bool:
            return eval_job_stop_requested(config.job_id)

        def wait_for_rate_slot() -> None:
            wait_for_global_rate_slot(
                rate_scope,
                current_interval(),
                disabled=bool(config.eval_config.mock),
                should_stop=should_stop,
                priority=current_priority(),
            )
            if should_stop():
                raise EvalJobStopped()

        if hasattr(runner.judge_client, "rate_limit_wait_callback"):
            runner.judge_client.rate_limit_wait_callback = wait_for_rate_slot

        def evaluate_task(case_index: int, case: Case) -> tuple[int, Case, EvalResult]:
            if should_stop():
                raise EvalJobStopped()
            wait_for_rate_slot()
            if should_stop():
                raise EvalJobStopped()
            return case_index, case, runner.evaluate_one(case)

        if tasks:
            _write_progress(
                config,
                done=skipped_count,
                total=total,
                skipped=skipped_count,
                evaluated=evaluated_count,
                fatal_count=sum(1 for item in results_by_key.values() if item.fatal_error),
                results=list(results_by_key.values()),
                message=f"开始评测：待评测 {len(tasks)} 条，跳过 {skipped_count} 条，并发数 {current_concurrency()}",
                started_at=started_at,
                extra={
                    "effective_request_interval": current_interval(),
                    "configured_request_interval": configured_interval,
                    "effective_concurrency": current_concurrency(),
                    "priority": current_priority(),
                },
            )

        if not output_path.exists():
            results_to_jsonl(list(results_by_key.values()), str(output_path))

        stopped = False
        with ThreadPoolExecutor(max_workers=min(100, max(1, len(tasks)))) as executor:
            futures: dict[Any, tuple[int, Case]] = {}
            next_task_index = 0

            def submit_next() -> None:
                nonlocal next_task_index
                if next_task_index >= len(tasks) or should_stop():
                    return
                case_index, case = tasks[next_task_index]
                futures[executor.submit(evaluate_task, case_index, case)] = (case_index, case)
                next_task_index += 1

            for _ in range(min(current_concurrency(), len(tasks))):
                submit_next()

            while futures:
                done_futures, _pending = wait(futures, return_when=FIRST_COMPLETED)
                for future in done_futures:
                    _, case = futures.pop(future)
                    try:
                        _, _, result = future.result()
                    except EvalJobStopped:
                        stopped = True
                        continue
                    except Exception as exc:
                        result = EvalResult.from_parse_failure(
                            case_id=case.case_id,
                            task_type=config.task_type,
                            raw=f"{type(exc).__name__}: {exc}",
                            model_name=case.model_name,
                            prompt_version=case.prompt_version,
                            judge_model=judge_model_key,
                            judge_prompt_version=config.judge_prompt_version,
                            extraction_prompt_version=config.extraction_prompt_version,
                            extraction_prompt_hash=config.extraction_prompt_hash,
                        )

                    results_by_key[eval_result_resume_key(result)] = result
                    append_result_to_jsonl(result, str(output_path))
                    evaluated_count += 1
                    done = skipped_count + evaluated_count
                    fatal_count = sum(1 for item in results_by_key.values() if item.fatal_error)
                    _write_progress(
                        config,
                        done=done,
                        total=total,
                        skipped=skipped_count,
                        evaluated=evaluated_count,
                        fatal_count=fatal_count,
                        results=list(results_by_key.values()),
                        message=f"已保存：整体 {done}/{total}，新增 {evaluated_count}/{len(tasks)}，当前样本：{case.case_id}",
                        started_at=started_at,
                        extra={
                            "effective_request_interval": current_interval(),
                            "effective_concurrency": current_concurrency(),
                            "priority": current_priority(),
                        },
                    )

                if stopped or should_stop():
                    stopped = True
                    for future in list(futures):
                        if future.cancel():
                            futures.pop(future, None)
                    if not futures:
                        break
                    continue

                while len(futures) < current_concurrency() and next_task_index < len(tasks) and not should_stop():
                    submit_next()

        if stopped or eval_job_stop_requested(config.job_id):
            results = list(results_by_key.values())
            results_to_jsonl(results, str(output_path))
            done = skipped_count + evaluated_count
            _write_progress(
                config,
                status="stopped",
                done=done,
                total=total,
                skipped=skipped_count,
                evaluated=evaluated_count,
                fatal_count=sum(1 for item in results if item.fatal_error),
                results=results,
                message=f"评测已终止：已完成 {done}/{total}，新增 {evaluated_count} 条，跳过 {skipped_count} 条；可用结果文件断点续跑。",
                started_at=started_at,
                extra={"finished_at": utc_now()},
            )
            return

        results = list(results_by_key.values())
        results_to_jsonl(results, str(output_path))
        _write_progress(
            config,
            status="completed",
            done=total,
            total=total,
            skipped=skipped_count,
            evaluated=evaluated_count,
            fatal_count=sum(1 for item in results if item.fatal_error),
            results=results,
            message=f"评测完成：新增 {evaluated_count} 条，跳过 {skipped_count} 条，结果文件共 {len(results)} 条",
            started_at=started_at,
            extra={"finished_at": utc_now()},
        )

    except Exception as exc:
        state = read_eval_job_state(config.job_id)
        _write_progress(
            config,
            status="failed",
            done=int(state.get("done", skipped_count + evaluated_count) or 0),
            total=total,
            skipped=int(state.get("skipped", skipped_count) or 0),
            evaluated=int(state.get("evaluated", evaluated_count) or 0),
            fatal_count=int(state.get("fatal_count", sum(1 for item in results_by_key.values() if item.fatal_error)) or 0),
            results=list(results_by_key.values()),
            message=f"评测失败：{type(exc).__name__}: {exc}",
            started_at=started_at,
            extra={
                "finished_at": utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )


def load_job_results_from_state(state: dict[str, Any]) -> list[EvalResult]:
    output_path = state.get("output_path", "")
    if not output_path or not Path(output_path).exists():
        return []
    return load_results(output_path)
