from __future__ import annotations

import hashlib
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field, replace
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

from src.eval.eval_runner import SCORING_SCHEMA_VERSION, EvalRunner
from src.eval.extraction_prompt_compare import compare_extraction_prompt_runs
from src.eval.extraction_prompt_comparison_advisor import call_comparison_model
from src.extraction.client import MemoryExtractionConfig
from src.extraction.memory_extractor import MemoryExtractionRunner
from src.loop.validation_gate import ValidationGateConfig
from src.persistence import atomic_write_bytes
from src.runtime_paths import DATA_DIR
from src.schema import (
    Case,
    EvalConfig,
    EvalResult,
    TaskType,
    append_result_to_jsonl,
    cases_from_jsonl,
    cases_to_jsonl,
    results_from_jsonl,
    results_to_jsonl,
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
from src.ui.data_service import (
    prepare_cases_from_run_output,
    prepare_long_memory_cases_from_run_output,
)
from src.ui.global_rate_limiter import api_rate_scope, wait_for_global_rate_slot
from src.ui.extraction_prompt_ab_export import write_extraction_prompt_diff_excel
from src.ui.state_io import atomic_write_json, state_file_lock
from src.ui.task_controls import (
    DEFAULT_PRIORITY,
    control_float,
    control_int,
    control_priority,
    init_task_controls,
    merge_task_controls,
    read_task_controls,
)


EXTRACTION_PROMPT_AB_JOBS_DIR = DATA_DIR / "extraction_prompt_ab_jobs"
PROGRESS_TOTAL = 1000


class ExtractionPromptAbStopped(Exception):
    pass


@dataclass
class ExtractionPromptAbJobConfig:
    job_id: str
    task_type: str
    input_path: str
    prompt_a_text: str
    prompt_a_version: str
    prompt_b_text: str
    prompt_b_version: str
    judge_prompt_text: str
    judge_prompt_version: str
    evaluation_rule_prompt_text: str
    evaluation_rule_prompt_version: str
    prompt_a_file: str = ""
    prompt_b_file: str = ""
    judge_prompt_file: str = ""
    evaluation_rule_prompt_file: str = ""
    prompt_a_create_text: str = ""
    prompt_b_create_text: str = ""
    prompt_a_hash: str = ""
    prompt_b_hash: str = ""
    evaluation_rule_prompt_hash: str = ""
    sheet_name: str | int | None = 0
    reviewer_filter: str = ""
    chunk_size: int = 10
    score_tolerance: float = 0.05
    extraction_config: MemoryExtractionConfig = field(default_factory=MemoryExtractionConfig)
    eval_config: EvalConfig = field(default_factory=EvalConfig)
    comparison_config: EvalConfig = field(default_factory=EvalConfig)
    enable_model_comparison: bool = False
    comparison_max_evidence: int = 8
    validation_config: ValidationGateConfig = field(default_factory=ValidationGateConfig)


def job_dir(job_id: str) -> Path:
    return task_job_dir(EXTRACTION_PROMPT_AB_JOBS_DIR, job_id)


def state_path(job_id: str) -> Path:
    return task_state_path(EXTRACTION_PROMPT_AB_JOBS_DIR, job_id)


def stop_path(job_id: str) -> Path:
    return task_stop_path(EXTRACTION_PROMPT_AB_JOBS_DIR, job_id)


def controls_path(job_id: str) -> Path:
    return job_dir(job_id) / "controls.json"


def extraction_path(job_id: str, label: str) -> Path:
    return job_dir(job_id) / f"extraction_{label.lower()}.xlsx"


def cases_path(job_id: str, label: str) -> Path:
    return job_dir(job_id) / f"cases_{label.lower()}.jsonl"


def missed_cases_path(job_id: str, label: str) -> Path:
    return job_dir(job_id) / f"missed_cases_{label.lower()}.jsonl"


def results_path(job_id: str, label: str) -> Path:
    return job_dir(job_id) / f"results_{label.lower()}.jsonl"


def report_path(job_id: str) -> Path:
    return job_dir(job_id) / "comparison.json"


def report_excel_path(job_id: str) -> Path:
    return job_dir(job_id) / "extraction_prompt_ab_comparison.xlsx"


def diff_excel_path(job_id: str) -> Path:
    return job_dir(job_id) / "extraction_prompt_ab_diff.xlsx"


def read_extraction_prompt_ab_job_state(job_id: str) -> dict[str, Any]:
    return read_json_state(state_path(job_id))


def write_extraction_prompt_ab_job_state(job_id: str, state: dict[str, Any]) -> None:
    path = state_path(job_id)
    with state_file_lock(path):
        state["heartbeat_at"] = utc_now()
        atomic_write_json(path, state)


def read_extraction_prompt_ab_job_controls(job_id: str) -> dict[str, Any]:
    return read_task_controls(controls_path(job_id))


def update_extraction_prompt_ab_job_controls(job_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    return merge_task_controls(controls_path(job_id), updates)


def list_extraction_prompt_ab_job_ids() -> list[str]:
    return list_task_job_ids(EXTRACTION_PROMPT_AB_JOBS_DIR)


def request_extraction_prompt_ab_stop(job_id: str) -> None:
    request_stop_file(stop_path(job_id))


def extraction_prompt_ab_stop_requested(job_id: str) -> bool:
    return stop_file_exists(stop_path(job_id))


def extraction_prompt_ab_job_stale_after_seconds(state: dict[str, Any]) -> float:
    config = state.get("config") if isinstance(state.get("config"), dict) else {}
    extraction = config.get("extraction_config") if isinstance(config.get("extraction_config"), dict) else {}
    evaluation = config.get("eval_config") if isinstance(config.get("eval_config"), dict) else {}
    comparison = config.get("comparison_config") if isinstance(config.get("comparison_config"), dict) else {}
    timeout = max(
        float(extraction.get("timeout") or 100),
        float(evaluation.get("judge_timeout") or 120),
        float(comparison.get("judge_timeout") or 120),
    )
    attempts = max(
        float(extraction.get("max_attempts") or 3),
        float(evaluation.get("judge_max_attempts") or 3),
        float(comparison.get("judge_max_attempts") or 3),
    )
    backoff = max(
        float(extraction.get("retry_sleep") or 15),
        float(evaluation.get("judge_qps_backoff") or 12),
        float(comparison.get("judge_qps_backoff") or 12),
    )
    return max(600.0, timeout * 2 + attempts * max(backoff, 5.0) + 180.0)


def extraction_prompt_ab_job_is_stale(state: dict[str, Any]) -> bool:
    if state.get("status") != "running":
        return False
    heartbeat = _parse_time(str(state.get("heartbeat_at") or state.get("updated_at") or ""))
    if heartbeat is None:
        return False
    if heartbeat.tzinfo is None:
        heartbeat = heartbeat.replace(tzinfo=utc_datetime().tzinfo)
    return (utc_datetime() - heartbeat).total_seconds() > extraction_prompt_ab_job_stale_after_seconds(state)


def mark_extraction_prompt_ab_job_interrupted(job_id: str) -> dict[str, Any]:
    state = read_extraction_prompt_ab_job_state(job_id)
    if not state or state.get("status") != "running":
        return state
    state.update(
        {
            "status": "interrupted",
            "stage": "已中断",
            "message": "提取提示词 A/B 任务长时间没有心跳，可能已随后台进程中断。",
            "finished_at": utc_now(),
            "updated_at": utc_now(),
        }
    )
    write_extraction_prompt_ab_job_state(job_id, state)
    return state


def extraction_prompt_ab_job_is_running(job_id: str) -> bool:
    state = read_extraction_prompt_ab_job_state(job_id)
    if extraction_prompt_ab_job_is_stale(state):
        mark_extraction_prompt_ab_job_interrupted(job_id)
        return False
    return state.get("status") == "running"


def _safe_config(config: ExtractionPromptAbJobConfig) -> dict[str, Any]:
    value = asdict(config)
    for key in (
        "prompt_a_text",
        "prompt_b_text",
        "prompt_a_create_text",
        "prompt_b_create_text",
        "judge_prompt_text",
        "evaluation_rule_prompt_text",
    ):
        value.pop(key, None)
    extraction = value.get("extraction_config")
    if isinstance(extraction, dict):
        extraction.pop("api_token", None)
        extraction["max_attempts"] = int(extraction.get("max_retries") or 0) + 1
    evaluation = value.get("eval_config")
    if isinstance(evaluation, dict):
        evaluation.pop("judge_api_bearer_token", None)
        evaluation["judge_max_attempts"] = int(evaluation.get("judge_max_retries") or 1)
    comparison = value.get("comparison_config")
    if isinstance(comparison, dict):
        comparison.pop("judge_api_bearer_token", None)
        comparison["judge_max_attempts"] = int(comparison.get("judge_max_retries") or 1)
    return value


def _write_state(
    config: ExtractionPromptAbJobConfig,
    *,
    stage: str,
    done: int,
    message: str,
    started_at: str,
    status: str = "running",
    phase_done: int | None = None,
    phase_total: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    state = {
        "job_id": config.job_id,
        "status": status,
        "stage": stage,
        "done": min(PROGRESS_TOTAL, max(0, int(done))),
        "total": PROGRESS_TOTAL,
        "phase_done": phase_done,
        "phase_total": phase_total,
        "message": message,
        "started_at": started_at,
        "updated_at": utc_now(),
        "config": _safe_config(config),
        "controls": read_extraction_prompt_ab_job_controls(config.job_id),
        "report_path": str(report_path(config.job_id)),
        "report_excel_path": str(report_excel_path(config.job_id)),
        "diff_excel_path": str(diff_excel_path(config.job_id)),
        "cases_a_path": str(cases_path(config.job_id, "A")),
        "cases_b_path": str(cases_path(config.job_id, "B")),
        "results_a_path": str(results_path(config.job_id, "A")),
        "results_b_path": str(results_path(config.job_id, "B")),
    }
    if extra:
        state.update(extra)
    write_extraction_prompt_ab_job_state(config.job_id, state)


def _current_controls(config: ExtractionPromptAbJobConfig) -> dict[str, Any]:
    return read_extraction_prompt_ab_job_controls(config.job_id)


def _current_priority(config: ExtractionPromptAbJobConfig) -> int:
    return control_priority(_current_controls(config))


def _current_extraction_concurrency(config: ExtractionPromptAbJobConfig) -> int:
    return control_int(
        _current_controls(config),
        "extraction_concurrency",
        min(100, max(1, int(config.extraction_config.concurrency or 1))),
        min_value=1,
        max_value=100,
    )


def _convert_extraction(
    config: ExtractionPromptAbJobConfig,
    label: str,
    output_path: Path,
    prompt_version: str,
) -> tuple[list[Case], list[Case], dict[str, Any]]:
    converter = (
        prepare_long_memory_cases_from_run_output
        if config.task_type == TaskType.LONG_MEMORY.value
        else prepare_cases_from_run_output
    )
    cases, missed, stats = converter(
        output_path,
        model=config.extraction_config.model or "unknown",
        prompt_version=prompt_version,
        chunk_size=max(1, int(config.chunk_size)),
        return_missed=True,
    )
    cases_to_jsonl(cases, str(cases_path(config.job_id, label)))
    cases_to_jsonl(missed, str(missed_cases_path(config.job_id, label)))
    return cases, missed, stats


def _run_extraction_side(
    config: ExtractionPromptAbJobConfig,
    *,
    label: str,
    prompt_text: str,
    create_prompt_text: str,
    prompt_version: str,
    progress_start: int,
    progress_end: int,
    started_at: str,
) -> tuple[list[Case], list[Case], dict[str, Any]]:
    if extraction_prompt_ab_stop_requested(config.job_id):
        raise ExtractionPromptAbStopped()
    side_config = replace(config.extraction_config)
    if str(side_config.prompt_cache_location or "none").lower() != "none":
        cache_text = "\n\n".join(filter(None, (create_prompt_text, prompt_text)))
        prompt_digest = hashlib.sha256(cache_text.encode("utf-8")).hexdigest()[:16]
        cache_prefix = str(side_config.prompt_cache_id or "memory_eval_extraction_ab").strip()
        side_config.prompt_cache_id = f"{cache_prefix}_{prompt_digest}"
    runner = MemoryExtractionRunner(
        config=side_config,
        prompt_text=prompt_text,
        task_type=TaskType(config.task_type),
        create_prompt_text=create_prompt_text or prompt_text,
        update_prompt_text=prompt_text,
    )
    output_path = extraction_path(config.job_id, label)

    def on_progress(done: int, total: int, message: str) -> None:
        fraction = done / total if total else 0.0
        weighted = progress_start + round((progress_end - progress_start) * fraction)
        _write_state(
            config,
            stage=f"提示词 {label}：记忆提取",
            done=weighted,
            phase_done=done,
            phase_total=total,
            message=f"提示词 {label}：{message}",
            started_at=started_at,
            extra={"current_side": label, "current_phase": "extraction"},
        )

    stats = runner.process_excel(
        config.input_path,
        output_path,
        sheet_name=config.sheet_name,
        reviewer_filter=config.reviewer_filter or None,
        chunk_size=max(1, int(config.chunk_size)),
        progress_callback=on_progress,
        should_stop=lambda: extraction_prompt_ab_stop_requested(config.job_id),
        emit_parallel_chunk_progress=True,
        priority_provider=lambda: _current_priority(config),
        concurrency_provider=lambda: _current_extraction_concurrency(config),
    )
    if stats.get("stopped") or extraction_prompt_ab_stop_requested(config.job_id):
        raise ExtractionPromptAbStopped()
    _write_state(
        config,
        stage=f"提示词 {label}：生成 case",
        done=progress_end,
        message=f"提示词 {label} 提取完成，正在生成同源评测 case。",
        started_at=started_at,
        extra={"current_side": label, "current_phase": "case_generation"},
    )
    cases, missed, case_stats = _convert_extraction(config, label, output_path, prompt_version)
    return cases, missed, {"extraction": stats, "case_generation": case_stats}


def _runtime_failure(runner: EvalRunner, case: Case, exc: Exception) -> EvalResult:
    input_hash = runner.case_input_hash(case)
    return EvalResult.from_parse_failure(
        case_id=case.case_id,
        task_type=case.task_type.value,
        raw=f"{type(exc).__name__}: {exc}",
        model_name=case.model_name,
        prompt_version=case.prompt_version,
        judge_model=runner.config.judge_model or "mock",
        judge_prompt_version=runner.resolved_judge_prompt_version,
        extraction_prompt_version=runner.extraction_prompt_version,
        extraction_prompt_hash=runner.extraction_prompt_hash,
        judge_prompt_hash=runner.judge_prompt_hash,
        scoring_schema_version=SCORING_SCHEMA_VERSION,
        dimension_weights_version=runner.dimension_weights_version,
        scoring_config_hash=runner.scoring_config_hash,
        case_input_hash=input_hash,
        evaluation_fingerprint=runner.evaluation_fingerprint(case),
    )


def _run_evaluation_side(
    config: ExtractionPromptAbJobConfig,
    *,
    label: str,
    cases: list[Case],
    progress_start: int,
    progress_end: int,
    started_at: str,
) -> list[EvalResult]:
    runner = EvalRunner(
        config=config.eval_config,
        task_type=TaskType(config.task_type),
        judge_prompt_version=config.judge_prompt_version,
        system_prompt_override=config.judge_prompt_text,
        extraction_prompt_text=config.evaluation_rule_prompt_text,
        extraction_prompt_version=config.evaluation_rule_prompt_version,
        extraction_prompt_hash=config.evaluation_rule_prompt_hash,
    )
    configured_concurrency = min(100, max(1, int(config.eval_config.judge_concurrency or 1)))
    configured_interval = float(config.eval_config.judge_request_interval or 0.0) if not config.eval_config.mock else 0.0
    backoff = float(config.eval_config.judge_qps_backoff or 0.0)
    rate_scope = api_rate_scope(config.eval_config.judge_api_base_url, config.eval_config.judge_api_bearer_token)

    def current_concurrency() -> int:
        return min(
            max(1, len(cases)),
            control_int(
                _current_controls(config),
                "judge_concurrency",
                configured_concurrency,
                min_value=1,
                max_value=100,
            ),
        )

    def current_interval() -> float:
        interval = control_float(
            _current_controls(config),
            "judge_request_interval",
            configured_interval,
            min_value=0.0,
            max_value=300.0,
        ) if not config.eval_config.mock else 0.0
        if current_concurrency() > 1 and not config.eval_config.mock:
            interval = max(interval, backoff)
        return interval

    def wait_for_rate_slot() -> None:
        wait_for_global_rate_slot(
            rate_scope,
            current_interval(),
            disabled=bool(config.eval_config.mock),
            should_stop=lambda: extraction_prompt_ab_stop_requested(config.job_id),
            priority=_current_priority(config),
        )

    if hasattr(runner.judge_client, "rate_limit_wait_callback"):
        runner.judge_client.rate_limit_wait_callback = wait_for_rate_slot

    output = results_path(config.job_id, label)
    results_to_jsonl([], str(output))
    results_by_index: dict[int, EvalResult] = {}
    case_iter = iter(enumerate(cases))
    futures: dict[Any, int] = {}
    completed = 0

    def evaluate_one(index: int, case: Case) -> tuple[int, EvalResult]:
        if extraction_prompt_ab_stop_requested(config.job_id):
            raise ExtractionPromptAbStopped()
        wait_for_rate_slot()
        if extraction_prompt_ab_stop_requested(config.job_id):
            raise ExtractionPromptAbStopped()
        try:
            return index, runner.evaluate_one(case)
        except Exception as exc:
            return index, _runtime_failure(runner, case, exc)

    def submit_next(executor: ThreadPoolExecutor) -> bool:
        if extraction_prompt_ab_stop_requested(config.job_id):
            return False
        try:
            index, case = next(case_iter)
        except StopIteration:
            return False
        futures[executor.submit(evaluate_one, index, case)] = index
        return True

    if cases:
        with ThreadPoolExecutor(max_workers=min(100, len(cases))) as executor:
            for _ in range(current_concurrency()):
                if not submit_next(executor):
                    break
            while futures:
                done_set, _ = wait(set(futures), return_when=FIRST_COMPLETED)
                for future in done_set:
                    fallback_index = futures.pop(future)
                    try:
                        index, result = future.result()
                    except ExtractionPromptAbStopped:
                        continue
                    except Exception as exc:
                        index = fallback_index
                        result = _runtime_failure(runner, cases[index], exc)
                    results_by_index[index] = result
                    append_result_to_jsonl(result, str(output))
                    completed += 1
                    fraction = completed / len(cases) if cases else 1.0
                    weighted = progress_start + round((progress_end - progress_start) * fraction)
                    _write_state(
                        config,
                        stage=f"提示词 {label}：Judge 评测",
                        done=weighted,
                        phase_done=completed,
                        phase_total=len(cases),
                        message=f"提示词 {label}：已评测 {completed}/{len(cases)} 个 case。",
                        started_at=started_at,
                        extra={
                            "current_side": label,
                            "current_phase": "evaluation",
                            "effective_judge_concurrency": current_concurrency(),
                            "effective_judge_request_interval": current_interval(),
                        },
                    )
                if extraction_prompt_ab_stop_requested(config.job_id):
                    for future in list(futures):
                        future.cancel()
                    break
                while len(futures) < current_concurrency() and submit_next(executor):
                    pass

    results = [results_by_index[index] for index in sorted(results_by_index)]
    results_to_jsonl(results, str(output))
    if extraction_prompt_ab_stop_requested(config.job_id):
        raise ExtractionPromptAbStopped()
    return results


def _run_comparison_model(
    config: ExtractionPromptAbJobConfig,
    report: dict[str, Any],
    *,
    prompt_a: str,
    prompt_b: str,
) -> dict[str, Any]:
    if not config.enable_model_comparison:
        return {
            "status": "skipped",
            "model": config.comparison_config.judge_model,
            "summary": "未启用独立对比总结模型。",
        }
    comparison_config = config.comparison_config
    errors = comparison_config.validate()
    if errors:
        return {
            "status": "failed",
            "model": comparison_config.judge_model,
            "preferred_version": "INSUFFICIENT",
            "confidence": "low",
            "summary": "对比模型配置不完整，统计结论不受影响。",
            "reasons": [],
            "risks": errors,
            "prompt_suggestions": [],
            "error": "；".join(errors),
        }

    rate_scope = api_rate_scope(
        comparison_config.judge_api_base_url,
        comparison_config.judge_api_bearer_token,
    )

    def wait_for_rate_slot() -> None:
        interval = control_float(
            _current_controls(config),
            "comparison_request_interval",
            float(comparison_config.judge_request_interval or 0.0),
            min_value=0.0,
            max_value=300.0,
        )
        wait_for_global_rate_slot(
            rate_scope,
            interval,
            disabled=bool(comparison_config.mock),
            should_stop=lambda: extraction_prompt_ab_stop_requested(config.job_id),
            priority=_current_priority(config),
        )

    result = call_comparison_model(
        comparison_config,
        report,
        prompt_a=prompt_a,
        prompt_b=prompt_b,
        max_evidence=max(1, int(config.comparison_max_evidence or 8)),
        rate_limit_wait_callback=wait_for_rate_slot,
        should_stop=lambda: extraction_prompt_ab_stop_requested(config.job_id),
    )
    if result.get("status") == "stopped":
        raise ExtractionPromptAbStopped()
    return result


def _write_report_excel(report: dict[str, Any], path: Path) -> None:
    model_roles = report.get("model_roles") if isinstance(report.get("model_roles"), dict) else {}
    model_comparison = (
        report.get("model_comparison")
        if isinstance(report.get("model_comparison"), dict)
        else {}
    )
    summary = {
        "recommendation": report.get("recommendation"),
        "recommendation_reason": report.get("recommendation_reason"),
        "extraction_model": model_roles.get("extraction_model", ""),
        "evaluation_model": model_roles.get("evaluation_model", ""),
        "comparison_model": model_roles.get("comparison_model", ""),
        "comparison_model_status": model_comparison.get("status", ""),
        "comparison_model_preference": model_comparison.get("preferred_version", ""),
        "comparison_model_summary": model_comparison.get("summary", ""),
        "identical_output_count": report.get("identical_output_count", 0),
        "judge_disagreement_on_identical_output_count": report.get(
            "judge_disagreement_on_identical_output_count", 0
        ),
        **{f"A_{key}": value for key, value in (report.get("quality_a") or {}).items()},
        **{f"B_{key}": value for key, value in (report.get("quality_b") or {}).items()},
    }
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame([summary]).to_excel(writer, sheet_name="结论", index=False)
        pd.DataFrame(report.get("dimension_summary") or []).to_excel(writer, sheet_name="维度对比", index=False)
        pd.DataFrame(report.get("rows") or []).to_excel(writer, sheet_name="逐样本对比", index=False)
        if model_comparison:
            comparison_sheet = {
                key: "；".join(value) if isinstance(value, list) else value
                for key, value in model_comparison.items()
            }
            pd.DataFrame([comparison_sheet]).to_excel(writer, sheet_name="模型综合意见", index=False)
    atomic_write_bytes(path, buffer.getvalue())


def load_extraction_prompt_ab_report(job_id: str) -> dict[str, Any]:
    path = report_path(job_id)
    return read_json_state(path) if path.exists() else {}


def ensure_extraction_prompt_ab_diff_excel(job_id: str) -> Path | None:
    output = diff_excel_path(job_id)
    if output.exists():
        return output
    extraction_a = extraction_path(job_id, "A")
    extraction_b = extraction_path(job_id, "B")
    report = load_extraction_prompt_ab_report(job_id)
    if not extraction_a.exists() or not extraction_b.exists() or not report:
        return None
    with state_file_lock(output):
        if not output.exists():
            write_extraction_prompt_diff_excel(
                extraction_a_path=extraction_a,
                extraction_b_path=extraction_b,
                comparison_rows=report.get("rows") or [],
                output_path=output,
                model_comparison=report.get("model_comparison") or None,
            )
    return output


def load_extraction_prompt_ab_side(
    job_id: str, label: str
) -> tuple[list[Case], list[Case], list[EvalResult]]:
    ready = cases_from_jsonl(str(cases_path(job_id, label))) if cases_path(job_id, label).exists() else []
    missed = cases_from_jsonl(str(missed_cases_path(job_id, label))) if missed_cases_path(job_id, label).exists() else []
    results = results_from_jsonl(str(results_path(job_id, label))) if results_path(job_id, label).exists() else []
    return ready, missed, results


def run_extraction_prompt_ab_job(config: ExtractionPromptAbJobConfig) -> None:
    started_at = utc_now()
    directory = job_dir(config.job_id)
    directory.mkdir(parents=True, exist_ok=True)
    if stop_path(config.job_id).exists():
        stop_path(config.job_id).unlink()
    init_task_controls(
        controls_path(config.job_id),
        {
            "priority": DEFAULT_PRIORITY,
            "extraction_concurrency": min(100, max(1, int(config.extraction_config.concurrency or 1))),
            "judge_concurrency": min(100, max(1, int(config.eval_config.judge_concurrency or 1))),
            "judge_request_interval": float(config.eval_config.judge_request_interval or 0.0),
            "comparison_request_interval": float(
                config.comparison_config.judge_request_interval or 0.0
            ),
        },
    )
    _write_state(
        config,
        stage="准备",
        done=0,
        message="正在准备同一数据、同一模型配置下的提取提示词 A/B 实验。",
        started_at=started_at,
    )

    try:
        cases_a, missed_a, stats_a = _run_extraction_side(
            config,
            label="A",
            prompt_text=config.prompt_a_text,
            create_prompt_text=config.prompt_a_create_text,
            prompt_version=config.prompt_a_version,
            progress_start=0,
            progress_end=200,
            started_at=started_at,
        )
        results_a = _run_evaluation_side(
            config,
            label="A",
            cases=cases_a,
            progress_start=220,
            progress_end=450,
            started_at=started_at,
        )
        cases_b, missed_b, stats_b = _run_extraction_side(
            config,
            label="B",
            prompt_text=config.prompt_b_text,
            create_prompt_text=config.prompt_b_create_text,
            prompt_version=config.prompt_b_version,
            progress_start=450,
            progress_end=650,
            started_at=started_at,
        )
        results_b = _run_evaluation_side(
            config,
            label="B",
            cases=cases_b,
            progress_start=670,
            progress_end=900,
            started_at=started_at,
        )

        _write_state(
            config,
            stage="计算统计结论",
            done=920,
            message="正在按评测人、session 和 chunk 配对，并计算置信区间与退化项。",
            started_at=started_at,
        )
        combined_prompt_a = "\n\n".join(
            filter(None, (config.prompt_a_create_text, config.prompt_a_text))
        )
        combined_prompt_b = "\n\n".join(
            filter(None, (config.prompt_b_create_text, config.prompt_b_text))
        )
        report = compare_extraction_prompt_runs(
            cases_a=cases_a,
            cases_b=cases_b,
            missed_cases_a=missed_a,
            missed_cases_b=missed_b,
            results_a=results_a,
            results_b=results_b,
            prompt_a=combined_prompt_a,
            prompt_b=combined_prompt_b,
            validation_config=config.validation_config,
            score_tolerance=config.score_tolerance,
        )
        report["model_roles"] = {
            "extraction_model": config.extraction_config.model,
            "evaluation_model": config.eval_config.judge_model,
            "comparison_model": (
                config.comparison_config.judge_model
                if config.enable_model_comparison
                else ""
            ),
        }
        _write_state(
            config,
            stage="生成模型综合意见",
            done=950,
            message="统计结论已完成，正在调用独立对比模型生成补充说明。",
            started_at=started_at,
            extra={"model_roles": report["model_roles"]},
        )
        report["model_comparison"] = _run_comparison_model(
            config,
            report,
            prompt_a=combined_prompt_a,
            prompt_b=combined_prompt_b,
        )
        atomic_write_json(report_path(config.job_id), report)
        _write_report_excel(report, report_excel_path(config.job_id))
        write_extraction_prompt_diff_excel(
            extraction_a_path=extraction_path(config.job_id, "A"),
            extraction_b_path=extraction_path(config.job_id, "B"),
            comparison_rows=report.get("rows") or [],
            output_path=diff_excel_path(config.job_id),
            model_comparison=report.get("model_comparison") or None,
        )
        _write_state(
            config,
            status="completed",
            stage="完成",
            done=PROGRESS_TOTAL,
            message=f"A/B 比较完成：{report.get('recommendation', '已生成结论')}。",
            started_at=started_at,
            extra={
                "recommendation": report.get("recommendation"),
                "recommendation_reason": report.get("recommendation_reason"),
                "quality_a": report.get("quality_a"),
                "quality_b": report.get("quality_b"),
                "validation_gate": report.get("validation_gate"),
                "winner_counts": report.get("winner_counts"),
                "diff_excel_path": str(diff_excel_path(config.job_id)),
                "stats_a": stats_a,
                "stats_b": stats_b,
                "finished_at": utc_now(),
            },
        )
    except ExtractionPromptAbStopped:
        state = read_extraction_prompt_ab_job_state(config.job_id)
        _write_state(
            config,
            status="stopped",
            stage="已终止",
            done=int(state.get("done", 0) or 0),
            message="提取提示词 A/B 任务已按终止请求停止，已完成的中间文件会保留。",
            started_at=started_at,
            extra={"finished_at": utc_now()},
        )
    except Exception as exc:
        state = read_extraction_prompt_ab_job_state(config.job_id)
        _write_state(
            config,
            status="failed",
            stage="失败",
            done=int(state.get("done", 0) or 0),
            message=f"提取提示词 A/B 失败：{type(exc).__name__}: {exc}",
            started_at=started_at,
            extra={
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "finished_at": utc_now(),
            },
        )
