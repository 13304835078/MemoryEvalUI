from __future__ import annotations

import json
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.eval.eval_runner import EvalRunner, SCORING_SCHEMA_VERSION
from src.eval.metrics import compute_aggregations, flatten_results
from src.eval.result_status import result_is_score_eligible
from src.extraction.memory_extractor import (
    MemoryExtractionConfig,
    MemoryExtractionRunner,
    sanitize_filename,
)
from src.runtime_paths import APP_HOME, DATA_DIR
from src.persistence import atomic_write_text
from src.schema import EvalConfig, EvalResult, TaskType, append_result_to_jsonl, results_to_jsonl
from src.ui.data_service import (
    prepare_cases_from_run_output,
    prepare_long_memory_cases_from_run_output,
    save_cases,
)
from src.ui.background_tasks import read_json_state
from src.ui.global_rate_limiter import api_rate_scope, set_current_task_priority, wait_for_global_rate_slot
from src.ui.prompt_advisor import call_prompt_advisor, collect_absolute_eval_evidence
from src.ui.prompt_editor import prompt_text_hash, save_prompt_version
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


PROJECT_ROOT = APP_HOME
CLOSED_LOOP_DIR = DATA_DIR / "closed_loop"


@dataclass
class ClosedLoopConfig:
    run_id: str
    input_excel_path: str
    sheet_name: str | int | None = 0
    reviewer_filter: str = ""
    rounds: int = 3
    chunk_size: int = 10
    max_cases_per_round: int = 0
    task_type: str = TaskType.USER_MD.value
    protocol_version: str = "v1_exploratory"
    discovery_ratio: float = 0.6
    validation_ratio: float = 0.2
    locked_test_ratio: float = 0.2
    split_seed: str = "memory-eval-v1"
    validation_min_score_delta: float = 0.03
    validation_min_end_to_end_delta: float = 0.0
    validation_max_coverage_drop: float = 0.005
    validation_max_case_regression_rate: float = 0.1
    validation_max_prompt_growth_ratio: float = 0.1
    validation_min_paired_cases: int = 8
    validation_min_paired_clusters: int = 2
    validation_confidence_level: float = 0.95
    validation_bootstrap_samples: int = 2000

    extraction_model: str = ""
    extraction_api_base: str = ""
    extraction_api_token: str = ""
    extraction_prompt_text: str = ""
    extraction_create_prompt_text: str = ""
    extraction_prompt_version: str = ""
    evaluation_rule_prompt_text: str = ""
    evaluation_rule_prompt_version: str = ""
    extraction_max_tokens: int = 50000
    extraction_request_interval: float = 10.0
    extraction_max_retries: int = 2
    extraction_retry_sleep: float = 15.0
    extraction_timeout: int = 120
    extraction_concurrency: int = 1
    extraction_send_enable_thinking: bool = True
    extraction_enable_thinking: bool = True

    judge_prompt_file: str = ""
    judge_prompt_text: str = ""
    judge_prompt_version: str = ""
    advisor_max_items: int = 40
    advisor_model: str = ""
    advisor_api_base: str = ""
    advisor_api_token: str = ""

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


def run_dir(run_id: str) -> Path:
    return CLOSED_LOOP_DIR / run_id


def state_path(run_id: str) -> Path:
    return run_dir(run_id) / "state.json"


def stop_path(run_id: str) -> Path:
    return run_dir(run_id) / "STOP"


def controls_path(run_id: str) -> Path:
    return run_dir(run_id) / "controls.json"


def read_loop_state(run_id: str) -> dict[str, Any]:
    return read_json_state(state_path(run_id))


def write_loop_state(run_id: str, state: dict[str, Any]) -> None:
    path = state_path(run_id)
    state["heartbeat_at"] = utc_now()
    atomic_write_json(path, state)


def read_loop_controls(run_id: str) -> dict[str, Any]:
    return read_task_controls(controls_path(run_id))


def update_loop_controls(run_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    return merge_task_controls(controls_path(run_id), updates)


def request_stop(run_id: str) -> None:
    path = stop_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, utc_now())


def stop_requested(run_id: str) -> bool:
    return stop_path(run_id).exists()


def loop_stale_after_seconds(state: dict[str, Any]) -> float:
    config = state.get("config") or {}
    eval_config = config.get("eval_config") if isinstance(config.get("eval_config"), dict) else {}
    judge_timeout = float(eval_config.get("judge_timeout") or 120)
    judge_retries = float(eval_config.get("judge_max_retries") or 3)
    judge_backoff = float(eval_config.get("judge_qps_backoff") or 12)
    extraction_timeout = float(config.get("extraction_timeout") or judge_timeout)
    extraction_retries = float(config.get("extraction_max_retries") or max(0, judge_retries - 1))
    extraction_backoff = float(config.get("extraction_retry_sleep") or judge_backoff)
    longest_call = max(
        judge_timeout * 2 + judge_retries * max(judge_backoff, 5.0),
        extraction_timeout * 2 + extraction_retries * max(extraction_backoff, 5.0),
    )
    return max(300.0, longest_call + 120.0)


def loop_state_is_stale(state: dict[str, Any]) -> bool:
    if state.get("status") != "running":
        return False
    heartbeat = _parse_time(str(state.get("heartbeat_at") or state.get("updated_at") or ""))
    if heartbeat is None:
        return False
    if heartbeat.tzinfo is None:
        heartbeat = heartbeat.replace(tzinfo=timezone.utc)
    elapsed = (datetime.now(timezone.utc) - heartbeat).total_seconds()
    return elapsed > loop_stale_after_seconds(state)


def mark_loop_interrupted(run_id: str) -> dict[str, Any]:
    state = read_loop_state(run_id)
    if not state or state.get("status") != "running":
        return state
    state["status"] = "interrupted"
    state["stage"] = "已中断"
    state["finished_at"] = utc_now()
    append_event(state, "后台闭环任务可能已中断：长时间没有心跳。", "warning")
    write_loop_state(run_id, state)
    return state


def loop_is_running(run_id: str) -> bool:
    state = read_loop_state(run_id)
    if loop_state_is_stale(state):
        mark_loop_interrupted(run_id)
        return False
    return state.get("status") == "running"


def append_event(state: dict[str, Any], message: str, level: str = "info") -> None:
    events = state.setdefault("events", [])
    events.append({
        "time": utc_now(),
        "level": level,
        "message": message,
    })
    del events[:-200]


def update_state(run_id: str, updater: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    with state_file_lock(state_path(run_id)):
        state = read_loop_state(run_id)
        updater(state)
        controls = read_loop_controls(run_id)
        if controls:
            state["controls"] = controls
        state["updated_at"] = utc_now()
        write_loop_state(run_id, state)
        return state


def _make_initial_state(config: ClosedLoopConfig) -> dict[str, Any]:
    safe_config = asdict(config)
    safe_config.pop("extraction_api_token", None)
    safe_config.pop("advisor_api_token", None)
    safe_config.pop("extraction_prompt_text", None)
    safe_config.pop("extraction_create_prompt_text", None)
    safe_config.pop("evaluation_rule_prompt_text", None)
    safe_config.pop("judge_prompt_text", None)
    if isinstance(safe_config.get("eval_config"), dict):
        safe_config["eval_config"].pop("judge_api_bearer_token", None)
        safe_config["eval_config"]["judge_max_attempts"] = int(
            safe_config["eval_config"].get("judge_max_retries") or 1
        )
    safe_config["extraction_max_attempts"] = int(safe_config.get("extraction_max_retries") or 0) + 1
    controls = read_loop_controls(config.run_id)
    return {
        "run_id": config.run_id,
        "status": "running",
        "stage": "初始化",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "config": safe_config,
        "controls": controls,
        "rounds": [],
        "events": [],
    }


def _advisor_eval_config(config: ClosedLoopConfig) -> EvalConfig:
    return replace(
        config.eval_config,
        judge_model=config.advisor_model or config.eval_config.judge_model,
        judge_api_base_url=config.advisor_api_base or config.eval_config.judge_api_base_url,
        judge_api_bearer_token=config.advisor_api_token or config.eval_config.judge_api_bearer_token,
    )


def _check_stop(run_id: str, message: str = "收到终止请求") -> None:
    if stop_requested(run_id):
        raise StopIteration(message)


def _current_loop_controls(config: ClosedLoopConfig) -> dict[str, Any]:
    return read_loop_controls(config.run_id)


def _current_priority(config: ClosedLoopConfig) -> int:
    return control_priority(_current_loop_controls(config))


def _current_target_rounds(config: ClosedLoopConfig) -> int:
    controls = _current_loop_controls(config)
    return control_int(
        controls,
        "target_rounds",
        int(config.rounds or 1),
        min_value=1,
        max_value=max(1, int(config.rounds or 1)),
    )


def _current_extraction_concurrency(config: ClosedLoopConfig) -> int:
    return control_int(
        _current_loop_controls(config),
        "extraction_concurrency",
        min(100, max(1, int(config.extraction_concurrency or 1))),
        min_value=1,
        max_value=100,
    )


def _current_judge_concurrency(config: ClosedLoopConfig, *, limit: int) -> int:
    return min(limit, control_int(
        _current_loop_controls(config),
        "judge_concurrency",
        min(100, max(1, int(getattr(config.eval_config, "judge_concurrency", 1) or 1))),
        min_value=1,
        max_value=100,
    ))


def _current_judge_interval(config: ClosedLoopConfig, *, concurrency: int) -> float:
    configured = float(getattr(config.eval_config, "judge_request_interval", 0.0) or 0.0)
    if config.eval_config.mock:
        return 0.0
    value = control_float(
        _current_loop_controls(config),
        "judge_request_interval",
        configured,
        min_value=0.0,
        max_value=300.0,
    )
    if concurrency > 1:
        value = max(value, float(getattr(config.eval_config, "judge_qps_backoff", 0.0) or 0.0))
    return value


def _round_record(state: dict[str, Any], round_index: int) -> dict[str, Any]:
    rounds = state.setdefault("rounds", [])
    while len(rounds) < round_index:
        rounds.append({"round": len(rounds) + 1})
    return rounds[round_index - 1]


def _save_candidate_prompt(
    candidate_prompt: str,
    round_index: int,
    task_type: TaskType,
) -> tuple[str, str]:
    task_slug = "long_memory" if task_type == TaskType.LONG_MEMORY else "user_md"
    version_name = (
        f"extract_{task_slug}_closed_loop_round_{round_index}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    saved = save_prompt_version(task_type.value, candidate_prompt, version_name, prompt_kind="extraction")
    return saved, candidate_prompt


def _classify_no_candidate_reason(
    *,
    results: list[EvalResult],
    evidence: list[dict[str, Any]],
    advisor_result: dict[str, Any] | None,
) -> dict[str, Any]:
    advisor_result = advisor_result or {}
    total = len(results)
    runtime_failure_count = sum(1 for result in results if not result_is_score_eligible(result))
    valid_count = max(0, total - runtime_failure_count)
    runtime_failure_rate = (runtime_failure_count / total) if total else 0.0
    issue_evidence_count = sum(
        1
        for item in evidence
        if item.get("evidence_mode") == "issue_or_low_score"
    )
    weak_context_count = sum(1 for item in evidence if item.get("evidence_mode") == "weak_context_from_result")
    risks = [str(item) for item in (advisor_result.get("risks") or []) if str(item).strip()]
    source = str(advisor_result.get("candidate_prompt_source") or "")
    error = str(advisor_result.get("error") or "")
    can_suggest = advisor_result.get("can_suggest")

    base = {
        "total_results": total,
        "valid_results": valid_count,
        "runtime_failure_results": runtime_failure_count,
        "runtime_failure_rate": round(runtime_failure_rate, 4),
        "evidence_count": len(evidence),
        "issue_evidence_count": issue_evidence_count,
        "weak_context_count": weak_context_count,
        "candidate_prompt_source": source,
        "can_suggest": can_suggest,
        "risks": risks[:8],
        "error": error,
    }

    if total == 0:
        return {
            **base,
            "category": "no_eval_results",
            "status": "paused_no_evidence",
            "stage": "无评测结果",
            "title": "没有可用于提示词改进的评测结果",
            "message": "本轮没有生成评测结果，不能判断是否需要修改提取提示词。",
        }

    if valid_count == 0 or (runtime_failure_rate >= 0.5 and issue_evidence_count == 0):
        return {
            **base,
            "category": "eval_chain_failed",
            "status": "paused_eval_failed",
            "stage": "评测链路失败",
            "title": "评测链路失败，未生成候选提示词",
            "message": "本轮有效 Judge 结果不足，主要证据是调用失败或 JSON 解析失败；这些不能作为修改提取提示词的依据。",
        }

    if issue_evidence_count == 0:
        return {
            **base,
            "category": "no_change_needed",
            "status": "completed_no_change",
            "stage": "无需修改提示词",
            "title": "本轮未发现需要自动修改提取提示词的稳定证据",
            "message": "有效评测结果没有低分、错误标签或 diagnostics 形成的明确问题证据；为了避免过拟合，本轮不生成候选提示词。",
        }

    if error or can_suggest is False:
        return {
            **base,
            "category": "advisor_failed",
            "status": "paused_advisor_failed",
            "stage": "提示词建议失败",
            "title": "提示词建议模型失败，未生成候选提示词",
            "message": "本轮存在有效问题证据，但提示词建议模型没有返回可解析、可应用的修改建议。",
        }

    if source == "no_valid_incremental_patch":
        return {
            **base,
            "category": "no_safe_patch",
            "status": "paused_no_safe_patch",
            "stage": "无可安全应用的修改",
            "title": "没有可安全应用的增量修改",
            "message": "本轮存在有效问题证据，但候选 patch 未通过校验或会导致提示词膨胀；系统没有自动采用。",
        }

    return {
        **base,
        "category": "no_candidate_unknown",
        "status": "paused_no_candidate",
        "stage": "未生成候选提示词",
        "title": "未生成候选提示词",
        "message": "本轮没有得到候选提示词；请查看提示词建议原始结果判断是证据不足、无需修改，还是建议模型输出不合格。",
    }


def _evaluate_cases(
    config: ClosedLoopConfig,
    cases,
    round_index: int,
    current_prompt_text: str,
    current_prompt_version: str,
    result_path: Path,
) -> list[EvalResult]:
    task_type = TaskType(config.task_type)
    evaluation_rule_prompt = config.evaluation_rule_prompt_text or config.extraction_prompt_text
    evaluation_rule_version = (
        config.evaluation_rule_prompt_version
        or config.extraction_prompt_version
        or "initial_extraction_prompt"
    )
    runner = EvalRunner(
        config=config.eval_config,
        task_type=task_type,
        prompt_file=config.judge_prompt_file,
        judge_prompt_version=config.judge_prompt_version,
        system_prompt_override=config.judge_prompt_text,
        extraction_prompt_text=evaluation_rule_prompt,
        extraction_prompt_version=evaluation_rule_version,
        extraction_prompt_hash=prompt_text_hash(evaluation_rule_prompt),
    )

    run_cases = cases[: config.max_cases_per_round] if config.max_cases_per_round and config.max_cases_per_round > 0 else cases
    results_by_index: dict[int, EvalResult] = {}
    result_path.parent.mkdir(parents=True, exist_ok=True)

    if not run_cases:
        results_to_jsonl([], str(result_path))
        return []
    results_to_jsonl([], str(result_path))

    update_state(config.run_id, lambda state: (
        state.update({"stage": f"第 {round_index} 轮：评测 0/{len(run_cases)}"}),
        _round_record(state, round_index).update({
            "eval_progress": f"0/{len(run_cases)}",
            "eval_concurrency": _current_judge_concurrency(config, limit=len(run_cases)),
        }),
    ))

    rate_scope = api_rate_scope(
        config.eval_config.judge_api_base_url,
        config.eval_config.judge_api_bearer_token,
    )

    def wait_for_rate_slot() -> None:
        concurrency = _current_judge_concurrency(config, limit=len(run_cases))
        wait_for_global_rate_slot(
            rate_scope,
            _current_judge_interval(config, concurrency=concurrency),
            disabled=bool(config.eval_config.mock),
            should_stop=lambda: stop_requested(config.run_id),
            priority=_current_priority(config),
        )
        _check_stop(config.run_id, "评测阶段收到终止请求")

    if hasattr(runner.judge_client, "rate_limit_wait_callback"):
        runner.judge_client.rate_limit_wait_callback = wait_for_rate_slot

    def evaluate_task(idx: int, case) -> tuple[int, EvalResult]:
        _check_stop(config.run_id, "评测阶段收到终止请求")
        wait_for_rate_slot()
        _check_stop(config.run_id, "评测阶段收到终止请求")
        return idx, runner.evaluate_one(case)

    case_iter = iter(enumerate(run_cases, start=1))
    futures = {}
    completed = 0

    def submit_next(executor: ThreadPoolExecutor) -> bool:
        try:
            idx, case = next(case_iter)
        except StopIteration:
            return False
        _check_stop(config.run_id, "评测阶段收到终止请求")
        futures[executor.submit(evaluate_task, idx, case)] = (idx, case)
        return True

    try:
        with ThreadPoolExecutor(max_workers=min(100, max(1, len(run_cases)))) as executor:
            for _ in range(_current_judge_concurrency(config, limit=len(run_cases))):
                if not submit_next(executor):
                    break

            while futures:
                _check_stop(config.run_id, "评测阶段收到终止请求")
                done, _ = wait(futures, timeout=1.0, return_when=FIRST_COMPLETED)
                if not done:
                    continue

                for future in done:
                    idx, case = futures.pop(future)
                    try:
                        _, result = future.result()
                    except StopIteration:
                        raise
                    except Exception as exc:
                        result = EvalResult.from_parse_failure(
                            case_id=case.case_id,
                            task_type=task_type.value,
                            raw=f"{type(exc).__name__}: {exc}",
                            model_name=case.model_name,
                            prompt_version=case.prompt_version,
                            judge_model=config.eval_config.judge_model or "mock",
                            judge_prompt_version=runner.resolved_judge_prompt_version,
                            extraction_prompt_version=evaluation_rule_version,
                            extraction_prompt_hash=prompt_text_hash(evaluation_rule_prompt),
                            judge_prompt_hash=runner.judge_prompt_hash,
                            scoring_schema_version=SCORING_SCHEMA_VERSION,
                            dimension_weights_version=runner.dimension_weights_version,
                            scoring_config_hash=runner.scoring_config_hash,
                            case_input_hash=runner.case_input_hash(case),
                            evaluation_fingerprint=runner.evaluation_fingerprint(case),
                        )

                    results_by_index[idx] = result
                    completed += 1
                    append_result_to_jsonl(result, str(result_path))
                    update_state(config.run_id, lambda state: (
                        state.update({"stage": f"第 {round_index} 轮：评测 {completed}/{len(run_cases)}"}),
                        _round_record(state, round_index).update({
                            "eval_progress": f"{completed}/{len(run_cases)}",
                            "eval_concurrency": _current_judge_concurrency(config, limit=len(run_cases)),
                            "latest_message": f"已完成评测：{case.case_id}",
                        }),
                    ))
                while len(futures) < _current_judge_concurrency(config, limit=len(run_cases)):
                    if not submit_next(executor):
                        break
    except StopIteration:
        for future in futures:
            future.cancel()
        raise

    ordered_results = [results_by_index[i] for i in sorted(results_by_index)]
    results_to_jsonl(ordered_results, str(result_path))
    return ordered_results


def _run_exploratory_closed_loop(config: ClosedLoopConfig) -> None:
    loop_dir = run_dir(config.run_id)
    loop_dir.mkdir(parents=True, exist_ok=True)
    if stop_path(config.run_id).exists():
        stop_path(config.run_id).unlink()
    init_task_controls(controls_path(config.run_id), {
        "priority": DEFAULT_PRIORITY,
        "target_rounds": int(config.rounds or 1),
        "extraction_concurrency": min(100, max(1, int(config.extraction_concurrency or 1))),
        "judge_concurrency": min(100, max(1, int(getattr(config.eval_config, "judge_concurrency", 1) or 1))),
        "judge_request_interval": float(getattr(config.eval_config, "judge_request_interval", 0.0) or 0.0),
    })

    state = _make_initial_state(config)
    evaluation_rule_prompt = config.evaluation_rule_prompt_text or config.extraction_prompt_text
    evaluation_rule_version = (
        config.evaluation_rule_prompt_version
        or config.extraction_prompt_version
        or "initial_extraction_prompt"
    )
    state["evaluation_contract"] = {
        "frozen": True,
        "version": evaluation_rule_version,
        "prompt_hash": prompt_text_hash(evaluation_rule_prompt),
        "description": "所有轮次均按实验启动时冻结的提取规则评测；候选提示词只负责生成输出。",
    }
    append_event(state, "闭环实验启动")
    append_event(state, f"评测规则已冻结：{evaluation_rule_version}")
    write_loop_state(config.run_id, state)

    current_prompt_text = config.extraction_prompt_text
    current_create_prompt_text = config.extraction_create_prompt_text or current_prompt_text
    current_prompt_version = config.extraction_prompt_version or "initial_extraction_prompt"
    task_type = TaskType(config.task_type)
    final_status = "completed"

    try:
        if not current_prompt_text.strip():
            raise ValueError("初始提取提示词为空")

        round_index = 1
        while round_index <= _current_target_rounds(config):
            _check_stop(config.run_id)
            round_dir = loop_dir / f"round_{round_index:02d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            prompt_hash = prompt_text_hash(current_prompt_text)

            update_state(config.run_id, lambda state: (
                state.update({"stage": f"第 {round_index} 轮：开始"}),
                _round_record(state, round_index).update({
                    "round": round_index,
                    "status": "running",
                    "prompt_version": current_prompt_version,
                    "prompt_hash": prompt_hash[:12],
                    "started_at": utc_now(),
                }),
                append_event(state, f"第 {round_index} 轮开始，提取提示词={current_prompt_version}"),
            ))

            extraction_config = MemoryExtractionConfig.from_eval_config(
                config.eval_config,
                model=config.extraction_model,
                max_tokens=config.extraction_max_tokens,
                request_interval=config.extraction_request_interval,
                max_retries=config.extraction_max_retries,
                retry_sleep=config.extraction_retry_sleep,
                enable_thinking=config.extraction_enable_thinking,
                timeout=config.extraction_timeout,
            )
            extraction_config.api_base = config.extraction_api_base or extraction_config.api_base
            extraction_config.api_token = config.extraction_api_token or extraction_config.api_token
            extraction_config.send_enable_thinking = config.extraction_send_enable_thinking
            extraction_config.concurrency = _current_extraction_concurrency(config)
            extraction_config.priority = _current_priority(config)
            extraction_output = round_dir / f"memory_extract_round_{round_index:02d}.xlsx"

            def extraction_progress(done: int, total: int, message: str) -> None:
                update_state(config.run_id, lambda state: (
                    state.update({"stage": f"第 {round_index} 轮：记忆提取"}),
                    _round_record(state, round_index).update({
                        "extraction_progress": f"{done}/{total}",
                        "latest_message": message,
                    }),
                ))

            runner = MemoryExtractionRunner(
                extraction_config,
                current_prompt_text,
                task_type=task_type,
                create_prompt_text=current_create_prompt_text,
                update_prompt_text=current_prompt_text,
            )
            extraction_stats = runner.process_excel(
                config.input_excel_path,
                extraction_output,
                sheet_name=config.sheet_name,
                reviewer_filter=config.reviewer_filter or None,
                chunk_size=config.chunk_size,
                progress_callback=extraction_progress,
                should_stop=lambda: stop_requested(config.run_id),
                emit_parallel_chunk_progress=True,
                priority_provider=lambda: _current_priority(config),
                concurrency_provider=lambda: _current_extraction_concurrency(config),
            )
            if extraction_stats.get("stopped"):
                raise StopIteration("记忆提取阶段收到终止请求")

            update_state(config.run_id, lambda state: (
                _round_record(state, round_index).update({
                    "extraction_output": str(extraction_output),
                    "extraction_stats": extraction_stats,
                    "latest_message": f"记忆提取完成：{extraction_output}",
                }),
                append_event(state, f"第 {round_index} 轮记忆提取完成：{extraction_output}"),
            ))

            _check_stop(config.run_id)
            update_state(config.run_id, lambda state: (
                state.update({"stage": f"第 {round_index} 轮：生成 case"}),
                _round_record(state, round_index).update({
                    "latest_message": "正在把提取结果转换为评测 case",
                }),
            ))
            converter = (
                prepare_long_memory_cases_from_run_output
                if task_type == TaskType.LONG_MEMORY
                else prepare_cases_from_run_output
            )
            cases, missed_cases, convert_stats = converter(
                extraction_output,
                model=config.extraction_model or "unknown",
                prompt_version=current_prompt_version,
                chunk_size=config.chunk_size,
                return_missed=True,
            )
            cases_path = save_cases(
                cases,
                f"closed_loop_{sanitize_filename(config.run_id)}_round_{round_index:02d}_cases.jsonl",
            )
            missed_path = ""
            if missed_cases:
                missed_path = save_cases(
                    missed_cases,
                    f"closed_loop_{sanitize_filename(config.run_id)}_round_{round_index:02d}_missed_cases.jsonl",
                )

            update_state(config.run_id, lambda state: (
                state.update({"stage": f"第 {round_index} 轮：生成 case"}),
                _round_record(state, round_index).update({
                    "cases_path": cases_path,
                    "missed_cases_path": missed_path,
                    "case_stats": convert_stats,
                    "latest_message": f"case 生成完成：完整 {len(cases)} 条，漏抽 {len(missed_cases)} 条",
                }),
                append_event(state, f"第 {round_index} 轮生成 case：{len(cases)} 条"),
            ))

            _check_stop(config.run_id)
            result_path = round_dir / f"eval_results_round_{round_index:02d}.jsonl"
            results = _evaluate_cases(
                config,
                cases,
                round_index,
                current_prompt_text,
                current_prompt_version,
                result_path,
            )
            stats = compute_aggregations(results)
            preview_rows = flatten_results(results[:20])
            update_state(config.run_id, lambda state: (
                state.update({"stage": f"第 {round_index} 轮：评测完成"}),
                _round_record(state, round_index).update({
                    "results_path": str(result_path),
                    "eval_stats": stats,
                    "eval_preview": preview_rows,
                }),
                append_event(state, f"第 {round_index} 轮评测完成，平均分={stats.get('avg_score_total', 0)}"),
            ))

            _check_stop(config.run_id)
            evidence = collect_absolute_eval_evidence(
                results,
                max_items=config.advisor_max_items,
                include_all=True,
            )
            update_state(config.run_id, lambda state: (
                state.update({"stage": f"第 {round_index} 轮：生成候选提取提示词"}),
                _round_record(state, round_index).update({
                    "advisor_evidence_count": len(evidence),
                    "latest_message": f"正在生成提示词改进建议，输入证据 {len(evidence)} 条",
                }),
            ))
            set_current_task_priority(_current_priority(config))
            advisor_result, raw = call_prompt_advisor(
                _advisor_eval_config(config),
                evidence=evidence,
                current_judge_prompt=config.judge_prompt_text,
                extraction_prompt=current_prompt_text,
                target="extraction_prompt",
                advisor_mode="absolute_eval",
                min_evidence=0,
            )
            candidate_prompt = str((advisor_result or {}).get("candidate_extraction_prompt") or "").strip()
            candidate_prompt_source = str((advisor_result or {}).get("candidate_prompt_source") or "")
            advisor_path = round_dir / f"advisor_round_{round_index:02d}.json"
            atomic_write_text(advisor_path, json.dumps({
                "result": advisor_result,
                "raw": raw,
            }, ensure_ascii=False, indent=2), encoding="utf-8")

            if not candidate_prompt:
                no_candidate_reason = _classify_no_candidate_reason(
                    results=results,
                    evidence=evidence,
                    advisor_result=advisor_result if isinstance(advisor_result, dict) else {},
                )
                final_status = str(no_candidate_reason.get("status") or "paused_no_candidate")
                update_state(config.run_id, lambda state: (
                    _round_record(state, round_index).update({
                        "advisor_path": str(advisor_path),
                        "candidate_prompt_saved": "",
                        "candidate_prompt_source": candidate_prompt_source,
                        "no_candidate_reason": no_candidate_reason,
                        "status": final_status,
                        "finished_at": utc_now(),
                        "latest_message": no_candidate_reason.get("message") or "未生成候选提取提示词。",
                    }),
                    append_event(
                        state,
                        f"第 {round_index} 轮{no_candidate_reason.get('title') or '未生成候选提取提示词'}",
                        "info" if no_candidate_reason.get("category") == "no_change_needed" else "warning",
                    ),
                ))
                break

            saved_prompt, saved_prompt_text = _save_candidate_prompt(
                candidate_prompt,
                round_index,
                task_type,
            )
            current_prompt_text = saved_prompt_text
            current_create_prompt_text = saved_prompt_text
            current_prompt_version = Path(saved_prompt).stem

            update_state(config.run_id, lambda state: (
                _round_record(state, round_index).update({
                    "advisor_path": str(advisor_path),
                    "candidate_prompt_saved": saved_prompt,
                    "candidate_prompt_source": candidate_prompt_source,
                    "status": "completed",
                    "finished_at": utc_now(),
                    "latest_message": f"候选提取提示词已保存：{saved_prompt}",
                }),
                append_event(state, f"第 {round_index} 轮候选提取提示词已保存：{saved_prompt}"),
            ))

            round_index += 1

        if final_status == "completed":
            update_state(config.run_id, lambda state: (
                state.update({"status": "completed", "stage": "完成", "finished_at": utc_now()}),
                append_event(state, "闭环实验完成"),
            ))
        else:
            update_state(config.run_id, lambda state: (
                state.update({
                    "status": final_status,
                    "stage": (
                        (_round_record(state, len(state.get("rounds") or [])).get("no_candidate_reason") or {}).get("stage")
                        if state.get("rounds")
                        else "未生成候选提示词"
                    ),
                    "finished_at": utc_now(),
                }),
                append_event(
                    state,
                    "闭环实验结束："
                    + str(
                        ((_round_record(state, len(state.get("rounds") or [])).get("no_candidate_reason") or {}).get("title"))
                        or "未生成候选提取提示词"
                    ),
                    "info" if final_status == "completed_no_change" else "warning",
                ),
            ))

    except StopIteration as exc:
        stop_message = str(exc)
        update_state(config.run_id, lambda state: (
            state.update({"status": "stopped", "stage": "已终止", "finished_at": utc_now()}),
            append_event(state, stop_message, "warning"),
        ))
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        error_traceback = traceback.format_exc()
        update_state(config.run_id, lambda state: (
            state.update({
                "status": "failed",
                "stage": "失败",
                "finished_at": utc_now(),
                "error": error_message,
                "traceback": error_traceback,
            }),
            append_event(state, f"闭环实验失败：{error_message}", "error"),
        ))


def run_closed_loop(config: ClosedLoopConfig) -> None:
    if config.protocol_version == "v2_holdout":
        from src.loop.trusted_closed_loop import run_trusted_closed_loop

        run_trusted_closed_loop(config)
        return
    _run_exploratory_closed_loop(config)
