from __future__ import annotations

import json
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.eval.eval_runner import EvalRunner
from src.eval.metrics import compute_aggregations, flatten_results
from src.extraction.memory_extractor import (
    MemoryExtractionConfig,
    MemoryExtractionRunner,
    sanitize_filename,
)
from src.schema import EvalConfig, EvalResult, TaskType, results_to_jsonl
from src.ui.data_service import prepare_cases_from_run_output, save_cases
from src.ui.prompt_advisor import call_prompt_advisor, collect_absolute_eval_evidence
from src.ui.prompt_editor import prompt_text_hash, save_prompt_version
from src.ui.state_io import atomic_write_json, state_file_lock


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLOSED_LOOP_DIR = PROJECT_ROOT / "data" / "closed_loop"


@dataclass
class ClosedLoopConfig:
    run_id: str
    input_excel_path: str
    sheet_name: str | int | None = 0
    reviewer_filter: str = ""
    rounds: int = 3
    chunk_size: int = 10
    max_cases_per_round: int = 0

    extraction_model: str = ""
    extraction_api_base: str = ""
    extraction_api_token: str = ""
    extraction_prompt_text: str = ""
    extraction_prompt_version: str = ""
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


def read_loop_state(run_id: str) -> dict[str, Any]:
    path = state_path(run_id)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_loop_state(run_id: str, state: dict[str, Any]) -> None:
    path = state_path(run_id)
    state["heartbeat_at"] = utc_now()
    atomic_write_json(path, state)


def request_stop(run_id: str) -> None:
    path = stop_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(utc_now(), encoding="utf-8")


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
        state["updated_at"] = utc_now()
        write_loop_state(run_id, state)
        return state


def _make_initial_state(config: ClosedLoopConfig) -> dict[str, Any]:
    safe_config = asdict(config)
    safe_config.pop("extraction_api_token", None)
    safe_config.pop("advisor_api_token", None)
    if isinstance(safe_config.get("eval_config"), dict):
        safe_config["eval_config"].pop("judge_api_bearer_token", None)
    return {
        "run_id": config.run_id,
        "status": "running",
        "stage": "初始化",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "config": safe_config,
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


def _round_record(state: dict[str, Any], round_index: int) -> dict[str, Any]:
    rounds = state.setdefault("rounds", [])
    while len(rounds) < round_index:
        rounds.append({"round": len(rounds) + 1})
    return rounds[round_index - 1]


def _save_candidate_prompt(candidate_prompt: str, round_index: int) -> tuple[str, str]:
    version_name = f"extract_closed_loop_round_{round_index}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    saved = save_prompt_version("user_md_update", candidate_prompt, version_name, prompt_kind="extraction")
    return saved, candidate_prompt


def _evaluate_cases(
    config: ClosedLoopConfig,
    cases,
    round_index: int,
    current_prompt_text: str,
    current_prompt_version: str,
    result_path: Path,
) -> list[EvalResult]:
    runner = EvalRunner(
        config=config.eval_config,
        task_type=TaskType.USER_MD,
        prompt_file=config.judge_prompt_file,
        judge_prompt_version=config.judge_prompt_version,
        system_prompt_override=config.judge_prompt_text,
        extraction_prompt_text=current_prompt_text,
        extraction_prompt_version=current_prompt_version,
        extraction_prompt_hash=prompt_text_hash(current_prompt_text),
    )

    run_cases = cases[: config.max_cases_per_round] if config.max_cases_per_round and config.max_cases_per_round > 0 else cases
    results_by_index: dict[int, EvalResult] = {}
    result_path.parent.mkdir(parents=True, exist_ok=True)

    request_interval = float(getattr(config.eval_config, "judge_request_interval", 0.0) or 0.0)
    concurrency = min(100, max(1, int(getattr(config.eval_config, "judge_concurrency", 1) or 1)))
    concurrency = min(concurrency, max(1, len(run_cases)))
    backoff_interval = float(getattr(config.eval_config, "judge_qps_backoff", 0.0) or 0.0)
    if concurrency > 1 and not config.eval_config.mock:
        request_interval = max(request_interval, backoff_interval)
    if result_path.exists():
        result_path.unlink()

    if not run_cases:
        results_to_jsonl([], str(result_path))
        return []

    update_state(config.run_id, lambda state: (
        state.update({"stage": f"第 {round_index} 轮：评测 0/{len(run_cases)}"}),
        _round_record(state, round_index).update({
            "eval_progress": f"0/{len(run_cases)}",
            "eval_concurrency": concurrency,
        }),
    ))

    rate_lock = threading.Lock()
    next_request_at = {"value": time.monotonic()}

    def wait_for_rate_slot() -> None:
        if request_interval <= 0 or config.eval_config.mock:
            return
        with rate_lock:
            now = time.monotonic()
            wait_seconds = max(0.0, next_request_at["value"] - now)
            next_request_at["value"] = max(now, next_request_at["value"]) + request_interval

        while wait_seconds > 0:
            _check_stop(config.run_id, "评测阶段收到终止请求")
            sleep_seconds = min(1.0, wait_seconds)
            time.sleep(sleep_seconds)
            wait_seconds -= sleep_seconds

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
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            for _ in range(concurrency):
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
                            task_type=TaskType.USER_MD.value,
                            raw=f"{type(exc).__name__}: {exc}",
                            model_name=case.model_name,
                            prompt_version=case.prompt_version,
                            judge_model=config.eval_config.judge_model or "mock",
                            judge_prompt_version=config.judge_prompt_version,
                            extraction_prompt_version=current_prompt_version,
                            extraction_prompt_hash=prompt_text_hash(current_prompt_text),
                        )

                    results_by_index[idx] = result
                    completed += 1
                    ordered_results = [results_by_index[i] for i in sorted(results_by_index)]
                    results_to_jsonl(ordered_results, str(result_path))
                    update_state(config.run_id, lambda state: (
                        state.update({"stage": f"第 {round_index} 轮：评测 {completed}/{len(run_cases)}"}),
                        _round_record(state, round_index).update({
                            "eval_progress": f"{completed}/{len(run_cases)}",
                            "latest_message": f"已完成评测：{case.case_id}",
                        }),
                    ))
                    submit_next(executor)
    except StopIteration:
        for future in futures:
            future.cancel()
        raise

    return [results_by_index[i] for i in sorted(results_by_index)]


def run_closed_loop(config: ClosedLoopConfig) -> None:
    loop_dir = run_dir(config.run_id)
    loop_dir.mkdir(parents=True, exist_ok=True)
    if stop_path(config.run_id).exists():
        stop_path(config.run_id).unlink()

    state = _make_initial_state(config)
    append_event(state, "闭环实验启动")
    write_loop_state(config.run_id, state)

    current_prompt_text = config.extraction_prompt_text
    current_prompt_version = config.extraction_prompt_version or "initial_extraction_prompt"
    final_status = "completed"

    try:
        if not current_prompt_text.strip():
            raise ValueError("初始提取提示词为空")

        for round_index in range(1, int(config.rounds) + 1):
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
            extraction_config.concurrency = min(100, max(1, int(config.extraction_concurrency or 1)))
            extraction_output = round_dir / f"memory_extract_round_{round_index:02d}.xlsx"

            def extraction_progress(done: int, total: int, message: str) -> None:
                update_state(config.run_id, lambda state: (
                    state.update({"stage": f"第 {round_index} 轮：记忆提取"}),
                    _round_record(state, round_index).update({
                        "extraction_progress": f"{done}/{total}",
                        "latest_message": message,
                    }),
                ))

            runner = MemoryExtractionRunner(extraction_config, current_prompt_text)
            extraction_stats = runner.process_excel(
                config.input_excel_path,
                extraction_output,
                sheet_name=config.sheet_name,
                reviewer_filter=config.reviewer_filter or None,
                chunk_size=config.chunk_size,
                progress_callback=extraction_progress,
                should_stop=lambda: stop_requested(config.run_id),
                emit_parallel_chunk_progress=True,
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
            cases, missed_cases, convert_stats = prepare_cases_from_run_output(
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
            advisor_path.write_text(json.dumps({
                "result": advisor_result,
                "raw": raw,
            }, ensure_ascii=False, indent=2), encoding="utf-8")

            if not candidate_prompt:
                final_status = "stopped_no_candidate"
                update_state(config.run_id, lambda state: (
                    _round_record(state, round_index).update({
                    "advisor_path": str(advisor_path),
                    "candidate_prompt_saved": "",
                    "status": "stopped_no_candidate",
                    "finished_at": utc_now(),
                    "latest_message": "未生成候选提取提示词，闭环停止",
                }),
                append_event(state, f"第 {round_index} 轮未生成候选提取提示词，闭环停止", "warning"),
            ))
                break

            saved_prompt, saved_prompt_text = _save_candidate_prompt(candidate_prompt, round_index)
            current_prompt_text = saved_prompt_text
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

        if final_status == "completed":
            update_state(config.run_id, lambda state: (
                state.update({"status": "completed", "stage": "完成", "finished_at": utc_now()}),
                append_event(state, "闭环实验完成"),
            ))
        else:
            update_state(config.run_id, lambda state: (
                state.update({"status": final_status, "stage": "未生成候选提示词", "finished_at": utc_now()}),
                append_event(state, "闭环实验停止：未生成候选提取提示词", "warning"),
            ))

    except StopIteration as exc:
        update_state(config.run_id, lambda state: (
            state.update({"status": "stopped", "stage": "已终止", "finished_at": utc_now()}),
            append_event(state, str(exc), "warning"),
        ))
    except Exception as exc:
        update_state(config.run_id, lambda state: (
            state.update({
                "status": "failed",
                "stage": "失败",
                "finished_at": utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }),
            append_event(state, f"闭环实验失败：{type(exc).__name__}: {exc}", "error"),
        ))
