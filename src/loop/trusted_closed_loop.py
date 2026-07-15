from __future__ import annotations

import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, TYPE_CHECKING

from src.eval.metrics import compute_aggregations, flatten_results
from src.eval.run_quality import compute_run_quality
from src.extraction.memory_extractor import MemoryExtractionConfig, MemoryExtractionRunner
from src.loop.dataset_split import split_excel_by_reviewer_session
from src.loop.validation_gate import ValidationGateConfig, evaluate_candidate_gate
from src.schema import TaskType, cases_to_jsonl
from src.ui.prompt_advisor import call_prompt_advisor, collect_absolute_eval_evidence
from src.ui.prompt_editor import prompt_text_hash
from src.ui.state_io import atomic_write_json
from src.ui.task_controls import DEFAULT_PRIORITY, init_task_controls

if TYPE_CHECKING:
    from src.loop.closed_loop import ClosedLoopConfig


def _gate_config(config: "ClosedLoopConfig") -> ValidationGateConfig:
    return ValidationGateConfig(
        min_score_delta=float(config.validation_min_score_delta),
        min_end_to_end_delta=float(config.validation_min_end_to_end_delta),
        max_extraction_coverage_drop=float(config.validation_max_coverage_drop),
        max_case_regression_rate=float(config.validation_max_case_regression_rate),
        max_prompt_growth_ratio=float(config.validation_max_prompt_growth_ratio),
        min_paired_cases=int(config.validation_min_paired_cases),
        min_paired_clusters=int(config.validation_min_paired_clusters),
        confidence_level=float(config.validation_confidence_level),
        bootstrap_samples=int(config.validation_bootstrap_samples),
    )


def _run_partition(
    core,
    config: "ClosedLoopConfig",
    *,
    partition: str,
    input_path: str,
    prompt_text: str,
    create_prompt_text: str,
    prompt_version: str,
    round_index: int,
    output_dir: Path,
) -> dict[str, Any]:
    core._check_stop(config.run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    task_type = TaskType(config.task_type)
    label_map = {"discovery": "Discovery", "validation": "Validation", "locked_test": "Locked Test"}
    label = label_map.get(partition, partition)
    extraction_output = output_dir / "memory_extraction.xlsx"

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
    extraction_config.concurrency = core._current_extraction_concurrency(config)
    extraction_config.priority = core._current_priority(config)

    def extraction_progress(done: int, total: int, message: str) -> None:
        core.update_state(config.run_id, lambda state: (
            state.update({"stage": f"第 {round_index} 轮：{label} 记忆提取 {done}/{total}"}),
            core._round_record(state, round_index).setdefault("partition_progress", {}).update({
                partition: {"stage": "extraction", "done": done, "total": total, "message": message}
            }),
        ))

    runner = MemoryExtractionRunner(
        extraction_config,
        prompt_text,
        task_type=task_type,
        create_prompt_text=create_prompt_text,
        update_prompt_text=prompt_text,
    )
    extraction_stats = runner.process_excel(
        input_path,
        extraction_output,
        sheet_name=0,
        reviewer_filter=None,
        chunk_size=config.chunk_size,
        progress_callback=extraction_progress,
        should_stop=lambda: core.stop_requested(config.run_id),
        emit_parallel_chunk_progress=True,
        priority_provider=lambda: core._current_priority(config),
        concurrency_provider=lambda: core._current_extraction_concurrency(config),
    )
    if extraction_stats.get("stopped"):
        raise StopIteration(f"{label} 提取阶段收到终止请求")

    converter = (
        core.prepare_long_memory_cases_from_run_output
        if task_type == TaskType.LONG_MEMORY
        else core.prepare_cases_from_run_output
    )
    cases, missed_cases, case_stats = converter(
        extraction_output,
        model=config.extraction_model or "unknown",
        prompt_version=prompt_version,
        chunk_size=config.chunk_size,
        return_missed=True,
    )
    # 可信协议必须完整评测固定分区。按前 N 条截断会让结果受原始行顺序影响，
    # 也会使 champion/candidate 的覆盖口径失真。
    run_cases = cases
    cases_path = output_dir / "cases.jsonl"
    missed_path = output_dir / "missed_cases.jsonl"
    cases_to_jsonl(run_cases, str(cases_path))
    cases_to_jsonl(missed_cases, str(missed_path))

    core.update_state(config.run_id, lambda state: (
        state.update({"stage": f"第 {round_index} 轮：{label} 执行评测"}),
        core._round_record(state, round_index).setdefault("partition_progress", {}).update({
            partition: {"stage": "evaluation", "done": 0, "total": len(run_cases), "message": "开始评测"}
        }),
    ))
    results_path = output_dir / "eval_results.jsonl"
    results = core._evaluate_cases(
        config,
        run_cases,
        round_index,
        prompt_text,
        prompt_version,
        results_path,
    )
    quality = compute_run_quality(results, cases=run_cases, missed_cases=missed_cases)
    payload = {
        "partition": partition,
        "prompt_version": prompt_version,
        "prompt_hash": prompt_text_hash(prompt_text),
        "extraction_output": str(extraction_output),
        "extraction_stats": extraction_stats,
        "cases_path": str(cases_path),
        "missed_cases_path": str(missed_path),
        "case_stats": case_stats,
        "results_path": str(results_path),
        "eval_stats": compute_aggregations(results),
        "run_quality": quality,
        "eval_preview": flatten_results(results[:20]),
    }
    core.update_state(config.run_id, lambda state: (
        core._round_record(state, round_index).setdefault("partition_runs", {}).update({
            f"{partition}:{prompt_version}": payload
        }),
        core._round_record(state, round_index).setdefault("partition_progress", {}).update({
            partition: {"stage": "completed", "done": len(results), "total": len(run_cases), "message": "完成"}
        }),
    ))
    return {**payload, "cases": run_cases, "missed_cases": missed_cases, "results": results}


def _public_run(run: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in run.items() if key not in {"cases", "missed_cases", "results"}}


def run_trusted_closed_loop(config: "ClosedLoopConfig") -> None:
    from src.loop import closed_loop as core

    loop_dir = core.run_dir(config.run_id)
    loop_dir.mkdir(parents=True, exist_ok=True)
    if core.stop_path(config.run_id).exists():
        core.stop_path(config.run_id).unlink()
    init_task_controls(core.controls_path(config.run_id), {
        "priority": DEFAULT_PRIORITY,
        "target_rounds": int(config.rounds or 1),
        "extraction_concurrency": min(100, max(1, int(config.extraction_concurrency or 1))),
        "judge_concurrency": min(100, max(1, int(config.eval_config.judge_concurrency or 1))),
        "judge_request_interval": float(config.eval_config.judge_request_interval or 0.0),
    })
    state = core._make_initial_state(config)
    state["protocol"] = {
        "version": "v2_holdout",
        "name": "Discovery / Validation / Locked Test",
        "judge_frozen": True,
        "advisor_visible_partitions": ["discovery"],
        "evaluation_rule_frozen": True,
    }
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
        "description": "候选提取提示词不能修改自己的评分规则。",
    }
    state["judge_snapshot"] = {
        "model": config.eval_config.judge_model,
        "judge_prompt_version": config.judge_prompt_version,
        "judge_prompt_hash": prompt_text_hash(config.judge_prompt_text),
        "temperature": config.eval_config.judge_temperature,
        "top_p": config.eval_config.judge_top_p,
        "top_k": config.eval_config.judge_top_k,
        "enable_thinking": config.eval_config.judge_enable_thinking,
    }
    core.append_event(state, "可信闭环启动：Judge 配置已冻结，Validation/Test 不会进入提示词建议证据。")
    core.write_loop_state(config.run_id, state)

    initial_prompt = config.extraction_prompt_text
    initial_create_prompt = config.extraction_create_prompt_text or initial_prompt
    current_prompt = initial_prompt
    current_create_prompt = initial_create_prompt
    current_version = config.extraction_prompt_version or "initial_extraction_prompt"
    final_status = "max_rounds_reached"
    champion_validation: dict[str, Any] | None = None
    previous_discovery_results: list[Any] = []

    try:
        if not initial_prompt.strip():
            raise ValueError("初始提取提示词为空")
        manifest = split_excel_by_reviewer_session(
            config.input_excel_path,
            loop_dir / "dataset_split",
            sheet_name=config.sheet_name,
            reviewer_filter=config.reviewer_filter,
            discovery_ratio=config.discovery_ratio,
            validation_ratio=config.validation_ratio,
            locked_test_ratio=config.locked_test_ratio,
            seed=config.split_seed,
            min_validation_reviewers=max(1, int(config.validation_min_paired_clusters)),
        )
        core.update_state(config.run_id, lambda state: (
            state.update({"split_manifest": manifest, "stage": "数据切分完成"}),
            core.append_event(state, f"固定切分完成：{manifest['partition_group_counts']}")
        ))

        round_index = 1
        while round_index <= core._current_target_rounds(config):
            core._check_stop(config.run_id)
            round_dir = loop_dir / f"round_{round_index:02d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            core.update_state(config.run_id, lambda state: (
                state.update({"stage": f"第 {round_index} 轮：Discovery"}),
                core._round_record(state, round_index).update({
                    "round": round_index,
                    "status": "running",
                    "champion_prompt_version": current_version,
                    "champion_prompt_hash": prompt_text_hash(current_prompt)[:12],
                    "started_at": core.utc_now(),
                }),
                core.append_event(state, f"第 {round_index} 轮开始，只从 Discovery 收集改词证据。"),
            ))

            discovery = _run_partition(
                core,
                config,
                partition="discovery",
                input_path=manifest["partition_paths"]["discovery"],
                prompt_text=current_prompt,
                create_prompt_text=current_create_prompt,
                prompt_version=current_version,
                round_index=round_index,
                output_dir=round_dir / "discovery_champion",
            )
            core.update_state(config.run_id, lambda state: core._round_record(state, round_index).update({
                "discovery": _public_run(discovery)
            }))
            if not discovery["run_quality"]["run_complete"]:
                final_status = "invalid_evaluation"
                core.update_state(config.run_id, lambda state: core._round_record(state, round_index).update({
                    "status": final_status,
                    "latest_message": "Discovery 存在未解决的接口/Judge 失败，禁止据此修改提示词。",
                }))
                break

            evidence = collect_absolute_eval_evidence(
                discovery["results"],
                max_items=config.advisor_max_items,
                include_all=True,
                positive_boundary_limit=min(3, max(1, config.advisor_max_items // 10)),
                regression_results=previous_discovery_results,
            )
            evidence_counts: dict[str, int] = {}
            for item in evidence:
                mode = str(item.get("evidence_mode") or "unknown")
                evidence_counts[mode] = evidence_counts.get(mode, 0) + 1
            core.update_state(config.run_id, lambda state: core._round_record(state, round_index).update({
                "advisor_evidence_count": len(evidence),
                "advisor_evidence_composition": evidence_counts,
                "latest_message": f"正在基于 Discovery 的 {len(evidence)} 条证据生成候选。",
            }))
            advisor_result, raw = call_prompt_advisor(
                core._advisor_eval_config(config),
                evidence=evidence,
                current_judge_prompt=config.judge_prompt_text,
                extraction_prompt=current_prompt,
                target="extraction_prompt",
                advisor_mode="absolute_eval",
                min_evidence=0,
            )
            advisor_path = round_dir / "advisor.json"
            atomic_write_json(advisor_path, {"result": advisor_result, "raw": raw})
            candidate_prompt = str((advisor_result or {}).get("candidate_extraction_prompt") or "").strip()
            candidate_prompt_source = str((advisor_result or {}).get("candidate_prompt_source") or "")
            if not candidate_prompt:
                reason = core._classify_no_candidate_reason(
                    results=discovery["results"], evidence=evidence, advisor_result=advisor_result
                )
                final_status = str(reason.get("status") or "paused_no_candidate")
                core.update_state(config.run_id, lambda state: core._round_record(state, round_index).update({
                    "advisor_path": str(advisor_path),
                    "candidate_prompt_source": candidate_prompt_source,
                    "no_candidate_reason": reason,
                    "status": final_status,
                    "finished_at": core.utc_now(),
                    "latest_message": reason.get("message", "未生成候选。"),
                }))
                break

            candidate_draft = round_dir / "candidate_prompt_draft.md"
            core.atomic_write_text(candidate_draft, candidate_prompt)
            candidate_version = f"candidate_round_{round_index}"

            if champion_validation is None:
                champion_validation = _run_partition(
                    core,
                    config,
                    partition="validation",
                    input_path=manifest["partition_paths"]["validation"],
                    prompt_text=current_prompt,
                    create_prompt_text=current_create_prompt,
                    prompt_version=current_version,
                    round_index=round_index,
                    output_dir=round_dir / "validation_champion",
                )
            candidate_validation = _run_partition(
                core,
                config,
                partition="validation",
                input_path=manifest["partition_paths"]["validation"],
                prompt_text=candidate_prompt,
                create_prompt_text=candidate_prompt,
                prompt_version=candidate_version,
                round_index=round_index,
                output_dir=round_dir / "validation_candidate",
            )
            gate_result = evaluate_candidate_gate(
                champion_validation["results"],
                candidate_validation["results"],
                champion_cases=champion_validation["cases"],
                candidate_cases=candidate_validation["cases"],
                champion_missed_cases=champion_validation["missed_cases"],
                candidate_missed_cases=candidate_validation["missed_cases"],
                champion_prompt=current_prompt,
                candidate_prompt=candidate_prompt,
                config=_gate_config(config),
            )
            core.update_state(config.run_id, lambda state: core._round_record(state, round_index).update({
                "advisor_path": str(advisor_path),
                "candidate_prompt_draft": str(candidate_draft),
                "candidate_prompt_source": candidate_prompt_source,
                "validation_champion": _public_run(champion_validation),
                "validation_candidate": _public_run(candidate_validation),
                "validation_gate": gate_result,
            }))
            if not gate_result["accepted"]:
                final_status = "invalid_evaluation" if (
                    not gate_result["champion_quality"]["run_complete"]
                    or not gate_result["candidate_quality"]["run_complete"]
                ) else "validation_rejected"
                core.update_state(config.run_id, lambda state: core._round_record(state, round_index).update({
                    "status": final_status,
                    "finished_at": core.utc_now(),
                    "latest_message": "候选未通过 Validation，不替换当前提取提示词。",
                }))
                break

            saved_prompt, _ = core._save_candidate_prompt(candidate_prompt, round_index, TaskType(config.task_type))
            current_prompt = candidate_prompt
            current_create_prompt = candidate_prompt
            current_version = Path(saved_prompt).stem
            champion_validation = candidate_validation
            core.update_state(config.run_id, lambda state: (
                core._round_record(state, round_index).update({
                    "candidate_prompt_saved": saved_prompt,
                    "status": "accepted",
                    "finished_at": core.utc_now(),
                    "latest_message": f"候选通过 Validation，已晋升为新版本：{current_version}",
                }),
                core.append_event(state, f"第 {round_index} 轮候选通过 Validation。"),
            ))
            previous_discovery_results = discovery["results"]
            round_index += 1

        if final_status == "invalid_evaluation":
            core.update_state(config.run_id, lambda state: (
                state.update({
                    "status": final_status,
                    "stage": "评测不完整",
                    "finished_at": core.utc_now(),
                }),
                core.append_event(state, "存在未解决的运行失败，已停止闭环且未消耗 Locked Test。", "warning"),
            ))
            return

        core._check_stop(config.run_id)
        test_dir = loop_dir / "locked_test"
        initial_test = _run_partition(
            core,
            config,
            partition="locked_test",
            input_path=manifest["partition_paths"]["locked_test"],
            prompt_text=initial_prompt,
            create_prompt_text=initial_create_prompt,
            prompt_version=config.extraction_prompt_version or "initial_extraction_prompt",
            round_index=max(1, len(core.read_loop_state(config.run_id).get("rounds") or [])),
            output_dir=test_dir / "initial",
        )
        if prompt_text_hash(initial_prompt) == prompt_text_hash(current_prompt):
            final_test = initial_test
        else:
            final_test = _run_partition(
                core,
                config,
                partition="locked_test",
                input_path=manifest["partition_paths"]["locked_test"],
                prompt_text=current_prompt,
                create_prompt_text=current_create_prompt,
                prompt_version=current_version,
                round_index=max(1, len(core.read_loop_state(config.run_id).get("rounds") or [])),
                output_dir=test_dir / "final",
            )
        test_report = evaluate_candidate_gate(
            initial_test["results"],
            final_test["results"],
            champion_cases=initial_test["cases"],
            candidate_cases=final_test["cases"],
            champion_missed_cases=initial_test["missed_cases"],
            candidate_missed_cases=final_test["missed_cases"],
            champion_prompt=initial_prompt,
            candidate_prompt=current_prompt,
            config=replace(_gate_config(config), min_score_delta=0.0),
        )
        if (
            not initial_test["run_quality"]["run_complete"]
            or not final_test["run_quality"]["run_complete"]
        ):
            final_status = "invalid_evaluation"
        core.update_state(config.run_id, lambda state: (
            state.update({
                "status": final_status,
                "stage": "Locked Test 完成" if final_status != "invalid_evaluation" else "评测不完整",
                "finished_at": core.utc_now(),
                "final_prompt_version": current_version,
                "final_prompt_hash": prompt_text_hash(current_prompt),
                "locked_test": {
                    "initial": _public_run(initial_test),
                    "final": _public_run(final_test),
                    "comparison": test_report,
                    "advisor_visible": False,
                },
            }),
            core.append_event(state, "Locked Test 已完成；该集合从未提供给提示词建议模型。"),
        ))
    except StopIteration as exc:
        core.update_state(config.run_id, lambda state: (
            state.update({"status": "stopped", "stage": "已终止", "finished_at": core.utc_now()}),
            core.append_event(state, str(exc), "warning"),
        ))
    except Exception as exc:
        core.update_state(config.run_id, lambda state: (
            state.update({
                "status": "failed",
                "stage": "失败",
                "finished_at": core.utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }),
            core.append_event(state, f"可信闭环失败：{type(exc).__name__}: {exc}", "error"),
        ))
