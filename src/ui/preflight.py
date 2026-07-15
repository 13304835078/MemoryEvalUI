from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import streamlit as st

from src.schema import EvalConfig, TaskType


PASS = "pass"
WARNING = "warning"
ERROR = "error"


@dataclass(frozen=True)
class PreflightCheck:
    code: str
    label: str
    status: str
    detail: str


def preflight_ok(checks: Iterable[PreflightCheck]) -> bool:
    return all(item.status != ERROR for item in checks)


def preflight_counts(checks: Iterable[PreflightCheck]) -> dict[str, int]:
    counts = {PASS: 0, WARNING: 0, ERROR: 0}
    for item in checks:
        counts[item.status] = counts.get(item.status, 0) + 1
    return counts


def build_eval_preflight(
    *,
    cases: list[Any],
    task_type: str,
    eval_config: EvalConfig,
    judge_prompt_text: str,
    extraction_prompt_text: str = "",
    extraction_prompt_selected: bool = False,
) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    checks.append(PreflightCheck(
        "cases",
        "评测样本",
        PASS if cases else ERROR,
        f"已加载 {len(cases)} 条样本。" if cases else "未加载样本。",
    ))

    task_values = {
        item.task_type.value if isinstance(getattr(item, "task_type", ""), TaskType)
        else str(getattr(item, "task_type", "") or "")
        for item in cases
    }
    task_values.discard("")
    task_match = not task_values or task_values == {str(task_type)}
    checks.append(PreflightCheck(
        "task_type",
        "任务类型",
        PASS if task_match else ERROR,
        "样本任务类型一致。" if task_match else f"样本包含任务类型 {sorted(task_values)}，与当前选择不一致。",
    ))

    config_errors = eval_config.validate()
    checks.append(PreflightCheck(
        "api_config",
        "模型与接口",
        PASS if not config_errors else ERROR,
        "模型与接口配置完整。" if not config_errors else "；".join(config_errors),
    ))
    checks.append(PreflightCheck(
        "judge_prompt",
        "裁判提示词",
        PASS if judge_prompt_text.strip() else ERROR,
        "裁判提示词已加载。" if judge_prompt_text.strip() else "裁判提示词为空或文件无法读取。",
    ))

    if extraction_prompt_selected:
        checks.append(PreflightCheck(
            "extraction_prompt",
            "提取规则",
            PASS if extraction_prompt_text.strip() else ERROR,
            "提取提示词已加载，将作为规则来源。" if extraction_prompt_text.strip() else "已选择提取提示词，但内容为空。",
        ))
    else:
        checks.append(PreflightCheck(
            "extraction_prompt",
            "提取规则",
            WARNING,
            "未使用提取提示词辅助评测，Judge 无法按真实提取规则生成引用。",
        ))

    empty_outputs = sum(1 for item in cases if not str(getattr(item, "candidate_output", "") or "").strip())
    if empty_outputs:
        checks.append(PreflightCheck(
            "empty_outputs",
            "候选输出",
            WARNING,
            f"{empty_outputs} 条样本的候选输出为空；若这不是预期的无更新结果，请先检查 case。",
        ))
    else:
        checks.append(PreflightCheck("empty_outputs", "候选输出", PASS, "候选输出均已提供。"))

    interval = float(eval_config.judge_request_interval or 0)
    concurrency = int(eval_config.judge_concurrency or 1)
    qps_warning = not eval_config.mock and concurrency > 1 and interval < 10
    checks.append(PreflightCheck(
        "rate_limit",
        "限流设置",
        WARNING if qps_warning else PASS,
        (
            "并发大于 1 且请求间隔小于 10 秒，0.10 QPS 接口很可能触发限流。"
            if qps_warning else f"并发 {concurrency}，请求启动间隔 {interval:.1f} 秒。"
        ),
    ))
    checks.append(PreflightCheck(
        "run_mode",
        "运行模式",
        WARNING if eval_config.mock else PASS,
        "当前为模拟模式，不会调用真实模型。" if eval_config.mock else "当前为真实接口调用。",
    ))
    return checks


def build_extraction_preflight(
    *,
    uploaded_name: str = "",
    local_path: str = "",
    prompt_text: str,
    eval_config: EvalConfig,
    model_name: str,
    concurrency: int,
    request_interval: float,
    auto_make_cases: bool,
    case_model_name: str = "",
    case_prompt_version: str = "",
) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    local = str(local_path or "").strip().strip('"')
    input_name = uploaded_name or local
    input_ok = bool(uploaded_name) or (bool(local) and Path(local).is_file())
    input_detail = f"将读取 {Path(input_name).name}。" if input_ok else "请上传 Excel，或填写存在的本地 Excel 路径。"
    checks.append(PreflightCheck("input", "输入 Excel", PASS if input_ok else ERROR, input_detail))

    suffix = Path(input_name).suffix.lower() if input_name else ""
    format_ok = not input_name or suffix in {".xlsx", ".xls"}
    checks.append(PreflightCheck(
        "input_format",
        "文件格式",
        PASS if format_ok else ERROR,
        "Excel 格式可识别。" if format_ok else f"不支持的文件格式：{suffix or '未知'}。",
    ))
    checks.append(PreflightCheck(
        "prompt",
        "提取提示词",
        PASS if prompt_text.strip() else ERROR,
        "提取提示词已加载。" if prompt_text.strip() else "提取提示词为空。",
    ))

    config_errors = eval_config.validate()
    if not model_name.strip():
        config_errors = [*config_errors, "提取模型名为空"]
    checks.append(PreflightCheck(
        "api_config",
        "模型与接口",
        PASS if not config_errors else ERROR,
        "模型与接口配置完整。" if not config_errors else "；".join(config_errors),
    ))

    qps_warning = not eval_config.mock and int(concurrency) > 1 and float(request_interval) < 10
    checks.append(PreflightCheck(
        "rate_limit",
        "限流设置",
        WARNING if qps_warning else PASS,
        (
            "并发大于 1 且请求间隔小于 10 秒，低 QPS 接口可能触发限流。"
            if qps_warning else f"并发 {int(concurrency)}，请求启动间隔 {float(request_interval):.1f} 秒。"
        ),
    ))

    case_fields_ok = not auto_make_cases or bool(case_model_name.strip() and case_prompt_version.strip())
    checks.append(PreflightCheck(
        "case_output",
        "评测 case 输出",
        PASS if case_fields_ok else ERROR,
        (
            "提取完成后将自动生成评测 case。" if auto_make_cases and case_fields_ok
            else "不自动生成评测 case。" if not auto_make_cases
            else "自动生成 case 时，模型名和提示词版本不能为空。"
        ),
    ))
    checks.append(PreflightCheck(
        "run_mode",
        "运行模式",
        WARNING if eval_config.mock else PASS,
        "当前为模拟模式。" if eval_config.mock else "当前为真实接口调用。",
    ))
    return checks


def build_closed_loop_preflight(
    *,
    uploaded_name: str = "",
    local_path: str = "",
    extraction_prompt_text: str,
    judge_prompt_text: str,
    eval_config: EvalConfig,
    extraction_model: str,
    extraction_api_base: str,
    extraction_api_token: str,
    advisor_model: str,
    advisor_api_base: str,
    advisor_api_token: str,
    rounds: int,
    concurrency: int,
    request_interval: float,
) -> list[PreflightCheck]:
    checks = build_extraction_preflight(
        uploaded_name=uploaded_name,
        local_path=local_path,
        prompt_text=extraction_prompt_text,
        eval_config=eval_config,
        model_name=extraction_model,
        concurrency=concurrency,
        request_interval=request_interval,
        auto_make_cases=True,
        case_model_name=extraction_model,
        case_prompt_version="closed-loop",
    )
    checks = [item for item in checks if item.code not in {"case_output", "run_mode"}]
    checks.append(PreflightCheck(
        "judge_prompt",
        "裁判提示词",
        PASS if judge_prompt_text.strip() else ERROR,
        "裁判提示词已加载。" if judge_prompt_text.strip() else "裁判提示词为空。",
    ))
    extraction_api_ok = eval_config.mock or bool(extraction_model.strip() and extraction_api_base.strip() and extraction_api_token.strip())
    checks.append(PreflightCheck(
        "extraction_api",
        "提取模型接口",
        PASS if extraction_api_ok else ERROR,
        "提取模型接口完整。" if extraction_api_ok else "提取模型、API 地址或 Token 不完整。",
    ))
    advisor_api_ok = eval_config.mock or bool(advisor_model.strip() and advisor_api_base.strip() and advisor_api_token.strip())
    checks.append(PreflightCheck(
        "advisor_api",
        "提示词改进接口",
        PASS if advisor_api_ok else ERROR,
        "提示词改进接口完整。" if advisor_api_ok else "提示词改进模型、API 地址或 Token 不完整。",
    ))
    checks.append(PreflightCheck(
        "rounds",
        "闭环轮次",
        PASS if int(rounds) <= 3 else WARNING,
        f"计划运行 {int(rounds)} 轮。" + ("建议先用 1-3 轮验证。" if int(rounds) > 3 else ""),
    ))
    checks.append(PreflightCheck(
        "run_mode",
        "运行模式",
        WARNING if eval_config.mock else PASS,
        "当前为模拟模式。" if eval_config.mock else "当前为真实接口调用。",
    ))
    return checks


def build_advisor_preflight(
    *,
    results_count: int,
    evidence_count: int,
    min_evidence: int,
    target: str,
    judge_prompt_text: str,
    extraction_prompt_text: str,
    eval_config: EvalConfig,
) -> list[PreflightCheck]:
    checks = [
        PreflightCheck(
            "results",
            "评测结果",
            PASS if results_count > 0 else ERROR,
            f"已加载 {results_count} 条评测结果。" if results_count > 0 else "未加载评测结果。",
        ),
        PreflightCheck(
            "evidence",
            "改进证据",
            PASS if evidence_count >= min_evidence else ERROR,
            f"已选择 {evidence_count} 条证据，最低要求 {min_evidence} 条。",
        ),
    ]
    needs_judge = target in {"judge_prompt", "both"}
    needs_extraction = target in {"extraction_prompt", "both"}
    checks.append(PreflightCheck(
        "judge_prompt",
        "当前裁判提示词",
        PASS if not needs_judge or judge_prompt_text.strip() else ERROR,
        (
            "本次不修改裁判提示词。"
            if not needs_judge
            else "当前裁判提示词已加载。"
            if judge_prompt_text.strip()
            else "本次目标需要裁判提示词，但内容为空。"
        ),
    ))
    checks.append(PreflightCheck(
        "extraction_prompt",
        "当前提取提示词",
        PASS if not needs_extraction or extraction_prompt_text.strip() else ERROR,
        (
            "本次不修改提取提示词。"
            if not needs_extraction
            else "当前提取提示词已加载。"
            if extraction_prompt_text.strip()
            else "本次目标需要提取提示词，但内容为空。"
        ),
    ))
    config_errors = eval_config.validate()
    checks.append(PreflightCheck(
        "api_config",
        "改提示词模型与接口",
        PASS if not config_errors else ERROR,
        "模型与接口配置完整。" if not config_errors else "；".join(config_errors),
    ))
    checks.append(PreflightCheck(
        "run_mode",
        "运行模式",
        WARNING if eval_config.mock else PASS,
        "当前为模拟模式。" if eval_config.mock else "当前为真实接口调用。",
    ))
    return checks


def build_ab_preflight(
    *,
    cases: list[Any],
    task_type: str,
    prompt_a_text: str,
    prompt_b_text: str,
    prompt_a_name: str,
    prompt_b_name: str,
    extraction_prompt_text: str,
    eval_config: EvalConfig,
) -> list[PreflightCheck]:
    checks = build_eval_preflight(
        cases=cases,
        task_type=task_type,
        eval_config=eval_config,
        judge_prompt_text=prompt_a_text,
        extraction_prompt_text=extraction_prompt_text,
        extraction_prompt_selected=bool(extraction_prompt_text),
    )
    checks = [item for item in checks if item.code not in {"judge_prompt", "empty_outputs"}]
    checks.extend([
        PreflightCheck(
            "prompt_a",
            "裁判提示词 A",
            PASS if prompt_a_text.strip() else ERROR,
            "提示词 A 已加载。" if prompt_a_text.strip() else "提示词 A 为空。",
        ),
        PreflightCheck(
            "prompt_b",
            "裁判提示词 B",
            PASS if prompt_b_text.strip() else ERROR,
            "提示词 B 已加载。" if prompt_b_text.strip() else "提示词 B 为空。",
        ),
        PreflightCheck(
            "single_variable",
            "A/B 单一变量",
            WARNING if prompt_a_name == prompt_b_name else PASS,
            "A 与 B 是同一提示词，对比不会产生有效变量。" if prompt_a_name == prompt_b_name else "A/B 仅替换裁判提示词。",
        ),
    ])
    model_count = len({str(getattr(item, "model_name", "") or "unknown") for item in cases})
    checks.append(PreflightCheck(
        "model_count",
        "被评测模型",
        WARNING if model_count > 1 else PASS,
        f"当前包含 {model_count} 个被评测模型。" + ("建议一次只比较一个模型。" if model_count > 1 else ""),
    ))
    return checks


def render_preflight(checks: list[PreflightCheck], *, title: str = "启动前检查") -> bool:
    counts = preflight_counts(checks)
    if counts[ERROR]:
        st.error(f"{title}未通过：{counts[ERROR]} 项错误，修正后才能启动。")
    elif counts[WARNING]:
        st.warning(f"{title}通过，但有 {counts[WARNING]} 项提醒。")
    else:
        st.success(f"{title}通过：{counts[PASS]} 项检查均正常。")

    with st.expander(f"查看{title}明细", expanded=bool(counts[ERROR])):
        labels = {PASS: "通过", WARNING: "提醒", ERROR: "错误"}
        rows = [
            {"状态": labels.get(item.status, item.status), "检查项": item.label, "说明": item.detail}
            for item in checks
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    return counts[ERROR] == 0
