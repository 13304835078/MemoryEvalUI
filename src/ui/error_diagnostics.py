from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.eval.result_status import result_is_score_eligible


@dataclass(frozen=True)
class FailureDiagnostic:
    code: str
    label: str
    summary: str
    action: str


def classify_failure_text(raw: str) -> FailureDiagnostic:
    text = str(raw or "")
    lower = text.lower()
    if "qps limit" in lower or "rate limit" in lower or "too many requests" in lower or "429" in lower:
        return FailureDiagnostic(
            "rate_limit",
            "接口限流",
            "请求频率超过服务端限制。",
            "降低并发、提高请求间隔，并确认全局是否还有其他任务共用同一接口。",
        )
    if "connection idle timeout" in lower or "idle timeout" in lower:
        return FailureDiagnostic(
            "idle_timeout",
            "连接空闲超时",
            "服务端在模型生成完成前关闭了长连接。",
            "缩短输入、减少输出长度或使用更快模型；单纯增加客户端超时通常无效。",
        )
    if "timeout" in lower or "timed out" in lower:
        return FailureDiagnostic(
            "timeout",
            "请求超时",
            "请求在配置的等待时间内没有完成。",
            "检查服务端负载，适当增加超时或缩短 Prompt 与输出长度。",
        )
    if "websocket" in lower or "connection reset" in lower or "connection aborted" in lower or "going away" in lower:
        return FailureDiagnostic(
            "connection",
            "网络连接中断",
            "请求连接在响应完成前被关闭。",
            "确认网络和服务状态；保留重试，并避免过高并发。",
        )
    if "json" in lower or "parse" in lower or "解析" in text:
        return FailureDiagnostic(
            "json_parse",
            "裁判输出无法解析",
            "模型返回了内容，但没有形成符合结果协议的 JSON。",
            "检查裁判提示词输出约束和原始响应；不要把已成功修复的结果视为失败。",
        )
    if "empty" in lower or "为空" in text or "no content" in lower:
        return FailureDiagnostic(
            "empty_response",
            "模型输出为空",
            "接口成功返回，但没有可用的裁判内容。",
            "检查模型响应字段、thinking 配置和最大输出长度。",
        )
    if "required" in lower or "missing field" in lower or "字段" in text:
        return FailureDiagnostic(
            "schema",
            "结果字段不完整",
            "模型 JSON 缺少评测协议要求的字段或字段类型不正确。",
            "查看原始响应，并修正裁判提示词中的 JSON schema 约束。",
        )
    if "api error" in lower or "http" in lower or "status code" in lower:
        return FailureDiagnostic(
            "api_error",
            "接口调用失败",
            "模型服务返回了错误响应。",
            "查看技术详情中的状态码和服务端消息，再决定重试或调整配置。",
        )
    return FailureDiagnostic(
        "unknown",
        "未分类运行错误",
        "当前错误无法自动归入已知类型。",
        "展开原始响应和错误堆栈进行排查。",
    )


def classify_eval_result(result: Any) -> FailureDiagnostic | None:
    if result_is_score_eligible(result):
        return None
    raw = str(
        getattr(result, "failure_message", "")
        or getattr(result, "raw_response", "")
        or getattr(result, "comment", "")
        or ""
    )
    return classify_failure_text(raw)
