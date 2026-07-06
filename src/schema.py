from dataclasses import dataclass, field, asdict
from typing import Optional
from enum import Enum
from datetime import datetime, timezone
import os

from .persistence import append_jsonl, atomic_write_jsonl, read_jsonl


class TaskType(str, Enum):
    RAW_DIALOGUE = "raw_dialogue"
    USER_MD = "user_md_update"
    DAY_MEMORY = "day_memory"
    LONG_MEMORY = "long_memory"
    SUMMARY = "summary"


EVALUATABLE_TASK_TYPES = (TaskType.USER_MD, TaskType.LONG_MEMORY)

TASK_TYPE_LABELS = {
    TaskType.USER_MD.value: "用户画像 USER.md",
    TaskType.LONG_MEMORY.value: "长期记忆 MEMORY.md",
}


@dataclass
class DialogueTurn:
    role: str
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Case:
    case_id: str
    task_type: TaskType
    session_id: str

    old_memory: Optional[str] = None
    dialogue: list[DialogueTurn] = field(default_factory=list)
    instructions: Optional[str] = None
    turn_range: Optional[list[int]] = None

    candidate_output: Optional[str] = None
    reference_output: Optional[str] = None

    model_name: str = "unknown"
    prompt_version: str = "unknown"
    metadata: dict = field(default_factory=dict)

    eval_result: Optional[dict] = None
    human_review: Optional[dict] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["task_type"] = self.task_type.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Case":
        data = dict(data)
        if "task_type" in data and isinstance(data["task_type"], str):
            data["task_type"] = TaskType(data["task_type"])
        if "dialogue" in data and isinstance(data["dialogue"], list):
            data["dialogue"] = [
                DialogueTurn(**t) if isinstance(t, dict) else t
                for t in data["dialogue"]
            ]
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def cases_to_jsonl(cases: list[Case], path: str) -> None:
    atomic_write_jsonl(path, (case.to_dict() for case in cases))


def cases_from_jsonl(path: str) -> list[Case]:
    return [Case.from_dict(row) for row in read_jsonl(path)]


def validate_case(case: Case) -> list[str]:
    errors = []
    if not case.case_id:
        errors.append("case_id is required")
    if not case.task_type:
        errors.append("task_type is required")
    if not case.session_id:
        errors.append("session_id is required")
    return errors


# ===================== Phase 2: 评测相关 =====================

SCORING_DIMENSIONS: dict[str, list[str]] = {
    "user_md_update": [
        "correctness", "coverage", "update_logic",
        "memory_boundary", "conciseness", "format",
    ],
    "day_memory": [],
    "long_memory": [
        "correctness", "coverage", "update_logic",
        "memory_boundary", "conciseness", "format",
    ],
    "summary": [],
}

DIMENSION_WEIGHTS: dict[str, dict[str, float]] = {
    "user_md_update": {
        "correctness": 0.30, "coverage": 0.20, "update_logic": 0.20,
        "memory_boundary": 0.15, "conciseness": 0.10, "format": 0.05,
    },
    "day_memory": {},
    "long_memory": {
        "correctness": 0.30, "coverage": 0.20, "update_logic": 0.20,
        "memory_boundary": 0.15, "conciseness": 0.10, "format": 0.05,
    },
    "summary": {},
}

VALID_ERROR_TAGS = {
    "hallucination", "wrong_fact", "missing_key_info", "over_memory",
    "short_term_pollution", "conflict_not_resolved", "duplicate_memory",
    "verbose_or_noisy", "format_error", "privacy_sensitive", "unclear_update",
}


@dataclass
class EvalConfig:
    judge_model: str = ""
    judge_api_base_url: str = ""
    judge_api_bearer_token: str = ""
    judge_max_tokens: int = 2000
    judge_timeout: int = 120
    judge_max_retries: int = 3

    judge_request_interval: float = 0.0 # 每条case之间等待多少秒
    judge_concurrency: int = 1 # 并发评测请求数，请求启动仍受 judge_request_interval 限制
    judge_qps_backoff: float = 12.0 # 遇到QPS limit后等待多少秒后再重试
    judge_enable_thinking: bool = False # Judge 只需要 JSON，默认关闭 thinking 降低耗时
    judge_send_enable_thinking: bool = True
    judge_send_skip_special_tokens: bool = True
    judge_skip_special_tokens: bool = False
    judge_temperature: float = 0.0
    judge_top_p: float = 1.0
    judge_top_k: int | None = None
    judge_stop: list[str] = field(default_factory=list)
    judge_stream: bool = False
    judge_stream_include_usage: bool = True
    judge_prompt_cache_id: str = ""
    judge_prompt_cache_location: str = "none" # none/top/extra/both

    judge_auth_type: str = "bearer" # bearer/hmac/none
    judge_bearer_header_name: str = "Authorization"
    judge_hmac_access_key: str = ""
    judge_hmac_secret_key: str = ""
    judge_hmac_access_key_header: str = "accessKey"
    judge_hmac_timestamp_header: str = "ts"
    judge_hmac_sign_header: str = "sign"

    judge_call_from: str = "default"
    judge_session_id: str = ""
    judge_interaction_id: int | None = None
    judge_moderation_action: str = ""
    judge_extra_body_json: str = "{}"
    judge_custom_headers_json: str = "{}"

    mock: bool = False

    @classmethod
    def from_env_and_args(
        cls, mock: bool = False, judge_model: str = "",
        api_base: str = "", api_token: str = "",
    ) -> "EvalConfig":
        config = cls(
            mock=mock,
            judge_model=judge_model or os.environ.get("EVAL_MODEL_NAME", ""),
            judge_api_base_url=api_base or os.environ.get("EVAL_API_BASE_URL", ""),
            judge_api_bearer_token=api_token or os.environ.get("EVAL_API_BEARER_TOKEN", ""),
            judge_max_tokens=int(os.environ.get("EVAL_MAX_TOKENS", "2000")),
            judge_timeout=int(os.environ.get("EVAL_TIMEOUT", "120")),
            judge_max_retries=int(os.environ.get("EVAL_MAX_RETRIES", "3")),

            # 新增
            judge_request_interval=float(os.environ.get("EVAL_REQUEST_INTERVAL", "0")),
            judge_concurrency=min(100, max(1, int(os.environ.get("EVAL_CONCURRENCY", "1")))),
            judge_qps_backoff=float(os.environ.get("EVAL_QPS_BACKOFF", "12")),
            judge_enable_thinking=os.environ.get("EVAL_ENABLE_THINKING", "false").lower() in {"1", "true", "yes"},
            judge_send_enable_thinking=os.environ.get("EVAL_SEND_ENABLE_THINKING", "true").lower() in {"1", "true", "yes"},
            judge_send_skip_special_tokens=os.environ.get("EVAL_SEND_SKIP_SPECIAL_TOKENS", "true").lower() in {"1", "true", "yes"},
            judge_skip_special_tokens=os.environ.get("EVAL_SKIP_SPECIAL_TOKENS", "false").lower() in {"1", "true", "yes"},
            judge_temperature=float(os.environ.get("EVAL_TEMPERATURE", "0")),
            judge_top_p=float(os.environ.get("EVAL_TOP_P", "1")),
            judge_top_k=int(os.environ["EVAL_TOP_K"]) if os.environ.get("EVAL_TOP_K") else None,
            judge_stream=os.environ.get("EVAL_STREAM", "false").lower() in {"1", "true", "yes"},
            judge_stream_include_usage=os.environ.get("EVAL_STREAM_INCLUDE_USAGE", "true").lower() in {"1", "true", "yes"},
            judge_auth_type=os.environ.get("EVAL_AUTH_TYPE", "bearer"),
            judge_call_from=os.environ.get("EVAL_CALL_FROM", "default"),
        )
        return config

    def validate(self) -> list[str]:
        errors = []
        if not self.mock:
            if not self.judge_model:
                errors.append("judge_model 未设置（EVAL_MODEL_NAME 或 --judge_model）")
            if not self.judge_api_base_url:
                errors.append("judge_api_base_url 未设置（EVAL_API_BASE_URL 或 --api_base）")
            if not self.judge_api_bearer_token:
                errors.append("judge_api_bearer_token 未设置（EVAL_API_BEARER_TOKEN 或 --api_token）")
        return errors


@dataclass
class EvalResult:
    case_id: str
    task_type: str
    score_total: float
    scores: dict[str, float] = field(default_factory=dict)
    comment: str = ""
    error_tags: list[str] = field(default_factory=list)
    fatal_error: bool = False
    # 被评测对象信息：candidate_output来自哪个模型/prompt
    model_name: str = "unknown"
    prompt_version: str = "unknown"
    # 裁判模型信息：谁负责打分
    judge_model: str = ""
    judge_prompt_version: str = ""
    extraction_prompt_version: str = ""
    extraction_prompt_hash: str = ""
    diagnostics: list[dict] = field(default_factory=list)
    rule_refs: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    output_refs: list[str] = field(default_factory=list)
    
    raw_response: Optional[str] = None
    timestamp: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EvalResult":
        data = dict(data)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_parse_failure(
        cls,
        case_id: str,
        task_type: str,
        raw: str,
        model_name: str = "unknown",
        prompt_version: str = "unknown",
        judge_model: str = "",
        judge_prompt_version: str = "",
        extraction_prompt_version: str = "",
        extraction_prompt_hash: str = "",
    ) -> "EvalResult":
        return cls(
            case_id=case_id,
            task_type=task_type,
            score_total=0.0,
            scores={},
            comment="Judge 调用失败或 JSON 解析失败",
            error_tags=["format_error"],
            fatal_error=True,
            model_name=model_name,
            prompt_version=prompt_version,
            judge_model=judge_model,
            judge_prompt_version=judge_prompt_version,
            extraction_prompt_version=extraction_prompt_version,
            extraction_prompt_hash=extraction_prompt_hash,
            diagnostics=[],
            rule_refs=[],
            evidence_refs=[],
            output_refs=[],
            raw_response=raw,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


def results_to_jsonl(results: list[EvalResult], path: str) -> None:
    atomic_write_jsonl(path, (result.to_dict() for result in results))


def append_result_to_jsonl(result: EvalResult, path: str) -> None:
    append_jsonl(path, result.to_dict())


def results_from_jsonl(path: str) -> list[EvalResult]:
    results_by_key: dict[tuple[str, str, str, str, str, str], EvalResult] = {}
    for row in read_jsonl(path):
        result = EvalResult.from_dict(row)
        key = (
            result.case_id,
            result.model_name or "unknown",
            result.prompt_version or "unknown",
            result.judge_model or "",
            result.judge_prompt_version or "",
            result.extraction_prompt_hash or "",
        )
        results_by_key[key] = result
    return list(results_by_key.values())
