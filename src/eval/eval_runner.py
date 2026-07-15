import hashlib
import json
import os
import logging
import re
import time
from datetime import datetime, timezone

from .. import runtime_paths
from ..schema import (
    Case, TaskType, EvalConfig, EvalResult,
    SCORING_DIMENSIONS, DIMENSION_WEIGHTS, VALID_ERROR_TAGS,
)
from ..extraction.contracts import get_extraction_task_profile
from ..llm_api import normalize_chat_completions_url
from .fingerprint import case_input_hash, evaluation_fingerprint
from .judge_client import RealJudgeClient, MockJudgeClient, JudgeClient

logger = logging.getLogger(__name__)
RULE_ID_RE = re.compile(r"(?<![A-Za-z0-9])R\d+(?![A-Za-z0-9])")
SCORING_SCHEMA_VERSION = "absolute_eval_schema_v2"

PROMPT_MAP = {
    TaskType.DAY_MEMORY: "judge_day_memory_v1.md",
    TaskType.SUMMARY: "judge_summary_v1.md",
}

JUDGE_PROMPT_VERSION = {
    TaskType.DAY_MEMORY: "v1",
    TaskType.SUMMARY: "v1",
}


def _normalize_str_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _extract_rule_reference_candidates(prompt_text: str, max_items: int = 80) -> list[str]:
    candidates: list[str] = []
    seen = set()
    for line in prompt_text.splitlines():
        text = line.strip()
        if not text.startswith("#"):
            continue
        text = text[:120]
        if text and text not in seen:
            seen.add(text)
            candidates.append(text)
        if len(candidates) >= max_items:
            break
    return candidates


def _filter_invalid_rule_id_refs(refs: list[str], extraction_prompt_text: str) -> list[str]:
    if not extraction_prompt_text:
        return refs
    allowed_rule_ids = set(RULE_ID_RE.findall(extraction_prompt_text))
    filtered: list[str] = []
    for ref in refs:
        ref_rule_ids = set(RULE_ID_RE.findall(ref))
        if ref_rule_ids and not ref_rule_ids.issubset(allowed_rule_ids):
            continue
        filtered.append(ref)
    return filtered


def _normalize_diagnostics(value) -> list[dict]:
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return []

    diagnostics = []
    for item in value:
        if not isinstance(item, dict):
            continue

        def pick(*keys: str):
            for key in keys:
                if key in item:
                    return item.get(key)
            return None

        normalized = {
            "dimension": str(pick("dimension", "维度") or "").strip(),
            "severity": str(pick("severity", "严重程度") or "").strip(),
            "rule_refs": _normalize_str_list(pick("rule_refs", "rule_references", "规则引用", "规则依据") or []),
            "evidence_refs": _normalize_str_list(pick("evidence_refs", "evidence_references", "证据引用", "事实证据") or []),
            "output_refs": _normalize_str_list(pick("output_refs", "output_references", "输出引用", "候选输出引用") or []),
            "reasoning_refs": _normalize_str_list(pick("reasoning_refs", "reasoning_references", "推理引用") or []),
            "reason": str(pick("reason", "rationale", "理由", "原因") or "").strip(),
        }
        diagnostics.append(normalized)
    return diagnostics


class EvalRunner:
    def __init__(
        self,
        config: EvalConfig,
        task_type: TaskType,
        prompts_dir: str = "",
        prompt_file: str = "",
        judge_prompt_version: str = "",
        system_prompt_override: str = "",
        extraction_prompt_text: str = "",
        extraction_prompt_version: str = "",
        extraction_prompt_hash: str = "",
    ):
        self.config = config
        self.task_type = TaskType(task_type)
        try:
            self.task_profile = get_extraction_task_profile(self.task_type)
        except ValueError:
            self.task_profile = None
        self.prompts_dir = prompts_dir or str(runtime_paths.PROMPTS_DIR / "judge")
        self.prompt_file = prompt_file
        self.judge_prompt_version = judge_prompt_version
        self.system_prompt_override = system_prompt_override
        self.extraction_prompt_text = extraction_prompt_text.strip()
        self.extraction_prompt_version = extraction_prompt_version
        self.extraction_prompt_hash = extraction_prompt_hash or self._hash_prompt(self.extraction_prompt_text)
        self.document_name = self.task_profile.document_name if self.task_profile else "候选输出"
        profile_version = self.task_profile.default_judge_prompt_version if self.task_profile else ""
        self.resolved_judge_prompt_version = (
            self.judge_prompt_version
            or profile_version
            or JUDGE_PROMPT_VERSION.get(self.task_type, "v1")
        )

        self.system_prompt = self._load_judge_prompt()
        if self.extraction_prompt_text:
            self.system_prompt = self._append_extraction_stability_contract(
                self.system_prompt,
                self.document_name,
            )
        self.judge_prompt_hash = self._hash_prompt(self.system_prompt)
        self.dimension_weights_version = f"{self.task_type.value}_weights_v1"
        self.scoring_config_hash = self._build_scoring_config_hash()
        if (
            str(getattr(self.config, "judge_prompt_cache_location", "none") or "none").lower() != "none"
            and not str(getattr(self.config, "judge_prompt_cache_id", "") or "").strip()
        ):
            cache_seed = hashlib.sha256(
                f"{self.task_type.value}|{self.judge_prompt_hash}|{self.extraction_prompt_hash}|{self.config.judge_model}".encode("utf-8")
            ).hexdigest()[:24]
            self.config.judge_prompt_cache_id = f"memory_eval_{self.task_type.value}_{cache_seed}"
        self.judge_client: JudgeClient = self._create_judge_client()

    @staticmethod
    def _hash_prompt(prompt_text: str) -> str:
        if not prompt_text:
            return ""
        return hashlib.sha1(prompt_text.encode("utf-8")).hexdigest()

    def _build_scoring_config_hash(self) -> str:
        payload = {
            "task_type": self.task_type.value,
            "mock": bool(self.config.mock),
            "judge_api_endpoint": normalize_chat_completions_url(self.config.judge_api_base_url),
            "judge_model": self.config.judge_model or "mock",
            "judge_prompt_hash": self.judge_prompt_hash,
            "extraction_prompt_hash": self.extraction_prompt_hash,
            "scoring_schema_version": SCORING_SCHEMA_VERSION,
            "dimension_weights_version": self.dimension_weights_version,
            "dimensions": SCORING_DIMENSIONS.get(self.task_type.value, []),
            "weights": DIMENSION_WEIGHTS.get(self.task_type.value, {}),
            "temperature": self.config.judge_temperature,
            "top_p": self.config.judge_top_p,
            "top_k": self.config.judge_top_k,
            "enable_thinking": self.config.judge_enable_thinking,
            "send_enable_thinking": self.config.judge_send_enable_thinking,
            "send_skip_special_tokens": self.config.judge_send_skip_special_tokens,
            "skip_special_tokens": self.config.judge_skip_special_tokens,
            "max_tokens": self.config.judge_max_tokens,
        }
        value = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def case_input_hash(self, case: Case) -> str:
        return case_input_hash(case)

    def evaluation_fingerprint(self, case: Case) -> str:
        return evaluation_fingerprint(self.case_input_hash(case), self.scoring_config_hash)

    @staticmethod
    def _append_extraction_stability_contract(system_prompt: str, document_name: str = "USER.md") -> str:
        contract = """

## 提取规则辅助评测稳定契约

本次用户消息可能包含“提取规则”。提取规则只用于判断候选 USER.md 是否遵守抽取/更新规则，不是用户事实来源。
事实判断只能依据旧 USER.md 和对话记录。新 USER.md 是被检查的候选输出，不是外部事实来源；模型 reasoning 只用于诊断提取过程，不能用于补充或证明用户事实。

输出要求：
- 保持原有必需 JSON 字段：score_total、scores、comment、error_tags、fatal_error。
- comment 必须简短引用提取 prompt 中真实存在的章节标题、编号或原文短句，例如“符合‘## 1. 只基于 user 提取 / A. 允许记录’；无明显边界污染”。
- 每条结果都必须填写顶层 rule_refs、evidence_refs、output_refs；即使满分也要引用支持“合规”的规则、事实证据和候选输出片段。
- 任一维度低于 5 分或 error_tags 非空时，必须补充至少一项 diagnostics；每项包含 dimension、severity、rule_refs、evidence_refs、output_refs、reason，可选 reasoning_refs。
- rule_refs 引用提取规则；evidence_refs 只能引用旧 USER.md 或对话；output_refs 引用新 USER.md；reasoning_refs 仅用于指出 reasoning 中的过程问题。
- rule_refs 必须逐字来自提取 prompt 中真实存在的编号、标题或短句；如果提取 prompt 没有 R1/R2/R3/R4 这类编号，禁止输出这类编号。
- 对同类错误使用相同扣分尺度，不因措辞或样本顺序改变评分。
- 没有明确证据或明确规则依据时，不要重扣分。
""".strip()
        contract = contract.replace("USER.md", document_name)
        if "提取规则辅助评测稳定契约" in system_prompt:
            return system_prompt
        return f"{system_prompt.rstrip()}\n\n{contract}\n"

    def _load_judge_prompt(self) -> str:
        if self.system_prompt_override:
            return self.system_prompt_override

        profile_prompt = self.task_profile.default_judge_prompt if self.task_profile else ""
        filename = self.prompt_file or profile_prompt or PROMPT_MAP.get(self.task_type)
        if not filename:
            raise NotImplementedError(f"task_type {self.task_type.value} 无对应 judge prompt 映射")

        prompt_path = filename
        if not os.path.isabs(prompt_path):
            prompt_path = os.path.join(self.prompts_dir, filename)

        if not os.path.isfile(prompt_path):
            bundled_path = runtime_paths.BUNDLED_PROMPTS_DIR / "judge" / filename
            if bundled_path.is_file():
                prompt_path = str(bundled_path)
            else:
                raise FileNotFoundError(f"Judge prompt 文件不存在: {prompt_path}")

        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def _create_judge_client(self) -> JudgeClient:
        if self.config.mock:
            return MockJudgeClient(self.config)
        return RealJudgeClient(self.config)

    def run(self, cases: list[Case], progress_callback=None) -> list[EvalResult]:
        results = []
        total = len(cases)

        request_interval = float(getattr(self.config, "judge_request_interval", 0.0) or 0.0)

        for i, case in enumerate(cases):
            # 限流：从第二条开始，每条 case 前等待
            if i > 0 and request_interval > 0 and not self.config.mock:
                logger.info(f"限流等待 {request_interval:.1f}s 后继续评测")
                if progress_callback:
                    progress_callback(i, total, case, None)
                time.sleep(request_interval)

            logger.info(f"评测 {i + 1}/{total}: {case.case_id}")

            result = self.evaluate_one(case)
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, total, case, result)

        return results

    def _build_user_message(self, case: Case) -> str:
        document_name = self.document_name
        old = case.old_memory or "（空）"
        dialogue_lines = []
        for turn in case.dialogue:
            dialogue_lines.append(f"- **{turn.role}**: {turn.content}")
        dialogue_text = "\n".join(dialogue_lines) if dialogue_lines else "（空）"
        candidate = case.candidate_output or ""
        reasoning = ""
        if isinstance(case.metadata, dict):
            reasoning = str(case.metadata.get("reasoning") or "").strip()
        reasoning_text = reasoning or "（空）"
        extraction_rules = self.extraction_prompt_text or ""
        extraction_section = ""
        if extraction_rules:
            rule_candidates = _extract_rule_reference_candidates(extraction_rules)
            rule_candidates_text = "\n".join(f"- {item}" for item in rule_candidates) or "（未提取到标题，可引用提取 prompt 原文短句）"
            extraction_section = (
                "## 提取规则（仅作为规则依据，不是事实来源）\n"
                f"下面是生成 {document_name} 时使用或参考的提取 prompt。评测时只能把它作为规则来源，"
                f"不能把其中的描述当作用户事实。事实依据仍然只能来自旧 {document_name}、"
                f"对话记录。新 {document_name} 是被检查对象；模型 reasoning 只用于过程诊断，不能证明用户事实。\n\n"
                f"{extraction_rules}\n\n"
                "## 可引用的提取规则标题清单\n"
                "下面这些标题来自提取 prompt。请优先在 comment、rule_refs 和 diagnostics.rule_refs 中引用这些真实标题；"
                "也可以引用提取 prompt 中真实存在的原文短句。\n"
                f"{rule_candidates_text}\n\n"
                "## 规则引用要求\n"
                "只要提供了提取规则，每条结果都必须在 JSON 顶层给出 rule_refs、evidence_refs、output_refs。\n"
                "- rule_refs: 必须逐字引用提取规则中真实存在的编号、标题或原文短片段；不要发明规则编号。\n"
                f"- evidence_refs: 只能引用旧 {document_name} 或对话记录中支持判断的事实证据。\n"
                f"- output_refs: 引用新 {document_name} 中对应的输出片段；"
                f"如果新 {document_name} 为空但合理，请写“新 {document_name} 为空”。\n"
                "- reasoning_refs: 可选，仅用于指出模型 reasoning 中的过程问题；不得把 reasoning 内容当作事实证据。\n"
                "- comment: 必须包含主要规则引用，例如“符合‘## 1. 只基于 user 提取 / A. 允许记录’；无明显边界污染”。\n"
                "如果提取规则中没有 R1/R2/R3/R4 这类编号，禁止在 rule_refs、diagnostics 或 comment 中输出这类编号。\n"
                "任一维度低于 5 分或 error_tags 非空时，必须在 diagnostics 中至少给出一项："
                "dimension、severity、rule_refs、evidence_refs、output_refs、reason，可选 reasoning_refs。\n"
                "如果没有明确证据，不要重扣分，但仍要引用支持当前判断的规则和证据。\n\n"
            )

        return (
            f"{extraction_section}"
            f"## 旧 {document_name}\n{old}\n\n"
            f"## 对话记录\n{dialogue_text}\n\n"
            f"## 模型 reasoning\n{reasoning_text}\n\n"
            f"## 新 {document_name}\n{candidate}"
        )

    def _parse_judge_result(self, case: Case, judge_response: dict,
        raw_response: str, prompt_version: str) -> EvalResult:
        input_hash = self.case_input_hash(case)
        scores = judge_response.get("scores", {})
        expected_dims = (
            list(self.task_profile.scoring_dimensions)
            if self.task_profile
            else SCORING_DIMENSIONS.get(self.task_type.value, [])
        )
        validated_scores = {}
        for dim in expected_dims:
            val = scores.get(dim, 0)
            if isinstance(val, (int, float)):
                validated_scores[dim] = float(max(0, min(5, val)))
            else:
                validated_scores[dim] = 0.0

        error_tags = judge_response.get("error_tags", [])
        if isinstance(error_tags, list):
            error_tags = [t for t in error_tags if t in VALID_ERROR_TAGS]
        else:
            error_tags = []

        weights = DIMENSION_WEIGHTS.get(self.task_type.value, {})
        if expected_dims and weights:
            score_total = round(sum(
                validated_scores.get(dim, 0.0) * weights.get(dim, 0.0)
                for dim in expected_dims
            ), 2)
        else:
            score_total = judge_response.get("score_total", 0.0)
            if isinstance(score_total, (int, float)):
                score_total = float(score_total)
            else:
                score_total = 0.0
            score_total = round(max(0.0, min(5.0, score_total)), 2)

        comment = judge_response.get("comment", "")
        fatal_error = judge_response.get("fatal_error", False)
        diagnostics = _normalize_diagnostics(judge_response.get("diagnostics", []))
        for item in diagnostics:
            item["rule_refs"] = _filter_invalid_rule_id_refs(
                _normalize_str_list(item.get("rule_refs", [])),
                self.extraction_prompt_text,
            )
        rule_refs = _filter_invalid_rule_id_refs(
            _normalize_str_list(judge_response.get("rule_refs", [])),
            self.extraction_prompt_text,
        )
        evidence_refs = _normalize_str_list(judge_response.get("evidence_refs", []))
        output_refs = _normalize_str_list(judge_response.get("output_refs", []))
        reasoning_refs = _normalize_str_list(judge_response.get("reasoning_refs", []))
        for item in diagnostics:
            rule_refs.extend(_normalize_str_list(item.get("rule_refs", [])))
            evidence_refs.extend(_normalize_str_list(item.get("evidence_refs", [])))
            output_refs.extend(_normalize_str_list(item.get("output_refs", [])))
            reasoning_refs.extend(_normalize_str_list(item.get("reasoning_refs", [])))

        return EvalResult(
            case_id=case.case_id,
            task_type=self.task_type.value,
            score_total=score_total,
            scores=validated_scores,
            comment=str(comment) if comment else "",
            error_tags=error_tags,
            fatal_error=bool(fatal_error),

            model_name=case.model_name,
            prompt_version=case.prompt_version,

            judge_model=self.config.judge_model or "mock",
            judge_prompt_version=prompt_version,
            extraction_prompt_version=self.extraction_prompt_version,
            extraction_prompt_hash=self.extraction_prompt_hash,
            judge_prompt_hash=self.judge_prompt_hash,
            scoring_schema_version=SCORING_SCHEMA_VERSION,
            dimension_weights_version=self.dimension_weights_version,
            scoring_config_hash=self.scoring_config_hash,
            case_input_hash=input_hash,
            evaluation_fingerprint=evaluation_fingerprint(input_hash, self.scoring_config_hash),
            diagnostics=diagnostics,
            rule_refs=_dedupe_preserve_order(rule_refs),
            evidence_refs=_dedupe_preserve_order(evidence_refs),
            output_refs=_dedupe_preserve_order(output_refs),
            reasoning_refs=_dedupe_preserve_order(reasoning_refs),
            raw_response=raw_response,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def evaluate_one(self, case: Case) -> EvalResult:
        prompt_version = self.resolved_judge_prompt_version

        user_message = self._build_user_message(case)
        judge_response, raw = self.judge_client.judge(self.system_prompt, user_message)

        if judge_response is None:
            input_hash = self.case_input_hash(case)
            result = EvalResult.from_parse_failure(
                case_id=case.case_id,
                task_type=self.task_type.value,
                raw=raw or "",
                model_name=case.model_name,
                prompt_version=case.prompt_version,
                judge_model=self.config.judge_model or "mock",
                judge_prompt_version=prompt_version,
                extraction_prompt_version=self.extraction_prompt_version,
                extraction_prompt_hash=self.extraction_prompt_hash,
                judge_prompt_hash=self.judge_prompt_hash,
                scoring_schema_version=SCORING_SCHEMA_VERSION,
                dimension_weights_version=self.dimension_weights_version,
                scoring_config_hash=self.scoring_config_hash,
                case_input_hash=input_hash,
                evaluation_fingerprint=evaluation_fingerprint(input_hash, self.scoring_config_hash),
            )
        else:
            result = self._parse_judge_result(case, judge_response, raw or "", prompt_version)

        return result
