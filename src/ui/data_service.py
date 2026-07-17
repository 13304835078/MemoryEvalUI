from __future__ import annotations

import json
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

from src.extraction.document_parser import normalize_memory_document_body
from src.extraction.contracts import (
    CallStatus,
    CaseStatus,
    InheritanceSource,
    ParseStatus,
    coerce_extraction_state,
    get_extraction_task_profile,
)
from src.schema import (
    Case,
    DialogueTurn,
    EvalResult,
    SCORING_DIMENSIONS,
    TaskType,
    cases_from_jsonl,
    cases_to_jsonl,
    append_result_to_jsonl,
    results_from_jsonl,
)
from src.eval.metrics import flatten_results
from src.eval.result_status import STATUS_LABELS, result_evaluation_status, result_is_score_eligible
from src.runtime_paths import APP_HOME, DATA_DIR, ensure_writable_layout
from src.persistence import atomic_write_bytes


PROJECT_ROOT = APP_HOME
RAW_DIR = DATA_DIR / "raw"
CASES_DIR = DATA_DIR / "cases"
RESULTS_DIR = DATA_DIR / "results"
UPLOAD_DIR = RAW_DIR / "uploads"


def _first_nonempty_cell(row: dict, columns: tuple[str, ...]) -> str:
    for column in columns:
        value = str(row.get(column, "")).strip()
        if value:
            return value
    return ""


def _safe_storage_filename(filename: str, default_name: str) -> str:
    name = Path(str(filename or "")).name.strip().strip(".")
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "_", name).strip()
    return name or default_name


def _status_skip_reason(status: str) -> str:
    normalized = str(status or "").strip().lower()
    safe = "".join(char if char.isalnum() else "_" for char in normalized).strip("_")
    return f"upstream_status_{safe or 'failed'}"


def ensure_dirs() -> None:
    ensure_writable_layout()
    for d in [RAW_DIR, CASES_DIR, RESULTS_DIR, UPLOAD_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def list_files(dir_path: str | Path, suffix: str | tuple[str, ...] = ".jsonl") -> list[str]:
    dir_path = Path(dir_path)
    if not dir_path.exists():
        return []
    if isinstance(suffix, str):
        suffixes = (suffix.lower(),)
    else:
        suffixes = tuple(item.lower() for item in suffix)
    return sorted([
        str(p)
        for p in dir_path.glob("*")
        if p.is_file() and p.name.lower().endswith(suffixes)
    ])


def list_case_files() -> list[str]:
    ensure_dirs()
    return list_files(CASES_DIR, ".jsonl")


def list_result_files() -> list[str]:
    ensure_dirs()
    return list_files(RESULTS_DIR, (".jsonl", ".csv", ".xlsx"))


def load_cases(path: str | Path) -> list[Case]:
    return cases_from_jsonl(str(path))


def load_results(path: str | Path) -> list[EvalResult]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return results_from_jsonl(str(path))
    if suffix == ".csv":
        return results_from_dataframe(pd.read_csv(path, encoding="utf-8-sig").fillna(""))
    if suffix == ".xlsx":
        return results_from_dataframe(pd.read_excel(path).fillna(""))
    raise ValueError(f"不支持的结果文件格式：{suffix}")


def load_results_bytes(content: bytes, filename: str) -> list[EvalResult]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".jsonl":
        rows = [
            EvalResult.from_dict(json.loads(line))
            for line in content.decode("utf-8-sig").splitlines()
            if line.strip()
        ]
        return rows
    if suffix == ".csv":
        return results_from_dataframe(pd.read_csv(BytesIO(content), encoding="utf-8-sig").fillna(""))
    if suffix == ".xlsx":
        return results_from_dataframe(pd.read_excel(BytesIO(content)).fillna(""))
    raise ValueError(f"不支持的结果文件格式：{suffix}")


def results_from_dataframe(df: pd.DataFrame) -> list[EvalResult]:
    if df.empty:
        return []
    required = {"case_id", "score_total"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"结果表缺少必要列：{missing}")

    dimensions = sorted({dimension for values in SCORING_DIMENSIONS.values() for dimension in values})
    results: list[EvalResult] = []
    row_errors: list[str] = []
    for index, row in df.fillna("").iterrows():
        case_id = _cell_text(row.get("case_id"))
        if not case_id:
            row_errors.append(f"第 {index + 2} 行缺少 case_id")
            continue
        try:
            score_total = float(row.get("score_total"))
        except (TypeError, ValueError):
            row_errors.append(f"第 {index + 2} 行 score_total 不是数字")
            continue

        scores: dict[str, float] = {}
        for dimension in dimensions:
            value = row.get(f"score_{dimension}", "")
            if value in ("", None):
                continue
            try:
                scores[dimension] = float(value)
            except (TypeError, ValueError):
                row_errors.append(f"第 {index + 2} 行 score_{dimension} 不是数字")

        result_data = {
            "case_id": case_id,
            "task_type": _cell_text(row.get("task_type")) or TaskType.USER_MD.value,
            "score_total": score_total,
            "scores": scores,
            "comment": _cell_text(row.get("comment")),
            "error_tags": _parse_table_list(row.get("error_tags"), primary_separator=","),
            "fatal_error": _parse_table_bool(row.get("fatal_error")),
            "model_name": _cell_text(row.get("model_name")) or "unknown",
            "prompt_version": _cell_text(row.get("prompt_version")) or "unknown",
            "judge_model": _cell_text(row.get("judge_model")),
            "judge_prompt_version": _cell_text(row.get("judge_prompt_version")),
            "extraction_prompt_version": _cell_text(row.get("extraction_prompt_version")),
            "extraction_prompt_hash": _cell_text(row.get("extraction_prompt_hash")),
            "judge_prompt_hash": _cell_text(row.get("judge_prompt_hash")),
            "scoring_schema_version": _cell_text(row.get("scoring_schema_version")),
            "dimension_weights_version": _cell_text(row.get("dimension_weights_version")),
            "scoring_config_hash": _cell_text(row.get("scoring_config_hash")),
            "case_input_hash": _cell_text(row.get("case_input_hash")),
            "evaluation_fingerprint": _cell_text(row.get("evaluation_fingerprint")),
            "diagnostics": _parse_diagnostics(row.get("diagnostics")),
            "rule_refs": _parse_table_list(row.get("rule_refs"), primary_separator=";"),
            "evidence_refs": _parse_table_list(row.get("evidence_refs"), primary_separator=";"),
            "output_refs": _parse_table_list(row.get("output_refs"), primary_separator=";"),
            "reasoning_refs": _parse_table_list(row.get("reasoning_refs"), primary_separator=";"),
            "failure_type": _cell_text(row.get("failure_type")),
            "failure_message": _cell_text(row.get("failure_message")),
            "timestamp": _cell_text(row.get("timestamp")),
        }
        evaluation_status = _cell_text(row.get("evaluation_status"))
        score_eligible_value = _cell_text(row.get("score_eligible"))
        if evaluation_status:
            result_data["evaluation_status"] = evaluation_status
        if score_eligible_value:
            result_data["score_eligible"] = _parse_table_bool(row.get("score_eligible"))
        results.append(EvalResult.from_dict(result_data))

    if row_errors:
        preview = "；".join(row_errors[:5])
        suffix = f"；另有 {len(row_errors) - 5} 个错误" if len(row_errors) > 5 else ""
        raise ValueError(f"结果表存在无法还原的行：{preview}{suffix}")
    return results


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _parse_table_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _cell_text(value).lower() in {"true", "1", "yes", "y", "是"}


def _parse_table_list(value: Any, *, primary_separator: str) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = _cell_text(value)
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass
    return [item.strip() for item in text.split(primary_separator) if item.strip()]


def _parse_diagnostics(value: Any) -> list[dict]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    text = _cell_text(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    return [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []


def resume_result_key(
    case_id: str,
    model_name: str = "unknown",
    prompt_version: str = "unknown",
    judge_model: str = "",
    judge_prompt_version: str = "",
    extraction_prompt_hash: str = "",
    evaluation_fingerprint: str = "",
) -> tuple[str, str, str, str, str, str, str]:
    return (
        case_id,
        model_name or "unknown",
        prompt_version or "unknown",
        judge_model or "",
        judge_prompt_version or "",
        extraction_prompt_hash or "",
        evaluation_fingerprint or "",
    )


def case_resume_key(
    case: Case,
    judge_model: str = "",
    judge_prompt_version: str = "",
    extraction_prompt_hash: str = "",
    evaluation_fingerprint: str = "",
) -> tuple[str, str, str, str, str, str, str]:
    return resume_result_key(
        case.case_id,
        case.model_name,
        case.prompt_version,
        judge_model,
        judge_prompt_version,
        extraction_prompt_hash,
        evaluation_fingerprint,
    )


def eval_result_resume_key(result: EvalResult) -> tuple[str, str, str, str, str, str, str]:
    return resume_result_key(
        result.case_id,
        result.model_name,
        result.prompt_version,
        result.judge_model,
        result.judge_prompt_version,
        result.extraction_prompt_hash,
        result.evaluation_fingerprint,
    )


def eval_result_row_key(row: Any) -> tuple[str, str, str, str, str, str, str]:
    """Build the same identity key from a dataframe row used by result pages."""
    return resume_result_key(
        str(row.get("case_id", "")),
        str(row.get("model_name", "unknown") or "unknown"),
        str(row.get("prompt_version", "unknown") or "unknown"),
        str(row.get("judge_model", "") or ""),
        str(row.get("judge_prompt_version", "") or ""),
        str(row.get("extraction_prompt_hash", "") or ""),
        str(row.get("evaluation_fingerprint", "") or ""),
    )


def append_result(path: str | Path, result: EvalResult) -> None:
    append_result_to_jsonl(result, str(path))


def save_uploaded_file(uploaded_file, suffix: str = "") -> str:
    """保存 Streamlit UploadedFile 到 data/raw/uploads，返回本地路径。"""
    ensure_dirs()

    original = _safe_storage_filename(uploaded_file.name, "upload")
    ext = suffix or Path(original).suffix
    ext = re.sub(r"[^A-Za-z0-9.]", "", str(ext or ""))
    if ext and not ext.startswith("."):
        ext = "." + ext
    stem = _safe_storage_filename(Path(original).stem, "upload")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = UPLOAD_DIR / f"{stem}_{ts}{ext}"

    atomic_write_bytes(out_path, uploaded_file.getvalue())

    return str(out_path)


def save_cases(cases: list[Case], filename: str = "") -> str:
    ensure_dirs()

    if not filename:
        filename = f"cases_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jsonl"
    filename = _safe_storage_filename(filename, "cases.jsonl")
    if not filename.lower().endswith(".jsonl"):
        filename += ".jsonl"

    path = CASES_DIR / filename
    cases_to_jsonl(cases, str(path))
    return str(path)


def _shorten(text: str | None, n: int = 120) -> str:
    if not text:
        return ""
    text = str(text).replace("\n", " ").strip()
    return text[:n] + ("..." if len(text) > n else "")


def dialogue_to_text(dialogue: list[DialogueTurn], max_turns: int = 8) -> str:
    lines = []
    for turn in dialogue[:max_turns]:
        role = turn.role
        content = _shorten(turn.content, 100)
        lines.append(f"{role}: {content}")
    if len(dialogue) > max_turns:
        lines.append(f"... 共 {len(dialogue)} turns")
    return "\n".join(lines)


def cases_to_dataframe(cases: list[Case]) -> pd.DataFrame:
    rows = []
    for c in cases:
        metadata = c.metadata or {}
        has_reasoning = bool(str(metadata.get("reasoning") or "").strip())
        has_raw_result = bool(str(metadata.get("raw_result") or "").strip())
        has_candidate = bool(c.candidate_output)
        extraction_status = metadata.get("extraction_status")
        if not extraction_status:
            extraction_status = "empty_user_md_with_reasoning" if not has_candidate and (has_reasoning or has_raw_result) else "has_user_md"
        rows.append({
            "case_id": c.case_id,
            "task_type": c.task_type.value if isinstance(c.task_type, TaskType) else str(c.task_type),
            "session_id": c.session_id,
            "model_name": c.model_name,
            "prompt_version": c.prompt_version,
            "extraction_status": extraction_status,
            "skip_reason": metadata.get("skip_reason", ""),
            "source_session_id": metadata.get("source_session_id", ""),
            "row_start": metadata.get("row_start", ""),
            "row_end": metadata.get("row_end", ""),
            "boundary_row": metadata.get("boundary_row", ""),
            "reviewer": metadata.get("reviewer", ""),
            "upstream_status": metadata.get("status", ""),
            "task_profile_id": metadata.get("task_profile_id", ""),
            "call_status": metadata.get("call_status", ""),
            "parse_status": metadata.get("parse_status", ""),
            "case_status": metadata.get("case_status", ""),
            "inheritance_source": metadata.get("inheritance_source", ""),
            "parse_method": metadata.get("parse_method", ""),
            "parse_confidence": metadata.get("parse_confidence", ""),
            "parse_warnings": metadata.get("parse_warnings", ""),
            "raw_output_preview": _shorten(metadata.get("raw_output") or metadata.get("raw_result")),
            "parsed_document_preview": _shorten(metadata.get("parsed_document")),
            "old_memory_preview": _shorten(c.old_memory),
            "candidate_output_preview": _shorten(c.candidate_output),
            "reasoning_preview": _shorten(metadata.get("reasoning")),
            "dialogue_turns": len(c.dialogue or []),
            "dialogue_preview": dialogue_to_text(c.dialogue or [], max_turns=4),
            "metadata": json.dumps(c.metadata, ensure_ascii=False),
        })
    return pd.DataFrame(rows)


def results_to_dataframe(results: list[EvalResult]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(flatten_results(results))


def _task_type_value(value: Any) -> str:
    return value.value if isinstance(value, TaskType) else str(value or "")


def case_match_key(case: Case) -> tuple[str, str, str, str]:
    return (
        _task_type_value(case.task_type),
        case.case_id,
        case.model_name or "unknown",
        case.prompt_version or "unknown",
    )


def result_match_key(result: EvalResult) -> tuple[str, str, str, str]:
    return (
        _task_type_value(result.task_type),
        result.case_id,
        result.model_name or "unknown",
        result.prompt_version or "unknown",
    )


def find_case_for_result(cases: list[Case], result: EvalResult) -> Case | None:
    """优先按 task_type+case_id+model_name+prompt_version 匹配；失败后只在同任务内按 case_id 匹配。"""
    exact = {case_match_key(c): c for c in cases}
    hit = exact.get(result_match_key(result))
    if hit:
        return hit

    result_task_type = _task_type_value(result.task_type)
    for c in cases:
        if _task_type_value(c.task_type) == result_task_type and c.case_id == result.case_id:
            return c

    return None


def merge_cases_results(cases: list[Case], results: list[EvalResult]) -> pd.DataFrame:
    result_rows = []
    for r in results:
        c = find_case_for_result(cases, r)

        row = {
            "case_id": r.case_id,
            "task_type": r.task_type,
            "model_name": r.model_name,
            "prompt_version": r.prompt_version,
            "score_total": r.score_total,
            "score_display": f"{r.score_total:.2f}" if result_is_score_eligible(r) else "未评分",
            "fatal_error": r.fatal_error,
            "evaluation_status": result_evaluation_status(r),
            "evaluation_status_label": STATUS_LABELS.get(result_evaluation_status(r), result_evaluation_status(r)),
            "score_eligible": result_is_score_eligible(r),
            "failure_type": r.failure_type,
            "failure_message": r.failure_message,
            "comment": r.comment,
            "error_tags": ",".join(r.error_tags or []),
            "judge_model": r.judge_model,
            "judge_prompt_version": r.judge_prompt_version,
            "extraction_prompt_version": r.extraction_prompt_version,
            "extraction_prompt_hash": r.extraction_prompt_hash,
            "judge_prompt_hash": r.judge_prompt_hash,
            "scoring_schema_version": r.scoring_schema_version,
            "dimension_weights_version": r.dimension_weights_version,
            "scoring_config_hash": r.scoring_config_hash,
            "case_input_hash": r.case_input_hash,
            "evaluation_fingerprint": r.evaluation_fingerprint,
            "rule_refs": "; ".join(r.rule_refs or []),
            "evidence_refs": "; ".join(r.evidence_refs or []),
            "output_refs": "; ".join(r.output_refs or []),
            "reasoning_refs": "; ".join(r.reasoning_refs or []),
            "diagnostics_count": len(r.diagnostics or []),
            "timestamp": r.timestamp,
        }

        for dim, score in (r.scores or {}).items():
            row[f"score_{dim}"] = score

        if c:
            row.update({
                "session_id": c.session_id,
                "old_memory_preview": _shorten(c.old_memory),
                "candidate_output_preview": _shorten(c.candidate_output),
                "reasoning_preview": _shorten(c.metadata.get("reasoning") if c.metadata else ""),
                "dialogue_turns": len(c.dialogue or []),
                "dialogue_preview": dialogue_to_text(c.dialogue or [], max_turns=4),
                "metadata": json.dumps(c.metadata, ensure_ascii=False),
            })
        else:
            row.update({
                "session_id": "",
                "old_memory_preview": "",
                "candidate_output_preview": "",
                "dialogue_turns": 0,
                "dialogue_preview": "",
                "metadata": "",
            })

        result_rows.append(row)

    return pd.DataFrame(result_rows)


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="results")
    return output.getvalue()


def prepare_cases_from_run_output(
    input_path: str | Path,
    model: str = "unknown",
    prompt_version: str = "unknown",
    chunk_size: int = 10,
    return_stats: bool = False,
    return_missed: bool = False,
    *,
    task_type: TaskType = TaskType.USER_MD,
    candidate_columns: tuple[str, ...] | None = None,
    raw_result_columns: tuple[str, ...] | None = None,
    explicit_old_columns: tuple[str, ...] | None = None,
    document_name: str | None = None,
    reset_on_reviewer_change: bool | None = None,
) -> list[Case] | tuple[list[Case], dict[str, Any]] | tuple[list[Case], list[Case], dict[str, Any]]:
    """把按 session/chunk 生成的记忆 Excel 转成评测 cases。

    逻辑与 run_user.py 保持一致：
    - 先按「轮次 == 1」切 session
    - 每个 session 内按 chunk_size 分 chunk
    - 最后不足 chunk_size 的尾段也作为一个 chunk
    - 每个 chunk 的结果只读取当前 chunk 最后一行
    - 如果末行没有候选结果、原始返回和 reasoning，视为上游漏抽
    - USER.md 按评测人跨 session 继承；长期记忆可按上游规则在评测人切换时清空
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")

    task_type = TaskType(task_type)
    task_profile = get_extraction_task_profile(task_type)
    candidate_columns = tuple(candidate_columns or task_profile.candidate_columns)
    raw_result_columns = tuple(raw_result_columns or task_profile.raw_output_columns)
    explicit_old_columns = tuple(explicit_old_columns or task_profile.old_document_columns)
    document_name = document_name or task_profile.document_name
    if reset_on_reviewer_change is None:
        reset_on_reviewer_change = task_profile.reset_on_reviewer_change

    input_path = Path(input_path)
    df = pd.read_excel(input_path).fillna("")
    rows = df.to_dict("records")
    has_explicit_old_column = any(column in df.columns for column in explicit_old_columns)

    if not rows:
        raise ValueError("输入 Excel 为空。")

    def is_session_start(row: dict, row_index: int) -> bool:
        if row_index == 0:
            return True
        turn = str(row.get("轮次", "")).strip()
        try:
            return int(float(turn)) == 1
        except ValueError:
            return False

    session_ranges: list[tuple[int, int]] = []
    session_starts = [i for i, row in enumerate(rows) if is_session_start(row, i)]
    for pos, start in enumerate(session_starts):
        end = session_starts[pos + 1] if pos + 1 < len(session_starts) else len(rows)
        if start < end:
            session_ranges.append((start, end))

    previous_user_md_by_reviewer: dict[str, str] = {}
    sequential_memory = ""
    last_reviewer = ""
    cases: list[Case] = []
    missed_cases: list[Case] = []
    skipped_chunks: list[dict[str, Any]] = []
    global_chunk_idx = 0
    total_chunks = 0
    call_status_counts: dict[str, int] = {}
    parse_status_counts: dict[str, int] = {}
    case_status_counts: dict[str, int] = {}

    for session_idx, (session_start, session_end) in enumerate(session_ranges):
        session_row = rows[session_start]
        source_session_id = str(session_row.get("session_id", "")).strip() or f"session_{session_idx + 1}"

        chunk_in_session = 0
        for start in range(session_start, session_end, chunk_size):
            total_chunks += 1
            end = min(start + chunk_size, session_end)
            boundary = end - 1

            dialogue: list[DialogueTurn] = []
            for j in range(start, end):
                row = rows[j]
                q = str(row.get("query", "")).strip()
                a = str(row.get("answer", "")).strip()

                if q:
                    dialogue.append(DialogueTurn(
                        role="user",
                        content=q,
                        metadata={"row_index": j + 1, "source_column": "query"},
                    ))
                if a:
                    dialogue.append(DialogueTurn(
                        role="assistant",
                        content=a,
                        metadata={"row_index": j + 1, "source_column": "answer"},
                    ))

            boundary_row = rows[boundary]
            current_user_md = _first_nonempty_cell(boundary_row, candidate_columns)
            result = _first_nonempty_cell(boundary_row, raw_result_columns)
            reasoning = str(boundary_row.get("reasoning", "")).strip()
            status = str(boundary_row.get("status", "")).strip()
            error = str(boundary_row.get("error", "")).strip()
            parse_method = str(boundary_row.get("parse_method", "")).strip()
            parse_confidence = boundary_row.get("parse_confidence", "")
            parse_warnings = str(boundary_row.get("parse_warnings", "")).strip()
            task_profile_id = str(boundary_row.get("task_profile_id", "")).strip() or task_profile.profile_id
            parsed_document = str(boundary_row.get("parsed_document", "")).strip()
            inheritance_source = str(boundary_row.get("inheritance_source", "")).strip()
            propagation_status = str(boundary_row.get("propagation_status", "")).strip()
            state = coerce_extraction_state(
                call_status=boundary_row.get("call_status", ""),
                parse_status=boundary_row.get("parse_status", ""),
                case_status=boundary_row.get("case_status", ""),
                legacy_status=status,
                has_effective_document=bool(current_user_md),
                has_raw_output=bool(result),
                has_reasoning=bool(reasoning),
            )
            if not current_user_md and result and state.parse_status == ParseStatus.RAW_FALLBACK:
                current_user_md = normalize_memory_document_body(result, document_name)
                if current_user_md:
                    parse_method = parse_method or "legacy_raw_fallback"
                    fallback_warning = "未可靠识别正文边界，已按原始输出生成候选 case"
                    parse_warnings = "；".join(filter(None, (parse_warnings, fallback_warning)))
                    if parse_confidence in (None, ""):
                        parse_confidence = 0.25
            if not inheritance_source:
                if state.parse_status == ParseStatus.RAW_FALLBACK and current_user_md:
                    inheritance_source = InheritanceSource.RAW_OUTPUT.value
                elif current_user_md:
                    inheritance_source = InheritanceSource.PARSED_DOCUMENT.value
                else:
                    inheritance_source = InheritanceSource.NONE.value
            low_confidence_candidate = (
                state.parse_status == ParseStatus.RAW_FALLBACK
                or propagation_status == "blocked_low_confidence"
            )
            if low_confidence_candidate:
                propagation_status = "blocked_low_confidence"
                propagation_warning = "低置信候选不会作为后续 case 的旧记忆"
                parse_warnings = "；".join(filter(None, (parse_warnings, propagation_warning)))
            reviewer = str(boundary_row.get("评测人", "")).strip()
            if not reviewer:
                for row in rows[start:end]:
                    reviewer = str(row.get("评测人", "")).strip()
                    if reviewer:
                        break

            safe_model = model or "unknown"
            safe_reviewer = reviewer or "unknown_reviewer"
            session_label = f"session_{source_session_id}"
            chunk_label = f"chunk_{chunk_in_session + 1}"
            if reset_on_reviewer_change:
                if last_reviewer and safe_reviewer != last_reviewer:
                    sequential_memory = ""
                previous_user_md = sequential_memory
                last_reviewer = safe_reviewer
            else:
                previous_user_md = previous_user_md_by_reviewer.get(safe_reviewer, "")
            if has_explicit_old_column:
                previous_user_md = _first_nonempty_cell(boundary_row, explicit_old_columns)

            for target, value in (
                (call_status_counts, state.call_status.value),
                (parse_status_counts, state.parse_status.value),
                (case_status_counts, state.case_status.value),
            ):
                target[value] = target.get(value, 0) + 1

            case_eligible = state.case_status in {CaseStatus.READY, CaseStatus.REVIEW_REQUIRED}
            if not case_eligible:
                if status:
                    skip_reason = _status_skip_reason(status)
                elif state.call_status != CallStatus.NOT_ATTEMPTED:
                    skip_reason = f"upstream_call_status_{state.call_status.value}"
                else:
                    missing_fields = "_".join(
                        [task_profile.legacy_candidate_column, task_profile.legacy_raw_output_column, "reasoning"]
                    ).replace(" ", "_").replace(".", "_")
                    skip_reason = f"chunk_last_row_missing_{missing_fields}"
                skip_detail = {
                    "source_file": input_path.name,
                    "source_session_id": source_session_id,
                    "session_index": session_idx,
                    "chunk_index": global_chunk_idx,
                    "chunk_index_in_session": chunk_in_session,
                    "chunk_size": chunk_size,
                    "row_start": start + 1,
                    "row_end": boundary + 1,
                    "boundary_row": boundary + 1,
                    "reviewer": reviewer,
                    "skip_reason": skip_reason,
                }
                skipped_chunks.append(skip_detail)
                missed_cases.append(Case(
                    case_id=f"missed_{safe_model}_{safe_reviewer}_{session_label}_{chunk_label}",
                    task_type=task_type,
                    session_id=source_session_id,
                    old_memory=previous_user_md if previous_user_md else None,
                    dialogue=dialogue,
                    candidate_output=None,
                    model_name=model or "unknown",
                    prompt_version=prompt_version or "unknown",
                    metadata={
                        **skip_detail,
                        "status": status,
                        "task_profile_id": task_profile_id,
                        "call_status": state.call_status.value,
                        "parse_status": state.parse_status.value,
                        "case_status": state.case_status.value,
                        "error": error,
                        "raw_result": result,
                        "raw_output": result,
                        "parsed_document": parsed_document,
                        "effective_document": current_user_md,
                        "old_effective_document": previous_user_md,
                        "inheritance_source": inheritance_source,
                        "propagation_status": propagation_status,
                        "reasoning": reasoning,
                        "parse_method": parse_method,
                        "parse_confidence": parse_confidence,
                        "parse_warnings": parse_warnings,
                        "loader": "prepare_cases_from_run_output",
                        "document_name": document_name,
                        "extraction_status": "missed_extraction",
                        "is_missed_case": True,
                    },
                ))
                global_chunk_idx += 1
                chunk_in_session += 1
                continue

            case_id = f"{safe_model}_{safe_reviewer}_{session_label}_{chunk_label}"

            case = Case(
                case_id=case_id,
                task_type=task_type,
                session_id=source_session_id,
                old_memory=previous_user_md if previous_user_md else None,
                dialogue=dialogue,
                candidate_output=current_user_md if current_user_md else None,
                model_name=model or "unknown",
                prompt_version=prompt_version or "unknown",
                metadata={
                    "source_file": input_path.name,
                    "source_session_id": source_session_id,
                    "session_index": session_idx,
                    "chunk_index": global_chunk_idx,
                    "chunk_index_in_session": chunk_in_session,
                    "chunk_size": chunk_size,
                    "row_start": start + 1,
                    "row_end": boundary + 1,
                    "boundary_row": boundary + 1,
                    "reviewer": reviewer,
                    "status": status,
                    "task_profile_id": task_profile_id,
                    "call_status": state.call_status.value,
                    "parse_status": state.parse_status.value,
                    "case_status": state.case_status.value,
                    "error": error,
                    "raw_result": result,
                    "raw_output": result,
                    "parsed_document": parsed_document,
                    "effective_document": current_user_md,
                    "old_effective_document": previous_user_md,
                    "inheritance_source": inheritance_source,
                    "propagation_status": propagation_status,
                    "reasoning": reasoning,
                    "parse_method": parse_method,
                    "parse_confidence": parse_confidence,
                    "parse_warnings": parse_warnings,
                    "loader": "prepare_cases_from_run_output",
                    "document_name": document_name,
                    "candidate_source_column": next(
                        (column for column in candidate_columns if str(boundary_row.get(column, "")).strip()),
                        candidate_columns[0],
                    ),
                    "extraction_status": (
                        "needs_parse_review"
                        if state.case_status == CaseStatus.REVIEW_REQUIRED
                        else f"has_{document_name.lower().replace('.', '_')}"
                    ),
                    "is_missed_case": False,
                },
            )
            cases.append(case)
            if reset_on_reviewer_change:
                if current_user_md and not low_confidence_candidate:
                    sequential_memory = current_user_md
            else:
                if not low_confidence_candidate:
                    previous_user_md_by_reviewer[safe_reviewer] = current_user_md
            global_chunk_idx += 1
            chunk_in_session += 1

    if not cases and not return_missed:
        raise ValueError(
            f"未生成任何 case：所有 chunk 的最后一行都没有 "
            f"{task_profile.legacy_candidate_column}/{task_profile.legacy_raw_output_column}/reasoning。"
        )

    stats = {
        "total_chunks": total_chunks,
        "generated_cases": len(cases),
        "missed_cases": len(missed_cases),
        "skipped_chunks": len(skipped_chunks),
        "skipped_chunk_details": skipped_chunks,
        "task_profile_id": task_profile.profile_id,
        "call_status_counts": call_status_counts,
        "parse_status_counts": parse_status_counts,
        "case_status_counts": case_status_counts,
        "missed_reason_counts": {
            reason: sum(1 for item in skipped_chunks if item.get("skip_reason") == reason)
            for reason in sorted({str(item.get("skip_reason") or "") for item in skipped_chunks})
            if reason
        },
    }
    if return_missed:
        return cases, missed_cases, stats

    if return_stats:
        return cases, stats

    return cases


def prepare_long_memory_cases_from_run_output(
    input_path: str | Path,
    model: str = "unknown",
    prompt_version: str = "unknown",
    chunk_size: int = 10,
    return_stats: bool = False,
    return_missed: bool = False,
) -> list[Case] | tuple[list[Case], dict[str, Any]] | tuple[list[Case], list[Case], dict[str, Any]]:
    """把长期记忆提取程序输出 Excel 转成 long_memory cases。"""
    return prepare_cases_from_run_output(
        input_path,
        model=model,
        prompt_version=prompt_version,
        chunk_size=chunk_size,
        return_stats=return_stats,
        return_missed=return_missed,
        task_type=TaskType.LONG_MEMORY,
    )
