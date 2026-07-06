from __future__ import annotations

import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

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
from src.runtime_paths import APP_HOME, DATA_DIR, ensure_writable_layout


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

        results.append(EvalResult(
            case_id=case_id,
            task_type=_cell_text(row.get("task_type")) or TaskType.USER_MD.value,
            score_total=score_total,
            scores=scores,
            comment=_cell_text(row.get("comment")),
            error_tags=_parse_table_list(row.get("error_tags"), primary_separator=","),
            fatal_error=_parse_table_bool(row.get("fatal_error")),
            model_name=_cell_text(row.get("model_name")) or "unknown",
            prompt_version=_cell_text(row.get("prompt_version")) or "unknown",
            judge_model=_cell_text(row.get("judge_model")),
            judge_prompt_version=_cell_text(row.get("judge_prompt_version")),
            extraction_prompt_version=_cell_text(row.get("extraction_prompt_version")),
            extraction_prompt_hash=_cell_text(row.get("extraction_prompt_hash")),
            diagnostics=_parse_diagnostics(row.get("diagnostics")),
            rule_refs=_parse_table_list(row.get("rule_refs"), primary_separator=";"),
            evidence_refs=_parse_table_list(row.get("evidence_refs"), primary_separator=";"),
            output_refs=_parse_table_list(row.get("output_refs"), primary_separator=";"),
            timestamp=_cell_text(row.get("timestamp")),
        ))

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
) -> tuple[str, str, str, str, str, str]:
    return (
        case_id,
        model_name or "unknown",
        prompt_version or "unknown",
        judge_model or "",
        judge_prompt_version or "",
        extraction_prompt_hash or "",
    )


def case_resume_key(
    case: Case,
    judge_model: str = "",
    judge_prompt_version: str = "",
    extraction_prompt_hash: str = "",
) -> tuple[str, str, str, str, str, str]:
    return resume_result_key(
        case.case_id,
        case.model_name,
        case.prompt_version,
        judge_model,
        judge_prompt_version,
        extraction_prompt_hash,
    )


def eval_result_resume_key(result: EvalResult) -> tuple[str, str, str, str, str, str]:
    return resume_result_key(
        result.case_id,
        result.model_name,
        result.prompt_version,
        result.judge_model,
        result.judge_prompt_version,
        result.extraction_prompt_hash,
    )


def append_result(path: str | Path, result: EvalResult) -> None:
    append_result_to_jsonl(result, str(path))


def save_uploaded_file(uploaded_file, suffix: str = "") -> str:
    """保存 Streamlit UploadedFile 到 data/raw/uploads，返回本地路径。"""
    ensure_dirs()

    original = uploaded_file.name
    ext = suffix or Path(original).suffix
    stem = Path(original).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = UPLOAD_DIR / f"{stem}_{ts}{ext}"

    with open(out_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    return str(out_path)


def save_cases(cases: list[Case], filename: str = "") -> str:
    ensure_dirs()

    if not filename:
        filename = f"cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    if not filename.endswith(".jsonl"):
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
            "fatal_error": r.fatal_error,
            "comment": r.comment,
            "error_tags": ",".join(r.error_tags or []),
            "judge_model": r.judge_model,
            "judge_prompt_version": r.judge_prompt_version,
            "extraction_prompt_version": r.extraction_prompt_version,
            "extraction_prompt_hash": r.extraction_prompt_hash,
            "rule_refs": "; ".join(r.rule_refs or []),
            "evidence_refs": "; ".join(r.evidence_refs or []),
            "output_refs": "; ".join(r.output_refs or []),
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
    candidate_columns: tuple[str, ...] = ("user.md",),
    raw_result_columns: tuple[str, ...] = ("result",),
    explicit_old_columns: tuple[str, ...] = (),
    document_name: str = "USER.md",
    reset_on_reviewer_change: bool = False,
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

            has_extraction = bool(current_user_md or result or reasoning)
            if not has_extraction:
                missing_fields = "_".join(
                    [candidate_columns[0], raw_result_columns[0], "reasoning"]
                ).replace(" ", "_").replace(".", "_")
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
                    "skip_reason": f"chunk_last_row_missing_{missing_fields}",
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
                        "error": error,
                        "raw_result": result,
                        "reasoning": reasoning,
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
                    "error": error,
                    "raw_result": result,
                    "reasoning": reasoning,
                    "loader": "prepare_cases_from_run_output",
                    "document_name": document_name,
                    "candidate_source_column": next(
                        (column for column in candidate_columns if str(boundary_row.get(column, "")).strip()),
                        candidate_columns[0],
                    ),
                    "extraction_status": f"has_{document_name.lower().replace('.', '_')}",
                    "is_missed_case": False,
                },
            )
            cases.append(case)
            if reset_on_reviewer_change:
                if current_user_md:
                    sequential_memory = current_user_md
            else:
                previous_user_md_by_reviewer[safe_reviewer] = current_user_md
            global_chunk_idx += 1
            chunk_in_session += 1

    if not cases and not return_missed:
        raise ValueError(
            f"未生成任何 case：所有 chunk 的最后一行都没有 "
            f"{candidate_columns[0]}/{raw_result_columns[0]}/reasoning。"
        )

    stats = {
        "total_chunks": total_chunks,
        "generated_cases": len(cases),
        "missed_cases": len(missed_cases),
        "skipped_chunks": len(skipped_chunks),
        "skipped_chunk_details": skipped_chunks,
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
        candidate_columns=("MEMORY.md", "生成的MEMORY.md正文", "memory.md"),
        raw_result_columns=("模型原始返回", "result"),
        explicit_old_columns=("旧MEMORY.md", "old_memory"),
        document_name="MEMORY.md",
        reset_on_reviewer_change=True,
    )
