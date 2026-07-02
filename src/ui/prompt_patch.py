from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from typing import Any


HEADING_RE = re.compile(r"(?m)^(#{1,6})\s+(.+?)\s*$")
LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)、]\s+)")
ALLOWED_OPS = {
    "insert_before_section",
    "insert_after_section",
    "append_to_section",
    "replace_within_section",
}


@dataclass(frozen=True)
class PromptSection:
    section_id: str
    title: str
    level: int
    start: int
    end: int
    text: str


def split_prompt_sections(prompt_text: str) -> list[PromptSection]:
    """Split prompt text at semantic Markdown/YAML-ish boundaries.

    The splitter is deliberately conservative: headings, paragraphs, and full
    list items are boundaries; sentence-length or fixed-width chunks are not.
    """
    text = prompt_text or ""
    if not text:
        return []

    heading_matches = list(HEADING_RE.finditer(text))
    if heading_matches:
        sections: list[PromptSection] = []
        for index, match in enumerate(heading_matches, 1):
            start = match.start()
            end = heading_matches[index].start() if index < len(heading_matches) else len(text)
            title = match.group(0).strip()
            sections.append(PromptSection(
                section_id=f"S{index:03d}",
                title=title,
                level=len(match.group(1)),
                start=start,
                end=end,
                text=text[start:end],
            ))
        return sections

    blocks = _split_non_markdown_blocks(text)
    sections = []
    for index, (start, end, title) in enumerate(blocks, 1):
        sections.append(PromptSection(
            section_id=f"S{index:03d}",
            title=title,
            level=0,
            start=start,
            end=end,
            text=text[start:end],
        ))
    return sections


def _split_non_markdown_blocks(text: str) -> list[tuple[int, int, str]]:
    blocks: list[tuple[int, int, str]] = []
    start = 0
    current = 0
    lines = text.splitlines(keepends=True)

    for line in lines:
        stripped = line.strip()
        is_boundary = not stripped
        if is_boundary and current > start:
            blocks.append((start, current, _block_title(text[start:current], len(blocks) + 1)))
            start = current + len(line)
        current += len(line)

    if start < len(text):
        blocks.append((start, len(text), _block_title(text[start:], len(blocks) + 1)))

    if len(blocks) <= 1:
        return [(0, len(text), "全文")]
    return blocks


def _block_title(text: str, index: int) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return f"段落 {index}: {stripped[:60]}"
    return f"段落 {index}"


def prompt_sections_for_model(prompt_text: str, max_sections: int = 120, preview_chars: int = 700) -> list[dict[str, Any]]:
    sections = split_prompt_sections(prompt_text)
    rows = []
    for section in sections[:max_sections]:
        rows.append({
            "section_id": section.section_id,
            "title": section.title,
            "level": section.level,
            "preview": _truncate(section.text.strip(), preview_chars),
        })
    return rows


def parse_prompt_patch(value: Any) -> dict[str, Any]:
    if not value:
        return {"mode": "incremental_patch", "edits": []}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"mode": "incremental_patch", "edits": []}
        value = parsed
    if isinstance(value, list):
        return {"mode": "incremental_patch", "edits": value}
    if isinstance(value, dict):
        edits = value.get("edits")
        if isinstance(edits, list):
            return {**value, "mode": value.get("mode") or "incremental_patch", "edits": edits}
    return {"mode": "incremental_patch", "edits": []}


def apply_prompt_patch(
    original_prompt: str,
    patch: Any,
    *,
    max_change_ratio: float = 0.25,
    min_change_chars: int = 800,
) -> dict[str, Any]:
    original = original_prompt or ""
    parsed_patch = parse_prompt_patch(patch)
    sections = split_prompt_sections(original)
    section_map = {section.section_id: section for section in sections}
    max_change_chars = max(min_change_chars, int(len(original) * max_change_ratio))

    operations: list[dict[str, Any]] = []
    applied_edits: list[dict[str, Any]] = []
    skipped_edits: list[dict[str, Any]] = []
    changed_chars = 0

    for index, raw_edit in enumerate(parsed_patch.get("edits") or [], 1):
        edit = _normalize_edit(raw_edit, index)
        ok, message = _validate_edit_basics(edit, section_map)
        if not ok:
            skipped_edits.append({**edit, "status": "skipped", "message": message})
            continue

        section = section_map[edit["target_id"]]
        op_result = _build_operation(original, section, edit)
        if not op_result["ok"]:
            skipped_edits.append({**edit, "status": "skipped", "message": op_result["message"]})
            continue

        next_changed_chars = changed_chars + int(op_result["change_chars"])
        if next_changed_chars > max_change_chars:
            skipped_edits.append({
                **edit,
                "status": "skipped",
                "message": f"累计修改超过安全上限 {max_change_chars} 字符；为避免整篇重写，未自动应用。",
            })
            continue

        operations.append({
            "index": index,
            "start": op_result["start"],
            "end": op_result["end"],
            "replacement": op_result["replacement"],
        })
        changed_chars = next_changed_chars
        applied_edits.append({
            **edit,
            "status": "applied",
            "message": "已应用",
            "change_chars": op_result["change_chars"],
        })

    candidate = _apply_operations(original, operations)
    diff = make_prompt_diff(original, candidate)
    return {
        "mode": parsed_patch.get("mode") or "incremental_patch",
        "candidate_prompt": candidate,
        "diff": diff,
        "applied_edits": applied_edits,
        "skipped_edits": skipped_edits,
        "sections": [
            {
                "section_id": section.section_id,
                "title": section.title,
                "level": section.level,
                "start": section.start,
                "end": section.end,
            }
            for section in sections
        ],
        "change_chars": changed_chars,
        "max_change_chars": max_change_chars,
        "change_ratio": round((changed_chars / len(original)) if original else 0.0, 4),
    }


def make_prompt_diff(old: str, new: str) -> str:
    if old == new:
        return ""
    return "\n".join(difflib.unified_diff(
        (old or "").splitlines(),
        (new or "").splitlines(),
        fromfile="original_prompt",
        tofile="candidate_prompt",
        lineterm="",
    ))


def _normalize_edit(raw_edit: Any, index: int) -> dict[str, Any]:
    item = raw_edit if isinstance(raw_edit, dict) else {}
    op = str(item.get("op") or item.get("operation") or "").strip()
    target_id = str(item.get("target_id") or item.get("section_id") or item.get("target") or "").strip()
    old_text = str(item.get("old_text") or item.get("target_text") or "").strip()
    new_text = str(item.get("new_text") or item.get("replacement_text") or item.get("text") or "").strip()
    text = str(item.get("text") or item.get("insert_text") or new_text or "").strip()
    return {
        "edit_id": str(item.get("edit_id") or f"E{index:03d}"),
        "op": op,
        "target_id": target_id,
        "old_text": old_text,
        "new_text": new_text,
        "text": text,
        "reason": str(item.get("reason") or "").strip(),
        "evidence_refs": _normalize_string_list(item.get("evidence_refs") or item.get("case_refs") or []),
    }


def _validate_edit_basics(edit: dict[str, Any], section_map: dict[str, PromptSection]) -> tuple[bool, str]:
    if edit["op"] not in ALLOWED_OPS:
        return False, f"不支持的操作：{edit['op']}。只允许 {sorted(ALLOWED_OPS)}。"
    if not edit["target_id"] or edit["target_id"] not in section_map:
        return False, f"目标 section 不存在：{edit['target_id']}"
    if not edit["evidence_refs"]:
        return False, "缺少 evidence_refs；为避免无依据改 prompt，未自动应用。"
    if edit["op"] == "replace_within_section" and not edit["old_text"]:
        return False, "replace_within_section 缺少 old_text。"
    if edit["op"] == "replace_within_section" and not edit["new_text"]:
        return False, "replace_within_section 缺少 new_text。"
    if edit["op"] != "replace_within_section" and not edit["text"]:
        return False, "插入类操作缺少 text。"
    return True, ""


def _build_operation(original: str, section: PromptSection, edit: dict[str, Any]) -> dict[str, Any]:
    op = edit["op"]
    if op == "replace_within_section":
        relative = section.text.find(edit["old_text"])
        if relative < 0:
            return {"ok": False, "message": "old_text 未在目标 section 中精确命中。"}
        start = section.start + relative
        end = start + len(edit["old_text"])
        return {
            "ok": True,
            "start": start,
            "end": end,
            "replacement": edit["new_text"],
            "change_chars": len(edit["old_text"]) + len(edit["new_text"]),
        }

    if op == "insert_before_section":
        replacement = _format_insert_text(edit["text"], before_text=original[:section.start], after_text=original[section.start:])
        return {
            "ok": True,
            "start": section.start,
            "end": section.start,
            "replacement": replacement,
            "change_chars": len(replacement),
        }

    if op == "insert_after_section":
        replacement = _format_insert_text(edit["text"], before_text=original[:section.end], after_text=original[section.end:])
        return {
            "ok": True,
            "start": section.end,
            "end": section.end,
            "replacement": replacement,
            "change_chars": len(replacement),
        }

    if op == "append_to_section":
        insert_at = _section_content_end(section)
        replacement = _format_insert_text(edit["text"], before_text=original[:insert_at], after_text=original[insert_at:])
        return {
            "ok": True,
            "start": insert_at,
            "end": insert_at,
            "replacement": replacement,
            "change_chars": len(replacement),
        }

    return {"ok": False, "message": f"不支持的操作：{op}"}


def _section_content_end(section: PromptSection) -> int:
    trailing = re.search(r"\s*$", section.text)
    if not trailing:
        return section.end
    return section.start + trailing.start()


def _format_insert_text(text: str, *, before_text: str, after_text: str) -> str:
    value = text.strip()
    if not value:
        return ""
    prefix = "" if not before_text or before_text.endswith("\n") else "\n"
    suffix = "" if not after_text or after_text.startswith("\n") else "\n"
    return f"{prefix}{value}{suffix}"


def _apply_operations(original: str, operations: list[dict[str, Any]]) -> str:
    candidate = original
    for operation in sorted(operations, key=lambda item: (item["start"], item["index"]), reverse=True):
        candidate = candidate[:operation["start"]] + operation["replacement"] + candidate[operation["end"]:]
    return candidate


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    stripped = str(value).strip()
    return [stripped] if stripped else []


def _truncate(value: Any, max_len: int) -> str:
    text = "" if value is None else str(value)
    return text[:max_len] + ("..." if len(text) > max_len else "")
