from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DocumentParseResult:
    document: str | None
    method: str
    confidence: float
    warnings: tuple[str, ...] = ()


def _strip_outer_fence(text: str) -> str:
    match = re.fullmatch(
        r"\s*```(?:markdown|md|text|json)?[ \t]*\r?\n?(.*?)\r?\n?```\s*",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(1) if match else text


def _wrapper_line_pattern(document_name: str) -> re.Pattern[str]:
    escaped_name = re.escape(document_name)
    return re.compile(
        rf"(?:"
        rf"#{{1,6}}\s*(?:Output|输出|{escaped_name})\s*[:：]?"
        rf"|---+\s*{escaped_name}\s*---+"
        rf"|(?:\*{{1,2}}|_{{1,2}})?(?:Output|输出)(?:\*{{1,2}}|_{{1,2}})?\s*[:：]?"
        rf")",
        flags=re.IGNORECASE,
    )


def normalize_memory_document_body(body: str | None, document_name: str) -> str:
    """Remove transport wrappers while preserving the generated Markdown structure."""
    if body is None:
        return ""

    text = _strip_outer_fence(str(body).strip())
    if not text:
        return ""

    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    wrapper = _wrapper_line_pattern(document_name)

    while lines and (not lines[0].strip() or wrapper.fullmatch(lines[0].strip())):
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    return "\n".join(lines)


def normalize_user_md_body(body: str | None) -> str:
    return normalize_memory_document_body(body, "USER.md")


def _json_document(value: Any, document_name: str, path: str = "") -> tuple[str, str] | None:
    if not isinstance(value, dict):
        return None

    normalized_name = document_name.lower().replace(".", "_")
    candidate_keys = (
        document_name,
        document_name.lower(),
        normalized_name,
        "document",
        "document_text",
        "content",
        "output",
        "result",
        "answer",
        "response",
    )
    for key in candidate_keys:
        if key not in value:
            continue
        item = value[key]
        key_path = f"{path}.{key}" if path else key
        if isinstance(item, str):
            return item, key_path

    for key in ("data", "result", "output", "response"):
        nested = value.get(key)
        if isinstance(nested, dict):
            key_path = f"{path}.{key}" if path else key
            found = _json_document(nested, document_name, key_path)
            if found is not None:
                return found
    return None


def _normalized_document_type(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def _explicit_markers(text: str, document_name: str) -> list[tuple[re.Match[str], str]]:
    escaped_name = re.escape(document_name)
    patterns = (
        (
            re.compile(r"(?im)^[ \t]*#{1,6}[ \t]*(?:Output|输出)[ \t]*[:：]?[ \t]*$"),
            "output_heading",
        ),
        (
            re.compile(
                r"(?im)^[ \t]*(?:\*{1,2}|_{1,2})?(?:Output|输出)"
                r"(?:\*{1,2}|_{1,2})?[ \t]*[:：]?[ \t]*$"
            ),
            "output_marker",
        ),
        (
            re.compile(rf"(?im)^[ \t]*---+[ \t]*{escaped_name}[ \t]*---+[ \t]*$"),
            "document_separator",
        ),
        (
            re.compile(rf"(?im)^[ \t]*#{{1,6}}[ \t]*{escaped_name}[ \t]*$"),
            "document_heading",
        ),
    )
    markers: list[tuple[re.Match[str], str]] = []
    for pattern, method in patterns:
        markers.extend((match, method) for match in pattern.finditer(text))
    return markers


def _contains_reasoning_heading(text: str) -> bool:
    return bool(re.search(
        r"(?im)^[ \t]*(?:#{1,6}[ \t]*)?"
        r"(?:(?:Think|Reasoning)\b|(?:思考|推理|分析)(?:过程)?(?=[ \t]*[:：]|[ \t]*$)).*$",
        text,
    ))


def _looks_like_structured_document(text: str) -> bool:
    if re.search(r"(?m)^[ \t]*#{1,6}[ \t]+\S+", text):
        return True
    if re.search(r"(?m)^[ \t]*(?:第[一二三四五六七八九十]+分区|[一二三四五六七八九十]+、)\S*", text):
        return True
    return False


def _looks_like_list_document(text: str) -> bool:
    return bool(re.search(r"(?m)^[ \t]*(?:[-*•]|\d+[.、])[ \t]+\S+", text))


def parse_memory_document(text: str | None, document_name: str) -> DocumentParseResult:
    if text is None:
        return DocumentParseResult(None, "none", 0.0)

    raw = str(text).strip()
    if not raw:
        return DocumentParseResult(None, "none", 0.0)
    raw = _strip_outer_fence(raw).strip()

    escaped_name = re.escape(document_name)
    envelope_start = re.search(
        rf"(?im)^[ \t]*<<<MEMORY_DOCUMENT_V1:{escaped_name}>>>[ \t]*$",
        raw,
    )
    if envelope_start:
        envelope_end = re.search(
            r"(?im)^[ \t]*<<<END_MEMORY_DOCUMENT_V1>>>[ \t]*$",
            raw[envelope_start.end():],
        )
        warnings: tuple[str, ...] = ()
        if envelope_end:
            end = envelope_start.end() + envelope_end.start()
            body = raw[envelope_start.end():end]
            confidence = 1.0
        else:
            body = raw[envelope_start.end():]
            confidence = 0.9
            warnings = ("统一输出协议缺少结束标记",)
        return DocumentParseResult(
            normalize_memory_document_body(body, document_name),
            "document_envelope_v1",
            confidence,
            warnings,
        )

    try:
        json_value = json.loads(raw)
    except json.JSONDecodeError:
        json_value = None
    if isinstance(json_value, dict) and json_value.get("document_type") not in (None, ""):
        declared_type = _normalized_document_type(json_value.get("document_type"))
        expected_type = _normalized_document_type(document_name)
        if declared_type != expected_type:
            return DocumentParseResult(
                None,
                "json_document_type_mismatch",
                0.0,
                (f"JSON 声明的文档类型为 {json_value.get('document_type')}，预期为 {document_name}",),
            )
    found_json = _json_document(json_value, document_name)
    if found_json is not None:
        body, key_path = found_json
        return DocumentParseResult(
            normalize_memory_document_body(body, document_name),
            f"json:{key_path}",
            0.98,
        )

    markers = _explicit_markers(raw, document_name)
    if markers:
        marker, method = max(markers, key=lambda item: item[0].end())
        body = raw[marker.end():]
        return DocumentParseResult(
            normalize_memory_document_body(body, document_name),
            method,
            0.97,
        )

    if _contains_reasoning_heading(raw):
        return DocumentParseResult(
            None,
            "reasoning_only",
            0.0,
            ("检测到推理段但没有明确输出边界",),
        )

    if _looks_like_structured_document(raw):
        return DocumentParseResult(
            normalize_memory_document_body(raw, document_name),
            "structured_markdown_fallback",
            0.78,
            ("未检测到明确输出标记，按结构化 Markdown 正文解析",),
        )

    if _looks_like_list_document(raw):
        return DocumentParseResult(
            normalize_memory_document_body(raw, document_name),
            "list_fallback",
            0.7,
            ("未检测到明确输出标记，按列表正文解析",),
        )

    return DocumentParseResult(
        None,
        "unrecognized",
        0.0,
        ("无法识别可靠的正文边界",),
    )


def extract_memory_document(text: str | None, document_name: str) -> str | None:
    return parse_memory_document(text, document_name).document


def extract_user_md(text: str | None) -> str | None:
    return extract_memory_document(text, "USER.md")


def extract_long_memory(text: str | None) -> str | None:
    return extract_memory_document(text, "MEMORY.md")
