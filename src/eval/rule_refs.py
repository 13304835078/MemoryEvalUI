from __future__ import annotations

import re
import unicodedata


_MARKDOWN_PREFIX_RE = re.compile(r"^\s*(?:#{1,6}|[-*+]\s+|\d+[.)]\s+)+")
_SPACE_RE = re.compile(r"\s+")


def normalize_reference(value: str | None) -> str:
    """Create a deterministic comparison key while preserving rule semantics."""
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    text = _MARKDOWN_PREFIX_RE.sub("", text)
    text = text.strip("`'\"“”‘’：:；;。,.， ")
    return _SPACE_RE.sub(" ", text).lower()


def normalize_reference_set(values: list[str] | None) -> set[str]:
    return {normalized for value in values or [] if (normalized := normalize_reference(value))}
