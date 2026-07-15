from __future__ import annotations

import difflib
import hashlib
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptDiffSummary:
    added_lines: int
    removed_lines: int
    old_chars: int
    new_chars: int
    growth_ratio: float
    duplicate_headings: tuple[str, ...]
    old_hash: str
    new_hash: str
    diff_text: str


def _headings(text: str) -> list[str]:
    return [line.strip() for line in str(text or "").splitlines() if line.lstrip().startswith("#")]


def analyze_prompt_diff(old_text: str, new_text: str, *, old_name: str = "原版本", new_name: str = "对照版本") -> PromptDiffSummary:
    old_lines = str(old_text or "").splitlines()
    new_lines = str(new_text or "").splitlines()
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines)
    added = 0
    removed = 0
    for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        if tag in {"insert", "replace"}:
            added += new_end - new_start
        if tag in {"delete", "replace"}:
            removed += old_end - old_start

    heading_counts = Counter(_headings(new_text))
    duplicates = tuple(sorted(heading for heading, count in heading_counts.items() if count > 1))
    old_chars = len(str(old_text or ""))
    new_chars = len(str(new_text or ""))
    growth = (new_chars - old_chars) / old_chars if old_chars else (1.0 if new_chars else 0.0)
    diff_text = "\n".join(difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=old_name,
        tofile=new_name,
        lineterm="",
    ))
    return PromptDiffSummary(
        added_lines=added,
        removed_lines=removed,
        old_chars=old_chars,
        new_chars=new_chars,
        growth_ratio=growth,
        duplicate_headings=duplicates,
        old_hash=hashlib.sha256(str(old_text or "").encode("utf-8")).hexdigest(),
        new_hash=hashlib.sha256(str(new_text or "").encode("utf-8")).hexdigest(),
        diff_text=diff_text,
    )

