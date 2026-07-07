from __future__ import annotations

import difflib
import re


def _clean(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _normalize_text_key(value: str) -> str:
    return "".join(str(value or "").lower().split())


_RULE_PREFIX_RE = re.compile(r"^\s*(?:#{1,6}\s*)?(?:[-*+]\s+|\d+[.)、]\s*)?")
_RULE_SIMILARITY_DROP_RE = re.compile(r"[\s`*_#>\-+•·\d.、,，。；;：:!?！？（）()\[\]【】《》“”\"'/\\|=]+")
_NO_RECORD_MARKERS = ("不记录", "不应记录", "不要记录", "不能记录", "不新增", "不沉淀", "排除")
_RECORD_MARKERS = ("可记录", "可以记录", "应记录", "需要记录", "必须记录", "可沉淀", "才沉淀")


def _normalize_rule_similarity_key(value: str) -> str:
    text = _RULE_PREFIX_RE.sub("", str(value or "").strip().lower())
    return _RULE_SIMILARITY_DROP_RE.sub("", text)


def _prompt_rule_units(prompt_text: str) -> list[dict[str, str]]:
    units: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw_line in str(prompt_text or "").splitlines():
        line = _clean(raw_line)
        if not line or line.startswith("#"):
            continue
        key = _normalize_rule_similarity_key(line)
        if len(key) < 8 or key in seen:
            continue
        seen.add(key)
        units.append({"text": line, "key": key})
    return units


def _is_similar_rule_key(candidate: str, existing: str) -> bool:
    if not candidate or not existing:
        return False
    min_len = min(len(candidate), len(existing))
    max_len = max(len(candidate), len(existing))
    if min_len < 8:
        return candidate == existing
    if candidate in existing or existing in candidate:
        return min_len >= 12 and (min_len / max_len) >= 0.58
    if min_len < 14:
        return False
    if difflib.SequenceMatcher(None, candidate, existing).ratio() >= 0.88:
        return True
    if _rule_decision_group(candidate) and _rule_decision_group(candidate) == _rule_decision_group(existing):
        return _ngram_containment(candidate, existing, 2) >= 0.60 and _ngram_containment(candidate, existing, 1) >= 0.78
    return False


def _rule_decision_group(value: str) -> str:
    if any(marker in value for marker in _NO_RECORD_MARKERS):
        return "no_record"
    if any(marker in value for marker in _RECORD_MARKERS):
        return "record"
    return ""


def _ngram_containment(left: str, right: str, n: int) -> float:
    left_grams = _char_ngrams(left, n)
    right_grams = _char_ngrams(right, n)
    if not left_grams or not right_grams:
        return 0.0
    return len(left_grams & right_grams) / min(len(left_grams), len(right_grams))


def _char_ngrams(value: str, n: int) -> set[str]:
    text = str(value or "")
    if len(text) < n:
        return {text} if text else set()
    return {text[index:index + n] for index in range(len(text) - n + 1)}


def _find_similar_existing_rule(text: str, existing_units: list[dict[str, str]]) -> dict[str, str] | None:
    key = _normalize_rule_similarity_key(text)
    if len(key) < 8:
        return None
    for unit in existing_units:
        if _is_similar_rule_key(key, unit["key"]):
            return unit
    return None


def _prune_duplicate_insert_text(text: str, existing_units: list[dict[str, str]]) -> tuple[str, list[dict[str, str]], list[dict[str, str]]]:
    kept_lines: list[str] = []
    kept_units: list[dict[str, str]] = []
    duplicate_rows: list[dict[str, str]] = []
    local_units = list(existing_units)
    for raw_line in str(text or "").splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        duplicate = _find_similar_existing_rule(line, local_units)
        if duplicate:
            duplicate_rows.append({"text": line.strip(), "existing_text": duplicate["text"]})
            continue
        kept_lines.append(line)
        key = _normalize_rule_similarity_key(line)
        if key:
            unit = {"text": line.strip(), "key": key}
            kept_units.append(unit)
            local_units.append(unit)
    return "\n".join(kept_lines).strip(), kept_units, duplicate_rows


def _prune_duplicate_replacement_text(
    old_text: str,
    new_text: str,
    existing_units: list[dict[str, str]],
) -> tuple[str, list[dict[str, str]], list[dict[str, str]]]:
    old_lines = str(old_text or "").splitlines()
    new_lines = str(new_text or "").splitlines()
    old_units = _prompt_rule_units(old_text)
    local_units = list(existing_units) + old_units
    keep_indexes = set(range(len(new_lines)))
    kept_units: list[dict[str, str]] = []
    duplicate_rows: list[dict[str, str]] = []
    inserted_indexes: set[int] = set()

    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
    for tag, _old_start, _old_end, new_start, new_end in matcher.get_opcodes():
        if tag == "insert":
            inserted_indexes.update(range(new_start, new_end))

    for index in sorted(inserted_indexes, reverse=True):
        line = new_lines[index]
        duplicate = _find_similar_existing_rule(line, local_units)
        if duplicate:
            duplicate_rows.append({"text": line.strip(), "existing_text": duplicate["text"]})
            keep_indexes.discard(index)
            continue
        key = _normalize_rule_similarity_key(line)
        if key:
            unit = {"text": line.strip(), "key": key}
            kept_units.append(unit)
            local_units.append(unit)

    return "\n".join(line for index, line in enumerate(new_lines) if index in keep_indexes).strip(), kept_units, duplicate_rows
