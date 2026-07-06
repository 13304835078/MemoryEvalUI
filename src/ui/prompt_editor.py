from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from src import runtime_paths
from src.extraction.memory_extractor import parse_generation_prompt_templates
from src.persistence import atomic_write_text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
USER_PROMPTS_ROOT = runtime_paths.PROMPTS_DIR
BUNDLED_PROMPTS_ROOT = runtime_paths.BUNDLED_PROMPTS_DIR
PROMPTS_DIR = USER_PROMPTS_ROOT / "judge"
GENERATION_PROMPTS_DIR = USER_PROMPTS_ROOT / "generation"
BUNDLED_JUDGE_PROMPTS_DIR = BUNDLED_PROMPTS_ROOT / "judge"
BUNDLED_GENERATION_PROMPTS_DIR = BUNDLED_PROMPTS_ROOT / "generation"
RULES_DIR = runtime_paths.BUNDLED_ROOT / "rules"


TASK_PROMPT_DEFAULTS = {
    "user_md_update": "judge_user_md_absolute_stable_with_rules_v1.md",
    "day_memory": "judge_day_memory_v1.md",
    "long_memory": "judge_long_memory_v1.md",
    "summary": "judge_summary_v1.md",
}

TASK_EXTRACTION_PROMPT_DEFAULTS = {
    "user_md_update": "extract_user_md_rules_example_v1.md",
    "day_memory": "",
    "long_memory": "extract_long_memory_v1.yaml",
    "summary": "",
}

TASK_RUBRIC_DEFAULTS = {
    "user_md_update": "rubric_user_md.md",
    "day_memory": "rubric_day_memory.md",
    "long_memory": "rubric_long_memory.md",
    "summary": "rubric_summary.md",
}


def list_prompt_files() -> list[str]:
    return _list_prompt_files(PROMPTS_DIR, BUNDLED_JUDGE_PROMPTS_DIR)


def list_extraction_prompt_files() -> list[str]:
    return _list_prompt_files(GENERATION_PROMPTS_DIR, BUNDLED_GENERATION_PROMPTS_DIR)


def get_default_prompt_file(task_type: str) -> str:
    return TASK_PROMPT_DEFAULTS.get(task_type, "")


def get_default_extraction_prompt_file(task_type: str) -> str:
    return TASK_EXTRACTION_PROMPT_DEFAULTS.get(task_type, "")


def get_prompt_path(prompt_file: str) -> Path:
    path = Path(prompt_file)
    if path.is_absolute():
        return path
    return PROMPTS_DIR / prompt_file


def get_extraction_prompt_path(prompt_file: str) -> Path:
    path = Path(prompt_file)
    if path.is_absolute():
        return path
    return GENERATION_PROMPTS_DIR / prompt_file


def load_prompt(prompt_file: str, prompt_kind: str = "judge") -> str:
    if not prompt_file:
        return ""

    path = _resolve_prompt_path(prompt_file, prompt_kind)
    if not path.exists():
        return ""

    text = path.read_text(encoding="utf-8")
    if prompt_kind == "extraction" and path.suffix.lower() in {".yaml", ".yml"}:
        return _extract_prompt_from_yaml(text, path)
    return text


def load_extraction_prompt_templates(prompt_file: str) -> dict[str, str]:
    if not prompt_file:
        return {"create": "", "update": ""}
    path = _resolve_prompt_path(prompt_file, "extraction")
    if not path.exists():
        return {"create": "", "update": ""}
    return parse_generation_prompt_templates(
        path.read_text(encoding="utf-8"),
        path.suffix,
    )


def save_prompt_version(task_type: str, content: str, version_name: str = "", prompt_kind: str = "judge") -> str:
    """保存一个新的 prompt 版本，返回文件名。"""
    target_dir = GENERATION_PROMPTS_DIR if prompt_kind == "extraction" else PROMPTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    if not version_name:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "extract" if prompt_kind == "extraction" else "judge"
        version_name = f"{prefix}_{task_type}_{ts}.md"

    if not version_name.endswith(".md"):
        version_name += ".md"

    version_name = Path(version_name).name
    path = target_dir / version_name
    atomic_write_text(path, content, encoding="utf-8")
    _write_prompt_metadata(path, task_type, prompt_kind, content)

    return version_name


def infer_prompt_version(prompt_file: str) -> str:
    if not prompt_file:
        return "unknown"
    return Path(prompt_file).stem


def prompt_text_hash(prompt_text: str) -> str:
    if not prompt_text:
        return ""
    return hashlib.sha1(prompt_text.encode("utf-8")).hexdigest()


def load_rubric(task_type: str) -> str:
    filename = TASK_RUBRIC_DEFAULTS.get(task_type, "")
    if not filename:
        return ""

    path = RULES_DIR / filename
    if not path.exists():
        return ""

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _list_prompt_files(user_dir: Path, bundled_dir: Path) -> list[str]:
    user_dir.mkdir(parents=True, exist_ok=True)
    names: set[str] = set()
    if bundled_dir.exists():
        names.update(p.name for p in bundled_dir.glob("*.md"))
        names.update(p.name for p in bundled_dir.glob("*.yaml"))
        names.update(p.name for p in bundled_dir.glob("*.yml"))
    names.update(p.name for p in user_dir.glob("*.md"))
    names.update(p.name for p in user_dir.glob("*.yaml"))
    names.update(p.name for p in user_dir.glob("*.yml"))
    return sorted(names)


def _extract_prompt_from_yaml(text: str, path: Path) -> str:
    return parse_generation_prompt_templates(text, path.suffix)["update"]


def _resolve_prompt_path(prompt_file: str, prompt_kind: str) -> Path:
    requested = Path(prompt_file)
    if requested.is_absolute():
        return requested

    if prompt_kind == "extraction":
        user_path = GENERATION_PROMPTS_DIR / prompt_file
        bundled_path = BUNDLED_GENERATION_PROMPTS_DIR / prompt_file
    else:
        user_path = PROMPTS_DIR / prompt_file
        bundled_path = BUNDLED_JUDGE_PROMPTS_DIR / prompt_file

    if user_path.exists():
        return user_path
    return bundled_path


def _write_prompt_metadata(path: Path, task_type: str, prompt_kind: str, content: str) -> None:
    metadata = {
        "filename": path.name,
        "task_type": task_type,
        "prompt_kind": prompt_kind,
        "sha1": prompt_text_hash(content),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "storage": "user_prompt_dir",
    }
    metadata_path = path.with_name(f"{path.name}.meta.json")
    atomic_write_text(
        metadata_path,
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
