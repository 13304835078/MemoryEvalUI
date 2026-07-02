from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = PROJECT_ROOT / "prompts" / "judge"
GENERATION_PROMPTS_DIR = PROJECT_ROOT / "prompts" / "generation"
RULES_DIR = PROJECT_ROOT / "rules"


TASK_PROMPT_DEFAULTS = {
    "user_md_update": "judge_user_md_absolute_stable_with_rules_v1.md",
    "day_memory": "judge_day_memory_v1.md",
    "long_memory": "judge_long_memory_v1.md",
    "summary": "judge_summary_v1.md",
}

TASK_EXTRACTION_PROMPT_DEFAULTS = {
    "user_md_update": "extract_user_md_rules_example_v1.md",
    "day_memory": "",
    "long_memory": "",
    "summary": "",
}

TASK_RUBRIC_DEFAULTS = {
    "user_md_update": "rubric_user_md.md",
    "day_memory": "rubric_day_memory.md",
    "long_memory": "rubric_long_memory.md",
    "summary": "rubric_summary.md",
}


def list_prompt_files() -> list[str]:
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted([p.name for p in PROMPTS_DIR.glob("*.md")])


def list_extraction_prompt_files() -> list[str]:
    GENERATION_PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted([p.name for p in GENERATION_PROMPTS_DIR.glob("*.md")])


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

    path = get_extraction_prompt_path(prompt_file) if prompt_kind == "extraction" else get_prompt_path(prompt_file)
    if not path.exists():
        return ""

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


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

    path = target_dir / version_name
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

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
