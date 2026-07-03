from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
IS_FROZEN = bool(getattr(sys, "frozen", False) or hasattr(sys, "_MEIPASS"))
BUNDLED_ROOT = Path(getattr(sys, "_MEIPASS", SOURCE_ROOT)).resolve()


def resolve_app_home() -> Path:
    configured = os.environ.get("MEMORY_EVAL_HOME", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    if IS_FROZEN:
        return Path(sys.executable).resolve().parent
    return SOURCE_ROOT


APP_HOME = resolve_app_home()
DATA_DIR = APP_HOME / "data"
CONFIG_DIR = APP_HOME / "config"
LOGS_DIR = APP_HOME / "logs"
PROMPTS_DIR = APP_HOME / "prompts"
BUNDLED_PROMPTS_DIR = BUNDLED_ROOT / "prompts"


def ensure_writable_layout() -> None:
    for directory in (
        DATA_DIR,
        CONFIG_DIR,
        LOGS_DIR,
        PROMPTS_DIR / "judge",
        PROMPTS_DIR / "generation",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    if not IS_FROZEN or APP_HOME == BUNDLED_ROOT:
        return
    for name in ("data", "config", "logs", "prompts"):
        _copy_missing_files(BUNDLED_ROOT / name, APP_HOME / name)


def _copy_missing_files(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    for source_file in source.rglob("*"):
        if not source_file.is_file():
            continue
        target = destination / source_file.relative_to(source)
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target)
