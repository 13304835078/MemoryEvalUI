from __future__ import annotations

import os
import re
import shutil
import sys
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable


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
BUNDLED_PROMPTS_DIR = BUNDLED_ROOT / "prompts"
_ACTIVE_WORKSPACE: ContextVar[str] = ContextVar("memory_eval_workspace", default="")


def activate_workspace(workspace_id: str = "") -> None:
    normalized = str(workspace_id or "").strip().lower()
    if normalized and not re.fullmatch(r"[0-9a-z_-]{3,64}", normalized):
        raise ValueError("workspace_id 格式不合法")
    _ACTIVE_WORKSPACE.set(normalized)


def active_workspace_id() -> str:
    return _ACTIVE_WORKSPACE.get() or os.environ.get("MEMORY_EVAL_WORKSPACE_ID", "").strip().lower()


def workspace_root() -> Path:
    workspace_id = active_workspace_id()
    return APP_HOME / "workspaces" / workspace_id if workspace_id else APP_HOME


class ContextualPath(os.PathLike[str]):
    """Path-like proxy resolved against the current Streamlit user workspace."""

    def __init__(self, resolver: Callable[[], Path]):
        self._resolver = resolver

    def current(self) -> Path:
        return Path(self._resolver())

    def __fspath__(self) -> str:
        return os.fspath(self.current())

    def __str__(self) -> str:
        return str(self.current())

    def __repr__(self) -> str:
        return f"ContextualPath({self.current()!r})"

    def __truediv__(self, value: Any) -> "ContextualPath":
        return ContextualPath(lambda: self.current() / value)

    def __rtruediv__(self, value: Any) -> Path:
        return Path(value) / self.current()

    @property
    def parent(self) -> "ContextualPath":
        return ContextualPath(lambda: self.current().parent)

    def joinpath(self, *parts: Any) -> "ContextualPath":
        return ContextualPath(lambda: self.current().joinpath(*parts))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.current(), name)

    def __eq__(self, other: object) -> bool:
        try:
            return self.current() == Path(other)  # type: ignore[arg-type]
        except TypeError:
            return False

    def __hash__(self) -> int:
        return hash(self.current())


DATA_DIR = ContextualPath(lambda: workspace_root() / "data")
CONFIG_DIR = ContextualPath(lambda: workspace_root() / "config")
LOGS_DIR = ContextualPath(lambda: workspace_root() / "logs")
PROMPTS_DIR = ContextualPath(lambda: workspace_root() / "prompts")


def ensure_writable_layout() -> None:
    for directory in (
        DATA_DIR,
        CONFIG_DIR,
        LOGS_DIR,
        PROMPTS_DIR / "judge",
        PROMPTS_DIR / "generation",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    # Preserve the historical packaged-layout migration only for the legacy
    # public workspace. Authenticated user workspaces never inherit another
    # workspace's writable data or local token configuration.
    if IS_FROZEN and APP_HOME != BUNDLED_ROOT and not active_workspace_id():
        for name, destination in (
            ("data", Path(DATA_DIR)),
            ("config", Path(CONFIG_DIR)),
            ("logs", Path(LOGS_DIR)),
            ("prompts", Path(PROMPTS_DIR)),
        ):
            _copy_missing_files(BUNDLED_ROOT / name, destination)


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
