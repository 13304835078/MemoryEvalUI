from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.persistence import atomic_write_text, file_lock


state_file_lock = file_lock


def atomic_write_json(path: str | Path, data: dict[str, Any], *, retries: int = 8) -> None:
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    atomic_write_text(path, payload, retries=retries)
