from __future__ import annotations

import hashlib
import json
from typing import Any

from src.schema import Case


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def hash_payload(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def case_input_hash(case: Case) -> str:
    """Hash every case field that can affect an absolute Judge decision."""
    reasoning = ""
    if isinstance(case.metadata, dict):
        reasoning = str(case.metadata.get("reasoning") or "")
    payload = {
        "task_type": case.task_type.value,
        "session_id": case.session_id,
        "old_memory": case.old_memory or "",
        "dialogue": [
            {
                "role": turn.role,
                "content": turn.content,
                "metadata": turn.metadata or {},
            }
            for turn in case.dialogue
        ],
        "instructions": case.instructions or "",
        "turn_range": case.turn_range or [],
        "candidate_output": case.candidate_output or "",
        "reference_output": case.reference_output or "",
        "reasoning": reasoning,
    }
    return hash_payload(payload)


def evaluation_fingerprint(case_hash: str, scoring_config_hash: str) -> str:
    return hash_payload({
        "case_input_hash": case_hash or "",
        "scoring_config_hash": scoring_config_hash or "",
    })
