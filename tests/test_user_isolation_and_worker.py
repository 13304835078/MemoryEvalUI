from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.runtime_paths import DATA_DIR, activate_workspace
from src.ui import task_worker, user_identity
from src.ui.config_store import CONFIG_PATH
from src.ui.data_service import RESULTS_DIR


def test_contextual_paths_follow_active_workspace():
    try:
        activate_workspace("user_alpha")
        alpha = (Path(DATA_DIR), Path(RESULTS_DIR), Path(CONFIG_PATH))
        activate_workspace("user_beta")
        beta = (Path(DATA_DIR), Path(RESULTS_DIR), Path(CONFIG_PATH))

        assert alpha != beta
        assert all("user_alpha" in str(path) for path in alpha)
        assert all("user_beta" in str(path) for path in beta)
    finally:
        activate_workspace("")


def test_work_id_binds_one_name_without_storing_plain_id(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(user_identity, "USER_REGISTRY_DIR", tmp_path / "users")
    monkeypatch.setattr(user_identity, "ensure_writable_layout", lambda: None)
    try:
        identity = user_identity.register_or_validate_identity("A00123", "张三")
        repeated = user_identity.register_or_validate_identity("A00123", "张三")

        assert identity == repeated
        profile_text = next((tmp_path / "users").glob("*.json")).read_text(encoding="utf-8")
        assert "A00123" not in profile_text
        with pytest.raises(ValueError, match="已绑定其他姓名"):
            user_identity.register_or_validate_identity("A00123", "李四")
    finally:
        activate_workspace("")


def test_current_identity_restores_streamlit_session(monkeypatch):
    monkeypatch.setattr(
        user_identity,
        "st",
        SimpleNamespace(
            session_state={
                user_identity.IDENTITY_SESSION_KEY: {
                    "workspace_id": "user_session_test",
                    "display_name": "测试用户",
                    "masked_work_id": "TE**01",
                }
            }
        ),
    )

    assert user_identity.current_identity() == user_identity.UserIdentity(
        "user_session_test",
        "测试用户",
        "TE**01",
    )


def test_worker_request_restores_secret_from_environment(tmp_path: Path, monkeypatch):
    request_path = tmp_path / "request.json"
    payload = {
        "version": 1,
        "kind": "fake",
        "job_id": "job1",
        "workspace_id": "user_test",
        "status": "queued",
        "config": {"eval_config": {"judge_api_bearer_token": ""}},
        "cases": [],
        "existing_results": [],
    }
    request_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    secrets = {"config.eval_config.judge_api_bearer_token": "Bearer secret"}
    encoded = base64.b64encode(json.dumps(secrets).encode("utf-8")).decode("ascii")
    monkeypatch.setenv(task_worker._SECRET_ENV, encoded)
    captured = {}
    monkeypatch.setattr(task_worker, "_run_request", lambda value: captured.update(value))

    try:
        assert task_worker.run_worker(request_path) == 0
        assert captured["config"]["eval_config"]["judge_api_bearer_token"] == "Bearer secret"
        persisted = json.loads(request_path.read_text(encoding="utf-8"))
        assert persisted["status"] == "finished"
        assert "secret" not in request_path.read_text(encoding="utf-8")
    finally:
        activate_workspace("")
