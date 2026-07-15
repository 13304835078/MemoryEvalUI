import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ui.config_store import build_eval_config, load_config, save_config
from src.extraction.client import MemoryExtractionConfig


def test_build_eval_config_speed_defaults():
    cfg = build_eval_config({}, mock=True)

    assert cfg.judge_enable_thinking is False
    assert cfg.judge_temperature == 0.0
    assert cfg.judge_concurrency == 1


def test_memory_extraction_config_converts_max_attempts_to_extra_retries():
    eval_cfg = build_eval_config({"judge_max_retries": 4}, mock=True)

    extraction_cfg = MemoryExtractionConfig.from_eval_config(eval_cfg)

    assert eval_cfg.judge_max_retries == 4
    assert extraction_cfg.max_retries == 3


def test_build_eval_config_interface_options():
    cfg = build_eval_config({
        "judge_auth_type": "hmac",
        "judge_hmac_access_key": "ak",
        "judge_hmac_secret_key": "sk",
        "judge_top_p": 0.8,
        "judge_top_k": 1,
        "judge_stop": "END\nSTOP",
        "judge_stream": True,
        "judge_concurrency": 3,
        "judge_send_enable_thinking": False,
        "judge_call_from": "memory_eval",
        "judge_extra_body_json": '{"session_id":"s1"}',
    }, mock=False)

    assert cfg.judge_auth_type == "bearer"
    assert cfg.judge_hmac_access_key == "ak"
    assert cfg.judge_top_p == 0.8
    assert cfg.judge_top_k == 1
    assert cfg.judge_stop == ["END", "STOP"]
    assert cfg.judge_stream is True
    assert cfg.judge_concurrency == 3
    assert cfg.judge_send_enable_thinking is False
    assert cfg.judge_call_from == "memory_eval"


def test_build_eval_config_preserves_zero_top_p():
    cfg = build_eval_config({"judge_top_p": 0.0}, mock=True)

    assert cfg.judge_top_p == 0.0


def test_load_config_marks_corrupt_file_and_uses_defaults(tmp_path):
    path = tmp_path / "local_config.json"
    path.write_text("{bad json", encoding="utf-8")

    cfg = load_config(path)

    assert cfg["mock"] is True
    assert cfg["_config_error"]
    backup_path = cfg["_config_corrupt_path"]
    assert backup_path
    assert os.path.exists(backup_path)
    assert open(backup_path, encoding="utf-8").read() == "{bad json"
    assert not path.exists()


def test_save_config_drops_transient_config_error_fields(tmp_path):
    path = tmp_path / "local_config.json"

    save_config({
        "api_base": "http://example.com",
        "_config_error": "bad",
        "_config_corrupt_path": "backup",
    }, path)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["api_base"] == "http://example.com"
    assert "_config_error" not in saved
    assert "_config_corrupt_path" not in saved
