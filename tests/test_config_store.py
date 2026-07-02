import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ui.config_store import build_eval_config


def test_build_eval_config_speed_defaults():
    cfg = build_eval_config({}, mock=True)

    assert cfg.judge_enable_thinking is False
    assert cfg.judge_temperature == 0.0
    assert cfg.judge_concurrency == 1


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
