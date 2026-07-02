import os
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.schema import EvalResult
from src.ui.human_review_eval import (
    append_human_review_result_row,
    build_pair_row,
    decide_gsb,
    eval_config_fingerprint,
    load_human_review_result_rows,
    load_human_review_cache,
    low_confidence_rows,
    make_human_review_pairs,
    make_human_review_pairs_with_stats,
    normalize_gsb,
    pair_cache_key,
    save_human_review_cache,
    stable_hash,
    summarize_pair_rows,
)


def test_make_human_review_pairs_splits_one_row_into_two_cases():
    df = pd.DataFrame({
        "轮次": [1, 2],
        "query": ["q1", "q2"],
        "answer": ["a1", "a2"],
        "评测人": ["alice", "alice"],
        "user.md-glm5-think": ["m1-old", "m1-new"],
        "user.md-ds-10.1.2": ["m2-old", "m2-new"],
        "GSB": ["G", "B"],
        "问题类型": ["事实", "边界"],
        "备注": ["r1", "r2"],
    })

    pairs = make_human_review_pairs(df)

    assert len(pairs) == 2
    assert pairs[0].case_model1.candidate_output == "m1-old"
    assert pairs[0].case_model2.candidate_output == "m2-old"
    assert pairs[0].case_model1.old_memory is None
    assert pairs[1].case_model1.old_memory == "m1-old"
    assert pairs[1].case_model2.old_memory == "m2-old"
    assert pairs[1].human_gsb == "B"


def test_make_human_review_pairs_skips_rows_without_valid_gsb():
    df = pd.DataFrame({
        "轮次": [1, 2, 3, 4],
        "query": ["q1", "q2", "q3", "q4"],
        "answer": ["a1", "a2", "a3", "a4"],
        "评测人": ["alice", "alice", "alice", "alice"],
        "user.md-glm5-think": ["m1-row1", "m1-row2", "m1-row3", "m1-row4"],
        "user.md-ds-10.1.2": ["m2-row1", "m2-row2", "m2-row3", "m2-row4"],
        "GSB": ["G", "", "unknown", "B"],
        "问题类型": ["事实", "", "", "边界"],
        "备注": ["", "", "", ""],
    })

    pairs, skipped = make_human_review_pairs_with_stats(df)

    assert len(pairs) == 2
    assert [p.row_number for p in pairs] == [2, 5]
    assert len(skipped) == 2
    assert {item["row_number"] for item in skipped} == {3, 4}
    assert all(item["skip_reason"] == "missing_or_invalid_gsb" for item in skipped)
    assert pairs[1].case_model1.old_memory == "m1-row1"
    assert pairs[1].case_model2.old_memory == "m2-row1"


def test_normalize_and_decide_gsb():
    assert normalize_gsb("模型1更好") == "G"
    assert normalize_gsb("平局") == "S"
    assert normalize_gsb("ds更好") == "B"
    assert decide_gsb(4.8, 4.4, margin=0.25) == "G"
    assert decide_gsb(4.4, 4.8, margin=0.25) == "B"
    assert decide_gsb(4.6, 4.5, margin=0.25) == "S"


def test_build_pair_row_and_summary():
    df = pd.DataFrame({
        "轮次": [1],
        "query": ["q"],
        "answer": ["a"],
        "评测人": ["alice"],
        "user.md-glm5-think": ["m1"],
        "user.md-ds-10.1.2": ["m2"],
        "GSB": ["G"],
        "问题类型": ["事实"],
        "备注": [""],
    })
    pair = make_human_review_pairs(df)[0]
    result1 = EvalResult(case_id="c1", task_type="user_md_update", score_total=4.8, comment="good")
    result2 = EvalResult(case_id="c2", task_type="user_md_update", score_total=4.0, comment="bad")

    row = build_pair_row(pair, result1, result2, margin=0.25)
    summary = summarize_pair_rows([row])

    assert row["自动GSB"] == "G"
    assert row["是否一致"] is True
    assert "glm5-think 更好" in row["自动判断备注"]
    assert summary["agreement_rate"] == 1.0


def test_human_review_cache_roundtrip():
    result1 = EvalResult(case_id="c1", task_type="user_md_update", score_total=4.8, comment="good")
    result2 = EvalResult(case_id="c2", task_type="user_md_update", score_total=4.0, comment="bad")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = f.name
    try:
        save_human_review_cache({"k": (result1, result2)}, tmp)
        restored = load_human_review_cache(tmp)

        assert restored["k"][0].case_id == "c1"
        assert restored["k"][1].score_total == 4.0
    finally:
        os.unlink(tmp)


def test_pair_cache_key_includes_repeat_count():
    df = pd.DataFrame({
        "轮次": [1],
        "query": ["q"],
        "answer": ["a"],
        "评测人": ["alice"],
        "user.md-glm5-think": ["m1"],
        "user.md-ds-10.1.2": ["m2"],
        "GSB": ["G"],
        "问题类型": ["事实"],
        "备注": [""],
    })
    pair = make_human_review_pairs(df)[0]

    assert pair_cache_key(pair, "judge", "prompt", 1) != pair_cache_key(pair, "judge", "prompt", 3)


def test_pair_cache_key_includes_prompt_and_config_hash():
    df = pd.DataFrame({
        "轮次": [1],
        "query": ["q"],
        "answer": ["a"],
        "评测人": ["alice"],
        "user.md-glm5-think": ["m1"],
        "user.md-ds-10.1.2": ["m2"],
        "GSB": ["G"],
        "问题类型": ["事实"],
        "备注": [""],
    })
    pair = make_human_review_pairs(df)[0]

    k1 = pair_cache_key(pair, "judge", "prompt", 1, judge_prompt_hash="h1", config_hash="c1")
    k2 = pair_cache_key(pair, "judge", "prompt", 1, judge_prompt_hash="h2", config_hash="c1")
    k3 = pair_cache_key(pair, "judge", "prompt", 1, judge_prompt_hash="h1", config_hash="c2")

    assert k1 != k2
    assert k1 != k3


def test_append_and_load_human_review_result_rows():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        tmp = f.name
    try:
        append_human_review_result_row(tmp, {"pair_id": "p1", "是否一致": False})
        append_human_review_result_row(tmp, {"pair_id": "p2", "是否一致": True})
        rows = load_human_review_result_rows(tmp)

        assert [r["pair_id"] for r in rows] == ["p1", "p2"]
    finally:
        os.unlink(tmp)


def test_low_confidence_rows():
    rows = [
        {"pair_id": "p1", "是否一致": False, "score_diff_model1_minus_model2": 1.0},
        {"pair_id": "p2", "是否一致": True, "score_diff_model1_minus_model2": 0.1},
        {"pair_id": "p3", "是否一致": True, "score_diff_model1_minus_model2": 1.0},
    ]

    low = low_confidence_rows(rows, margin=0.25, band=0.15)

    assert [r["pair_id"] for r in low] == ["p1", "p2"]


def test_stable_hash_and_config_fingerprint_are_stable():
    class Cfg:
        judge_model = "m"
        judge_max_tokens = 100
        judge_temperature = 0.0
        judge_enable_thinking = False
        judge_timeout = 30
        judge_max_retries = 2

    assert stable_hash({"b": 2, "a": 1}) == stable_hash({"a": 1, "b": 2})
    assert eval_config_fingerprint(Cfg()) == eval_config_fingerprint(Cfg())
