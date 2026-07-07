from src.loop.progress import compute_closed_loop_progress, parse_progress_fraction, round_progress_fraction


def test_parse_progress_fraction():
    assert parse_progress_fraction("3/10") == 0.3
    assert parse_progress_fraction("10 / 10") == 1.0
    assert parse_progress_fraction("11/10") == 1.0
    assert parse_progress_fraction("bad") is None


def test_round_progress_fraction_uses_stage_weights():
    assert round_progress_fraction({}) == 0.0
    assert round_progress_fraction({"extraction_progress": "5/10"}) == 0.2
    assert round_progress_fraction({"extraction_stats": {"generated": 10}}) == 0.35
    assert round_progress_fraction({"case_stats": {"generated_cases": 10}}) == 0.4
    assert round_progress_fraction({"eval_progress": "5/10"}) == 0.65
    assert round_progress_fraction({"eval_stats": {"avg_score_total": 4.8}}) == 0.85
    assert round_progress_fraction({"advisor_evidence_count": 5}) == 0.9
    assert round_progress_fraction({"candidate_prompt_saved": "prompt.md"}) == 0.98
    assert round_progress_fraction({"status": "completed"}) == 1.0
    assert round_progress_fraction({"status": "completed_no_change"}) == 1.0


def test_compute_closed_loop_progress_across_rounds():
    state = {
        "status": "running",
        "config": {"rounds": 3},
        "rounds": [
            {"status": "completed"},
            {"eval_progress": "5/10"},
        ],
    }

    progress = compute_closed_loop_progress(state)

    assert progress["current_round"] == 2
    assert progress["total_rounds"] == 3
    assert round(progress["current_round_fraction"], 2) == 0.65
    assert round(progress["overall_fraction"], 4) == round((1 + 0.65 + 0) / 3, 4)


def test_compute_closed_loop_progress_completed_is_full():
    state = {"status": "completed", "config": {"rounds": 2}, "rounds": [{"status": "completed"}]}

    progress = compute_closed_loop_progress(state)

    assert progress["overall_fraction"] == 1.0
    assert progress["current_round_fraction"] == 1.0


def test_compute_closed_loop_progress_completed_no_change_is_full():
    state = {
        "status": "completed_no_change",
        "config": {"rounds": 3},
        "rounds": [{"status": "completed_no_change"}],
    }

    progress = compute_closed_loop_progress(state)

    assert progress["overall_fraction"] == 1.0
    assert progress["current_step"] == "无需修改提示词，闭环结束"


def test_compute_closed_loop_progress_uses_runtime_target_rounds():
    state = {
        "status": "running",
        "config": {"rounds": 3},
        "controls": {"target_rounds": 2},
        "rounds": [
            {"status": "completed"},
        ],
    }

    progress = compute_closed_loop_progress(state)

    assert progress["current_round"] == 2
    assert progress["total_rounds"] == 2
    assert round(progress["overall_fraction"], 4) == 0.5
