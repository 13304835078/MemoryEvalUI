import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.schema import Case, TaskType, EvalConfig, results_from_jsonl, DialogueTurn
from src.eval.eval_runner import EvalRunner
from src.ui.data_service import case_resume_key, eval_result_resume_key


def make_simple_cases() -> list[Case]:
    return [
        Case(
            case_id="eval_test_1",
            task_type=TaskType.USER_MD,
            session_id="s1",
            old_memory="- 姓名: 张三",
            dialogue=[
                DialogueTurn(role="user", content="我今年30岁了"),
                DialogueTurn(role="assistant", content="好的，已更新年龄"),
            ],
            candidate_output="- 姓名: 张三\n- 年龄: 30",
            model_name="test-model",
            prompt_version="v1",
        ),
        Case(
            case_id="eval_test_2",
            task_type=TaskType.USER_MD,
            session_id="s2",
            old_memory="",
            dialogue=[DialogueTurn(role="user", content="帮我查天气")],
            candidate_output="- 查询过天气\n- 喜欢晴天",  # bad: single query logged as preference
            model_name="test-model",
            prompt_version="v1",
        ),
    ]


def test_runner_mock_flow():
    config = EvalConfig(mock=True, judge_model="mock")
    cases = make_simple_cases()
    runner = EvalRunner(config, TaskType.USER_MD)
    results = runner.run(cases)

    assert len(results) == 2
    assert results[0].case_id == "eval_test_1"
    assert results[0].fatal_error is False
    assert results[0].score_total >= 0.0
    assert "correctness" in results[0].scores
    assert results[0].judge_prompt_version == "judge_user_md_absolute_stable_with_rules_v1"

    assert results[1].case_id == "eval_test_2"


def test_runner_build_user_message():
    config = EvalConfig(mock=True)
    runner = EvalRunner(config, TaskType.USER_MD)
    case = make_simple_cases()[0]
    case.metadata["reasoning"] = "用户明确说自己今年30岁，应更新年龄。"
    msg = runner._build_user_message(case)
    assert "## 旧 USER.md" in msg
    assert "张三" in msg
    assert "## 对话记录" in msg
    assert "我今年30岁了" in msg
    assert "## 模型 reasoning" in msg
    assert "应更新年龄" in msg
    assert "## 新 USER.md" in msg
    assert "年龄: 30" in msg
    assert "## 提取规则" not in msg


def test_runner_build_user_message_with_extraction_prompt():
    config = EvalConfig(mock=True)
    runner = EvalRunner(
        config,
        TaskType.USER_MD,
        extraction_prompt_text="## 1. 只基于 user 提取\n只记录用户自己明确表达的长期稳定事实。",
        extraction_prompt_version="extract_v1",
    )
    case = make_simple_cases()[0]
    msg = runner._build_user_message(case)

    assert "## 提取规则（仅作为规则依据，不是事实来源）" in msg
    assert "## 1. 只基于 user 提取" in msg
    assert "不能把其中的描述当作用户事实" in msg
    assert "模型 reasoning 只用于过程诊断，不能证明用户事实" in msg
    assert "## 可引用的提取规则标题清单" in msg
    assert "- ## 1. 只基于 user 提取" in msg
    assert "不要发明规则编号" in msg
    assert "符合 R2/R4" not in msg
    assert "diagnostics" in msg
    assert "## 旧 USER.md" in msg
    assert "提取规则辅助评测稳定契约" in runner.system_prompt
    assert runner.extraction_prompt_hash


def test_runner_records_scoring_comparability_metadata():
    runner = EvalRunner(EvalConfig(mock=True, judge_model="mock"), TaskType.USER_MD)
    result = runner.evaluate_one(make_simple_cases()[0])

    assert result.score_eligible is True
    assert result.evaluation_status == "success"
    assert result.judge_prompt_hash
    assert result.scoring_schema_version == "absolute_eval_schema_v2"
    assert result.dimension_weights_version == "user_md_update_weights_v1"
    assert result.scoring_config_hash
    assert result.case_input_hash
    assert result.evaluation_fingerprint


def test_evaluation_fingerprint_changes_with_case_prompt_and_sampling_config():
    case = make_simple_cases()[0]
    base = EvalRunner(
        EvalConfig(mock=True, judge_model="mock", judge_max_tokens=2000),
        TaskType.USER_MD,
        extraction_prompt_text="# 规则 A",
    )
    changed_prompt = EvalRunner(
        EvalConfig(mock=True, judge_model="mock", judge_max_tokens=2000),
        TaskType.USER_MD,
        extraction_prompt_text="# 规则 B",
    )
    changed_config = EvalRunner(
        EvalConfig(mock=True, judge_model="mock", judge_max_tokens=3000),
        TaskType.USER_MD,
        extraction_prompt_text="# 规则 A",
    )
    real_mode = EvalRunner(
        EvalConfig(mock=False, judge_model="mock", judge_max_tokens=2000),
        TaskType.USER_MD,
        extraction_prompt_text="# 规则 A",
    )
    changed_case = Case.from_dict(case.to_dict())
    changed_case.candidate_output = "不同候选输出"

    assert base.evaluation_fingerprint(case) != base.evaluation_fingerprint(changed_case)
    assert base.evaluation_fingerprint(case) != changed_prompt.evaluation_fingerprint(case)
    assert base.evaluation_fingerprint(case) != changed_config.evaluation_fingerprint(case)
    assert base.evaluation_fingerprint(case) != real_mode.evaluation_fingerprint(case)

    result = base.evaluate_one(case)
    assert case_resume_key(
        case,
        base.config.judge_model or "mock",
        base.resolved_judge_prompt_version,
        base.extraction_prompt_hash,
        base.evaluation_fingerprint(case),
    ) == eval_result_resume_key(result)
    assert case_resume_key(
        changed_case,
        base.config.judge_model or "mock",
        base.resolved_judge_prompt_version,
        base.extraction_prompt_hash,
        base.evaluation_fingerprint(changed_case),
    ) != eval_result_resume_key(result)


def test_long_memory_message_uses_memory_document_labels():
    case = Case(
        case_id="memory_case",
        task_type=TaskType.LONG_MEMORY,
        session_id="s1",
        old_memory="- 计划：准备考研",
        dialogue=[DialogueTurn(role="user", content="考研目标改为明年")],
        candidate_output="- 计划：准备明年考研",
        metadata={"reasoning": "用户明确修改计划时间。"},
    )
    runner = EvalRunner(
        config=EvalConfig(mock=True),
        task_type=TaskType.LONG_MEMORY,
        extraction_prompt_text="# 记忆更新规则\n新信息优先覆盖旧信息。",
    )

    msg = runner._build_user_message(case)

    assert "## 旧 MEMORY.md" in msg
    assert "## 新 MEMORY.md" in msg
    assert "生成 MEMORY.md 时使用" in msg
    assert "## 旧 USER.md" not in msg
    assert "候选 MEMORY.md" in runner.system_prompt


def test_long_memory_score_is_recomputed_with_task_weights():
    case = Case(
        case_id="memory_case",
        task_type=TaskType.LONG_MEMORY,
        session_id="s1",
        candidate_output="- 计划：准备考研",
    )
    runner = EvalRunner(
        config=EvalConfig(mock=True),
        task_type=TaskType.LONG_MEMORY,
    )
    response = {
        "score_total": 1,
        "scores": {
            "correctness": 5,
            "coverage": 4,
            "update_logic": 3,
            "memory_boundary": 2,
            "conciseness": 5,
            "format": 5,
        },
        "comment": "存在更新问题。",
        "error_tags": [],
        "fatal_error": False,
    }

    result = runner._parse_judge_result(case, response, "{}", "judge_long_memory_v1")

    assert result.task_type == TaskType.LONG_MEMORY.value
    assert result.score_total == 3.95


def test_runner_recomputes_weighted_score_total():
    config = EvalConfig(mock=True)
    runner = EvalRunner(config, TaskType.USER_MD)
    case = make_simple_cases()[0]
    result = runner._parse_judge_result(
        case,
        {
            "score_total": 5.0,
            "scores": {
                "correctness": 1,
                "coverage": 2,
                "update_logic": 3,
                "memory_boundary": 4,
                "conciseness": 5,
                "format": 5,
            },
            "comment": "ok",
            "error_tags": [],
            "fatal_error": False,
        },
        raw_response="{}",
        prompt_version="judge_user_md_v1",
    )

    assert result.score_total == 2.65


def test_runner_parses_diagnostics_and_references():
    config = EvalConfig(mock=True)
    runner = EvalRunner(
        config,
        TaskType.USER_MD,
        extraction_prompt_text="R2: 不记录临时请求。",
        extraction_prompt_version="extract_v1",
        extraction_prompt_hash="hash1",
    )
    case = make_simple_cases()[1]
    result = runner._parse_judge_result(
        case,
        {
            "score_total": 3.0,
            "scores": {
                "correctness": 3,
                "coverage": 4,
                "update_logic": 3,
                "memory_boundary": 2,
                "conciseness": 4,
                "format": 5,
            },
            "comment": "写入一次性请求。",
            "error_tags": ["over_memory"],
            "fatal_error": False,
            "diagnostics": [
                {
                    "dimension": "memory_boundary",
                    "severity": "medium",
                    "rule_refs": ["R2"],
                    "evidence_refs": ["帮我查天气"],
                    "output_refs": ["喜欢晴天"],
                    "reason": "一次性查询被写成偏好。",
                }
            ],
        },
        raw_response="{}",
        prompt_version="judge_v1",
    )

    assert result.extraction_prompt_version == "extract_v1"
    assert result.extraction_prompt_hash == "hash1"
    assert result.diagnostics[0]["dimension"] == "memory_boundary"
    assert result.rule_refs == ["R2"]
    assert result.evidence_refs == ["帮我查天气"]
    assert result.output_refs == ["喜欢晴天"]


def test_runner_drops_rule_ids_missing_from_extraction_prompt():
    config = EvalConfig(mock=True)
    runner = EvalRunner(
        config,
        TaskType.USER_MD,
        extraction_prompt_text="## 3. 单次任务和稳定偏好要拆开判断\n不要记录一次性查询。",
        extraction_prompt_version="extract_v1",
    )
    case = make_simple_cases()[1]
    result = runner._parse_judge_result(
        case,
        {
            "score_total": 3.0,
            "scores": {
                "correctness": 3,
                "coverage": 4,
                "update_logic": 3,
                "memory_boundary": 2,
                "conciseness": 4,
                "format": 5,
            },
            "comment": "写入一次性请求。",
            "error_tags": ["over_memory"],
            "fatal_error": False,
            "diagnostics": [
                {
                    "dimension": "memory_boundary",
                    "severity": "medium",
                    "rule_refs": ["R3", "## 3. 单次任务和稳定偏好要拆开判断"],
                    "evidence_refs": ["帮我查天气"],
                    "output_refs": ["喜欢晴天"],
                    "reason": "一次性查询被写成偏好。",
                }
            ],
            "rule_refs": ["R4", "## 3. 单次任务和稳定偏好要拆开判断"],
        },
        raw_response="{}",
        prompt_version="judge_v1",
    )

    assert result.rule_refs == ["## 3. 单次任务和稳定偏好要拆开判断"]
    assert result.diagnostics[0]["rule_refs"] == ["## 3. 单次任务和稳定偏好要拆开判断"]


def test_runner_loads_prompt():
    config = EvalConfig(mock=True)
    runner = EvalRunner(config, TaskType.USER_MD)
    sp = runner.system_prompt
    assert len(sp) > 100
    assert "correctness" in sp.lower()
    assert "score_total" in sp


def test_runner_jsonl_roundtrip():
    config = EvalConfig(mock=True)
    cases = make_simple_cases()
    runner = EvalRunner(config, TaskType.USER_MD)
    results = runner.run(cases)

    with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False, encoding="utf-8") as f:
        tmp = f.name
    try:
        from src.schema import results_to_jsonl
        results_to_jsonl(results, tmp)
        restored = results_from_jsonl(tmp)
        assert len(restored) == 2
        assert restored[0].case_id == "eval_test_1"
        assert isinstance(restored[0].scores, dict)
    finally:
        os.unlink(tmp)
