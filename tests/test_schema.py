import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.schema import (
    Case,
    DialogueTurn,
    EvalResult,
    TaskType,
    cases_to_jsonl,
    cases_from_jsonl,
    results_to_jsonl,
    results_from_jsonl,
    validate_case,
)
import tempfile


def test_case_creation():
    case = Case(
        case_id="test_001",
        task_type=TaskType.USER_MD,
        session_id="session_001",
        old_memory="- 姓名: 张三",
        dialogue=[
            DialogueTurn(role="user", content="我今年25岁"),
            DialogueTurn(role="assistant", content="好的，已记录"),
        ],
        candidate_output="- 姓名: 张三\n- 年龄: 25",
        model_name="test-model",
        prompt_version="v1",
    )
    assert case.case_id == "test_001"
    assert case.task_type == TaskType.USER_MD
    assert len(case.dialogue) == 2
    assert case.dialogue[0].role == "user"
    assert case.dialogue[0].content == "我今年25岁"


def test_case_serialization():
    case = Case(
        case_id="test_002",
        task_type=TaskType.DAY_MEMORY,
        session_id="s2",
        dialogue=[DialogueTurn(role="user", content="hello")],
    )
    d = case.to_dict()
    assert d["case_id"] == "test_002"
    assert d["task_type"] == "day_memory"
    assert d["dialogue"] == [{"role": "user", "content": "hello", "metadata": {}}]

    restored = Case.from_dict(d)
    assert restored.case_id == case.case_id
    assert restored.task_type == case.task_type
    assert restored.dialogue[0].content == "hello"


def test_jsonl_roundtrip():
    cases = [
        Case(case_id="c1", task_type=TaskType.USER_MD, session_id="s1",
             dialogue=[DialogueTurn(role="user", content="hi")]),
        Case(case_id="c2", task_type=TaskType.SUMMARY, session_id="s2"),
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        tmp = f.name
    try:
        cases_to_jsonl(cases, tmp)
        restored = cases_from_jsonl(tmp)
        assert len(restored) == 2
        assert restored[0].case_id == "c1"
        assert restored[1].case_id == "c2"
        assert restored[0].dialogue[0].content == "hi"
    finally:
        os.unlink(tmp)


def test_validate_case():
    valid_case = Case(case_id="ok", task_type=TaskType.USER_MD, session_id="s1")
    assert validate_case(valid_case) == []

    invalid_case = Case(case_id="", task_type=TaskType.USER_MD, session_id="")
    errs = validate_case(invalid_case)
    assert len(errs) == 2
    assert "case_id" in errs[0]
    assert "session_id" in errs[1]


def test_eval_result_new_fields_jsonl_roundtrip():
    result = EvalResult(
        case_id="c1",
        task_type="user_md_update",
        score_total=3.5,
        extraction_prompt_version="extract_v1",
        extraction_prompt_hash="hash1",
        diagnostics=[{
            "dimension": "coverage",
            "severity": "medium",
            "rule_refs": ["R4"],
            "evidence_refs": ["用户说喜欢粤菜"],
            "output_refs": ["未记录粤菜偏好"],
            "reason": "漏掉稳定偏好。",
        }],
        rule_refs=["R4"],
        evidence_refs=["用户说喜欢粤菜"],
        output_refs=["未记录粤菜偏好"],
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        tmp = f.name
    try:
        results_to_jsonl([result], tmp)
        restored = results_from_jsonl(tmp)[0]
        assert restored.extraction_prompt_version == "extract_v1"
        assert restored.extraction_prompt_hash == "hash1"
        assert restored.diagnostics[0]["dimension"] == "coverage"
        assert restored.rule_refs == ["R4"]
    finally:
        os.unlink(tmp)


def test_eval_result_old_json_is_compatible():
    old = {
        "case_id": "c1",
        "task_type": "user_md_update",
        "score_total": 5.0,
        "scores": {},
    }
    result = EvalResult.from_dict(old)
    assert result.extraction_prompt_version == ""
    assert result.diagnostics == []
