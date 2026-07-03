import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.schema import EvalConfig
from src.ui.prompt_advisor import (
    _split_complete_blocks,
    build_advisor_user_message,
    call_prompt_advisor,
    collect_absolute_eval_evidence,
    collect_gsb_evidence,
    collect_review_evidence,
)
from src.schema import EvalResult


def test_collect_gsb_evidence_only_uses_mismatches():
    df = pd.DataFrame({
        "row_number": [2, 3],
        "pair_id": ["p1", "p2"],
        "人工GSB": ["G", "S"],
        "自动GSB": ["B", "S"],
        "是否一致": [False, True],
        "问题类型": ["事实", "格式"],
        "备注": ["人工认为模型1更准", ""],
        "query": ["q1", "q2"],
        "answer": ["a1", "a2"],
        "m1_score": [4.5, 4.0],
        "m2_score": [4.8, 4.0],
        "m1_judge备注": ["ok1", "ok"],
        "m2_judge备注": ["ok2", "ok"],
    })

    evidence = collect_gsb_evidence(df)

    assert len(evidence) == 1
    assert evidence[0]["pair_id"] == "p1"
    assert evidence[0]["human_gsb"] == "G"
    assert evidence[0]["auto_gsb"] == "B"


def test_collect_review_evidence_requires_human_fields():
    df = pd.DataFrame({
        "case_id": ["c1", "c2"],
        "human_score": ["", 3.0],
        "human_comment": ["", "漏记"],
        "score_total": [5.0, 4.8],
    })

    evidence = collect_review_evidence(df)

    assert len(evidence) == 1
    assert evidence[0]["case_id"] == "c2"
    assert evidence[0]["human_comment"] == "漏记"


def test_prompt_advisor_refuses_insufficient_evidence():
    result, raw = call_prompt_advisor(
        EvalConfig(mock=True),
        evidence=[{"row_id": "1"}],
        current_judge_prompt="prompt",
    )

    assert raw == ""
    assert result["can_suggest"] is False
    assert "证据少于 3 条" in result["evidence_summary"]


def test_collect_absolute_eval_evidence_uses_single_model_results():
    results = [
        EvalResult(
            case_id="good",
            task_type="user_md_update",
            score_total=5.0,
            scores={"correctness": 5},
            comment="ok",
        ),
        EvalResult(
            case_id="low",
            task_type="user_md_update",
            score_total=4.2,
            scores={"correctness": 4},
            comment="漏记稳定偏好",
            error_tags=["missing_key_info"],
            diagnostics=[{
                "dimension": "coverage",
                "rule_refs": ["### A4. 兴趣爱好"],
                "evidence_refs": ["用户说喜欢粤菜"],
                "output_refs": ["未记录粤菜"],
                "reason": "漏记",
            }],
            rule_refs=["### A4. 兴趣爱好"],
            evidence_refs=["用户说喜欢粤菜"],
            output_refs=["未记录粤菜"],
        ),
        EvalResult(
            case_id="fatal",
            task_type="user_md_update",
            score_total=0.0,
            fatal_error=True,
            comment="JSON解析失败",
        ),
    ]

    evidence = collect_absolute_eval_evidence(results, score_threshold=4.8)

    assert [item["case_id"] for item in evidence] == ["fatal", "low"]
    assert evidence[0]["fatal_error"] is True
    assert evidence[1]["error_tags"] == ["missing_key_info"]
    assert evidence[1]["rule_refs"] == ["### A4. 兴趣爱好"]


def test_collect_absolute_eval_evidence_can_include_all_as_weak_context():
    results = [
        EvalResult(
            case_id="good",
            task_type="user_md_update",
            score_total=5.0,
            scores={"correctness": 5},
            comment="ok",
        )
    ]

    normal = collect_absolute_eval_evidence(results, score_threshold=4.8)
    experimental = collect_absolute_eval_evidence(results, score_threshold=4.8, include_all=True)

    assert normal == []
    assert len(experimental) == 1
    assert experimental[0]["case_id"] == "good"
    assert experimental[0]["evidence_mode"] == "weak_context_from_result"


def test_build_advisor_user_message_marks_extraction_loop_constraints():
    evidence = [{"case_id": "c1", "evidence_mode": "weak_context_from_result"}]

    message = build_advisor_user_message(
        evidence,
        current_judge_prompt="judge",
        extraction_prompt="extract",
        target="extraction_prompt",
        advisor_mode="absolute_eval",
    )

    assert '"target": "extraction_prompt"' in message
    assert '"weak_context_count": 1' in message
    assert "候选提取 prompt" in message
    assert "自我强化" in message


def test_build_advisor_user_message_uses_sections_without_full_extraction_prompt():
    extraction_prompt = "## 规则\n" + ("- 很长的原始提取规则。\n" * 200)

    message = build_advisor_user_message(
        [{"case_id": "c1"}],
        current_judge_prompt="judge",
        extraction_prompt=extraction_prompt,
        target="extraction_prompt",
        advisor_mode="absolute_eval",
    )
    payload = json.loads(message)

    assert payload["extraction_prompt_hash"]
    assert payload["extraction_prompt_sections"]
    assert "未发送提取 prompt 全文" in payload["original_extraction_prompt"]
    assert payload["output_schema"]["candidate_extraction_prompt"] == ""


def test_prompt_advisor_refuses_absolute_and_gsb_modes_separately():
    abs_result, _ = call_prompt_advisor(
        EvalConfig(mock=True),
        evidence=[{"case_id": "1"}],
        current_judge_prompt="prompt",
        advisor_mode="absolute_eval",
    )
    gsb_result, _ = call_prompt_advisor(
        EvalConfig(mock=True),
        evidence=[{"row_id": "1"}],
        current_judge_prompt="prompt",
        advisor_mode="gsb_alignment",
    )

    assert "评测结果证据少于 3 条" in abs_result["evidence_summary"]
    assert "人工证据少于 3 条" in gsb_result["evidence_summary"]


def test_prompt_advisor_mock_with_enough_evidence_does_not_call_network():
    result, raw = call_prompt_advisor(
        EvalConfig(mock=True),
        evidence=[{"case_id": "1"}, {"case_id": "2"}, {"case_id": "3"}],
        current_judge_prompt="judge prompt",
        extraction_prompt="## 规则\nextract prompt",
        target="extraction_prompt",
        min_evidence=3,
    )

    assert result["can_suggest"] is True
    assert result["candidate_prompt_source"] == "applied_incremental_patch"
    assert "MOCK" in result["candidate_extraction_prompt"]
    assert result["extraction_prompt_patch_result"]["applied_edits"]
    assert "[MOCK]" in raw


def test_prompt_advisor_applies_incremental_patch_and_rejects_full_rewrite(monkeypatch):
    extraction_prompt = "## 规则\n- 单次评价默认不记录。\n"
    advisor_result = {
        "can_suggest": True,
        "evidence_summary": "ok",
        "diagnoses": [],
        "judge_prompt_changes": [],
        "candidate_judge_prompt": "",
        "extraction_prompt_notes": "补充边界",
        "extraction_prompt_patch": {
            "edits": [
                {
                    "op": "append_to_section",
                    "target_id": "S001",
                    "text": "- 影视作品单次评价默认视为即时评价，除非明确表达长期偏好。",
                    "reason": "边界不清",
                    "evidence_refs": ["case_1"],
                }
            ]
        },
        "candidate_extraction_prompt": "这是一整版模型重写，系统不应直接采用。",
        "risks": [],
        "validation_plan": [],
    }

    monkeypatch.setattr(
        "src.ui.prompt_advisor.requests.post",
        lambda *args, **kwargs: _FakeAdvisorResponse({
            "choices": [{"message": {"content": json.dumps(advisor_result, ensure_ascii=False)}}]
        }),
    )

    result, _raw = call_prompt_advisor(
        EvalConfig(judge_max_retries=1),
        evidence=[{"case_id": "case_1"}, {"case_id": "case_2"}, {"case_id": "case_3"}],
        current_judge_prompt="judge",
        extraction_prompt=extraction_prompt,
        target="extraction_prompt",
        min_evidence=3,
    )

    assert result["candidate_prompt_source"] == "applied_incremental_patch"
    assert result["model_candidate_extraction_prompt"].startswith("这是一整版")
    assert result["candidate_extraction_prompt"] != result["model_candidate_extraction_prompt"]
    assert "影视作品单次评价" in result["candidate_extraction_prompt"]


def test_prompt_advisor_does_not_use_full_rewrite_without_valid_patch(monkeypatch):
    advisor_result = {
        "can_suggest": True,
        "evidence_summary": "ok",
        "diagnoses": [],
        "judge_prompt_changes": [],
        "candidate_judge_prompt": "",
        "extraction_prompt_notes": "",
        "candidate_extraction_prompt": "完整重写版本",
        "risks": [],
        "validation_plan": [],
    }

    monkeypatch.setattr(
        "src.ui.prompt_advisor.requests.post",
        lambda *args, **kwargs: _FakeAdvisorResponse({
            "choices": [{"message": {"content": json.dumps(advisor_result, ensure_ascii=False)}}]
        }),
    )

    result, _raw = call_prompt_advisor(
        EvalConfig(judge_max_retries=1),
        evidence=[{"case_id": "case_1"}, {"case_id": "case_2"}, {"case_id": "case_3"}],
        current_judge_prompt="judge",
        extraction_prompt="## 规则\n- 原规则。\n",
        target="extraction_prompt",
        min_evidence=3,
    )

    assert result["candidate_prompt_source"] == "no_valid_incremental_patch"
    assert result["candidate_extraction_prompt"] == ""
    assert result["model_candidate_extraction_prompt"] == "完整重写版本"


class _FakeAdvisorResponse:
    def __init__(self, data):
        self._data = data
        self.text = json.dumps(data, ensure_ascii=False)

    def json(self):
        return self._data

    def raise_for_status(self):
        return None


class _FakeAdvisorStreamResponse:
    def __init__(self, content: str):
        chunks = [
            {"choices": [{"delta": {"content": content}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        self._lines = [f"data: {json.dumps(chunk, ensure_ascii=False)}" for chunk in chunks] + ["data: [DONE]"]
        self.text = "\n".join(self._lines)

    def iter_lines(self, decode_unicode=True):
        return iter(self._lines)

    def raise_for_status(self):
        return None


def test_prompt_advisor_caps_extraction_advisor_tokens_and_payload(monkeypatch):
    advisor_result = {
        "can_suggest": True,
        "evidence_summary": "ok",
        "diagnoses": [],
        "judge_prompt_changes": [],
        "candidate_judge_prompt": "",
        "extraction_prompt_notes": "",
        "candidate_extraction_prompt": "",
        "risks": [],
        "validation_plan": [],
    }
    captured = {}

    def fake_post(_url, headers, data, timeout):
        captured["payload"] = json.loads(data.decode("utf-8"))
        return _FakeAdvisorResponse({
            "choices": [{"message": {"content": json.dumps(advisor_result, ensure_ascii=False)}}]
        })

    monkeypatch.setattr("src.ui.prompt_advisor.requests.post", fake_post)

    result, _raw = call_prompt_advisor(
        EvalConfig(judge_max_retries=1, judge_max_tokens=8000),
        evidence=[{"case_id": "case_1", "comment": "长评语" * 500} for _ in range(12)],
        current_judge_prompt="judge prompt" * 3000,
        extraction_prompt="## 规则\n" + ("extract prompt\n" * 3000),
        target="extraction_prompt",
        min_evidence=3,
    )

    user_payload = json.loads(captured["payload"]["messages"][1]["content"])
    assert result["can_suggest"] is True
    assert captured["payload"]["max_tokens"] == 1200
    assert user_payload["stage"] == "1_intent_localization"
    assert user_payload["prompt_global_outline"]
    assert len(captured["payload"]["messages"][1]["content"]) < 40000


def test_prompt_advisor_two_stage_groups_cases_by_section(monkeypatch):
    extraction_prompt = "## A4 兴趣爱好\n- 单次评价默认不记录。\n\n## A5 排除规则\n- 只记录长期稳定特征。\n"
    stage1 = {
        "can_suggest": True,
        "evidence_summary": "同一规则边界不清",
        "diagnoses": [],
        "judge_prompt_changes": [],
        "candidate_judge_prompt": "",
        "extraction_prompt_notes": "需要澄清单次影视评价边界",
        "patch_intents": [
            {
                "intent_id": "I001",
                "section_id": "S001",
                "problem_type": "missing_boundary",
                "issue_summary": "单次影视评价是否记录边界不清",
                "proposed_direction": "补充长期偏好的判定条件",
                "confidence": "medium",
                "evidence_refs": ["case_1"],
            },
            {
                "intent_id": "I002",
                "section_id": "S001",
                "problem_type": "missing_boundary",
                "issue_summary": "同一边界在另一条样本重复出现",
                "proposed_direction": "合并为同一条边界规则",
                "confidence": "medium",
                "evidence_refs": ["case_2"],
            },
        ],
        "risks": [],
        "validation_plan": [],
    }
    stage2 = {
        "can_suggest": True,
        "section_id": "S001",
        "section_hash": "",
        "extraction_prompt_patch": {
            "edits": [
                {
                    "op": "append_to_section",
                    "target_id": "S001",
                    "text": "- 单次影视作品评价只有明确表达长期偏好或稳定审美时才沉淀。",
                    "reason": "合并同组样本的边界问题",
                    "evidence_refs": ["case_1"],
                },
                {
                    "op": "append_to_section",
                    "target_id": "S001",
                    "text": "- 单次影视作品评价只有明确表达长期偏好或稳定审美时才沉淀。",
                    "reason": "重复证据合并",
                    "evidence_refs": ["case_2"],
                },
            ]
        },
        "section_notes": "合并为一条规则",
        "risks": [],
    }
    responses = [
        _FakeAdvisorResponse({"choices": [{"message": {"content": json.dumps(stage1, ensure_ascii=False)}}]}),
        _FakeAdvisorResponse({"choices": [{"message": {"content": json.dumps(stage2, ensure_ascii=False)}}]}),
    ]
    sent_messages = []

    def fake_post(_url, headers, data, timeout):
        payload = json.loads(data.decode("utf-8"))
        sent_messages.append(json.loads(payload["messages"][1]["content"]))
        return responses.pop(0)

    monkeypatch.setattr("src.ui.prompt_advisor.requests.post", fake_post)

    result, raw = call_prompt_advisor(
        EvalConfig(judge_max_retries=1, judge_request_interval=0),
        evidence=[{"case_id": "case_1", "comment": "漏记"}, {"case_id": "case_2", "comment": "漏记"}],
        current_judge_prompt="judge",
        extraction_prompt=extraction_prompt,
        target="extraction_prompt",
        min_evidence=2,
    )

    assert result["advisor_flow"] == "two_stage_extraction_prompt_advisor"
    assert len(sent_messages) == 2
    assert sent_messages[0]["stage"] == "1_intent_localization"
    assert sent_messages[1]["stage"] == "2_section_patch"
    assert len(result["extraction_prompt_patch_result"]["applied_edits"]) == 1
    applied = result["extraction_prompt_patch_result"]["applied_edits"][0]
    assert applied["evidence_refs"] == ["case_1", "case_2"]
    assert "单次影视作品评价" in result["candidate_extraction_prompt"]
    assert json.loads(raw)["mode"] == "two_stage_extraction_prompt_advisor"


def test_prompt_advisor_skips_case_specific_or_overlong_patch(monkeypatch):
    extraction_prompt = "## A4 兴趣爱好\n- 单次评价默认不记录。\n"
    stage1 = {
        "can_suggest": True,
        "evidence_summary": "同一规则边界不清",
        "diagnoses": [],
        "patch_intents": [
            {
                "intent_id": "I001",
                "section_id": "S001",
                "problem_type": "missing_boundary",
                "issue_summary": "边界不清",
                "proposed_direction": "补充通用规则",
                "confidence": "medium",
                "evidence_refs": ["case_1"],
            }
        ],
        "risks": [],
        "validation_plan": [],
    }
    stage2 = {
        "can_suggest": True,
        "section_id": "S001",
        "extraction_prompt_patch": {
            "edits": [
                {
                    "op": "append_to_section",
                    "target_id": "S001",
                    "text": "- 针对 case_1，用户说喜欢某部具体作品时需要记录这一部作品的评价。",
                    "reason": "过细",
                    "evidence_refs": ["case_1"],
                },
                {
                    "op": "append_to_section",
                    "target_id": "S001",
                    "text": "- " + ("这是一条过长的细节规则。" * 80),
                    "reason": "过长",
                    "evidence_refs": ["case_1"],
                },
            ]
        },
        "section_notes": "bad",
    }
    responses = [
        _FakeAdvisorResponse({"choices": [{"message": {"content": json.dumps(stage1, ensure_ascii=False)}}]}),
        _FakeAdvisorResponse({"choices": [{"message": {"content": json.dumps(stage2, ensure_ascii=False)}}]}),
    ]

    def fake_post(_url, headers, data, timeout):
        return responses.pop(0)

    monkeypatch.setattr("src.ui.prompt_advisor.requests.post", fake_post)

    result, _raw = call_prompt_advisor(
        EvalConfig(judge_max_retries=1, judge_request_interval=0),
        evidence=[{"case_id": "case_1", "comment": "漏记"}],
        current_judge_prompt="judge",
        extraction_prompt=extraction_prompt,
        target="extraction_prompt",
        min_evidence=1,
    )

    assert result["candidate_prompt_source"] == "no_valid_incremental_patch"
    skipped = result["extraction_prompt_patch_skipped_before_apply"]
    assert len(skipped) >= 2
    assert any("具体 case" in item.get("message", "") for item in skipped)
    assert any("过长" in item.get("message", "") for item in skipped)


def test_prompt_advisor_retries_idle_timeout_with_compacted_payload(monkeypatch):
    evidence = [
        {
            "case_id": f"case_{idx}",
            "comment": "很长的评语" * 300,
            "diagnostics": [{"reason": "很长的诊断" * 200, "rule_refs": ["规则"] * 20}],
            "rule_refs": ["规则"] * 20,
            "evidence_refs": ["证据"] * 20,
            "output_refs": ["输出"] * 20,
        }
        for idx in range(8)
    ]
    advisor_result = {
        "can_suggest": True,
        "evidence_summary": "ok",
        "diagnoses": [],
        "judge_prompt_changes": [],
        "candidate_judge_prompt": "",
        "extraction_prompt_notes": "",
        "candidate_extraction_prompt": "",
        "risks": [],
        "validation_plan": [],
    }
    responses = [
        _FakeAdvisorResponse({"error": {"message": "websocket: close 1001 (going away): Connection Idle Timeout"}}),
        _FakeAdvisorResponse({"choices": [{"message": {"content": json.dumps(advisor_result, ensure_ascii=False)}}]}),
    ]
    payload_lengths = []
    sleeps = []

    def fake_post(_url, headers, data, timeout):
        payload_lengths.append(len(data))
        return responses.pop(0)

    monkeypatch.setattr("src.ui.prompt_advisor.requests.post", fake_post)
    monkeypatch.setattr("src.ui.prompt_advisor.time.sleep", lambda seconds: sleeps.append(seconds))

    result, raw = call_prompt_advisor(
        EvalConfig(judge_max_retries=2, judge_qps_backoff=1.0),
        evidence=evidence,
        current_judge_prompt="judge prompt" * 3000,
        extraction_prompt="extract prompt" * 3000,
        target="judge_prompt",
        min_evidence=3,
    )

    assert result["can_suggest"] is True
    assert json.loads(raw)["can_suggest"] is True
    assert len(payload_lengths) == 2
    assert payload_lengths[1] < payload_lengths[0]
    assert sleeps and sleeps[0] >= 2


def test_split_complete_blocks_never_cuts_paragraph_or_single_long_line():
    first = "第一段" * 30
    second = "第二段" * 30
    oversized = "超长单行" * 80

    blocks = _split_complete_blocks(
        f"{first}\n\n{second}\n\n{oversized}",
        max_chars=150,
    )

    combined = "\n\n".join(block["text"] for block in blocks)
    assert first in combined
    assert second in combined
    assert oversized in combined
    oversized_block = next(block for block in blocks if oversized in block["text"])
    assert oversized_block["editable"] is False


def test_extraction_advisor_batches_stage1_evidence(monkeypatch):
    sent_payloads = []
    response = {
        "can_suggest": True,
        "evidence_summary": "无需修改",
        "diagnoses": [],
        "patch_intents": [],
        "risks": [],
        "validation_plan": [],
    }

    def fake_post(_url, headers, data, timeout):
        payload = json.loads(data.decode("utf-8"))
        sent_payloads.append(payload)
        return _FakeAdvisorResponse({
            "choices": [{"message": {"content": json.dumps(response, ensure_ascii=False)}}]
        })

    monkeypatch.setattr("src.ui.prompt_advisor.requests.post", fake_post)
    monkeypatch.setattr("src.ui.prompt_advisor.time.sleep", lambda _seconds: None)

    result, _raw = call_prompt_advisor(
        EvalConfig(judge_max_retries=1, judge_request_interval=0),
        evidence=[
            {"case_id": f"case_{index}", "comment": "评语" * 300}
            for index in range(15)
        ],
        current_judge_prompt="judge",
        extraction_prompt="## 规则\n- 原规则。\n",
        target="extraction_prompt",
        min_evidence=1,
    )

    assert len(sent_payloads) == 8
    batch_sizes = [
        len(json.loads(payload["messages"][1]["content"])["evidence"])
        for payload in sent_payloads
    ]
    assert batch_sizes == [2, 2, 2, 2, 2, 2, 2, 1]
    assert all(payload["max_tokens"] == 1200 for payload in sent_payloads)
    assert len(result["extraction_prompt_request_metrics"]) == 8
    assert result["evidence_usage"]["selected_count"] == 15
    assert result["evidence_usage"]["initial_used_count"] == 15
    assert result["evidence_usage"]["all_selected_used_initially"] is True


def test_stage2_uses_complete_target_blocks_instead_of_whole_sections(monkeypatch):
    paragraphs = [f"规则段落{index}：" + ("内容" * 180) for index in range(20)]
    extraction_prompt = "## 长章节\n" + "\n\n".join(paragraphs) + "\n\n## 邻接章节\n- 邻接规则。\n"
    sent_messages = []

    def fake_post(_url, headers, data, timeout):
        payload = json.loads(data.decode("utf-8"))
        message = json.loads(payload["messages"][1]["content"])
        sent_messages.append(message)
        if message["stage"] == "1_intent_localization":
            result = {
                "can_suggest": True,
                "evidence_summary": "需要澄清",
                "patch_intents": [{
                    "intent_id": "I001",
                    "section_id": "S001",
                    "problem_type": "missing_boundary",
                    "issue_summary": "规则边界不清",
                    "proposed_direction": "补充通用边界",
                    "confidence": "medium",
                    "evidence_refs": ["case_1"],
                }],
                "risks": [],
                "validation_plan": [],
            }
        else:
            result = {
                "can_suggest": True,
                "section_id": "S001",
                "extraction_prompt_patch": {"mode": "incremental_patch", "edits": []},
                "section_notes": "原规则已覆盖",
                "risks": [],
            }
        return _FakeAdvisorResponse({
            "choices": [{"message": {"content": json.dumps(result, ensure_ascii=False)}}]
        })

    monkeypatch.setattr("src.ui.prompt_advisor.requests.post", fake_post)

    result, _raw = call_prompt_advisor(
        EvalConfig(judge_max_retries=1, judge_request_interval=0),
        evidence=[{"case_id": "case_1", "comment": "遗漏了稳定偏好"}],
        current_judge_prompt="judge",
        extraction_prompt=extraction_prompt,
        target="extraction_prompt",
        min_evidence=1,
    )

    stage2 = sent_messages[1]
    context = stage2["target_section_blocks"]
    assert "editable_sections" not in stage2
    assert context["block_count"] > 2
    assert len(context["editable_blocks"]) <= 2
    assert all(len(block["full_text"]) <= 2400 for block in context["editable_blocks"])
    assert len(json.dumps(stage2, ensure_ascii=False)) < 15000
    assert result["extraction_prompt_request_metrics"][1]["stage"] == "2_段落级编辑"


def test_extraction_stage_retry_reduces_same_batch_payload(monkeypatch):
    success = {
        "can_suggest": True,
        "evidence_summary": "无需修改",
        "patch_intents": [],
        "risks": [],
        "validation_plan": [],
    }
    responses = [
        _FakeAdvisorResponse({"error": {"message": "websocket: close 1001 Connection Idle Timeout"}}),
        _FakeAdvisorResponse({"choices": [{"message": {"content": json.dumps(success, ensure_ascii=False)}}]}),
    ]
    payloads = []

    def fake_post(_url, headers, data, timeout):
        payloads.append(json.loads(data.decode("utf-8")))
        return responses.pop(0)

    monkeypatch.setattr("src.ui.prompt_advisor.requests.post", fake_post)
    monkeypatch.setattr("src.ui.prompt_advisor.time.sleep", lambda _seconds: None)

    result, _raw = call_prompt_advisor(
        EvalConfig(judge_max_retries=2, judge_qps_backoff=0),
        evidence=[
            {"case_id": f"case_{index}", "comment": "较长评语" * 100}
            for index in range(2)
        ],
        current_judge_prompt="judge",
        extraction_prompt="\n\n".join(f"## 规则{index}\n- 内容。" for index in range(20)),
        target="extraction_prompt",
        min_evidence=1,
    )

    first_message = payloads[0]["messages"][1]["content"]
    second_message = payloads[1]["messages"][1]["content"]
    assert result["can_suggest"] is True
    assert len(second_message) < len(first_message)
    assert payloads[0]["max_tokens"] == 1200
    assert payloads[1]["max_tokens"] == 800
    assert len(json.loads(second_message)["evidence"]) == 2


def test_matching_rule_refs_skip_model_localization_call(monkeypatch):
    sent_messages = []

    def fake_post(_url, headers, data, timeout):
        payload = json.loads(data.decode("utf-8"))
        message = json.loads(payload["messages"][1]["content"])
        sent_messages.append(message)
        assert message["stage"] == "2_section_patch"
        result = {
            "can_suggest": True,
            "section_id": "S001",
            "extraction_prompt_patch": {"mode": "incremental_patch", "edits": []},
            "section_notes": "无需修改",
            "risks": [],
        }
        return _FakeAdvisorResponse({
            "choices": [{"message": {"content": json.dumps(result, ensure_ascii=False)}}]
        })

    monkeypatch.setattr("src.ui.prompt_advisor.requests.post", fake_post)

    result, _raw = call_prompt_advisor(
        EvalConfig(judge_max_retries=1, judge_request_interval=0),
        evidence=[{
            "case_id": "case_1",
            "comment": "兴趣爱好规则边界可能不清",
            "rule_refs": ["## A4 兴趣爱好"],
        }],
        current_judge_prompt="judge",
        extraction_prompt="## A4 兴趣爱好\n- 仅记录长期偏好。\n\n## A5 排除规则\n- 排除瞬时信息。\n",
        target="extraction_prompt",
        min_evidence=1,
    )

    assert len(sent_messages) == 1
    assert result["extraction_prompt_request_metrics"][0]["stage"] == "1_本地规则定位"
    assert result["extraction_prompt_request_metrics"][0]["request_chars"] == 0


def test_extraction_advisor_accepts_streaming_json(monkeypatch):
    stage2_result = {
        "can_suggest": True,
        "section_id": "S001",
        "extraction_prompt_patch": {"mode": "incremental_patch", "edits": []},
        "section_notes": "无需修改",
        "risks": [],
    }
    captured = {}

    def fake_post(_url, headers, data, timeout):
        captured["payload"] = json.loads(data.decode("utf-8"))
        return _FakeAdvisorStreamResponse(json.dumps(stage2_result, ensure_ascii=False))

    monkeypatch.setattr("src.ui.prompt_advisor.requests.post", fake_post)

    result, _raw = call_prompt_advisor(
        EvalConfig(judge_max_retries=1, judge_request_interval=0),
        evidence=[{
            "case_id": "case_1",
            "comment": "规则边界可能不清",
            "rule_refs": ["## A4 兴趣爱好"],
        }],
        current_judge_prompt="judge",
        extraction_prompt="## A4 兴趣爱好\n- 仅记录长期偏好。\n",
        target="extraction_prompt",
        min_evidence=1,
    )

    assert captured["payload"]["stream"] is True
    assert result["can_suggest"] is True
    assert result["extraction_prompt_stage2_summaries"][0]["section_notes"] == "无需修改"
