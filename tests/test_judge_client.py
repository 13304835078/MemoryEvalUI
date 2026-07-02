import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.schema import EvalConfig
from src.eval.judge_client import MockJudgeClient, RealJudgeClient, JUDGE_RESULT_SCHEMA


def test_mock_judge_pass_all():
    config = EvalConfig(mock=True)
    client = MockJudgeClient(config, strategy="pass_all")
    user_msg = "## 新 USER.md\n- 姓名: 张三\n- 职业: 工程师\n"
    result, raw = client.judge("system prompt", user_msg)
    assert result is not None
    assert result["score_total"] == 5.0
    for dim in JUDGE_RESULT_SCHEMA["dimension_keys"]:
        assert result["scores"][dim] == 5
    assert result["error_tags"] == []
    assert result["fatal_error"] is False
    assert raw is not None
    parsed = json.loads(raw)
    assert parsed["score_total"] == 5.0


def test_mock_judge_random():
    config = EvalConfig(mock=True)
    client = MockJudgeClient(config, strategy="random")
    result, raw = client.judge("prompt", "## 新 USER.md\n- content\n")
    assert result is not None
    assert 2.0 <= result["score_total"] <= 5.0
    for dim in JUDGE_RESULT_SCHEMA["dimension_keys"]:
        assert 3 <= result["scores"][dim] <= 5


def test_mock_judge_empty_candidate():
    config = EvalConfig(mock=True)
    client = MockJudgeClient(config, strategy="pass_all")
    user_msg = "## 新 USER.md\n\n"
    result, raw = client.judge("prompt", user_msg)
    assert result is not None
    assert result["fatal_error"] is True
    assert result["score_total"] == 0.0


def test_mock_judge_empty_candidate_with_reasoning():
    config = EvalConfig(mock=True)
    client = MockJudgeClient(config, strategy="pass_all")
    user_msg = "## 模型 reasoning\n对话没有稳定用户画像信息，无需更新。\n\n## 新 USER.md\n\n"
    result, raw = client.judge("prompt", user_msg)
    assert result is not None
    assert result["fatal_error"] is False
    assert result["score_total"] == 5.0


def test_real_judge_json_parse():
    text = '{"score_total":4.2,"scores":{"correctness":5,"coverage":4,"update_logic":4,"memory_boundary":3,"conciseness":4,"format":5},"comment":"ok","error_tags":[],"fatal_error":false}'
    result = RealJudgeClient._parse_json_response(text)
    assert result is not None
    assert result["score_total"] == 4.2


def test_real_judge_json_parse_with_markdown_wrapper():
    text = '```json\n{"score_total":3.0,"scores":{"correctness":3,"coverage":3,"update_logic":3,"memory_boundary":3,"conciseness":3,"format":3},"comment":"test","error_tags":["over_memory"],"fatal_error":false}\n```'
    result = RealJudgeClient._parse_json_response(text)
    assert result is not None
    assert result["score_total"] == 3.0
    assert result["error_tags"] == ["over_memory"]


def test_real_judge_json_parse_invalid():
    result = RealJudgeClient._parse_json_response("这是纯文本不是JSON")
    assert result is None


def test_real_judge_normalize_common_aliases():
    raw = {
        "total_score": 4.1,
        "dimension_scores": {
            "正确性": 5,
            "完整性": 4,
            "更新合理性": 4,
            "记忆边界": 3,
            "简洁性": 4,
            "格式": 5,
        },
        "reason": "整体可用",
        "errors": "over_memory, missing_key_info",
        "fatal": "false",
    }
    result = RealJudgeClient._normalize_judge_result(raw)
    valid, reason = RealJudgeClient._is_valid_judge_result(result)
    assert valid, reason
    assert result["score_total"] == 4.1
    assert result["scores"]["correctness"] == 5
    assert result["scores"]["memory_boundary"] == 3
    assert result["comment"] == "整体可用"
    assert result["error_tags"] == ["over_memory", "missing_key_info"]
    assert result["fatal_error"] is False


def test_real_judge_normalize_optional_reference_aliases():
    raw = {
        "score_total": 3,
        "scores": {
            "正确性": 3,
            "完整性": 3,
            "更新合理性": 3,
            "记忆边界": 2,
            "简洁性": 4,
            "格式": 5,
        },
        "comment": "边界问题",
        "error_tags": [],
        "fatal_error": False,
        "扣分点": [{"维度": "memory_boundary", "规则引用": ["R2"]}],
        "规则引用": ["R2"],
        "证据引用": ["帮我查天气"],
        "输出引用": ["喜欢晴天"],
    }
    result = RealJudgeClient._normalize_judge_result(raw)
    valid, reason = RealJudgeClient._is_valid_judge_result(result)
    assert valid, reason
    assert result["diagnostics"] == [{"维度": "memory_boundary", "规则引用": ["R2"]}]
    assert result["rule_refs"] == ["R2"]
    assert result["evidence_refs"] == ["帮我查天气"]
    assert result["output_refs"] == ["喜欢晴天"]


def test_real_judge_build_payload_uses_core_options_only():
    config = EvalConfig(
        judge_model="m",
        judge_temperature=0.0,
        judge_top_p=0.7,
        judge_top_k=1,
        judge_stop=["END"],
        judge_stream=True,
        judge_stream_include_usage=True,
        judge_send_enable_thinking=False,
        judge_send_skip_special_tokens=True,
        judge_skip_special_tokens=True,
        judge_call_from="memory_eval",
        judge_session_id="s1",
        judge_interaction_id=2,
        judge_prompt_cache_id="task_1",
        judge_prompt_cache_location="both",
        judge_moderation_action="dudu",
        judge_extra_body_json='{"custom":"x"}',
    )
    client = RealJudgeClient(config)
    payload = client._build_payload("system", "user")

    assert payload["top_p"] == 0.7
    assert payload["top_k"] == 1
    assert payload["stream"] is False
    assert "stream_options" not in payload
    assert "stop" not in payload
    assert "promptCacheId" not in payload
    assert "call_from" not in payload["extra_body"]
    assert "session_id" not in payload["extra_body"]
    assert "interaction_id" not in payload["extra_body"]
    assert "enable_thinking" not in payload["extra_body"]
    assert payload["extra_body"]["skip_special_tokens"] is False
    assert "moderation_options" not in payload["extra_body"]
    assert "custom" not in payload["extra_body"]


def test_real_judge_build_headers_uses_bearer_only():
    config = EvalConfig(
        judge_auth_type="hmac",
        judge_api_bearer_token="token",
        judge_hmac_access_key="ak",
        judge_hmac_secret_key="sk",
    )
    client = RealJudgeClient(config)
    headers = client._build_headers()

    assert headers["Content-Type"] == "application/json"
    assert headers["Authorization"] == "Bearer token"
    assert "accessKey" not in headers
    assert "ts" not in headers
    assert "sign" not in headers


def test_real_judge_rate_limit_backoff_parses_qps_limit():
    config = EvalConfig(judge_qps_backoff=2.0)
    client = RealJudgeClient(config)

    assert RealJudgeClient._parse_qps_limit("QPS limit exceeded, limit:0.10") == 0.10
    assert client._get_rate_limit_backoff("QPS limit exceeded, limit:0.10") >= 11.0


def test_real_judge_detects_retryable_idle_timeout():
    assert RealJudgeClient._is_retryable_transient_error(
        "websocket: close 1001 (going away): Connection Idle Timeout"
    )
    assert RealJudgeClient._is_retryable_transient_error("Gateway Timeout 504")


class _FakeJsonResponse:
    def __init__(self, data):
        self._data = data
        self.text = json.dumps(data, ensure_ascii=False)

    def json(self):
        return self._data

    def raise_for_status(self):
        return None


def test_real_judge_rate_limit_retry_uses_global_wait_callback(monkeypatch):
    ok_result = {
        "score_total": 5.0,
        "scores": {key: 5 for key in JUDGE_RESULT_SCHEMA["dimension_keys"]},
        "comment": "ok",
        "error_tags": [],
        "fatal_error": False,
    }
    responses = [
        _FakeJsonResponse({"error": {"message": "QPS limit exceeded, limit:0.10"}}),
        _FakeJsonResponse({"choices": [{"message": {"content": json.dumps(ok_result, ensure_ascii=False)}}]}),
    ]
    sleeps = []
    waits = []

    monkeypatch.setattr("src.eval.judge_client.requests.post", lambda *args, **kwargs: responses.pop(0))
    monkeypatch.setattr("src.eval.judge_client.time.sleep", lambda seconds: sleeps.append(seconds))

    config = EvalConfig(judge_max_retries=2, judge_qps_backoff=1.0)
    client = RealJudgeClient(config)
    client.rate_limit_wait_callback = lambda: waits.append("waited")

    result, raw = client.judge("system", "user")

    assert result is not None
    assert result["score_total"] == 5.0
    assert waits == ["waited"]
    assert sleeps and sleeps[0] >= 11.0
    assert raw is not None


class _FakeStreamResponse:
    def __init__(self, lines):
        self._lines = lines

    def iter_lines(self, decode_unicode=True):
        yield from self._lines


def test_real_judge_extract_stream_content():
    config = EvalConfig()
    client = RealJudgeClient(config)
    response = _FakeStreamResponse([
        'data: {"choices":[{"delta":{"content":"{\\"score_total\\":5,"},"finish_reason":""}]}',
        'data: {"choices":[{"delta":{"content":"\\"scores\\":{},\\"comment\\":\\"ok\\",\\"error_tags\\":[],\\"fatal_error\\":false}"},"finish_reason":"stop"}]}',
        "data: [DONE]",
    ])
    content, error = client._extract_stream_content(response)

    assert error == ""
    assert json.loads(content)["score_total"] == 5
