import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ui.prompt_patch import apply_prompt_patch, prompt_sections_for_model, split_prompt_sections


def test_split_prompt_sections_keeps_markdown_section_body_together():
    prompt = """# Role
你是提取专家。

## 1. 只基于 user 提取
- 只能记录用户自己的稳定信息。
- 不记录 assistant 事实。

### A4. 兴趣爱好
- 用户明确表达长期偏好时可以记录。
"""

    sections = split_prompt_sections(prompt)

    assert [section.section_id for section in sections] == ["S001", "S002", "S003"]
    assert sections[1].title == "## 1. 只基于 user 提取"
    assert "- 只能记录用户自己的稳定信息。" in sections[1].text
    assert "### A4. 兴趣爱好" not in sections[1].text
    assert "- 用户明确表达长期偏好时可以记录。" in sections[2].text


def test_prompt_sections_for_model_contains_ids_and_previews():
    rows = prompt_sections_for_model("## A\n正文\n\n## B\n更多正文", preview_chars=10)

    assert rows[0]["section_id"] == "S001"
    assert rows[0]["title"] == "## A"
    assert rows[0]["preview"].startswith("## A")


def test_apply_prompt_patch_appends_and_replaces_with_evidence_refs():
    prompt = """## 规则
- 单次评价默认不记录。
"""
    patch = {
        "edits": [
            {
                "op": "replace_within_section",
                "target_id": "S001",
                "old_text": "- 单次评价默认不记录。",
                "new_text": "- 单次评价默认不记录；除非用户明确表达长期、稳定、可复用偏好。",
                "reason": "边界需要更清晰",
                "evidence_refs": ["case_1"],
            },
            {
                "op": "append_to_section",
                "target_id": "S001",
                "text": "- 对影视作品的单次服化道评价默认视为即时评价。",
                "reason": "补充影视评价边界",
                "evidence_refs": ["case_2"],
            },
        ]
    }

    result = apply_prompt_patch(prompt, patch)

    assert len(result["applied_edits"]) == 2
    assert "长期、稳定、可复用偏好" in result["candidate_prompt"]
    assert "单次服化道评价" in result["candidate_prompt"]
    assert result["diff"]


def test_apply_prompt_patch_skips_unverified_or_oversized_edits():
    prompt = "## 规则\n- 原规则。\n"
    patch = {
        "edits": [
            {
                "op": "append_to_section",
                "target_id": "S001",
                "text": "- 没有证据引用。",
                "reason": "缺证据",
                "evidence_refs": [],
            },
            {
                "op": "append_to_section",
                "target_id": "S001",
                "text": "超长修改" * 500,
                "reason": "过大",
                "evidence_refs": ["case_1"],
            },
        ]
    }

    result = apply_prompt_patch(prompt, patch, max_change_ratio=0.1, min_change_chars=20)

    assert result["applied_edits"] == []
    assert len(result["skipped_edits"]) == 2
    assert result["candidate_prompt"] == prompt


def test_append_patch_removes_duplicate_heading_and_diff_markers():
    prompt = "## 基本原则\n- 只记录长期稳定信息。\n"
    patch = {
        "edits": [
            {
                "op": "append_to_section",
                "target_id": "S001",
                "text": "+## 基本原则\n+-严格过滤瞬时信息。",
                "reason": "模型误把 diff 和标题放进 patch",
                "evidence_refs": ["case_1"],
            }
        ]
    }

    result = apply_prompt_patch(prompt, patch)

    assert len(result["applied_edits"]) == 1
    assert result["candidate_prompt"].count("## 基本原则") == 1
    assert "+##" not in result["candidate_prompt"]
    assert "+-" not in result["candidate_prompt"]
    assert "- 严格过滤瞬时信息。" in result["candidate_prompt"]
    assert "章节标题" in result["applied_edits"][0]["message"]


def test_append_patch_aligns_with_target_section_indent():
    prompt = "      ## 基本原则\n      - 只记录长期稳定信息。\n"
    patch = {
        "edits": [
            {
                "op": "append_to_section",
                "target_id": "S001",
                "text": "## 基本原则\n- 严格过滤瞬时信息。",
                "reason": "模型返回了重复标题且没有缩进",
                "evidence_refs": ["case_1"],
            }
        ]
    }

    result = apply_prompt_patch(prompt, patch)

    assert len(result["applied_edits"]) == 1
    assert result["candidate_prompt"].count("## 基本原则") == 1
    assert "\n      - 严格过滤瞬时信息。" in result["candidate_prompt"]
    assert "\n- 严格过滤瞬时信息。" not in result["candidate_prompt"]
    assert result["applied_edits"][0]["applied_text"] == "      - 严格过滤瞬时信息。"


def test_append_patch_skips_when_only_duplicate_heading_remains():
    prompt = "## 基本原则\n- 只记录长期稳定信息。\n"
    patch = {
        "edits": [
            {
                "op": "append_to_section",
                "target_id": "S001",
                "text": "## 基本原则",
                "reason": "只有重复标题",
                "evidence_refs": ["case_1"],
            }
        ]
    }

    result = apply_prompt_patch(prompt, patch)

    assert result["applied_edits"] == []
    assert len(result["skipped_edits"]) == 1
    assert result["candidate_prompt"] == prompt
    assert "章节标题" in result["skipped_edits"][0]["message"]


def test_append_patch_requires_list_marker_in_list_section():
    prompt = "## B1 用户画像\n- 姓名、昵称\n- 性别\n"
    patch = {
        "edits": [
            {
                "op": "append_to_section",
                "target_id": "S001",
                "text": "严禁记录泛化的娱乐偏好。",
                "reason": "模型返回了非列表段落",
                "evidence_refs": ["case_1"],
            }
        ]
    }

    result = apply_prompt_patch(prompt, patch)

    assert result["applied_edits"] == []
    assert result["candidate_prompt"] == prompt
    assert "列表结构" in result["skipped_edits"][0]["message"]


def test_apply_prompt_patch_deletes_exact_text_with_evidence_refs():
    prompt = "## 规则\n- 保留规则。\n- 冗余规则。\n"
    patch = {
        "edits": [
            {
                "op": "delete_within_section",
                "target_id": "S001",
                "old_text": "- 冗余规则。\n",
                "reason": "删除重复规则",
                "evidence_refs": ["case_1"],
            }
        ]
    }

    result = apply_prompt_patch(prompt, patch)

    assert len(result["applied_edits"]) == 1
    assert "- 保留规则。" in result["candidate_prompt"]
    assert "冗余规则" not in result["candidate_prompt"]
