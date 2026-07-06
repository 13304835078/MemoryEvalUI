from __future__ import annotations

import json
from pathlib import Path

from src.ui import prompt_editor


def test_prompt_list_combines_bundled_and_user_with_user_override(monkeypatch, tmp_path):
    user_judge = tmp_path / "home" / "prompts" / "judge"
    bundled_judge = tmp_path / "_internal" / "prompts" / "judge"
    user_judge.mkdir(parents=True)
    bundled_judge.mkdir(parents=True)
    (bundled_judge / "base.md").write_text("bundled base", encoding="utf-8")
    (bundled_judge / "shared.md").write_text("bundled shared", encoding="utf-8")
    (user_judge / "shared.md").write_text("user shared", encoding="utf-8")

    monkeypatch.setattr(prompt_editor, "PROMPTS_DIR", user_judge)
    monkeypatch.setattr(prompt_editor, "BUNDLED_JUDGE_PROMPTS_DIR", bundled_judge)

    assert prompt_editor.list_prompt_files() == ["base.md", "shared.md"]
    assert prompt_editor.load_prompt("base.md") == "bundled base"
    assert prompt_editor.load_prompt("shared.md") == "user shared"


def test_save_prompt_version_writes_user_dir_and_metadata(monkeypatch, tmp_path):
    user_generation = tmp_path / "home" / "prompts" / "generation"
    bundled_generation = tmp_path / "_internal" / "prompts" / "generation"
    bundled_generation.mkdir(parents=True)

    monkeypatch.setattr(prompt_editor, "GENERATION_PROMPTS_DIR", user_generation)
    monkeypatch.setattr(prompt_editor, "BUNDLED_GENERATION_PROMPTS_DIR", bundled_generation)

    saved = prompt_editor.save_prompt_version(
        "user_md_update",
        "new extraction prompt",
        "../candidate_prompt",
        prompt_kind="extraction",
    )

    assert saved == "candidate_prompt.md"
    prompt_path = user_generation / saved
    metadata_path = user_generation / f"{saved}.meta.json"
    assert prompt_path.read_text(encoding="utf-8") == "new extraction prompt"
    assert not (bundled_generation / saved).exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["filename"] == saved
    assert metadata["task_type"] == "user_md_update"
    assert metadata["prompt_kind"] == "extraction"
    assert metadata["sha1"] == prompt_editor.prompt_text_hash("new extraction prompt")


def test_extract_long_memory_update_prompt_from_nested_yaml():
    text = """
memory_extraction:
  create_template: |
    create prompt
  update_template: |
    update prompt
    second line
"""

    prompt = prompt_editor._extract_prompt_from_yaml(text, Path("memory.yaml"))

    assert prompt == "update prompt\nsecond line"
