from __future__ import annotations

from src import runtime_paths


def test_resolve_app_home_prefers_environment(monkeypatch, tmp_path):
    configured = tmp_path / "memory-home"
    monkeypatch.setenv("MEMORY_EVAL_HOME", str(configured))

    assert runtime_paths.resolve_app_home() == configured.resolve()


def test_resolve_app_home_uses_executable_directory_when_frozen(monkeypatch, tmp_path):
    executable = tmp_path / "release" / "MemoryEvalUI.exe"
    monkeypatch.delenv("MEMORY_EVAL_HOME", raising=False)
    monkeypatch.setattr(runtime_paths, "IS_FROZEN", True)
    monkeypatch.setattr(runtime_paths.sys, "executable", str(executable))

    assert runtime_paths.resolve_app_home() == executable.parent.resolve()


def test_legacy_writable_files_are_migrated_without_overwrite(monkeypatch, tmp_path):
    bundled = tmp_path / "_internal"
    app_home = tmp_path / "release"
    (bundled / "data" / "results").mkdir(parents=True)
    (bundled / "data" / "results" / "old.jsonl").write_text("old", encoding="utf-8")
    (bundled / "config").mkdir(parents=True)
    (bundled / "config" / "local_config.json").write_text("internal", encoding="utf-8")
    (bundled / "prompts" / "generation").mkdir(parents=True)
    (bundled / "prompts" / "generation" / "extract.md").write_text("bundled prompt", encoding="utf-8")
    (app_home / "prompts" / "generation").mkdir(parents=True)
    (app_home / "prompts" / "generation" / "extract.md").write_text("external prompt", encoding="utf-8")
    (app_home / "config").mkdir(parents=True)
    (app_home / "config" / "local_config.json").write_text("external", encoding="utf-8")

    monkeypatch.setattr(runtime_paths, "IS_FROZEN", True)
    monkeypatch.setattr(runtime_paths, "BUNDLED_ROOT", bundled)
    monkeypatch.setattr(runtime_paths, "APP_HOME", app_home)
    monkeypatch.setattr(runtime_paths, "DATA_DIR", app_home / "data")
    monkeypatch.setattr(runtime_paths, "CONFIG_DIR", app_home / "config")
    monkeypatch.setattr(runtime_paths, "LOGS_DIR", app_home / "logs")
    monkeypatch.setattr(runtime_paths, "PROMPTS_DIR", app_home / "prompts")
    monkeypatch.setattr(runtime_paths, "BUNDLED_PROMPTS_DIR", bundled / "prompts")

    runtime_paths.ensure_writable_layout()

    assert (app_home / "data" / "results" / "old.jsonl").read_text(encoding="utf-8") == "old"
    assert (app_home / "config" / "local_config.json").read_text(encoding="utf-8") == "external"
    assert (app_home / "prompts" / "generation" / "extract.md").read_text(encoding="utf-8") == "external prompt"
