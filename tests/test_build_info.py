from __future__ import annotations

import json

from src import build_info


def test_build_info_prefers_app_home_file(monkeypatch, tmp_path):
    app_home = tmp_path / "app"
    bundled = tmp_path / "_internal"
    app_home.mkdir()
    bundled.mkdir()
    (bundled / "build_info.json").write_text(json.dumps({"version": "bundled"}), encoding="utf-8")
    (app_home / "build_info.json").write_text(json.dumps({"version": "external"}), encoding="utf-8")

    monkeypatch.setattr(build_info.runtime_paths, "APP_HOME", app_home)
    monkeypatch.setattr(build_info.runtime_paths, "BUNDLED_ROOT", bundled)
    monkeypatch.setattr(build_info.runtime_paths, "SOURCE_ROOT", tmp_path / "source")

    info = build_info.get_build_info()

    assert info["version"] == "external"
    assert build_info.format_build_label(info).startswith("external /")


def test_build_info_falls_back_to_dev(monkeypatch, tmp_path):
    monkeypatch.setattr(build_info.runtime_paths, "APP_HOME", tmp_path / "missing-app")
    monkeypatch.setattr(build_info.runtime_paths, "BUNDLED_ROOT", tmp_path / "missing-bundled")
    monkeypatch.setattr(build_info.runtime_paths, "SOURCE_ROOT", tmp_path / "missing-source")
    monkeypatch.setattr(build_info, "_git_value", lambda *_args: "")

    info = build_info.get_build_info()

    assert info["version"] == "dev"
    assert info["build_mode"] == "development"
