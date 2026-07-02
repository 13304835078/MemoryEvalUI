# MemoryEvalUI.spec
# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    copy_metadata,
)

project_root = Path.cwd()


def add_data_if_exists(source, target):
    path = project_root / source
    if path.exists():
        datas.append((source, target))


datas = [
    ("app.py", "."),
    (".streamlit", ".streamlit"),
    ("pages", "pages"),
    ("src", "src"),  # Keep src as a real directory under _internal/src.
    ("prompts", "prompts"),
    ("rules", "rules"),
]

# Only package example config and empty directory placeholders. Do not ship
# local tokens, uploaded Excel files, or evaluation results to collaborators.
add_data_if_exists("config/.gitkeep", "config")
add_data_if_exists("config/local_config.example.json", "config")
add_data_if_exists("data/cases/.gitkeep", "data/cases")
add_data_if_exists("data/raw/uploads/.gitkeep", "data/raw/uploads")
add_data_if_exists("data/results/.gitkeep", "data/results")
add_data_if_exists("logs/.gitkeep", "logs")

datas += collect_data_files("streamlit")

for pkg in [
    "streamlit",
    "altair",
    "pyarrow",
    "pandas",
    "numpy",
    "packaging",
    "protobuf",
    "tornado",
    "watchdog",
    "blinker",
    "click",
    "cachetools",
    "requests",
    "jinja2",
    "tenacity",
    "typing_extensions",
    "openpyxl",
    "pyyaml",
    "starlette",
    "uvicorn",
    "anyio",
    "python_multipart",
]:
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass

hiddenimports = []
hiddenimports += collect_submodules("streamlit")

# Keep this import collection, but local src files are still copied through datas.
try:
    hiddenimports += collect_submodules("src")
except Exception:
    pass

hiddenimports += [
    "pandas",
    "openpyxl",
    "requests",
    "yaml",
    "starlette",
    "uvicorn",
    "anyio",
    "python_multipart",
    "python_multipart.multipart",
    "multipart",
    "multipart.multipart",
    "httptools",
    "websockets",
    "streamlit.web.cli",
    "streamlit.runtime.scriptrunner",
    "streamlit.runtime",
    "streamlit.runtime.runtime",
    "streamlit.web.bootstrap",
    "streamlit.web.server.server",
    "streamlit.web.server.websocket_headers",
    "tornado",
    "tornado.web",
    "tornado.websocket",
    "tornado.httpserver",
    "tornado.ioloop",
    "watchdog",
    "watchdog.observers",
    "watchdog.events",
    "altair",
    "pyarrow",
]

for pkg in [
    "starlette",
    "uvicorn",
    "anyio",
    "python_multipart",
    "multipart",
    "httptools",
    "websockets",
]:
    try:
        hiddenimports += collect_submodules(pkg)
    except Exception:
        pass

a = Analysis(
    ["run_streamlit.py"],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "pytest",
        "tests",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MemoryEvalUI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="MemoryEvalUI",
)
