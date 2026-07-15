from __future__ import annotations

import os
import sys
from pathlib import Path

from streamlit.web import cli as stcli


def resource_path(relative_path: str) -> Path:
    base_path = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base_path / relative_path


def main() -> None:
    if "--background-worker" in sys.argv:
        index = sys.argv.index("--background-worker")
        if index + 1 >= len(sys.argv):
            raise SystemExit("缺少后台任务 request.json 路径")
        from src.ui.task_worker import run_worker

        raise SystemExit(run_worker(sys.argv[index + 1]))

    app_path = resource_path("app.py")
    os.environ.setdefault("STREAMLIT_GLOBAL_DEVELOPMENT_MODE", "false")
    os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--global.developmentMode=false",
        "--server.fileWatcherType=none",
        "--server.maxUploadSize=1024",
        "--server.enableXsrfProtection=true",
        "--server.enableCORS=true",
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
