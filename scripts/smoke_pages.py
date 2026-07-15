from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest


def discover_pages(project_root: Path) -> list[Path]:
    return [project_root / "app.py", *sorted((project_root / "pages").glob("*.py"))]


def run_smoke(project_root: Path, timeout: float = 30.0) -> list[str]:
    failures: list[str] = []
    os.environ["MEMORY_EVAL_TEST_BYPASS_IDENTITY"] = "1"
    for path in discover_pages(project_root):
        relative_path = path.relative_to(project_root)
        try:
            app = AppTest.from_file(str(path)).run(timeout=timeout)
        except Exception as exc:
            failures.append(f"{relative_path}: 页面加载异常: {exc}")
            continue

        if app.exception:
            details = "; ".join(str(item.value) for item in app.exception)
            failures.append(f"{relative_path}: {details}")
            continue

        print(f"[页面冒烟] {relative_path}: 通过")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="加载全部 Streamlit 页面并检查未捕获异常")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    failures = run_smoke(project_root, timeout=args.timeout)
    if failures:
        print("\n页面冒烟测试失败：", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("全部 Streamlit 页面冒烟测试通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
