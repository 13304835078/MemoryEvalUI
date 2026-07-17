"""
Memory Eval UI — 一键交互入口

运行方式:
    cd memory_eval_ui
    python run.py
"""

import os
import sys
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def pick_file(prompt: str, base_dir: str, pattern: str = "*") -> str:
    """列出目录下文件供用户选择，或直接输入路径"""
    files = []
    for root, dirs, filenames in os.walk(base_dir):
        for f in filenames:
            if pattern == "*" or f.endswith(pattern):
                files.append(os.path.join(root, f))
        break

    print(f"\n{prompt}")
    if files:
        for i, f in enumerate(files, 1):
            print(f"  {i}. {os.path.basename(f)}")
        print("  0. 手动输入路径")
    else:
        print("  (目录为空)")

    choice = input("请选择序号: ").strip()
    if choice.isdigit():
        idx = int(choice)
        if 1 <= idx <= len(files):
            return files[idx - 1]
    return input("请输入文件路径: ").strip()


def run_cmd(cmd: list[str]) -> None:
    print(f"\n>>> 执行: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=PROJECT_ROOT)


def menu_convert_excel():
    """Excel → 标准 Case"""
    input_path = pick_file("选择 Excel 文件:", os.path.join(PROJECT_ROOT, "data", "raw"), ".xlsx")
    if not input_path:
        return

    task_types = ["user_md_update", "day_memory", "long_memory", "summary"]
    print("\n任务类型:")
    for i, t in enumerate(task_types, 1):
        print(f"  {i}. {t}")
    task = input("请选择序号 [1]: ").strip()
    task_type = task_types[int(task) - 1] if task.isdigit() and 1 <= int(task) <= 4 else "user_md_update"

    output_name = input("输出文件名 [cases.jsonl]: ").strip() or "cases.jsonl"
    output_path = os.path.join(PROJECT_ROOT, "data", "cases", output_name)

    sheet = input("Excel sheet 名称（直接回车跳过）: ").strip()

    cmd = [
        sys.executable, os.path.join(PROJECT_ROOT, "scripts", "convert_to_cases.py"),
        "--input", input_path,
        "--output", output_path,
        "--task", task_type,
    ]
    if sheet:
        cmd += ["--sheet", sheet]
    run_cmd(cmd)


def menu_prepare_from_run_output():
    """run_user.py 输出 → 评测 Case"""
    input_path = pick_file(
        "选择 run_user.py 的输出 Excel:",
        os.path.join(os.path.dirname(PROJECT_ROOT), "output"),
        ".xlsx",
    )
    if not input_path:
        return

    model = input("模型名称 [unknown]: ").strip() or "unknown"
    prompt_ver = input("Prompt 版本 [unknown]: ").strip() or "unknown"
    chunk_size = input("chunk_size [10]: ").strip() or "10"
    output_name = input("输出文件名 (如 glm5_cases.jsonl): ").strip() or "cases.jsonl"
    output_path = os.path.join(PROJECT_ROOT, "data", "cases", output_name)

    run_cmd([
        sys.executable, os.path.join(PROJECT_ROOT, "scripts", "prepare_eval_from_run_output.py"),
        "--input", input_path,
        "--output", output_path,
        "--model", model,
        "--prompt_version", prompt_ver,
        "--chunk_size", chunk_size,
    ])


def menu_run_eval():
    """运行评测"""
    case_path = pick_file("选择 case 文件:", os.path.join(PROJECT_ROOT, "data", "cases"), ".jsonl")
    if not case_path:
        return

    print("\n评测模式:")
    print("  1. Mock (离线，无需 API)")
    print("  2. 真实 API (需配置环境变量)")
    mode = input("请选择 [1]: ").strip()
    mock = mode != "2"

    task_types = ["user_md_update", "day_memory", "long_memory", "summary"]
    print("\n任务类型:")
    for i, t in enumerate(task_types, 1):
        print(f"  {i}. {t}")
    task = input("请选择序号 [1]: ").strip()
    task_type = task_types[int(task) - 1] if task.isdigit() and 1 <= int(task) <= 4 else "user_md_update"

    limit = input("评测条数限制（0=全量）[0]: ").strip() or "0"

    output_name = input("输出文件名 [results.jsonl]: ").strip() or "results.jsonl"
    output_path = os.path.join(PROJECT_ROOT, "data", "results", output_name)

    cmd = [
        sys.executable, os.path.join(PROJECT_ROOT, "scripts", "run_eval.py"),
        "--case_path", case_path,
        "--output_path", output_path,
        "--task_type", task_type,
        "--limit", limit,
    ]
    if mock:
        cmd.append("--mock")
    else:
        if not os.environ.get("EVAL_API_BASE_URL"):
            base = input("API 地址: ").strip()
            if base:
                cmd += ["--api_base", base]
        if not os.environ.get("EVAL_MODEL_NAME"):
            model = input("裁判模型名: ").strip()
            if model:
                cmd += ["--judge_model", model]
        if not os.environ.get("EVAL_API_BEARER_TOKEN"):
            token = input("Bearer Token: ").strip()
            if token:
                cmd += ["--api_token", token]

    run_cmd(cmd)


def menu_analyze():
    """分析结果"""
    result_path = pick_file("选择结果文件:", os.path.join(PROJECT_ROOT, "data", "results"), ".jsonl")
    if not result_path:
        return

    print("\n分组方式（可选）:")
    print("  1. 不分组")
    print("  2. 按 judge_model 分组")
    print("  3. 按 judge_prompt_version 分组")
    gmap = {
        "2": "model_name",
        "3": "prompt_version",
        "4": "judge_model",
        "5": "judge_prompt_version",
    }
    g_choice = input("请选择 [1]: ").strip()
    group = gmap.get(g_choice, "")

    cmd = [
        sys.executable, os.path.join(PROJECT_ROOT, "scripts", "analyze_results.py"),
        "--result_path", result_path,
    ]
    if group:
        cmd += ["--group_by", group]
    run_cmd(cmd)

# 5. 启动 Streamlit UI
def menu_start_ui():
    run_cmd([
        sys.executable, "-m", "streamlit", "run",
        os.path.join(PROJECT_ROOT, "app.py"),
    ])

def main():
    print("\n" + "=" * 55)
    print("    Memory Eval UI — 评测工具")
    print("=" * 55)

    menu = {
        "1": ("Excel → 标准 Case", menu_convert_excel),
        "2": ("run_user.py 输出 → 评测 Case", menu_prepare_from_run_output),
        "3": ("运行评测 (Mock / 真实 API)", menu_run_eval),
        "4": ("分析评测结果", menu_analyze),
        "5": ("启动 Streamlit UI", menu_start_ui),
        "0": ("退出", None),
    }

    while True:
        print()
        for key, (label, _) in menu.items():
            print(f"  {key}. {label}")
        choice = input("\n请选择操作: ").strip()

        if choice == "0":
            print("再见！")
            break
        if choice in menu and menu[choice][1]:
            try:
                menu[choice][1]()
            except KeyboardInterrupt:
                print("\n已取消")
            except Exception as e:
                print(f"出错: {e}")
        else:
            print("无效选择，请重试。")


if __name__ == "__main__":
    main()
