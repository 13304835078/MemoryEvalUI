"""Memory Eval UI Harness - CLI 工具：将原始输入转换为标准 case 格式

用法:
    # Excel 输入
    python scripts/convert_to_cases.py \\
        --input data/raw/dialogue.xlsx \\
        --task user_md_update \\
        --output data/cases/user_md_cases.jsonl

    # JSON / JSONL 输入
    python scripts/convert_to_cases.py \\
        --input data/raw/cases.json \\
        --output data/cases/out.jsonl

    # Markdown 目录输入
    python scripts/convert_to_cases.py \\
        --input-dir data/raw/md_cases/ \\
        --task user_md_update \\
        --output data/cases/md_cases.jsonl
"""

import os
import sys
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.schema import TaskType, cases_to_jsonl, validate_case
from src.loaders import ExcelLoader, JsonLoader, MdLoader


def main():
    parser = argparse.ArgumentParser(description="Memory Eval UI - Case Converter")
    parser.add_argument("--input", type=str, help="输入文件路径 (.xlsx / .json / .jsonl)")
    parser.add_argument("--input-dir", type=str, help="Markdown case 目录（用于 MD loader）")
    parser.add_argument("--task", type=str, default="user_md_update",
                        choices=[t.value for t in TaskType],
                        help="任务类型")
    parser.add_argument("--output", type=str, required=True, help="输出 JSONL 文件路径")
    parser.add_argument("--sheet", type=str, default="", help="Excel sheet 名称")
    args = parser.parse_args()

    task_type = TaskType(args.task)

    if args.input_dir:
        loader = MdLoader(task_type)
        cases = loader.load(args.input_dir)
    elif args.input:
        ext = os.path.splitext(args.input)[1].lower()
        if ext in (".xlsx", ".xls"):
            loader = ExcelLoader(task_type, sheet_name=args.sheet)
        elif ext in (".json", ".jsonl"):
            loader = JsonLoader(task_type)
        else:
            print(f"不支持的文件格式: {ext}")
            sys.exit(1)
        cases = loader.load(args.input)
    else:
        print("请指定 --input 或 --input-dir")
        sys.exit(1)

    validation_errors = []
    for i, case in enumerate(cases):
        errs = validate_case(case)
        if errs:
            validation_errors.append((i, case.case_id, errs))
            print(f"  [WARN] Case {i} ({case.case_id}) 校验问题: {errs}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cases_to_jsonl(cases, args.output)

    print(f"完成: {len(cases)} 个 case → {args.output}")
    if validation_errors:
        print(f"注意: {len(validation_errors)} 个 case 有校验警告")


if __name__ == "__main__":
    main()
