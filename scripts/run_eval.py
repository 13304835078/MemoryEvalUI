"""Memory Eval UI - 评测运行 CLI

用法:
    python scripts/run_eval.py --case_path data/cases/xxx.jsonl \
        --output_path data/results/xxx.jsonl --task_type user_md_update --mock
"""

import os
import sys
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
import argparse
import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.schema import TaskType, EvalConfig, cases_from_jsonl, results_to_jsonl
from src.eval.eval_runner import EvalRunner


def main():
    parser = argparse.ArgumentParser(description="Memory Eval UI - Run Evaluation")
    parser.add_argument("--case_path", type=str, required=True, help="输入 case JSONL 路径")
    parser.add_argument("--output_path", type=str, required=True, help="输出 result JSONL 路径")
    parser.add_argument("--task_type", type=str, default="user_md_update",
                        choices=[t.value for t in TaskType], help="任务类型")
    parser.add_argument("--mock", action="store_true", help="使用 mock judge（不需要 API）")
    parser.add_argument("--limit", type=int, default=0, help="评测条数限制（0=全量）")
    parser.add_argument("--judge_model", type=str, default="", help="裁判模型名")
    parser.add_argument("--api_base", type=str, default="", help="API 地址")
    parser.add_argument("--api_token", type=str, default="", help="Bearer token")
    parser.add_argument("--prompt_file", type=str, default="", help="Judge prompt 文件名或路径")
    parser.add_argument("--judge_prompt_version", type=str, default="", help="Judge prompt 版本")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    config = EvalConfig.from_env_and_args(
        mock=args.mock,
        judge_model=args.judge_model,
        api_base=args.api_base,
        api_token=args.api_token,
    )

    errs = config.validate()
    if errs:
        print("配置错误:")
        for e in errs:
            print(f"  - {e}")
        sys.exit(1)

    task_type = TaskType(args.task_type)

    print(f"加载 cases: {args.case_path}")
    cases = cases_from_jsonl(args.case_path)
    if args.limit > 0:
        cases = cases[:args.limit]
    print(f"共 {len(cases)} 条 case")

    runner = EvalRunner(
        config,
        task_type,
        prompt_file=args.prompt_file,
        judge_prompt_version=args.judge_prompt_version,
    )
    results = runner.run(cases)

    results_to_jsonl(results, args.output_path)
    print(f"完成: {len(results)} 条结果 → {args.output_path}")

    fatal = sum(1 for r in results if r.fatal_error)
    if fatal:
        print(f"注意: {fatal} 条 fatal_error")


if __name__ == "__main__":
    main()
