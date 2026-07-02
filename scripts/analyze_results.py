"""Memory Eval UI - 结果分析 CLI

用法:
    python scripts/analyze_results.py \
  --result_path data/results/demo_results.jsonl \
  --group_by model_name
"""

import os
import sys

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.schema import results_from_jsonl, TaskType
from src.eval.metrics import compute_aggregations, group_by, print_summary, DIM_LABELS

TASK_TITLES = {
    "user_md_update": "USER.md 更新评测统计",
    "day_memory": "天级记忆评测统计",
    "long_memory": "长期记忆评测统计",
    "summary": "摘要评测统计",
}


def main():
    parser = argparse.ArgumentParser(description="Memory Eval UI - 结果分析")
    parser.add_argument("--result_path", type=str, required=True, help="输入 result JSONL 路径")
    parser.add_argument("--group_by", type=str, default="", help="按字段分组统计（如 judge_model）")
    args = parser.parse_args()

    results = results_from_jsonl(args.result_path)
    if not results:
        print("结果文件为空，无数据可分析。")
        sys.exit(1)

    task = results[0].task_type
    title = TASK_TITLES.get(task, f"评测统计 ({task})")

    stats = compute_aggregations(results)
    print_summary(stats, title=title)

    if args.group_by:
        groups = group_by(results, args.group_by)
        print(f"\n  {'=' * 60}")
        print(f"  按 {args.group_by} 分组对比")
        print(f"  {'=' * 60}")
        for key, grp in groups.items():
            gs = compute_aggregations(grp)
            bar_max = 30
            print(f"\n  > {key}  ({gs['total_cases']} cases, 均分 {gs['avg_score_total']:.2f})")
            dims = gs.get("avg_dimension_scores", {})
            for dim, score in dims.items():
                label = DIM_LABELS.get(dim, dim)
                bar_len = int(score / 5 * bar_max)
                bar = "█" * bar_len + "░" * (bar_max - bar_len)
                print(f"      {label:<8} {score:.1f}  {bar}")
        print()


if __name__ == "__main__":
    main()
