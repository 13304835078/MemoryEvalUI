"""将 run_user.py 输出 Excel 转换为标准评测 Case

切分逻辑与 run_user.py 保持一致：
  - 先按「轮次 == 1」切 session
  - 每个 session 内按 chunk_size 分 chunk
  - 最后不足 chunk_size 的尾段也作为一个 chunk
  - 每个 chunk 的结果只读取当前 chunk 最后一行的 user.md/result/reasoning
  - 如果 chunk 最后一行没有 user.md/result/reasoning，视为上游漏抽，跳过该 chunk
  - old_memory 按评测人跨 session 继承上一条 user.md

用法:
    python scripts/prepare_eval_from_run_output.py \
        --input ../output/eval_results_user_AGENT-GLM5-PERF.xlsx \
        --output data/cases/glm5_user_md_cases.jsonl \
        --model GLM5 --prompt_version user_10.2 --chunk_size 10
"""

import os
import sys
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.schema import cases_to_jsonl
from src.ui.data_service import prepare_cases_from_run_output


def main():
    parser = argparse.ArgumentParser(description="将 run_user.py 输出转换为标准评测 Case")
    parser.add_argument("--input", type=str, required=True, help="run_user.py 输出 Excel 路径")
    parser.add_argument("--output", type=str, required=True, help="输出 JSONL 路径")
    parser.add_argument("--missed-output", type=str, default="", help="漏抽 case JSONL 输出路径，可选")
    parser.add_argument("--model", type=str, default="unknown", help="模型名称")
    parser.add_argument("--prompt_version", type=str, default="unknown", help="Prompt 版本")
    parser.add_argument("--chunk_size", type=int, default=10, help="run_user.py 提取 USER.md 的 chunk size")
    args = parser.parse_args()

    cases, missed_cases, stats = prepare_cases_from_run_output(
        args.input,
        model=args.model,
        prompt_version=args.prompt_version,
        chunk_size=args.chunk_size,
        return_missed=True,
    )
    print(
        f"总 chunk: {stats['total_chunks']}，"
        f"生成 case: {stats['generated_cases']}，"
        f"漏抽 case: {stats['missed_cases']}"
    )
    for item in stats.get("skipped_chunk_details", []):
        print(
            "  [SKIP] "
            f"session={item.get('source_session_id')} "
            f"rows={item.get('row_start')}-{item.get('row_end')} "
            f"reviewer={item.get('reviewer')} "
            f"reason={item.get('skip_reason')}"
        )

    cases_to_jsonl(cases, args.output)
    print(f"完成: {len(cases)} 条 case → {args.output}")
    if missed_cases:
        missed_output = args.missed_output
        if not missed_output:
            root, ext = os.path.splitext(args.output)
            missed_output = f"{root}_missed{ext or '.jsonl'}"
        cases_to_jsonl(missed_cases, missed_output)
        print(f"漏抽 case: {len(missed_cases)} 条 → {missed_output}")


if __name__ == "__main__":
    main()
