from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from statistics import median
from threading import Lock

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.eval_runner import EvalRunner
from src.schema import EvalResult, TaskType
from src.ui.config_store import build_eval_config, load_config
from src.ui.data_service import dataframe_to_excel_bytes, save_uploaded_file
from src.ui.human_review_eval import (
    HUMAN_REVIEW_CACHE_PATH,
    append_human_review_result_row,
    build_pair_row,
    eval_config_fingerprint,
    get_human_review_run_path,
    list_human_review_run_files,
    load_human_review_cache,
    load_human_review_result_rows,
    low_confidence_rows,
    make_human_review_run_id,
    make_human_review_pairs_with_stats,
    pair_cache_key,
    read_human_review_excel,
    save_human_review_cache,
    stable_hash,
    summarize_pair_rows,
)
from src.ui.prompt_editor import infer_prompt_version, list_prompt_files, load_prompt


st.title("人工审核结果评估")
st.caption("上传人工审核表，只评估有有效 G/S/B 标注的行；系统会分别给两个模型的用户画像打分，再按分差自动判定 G/S/B。")

if "ui_config" not in st.session_state:
    st.session_state.ui_config = load_config()
if "human_review_pairs" not in st.session_state:
    st.session_state.human_review_pairs = []
if "human_review_rows" not in st.session_state:
    st.session_state.human_review_rows = []
if "human_review_cache" not in st.session_state:
    st.session_state.human_review_cache = {}


def display_table(df: pd.DataFrame, **kwargs) -> None:
    rename = {
        "pair_id": "配对编号",
        "row_number": "Excel行号",
        "round": "轮次",
        "skip_reason": "跳过原因",
        "raw_gsb": "原始GSB",
        "has_model1_output": "模型1有结果",
        "has_model2_output": "模型2有结果",
        "from_cache": "来自缓存",
        "cache_key": "缓存键",
        "run_id": "运行编号",
        "judge_prompt_hash": "裁判提示词指纹",
        "config_hash": "配置指纹",
        "judge_prompt_version": "裁判提示词版本",
        "score_diff_model1_minus_model2": "模型1减模型2分差",
        "query": "用户问题",
        "answer": "助手回答",
        "模型1_score_total": "模型1总分",
        "模型2_score_total": "模型2总分",
        "模型1_score_correctness": "模型1正确性",
        "模型2_score_correctness": "模型2正确性",
        "模型1_score_completeness": "模型1完整性",
        "模型2_score_completeness": "模型2完整性",
        "模型1_score_conciseness": "模型1简洁性",
        "模型2_score_conciseness": "模型2简洁性",
        "模型1_score_instruction_following": "模型1指令遵循",
        "模型2_score_instruction_following": "模型2指令遵循",
    }
    st.dataframe(df.rename(columns=rename), use_container_width=True, hide_index=True, **kwargs)


def build_gsb_confusion_matrix(df: pd.DataFrame) -> pd.DataFrame:
    labels = ["G", "S", "B"]
    matrix = pd.crosstab(df["人工GSB"], df["自动GSB"], dropna=False)
    matrix = matrix.reindex(index=labels, columns=labels, fill_value=0)
    matrix.index.name = "人工GSB"
    matrix.columns.name = "自动GSB"
    matrix["合计"] = matrix.sum(axis=1)
    total_row = pd.DataFrame([matrix.sum(axis=0)], index=["合计"])
    return pd.concat([matrix, total_row])


def highlight_confusion_diagonal(data: pd.DataFrame) -> pd.DataFrame:
    styles = pd.DataFrame("", index=data.index, columns=data.columns)
    for label in ("G", "S", "B"):
        if label in styles.index and label in styles.columns:
            styles.loc[label, label] = "background-color: #d1fae5; font-weight: 700"
    if "合计" in styles.index:
        styles.loc["合计", :] = "background-color: #f3f4f6; font-weight: 700"
    if "合计" in styles.columns:
        styles.loc[:, "合计"] = "background-color: #f3f4f6; font-weight: 700"
    return styles


st.info("使用步骤：1. 上传 Excel 或填写本地路径；2. 确认模型列名；3. 读取有效 G/S/B 行；4. 开始评估；5. 查看一致率和低置信复核队列。")

st.subheader("数据")
uploaded = st.file_uploader("上传人工审核结果表（Excel）", type=["xlsx", "xls"])
local_excel_path = st.text_input(
    "或输入本地 Excel 路径",
    value="",
    placeholder=r"C:\Users\...\人工审核结果.xlsx",
)
sheet_name = st.text_input("Sheet 名称或序号，可留空", value="")

with st.expander("模型列设置", expanded=True):
    col_a, col_b = st.columns(2)
    with col_a:
        model1_name = st.text_input("模型1名称（G 表示模型1更好）", value="glm5-think")
        model1_column = st.text_input("模型1用户画像列名", value="user.md-glm5-think")
    with col_b:
        model2_name = st.text_input("模型2名称（B 表示模型2更好）", value="ds-10.1.2")
        model2_column = st.text_input("模型2用户画像列名", value="user.md-ds-10.1.2")

with st.expander("高级数据设置", expanded=False):
    prompt_version_for_cases = st.text_input("生成样本的版本标记", value="human_review")

st.info(
    "GSB 采用模型1视角：G=模型1更好，S=基本持平，B=模型2更好。"
    "为降低随机性，推荐分别评分后用固定分差阈值判定；不要让裁判模型直接二选一。"
)

input_ready = uploaded is not None or bool(local_excel_path.strip())
if input_ready and st.button("读取有效 G/S/B 行", use_container_width=True):
    try:
        if uploaded is not None:
            excel_path = save_uploaded_file(uploaded, suffix=Path(uploaded.name).suffix)
        else:
            excel_path = local_excel_path.strip().strip('"')
            if not Path(excel_path).is_file():
                raise FileNotFoundError(f"本地文件不存在：{excel_path}")

        sheet_arg: str | int | None = 0
        if sheet_name.strip():
            sheet_arg = int(sheet_name) if sheet_name.strip().isdigit() else sheet_name.strip()

        df = read_human_review_excel(excel_path, sheet_name=sheet_arg)
        pairs, skipped_rows = make_human_review_pairs_with_stats(
            df,
            model1_column=model1_column,
            model2_column=model2_column,
            model1_name=model1_name,
            model2_name=model2_name,
            prompt_version=prompt_version_for_cases,
            require_gsb=True,
        )
        st.session_state.human_review_pairs = pairs
        st.session_state.human_review_skipped_rows = skipped_rows
        st.session_state.human_review_rows = []
        st.session_state.human_review_source = excel_path
        st.success(f"已读取 {len(pairs)} 行有效 G/S/B 配对样本：{excel_path}")
        if skipped_rows:
            st.warning(f"已跳过 {len(skipped_rows)} 行无 G/S/B、无效 G/S/B 或空行。")
    except Exception as e:
        st.error(f"读取失败：{e}")

pairs = st.session_state.get("human_review_pairs", [])
if pairs:
    skipped_rows = st.session_state.get("human_review_skipped_rows", [])
    if skipped_rows:
        with st.expander("查看跳过的行", expanded=False):
            display_table(pd.DataFrame(skipped_rows))
    preview_df = pd.DataFrame([
        {
            "pair_id": p.pair_id,
            "row_number": p.row_number,
            "轮次": p.round_value,
            "评测人": p.reviewer,
            "人工GSB": p.human_gsb,
            "问题类型": p.issue_type,
            f"{p.model1_name}预览": p.model1_output[:80],
            f"{p.model2_name}预览": p.model2_output[:80],
        }
        for p in pairs
    ])
    st.markdown("**已读取样本预览**")
    display_table(preview_df.head(50))

st.divider()
st.subheader("评估")

cfg = dict(st.session_state.ui_config)
prompt_files = list_prompt_files()
default_human_prompt = "judge_user_md_human_aligned_v1.md"
selected_prompt_file = st.selectbox(
    "裁判提示词文件",
    prompt_files,
    index=prompt_files.index(default_human_prompt)
    if default_human_prompt in prompt_files
    else (prompt_files.index("judge_user_md_v1.md") if "judge_user_md_v1.md" in prompt_files else 0),
)
judge_prompt_version = infer_prompt_version(selected_prompt_file)
limit = st.number_input("评估行数（0 表示全部）", min_value=0, value=0, step=1)

with st.expander("高级评估设置", expanded=False):
    mock = st.checkbox("模拟模式（Mock）", value=bool(cfg.get("mock", True)))
    margin = st.number_input(
        "判定为胜负的分差阈值",
        min_value=0.0,
        max_value=2.0,
        value=0.25,
        step=0.05,
        help="模型1总分 - 模型2总分大于该阈值判 G，小于负阈值判 B，否则判 S。",
    )
    repeat_count = st.selectbox(
        "每个样本重复评测次数",
        [1, 3, 5],
        index=0,
        help="重复评测会取总分中位数对应的结果，能降低随机性，但会按倍数增加调用成本。",
    )
    use_cache = st.checkbox("复用页面临时缓存", value=True)
    use_disk_cache = st.checkbox("复用磁盘缓存", value=True)
    max_workers = st.number_input(
        "同时评估的行数",
        min_value=1,
        max_value=100,
        value=min(100, max(1, int(cfg.get("judge_concurrency", 2) or 2))),
        step=1,
        help="每行包含两个模型评分。真实调用会按配置页的请求间隔做全局排队；提高并发主要有利于缓存命中或无请求间隔场景。",
    )
    st.caption(f"磁盘缓存：{HUMAN_REVIEW_CACHE_PATH}")
    if st.button("清空人工审核评估缓存", use_container_width=True):
        st.session_state.human_review_cache = {}
        if HUMAN_REVIEW_CACHE_PATH.exists():
            HUMAN_REVIEW_CACHE_PATH.unlink()
        st.success("已清空缓存")

with st.expander("历史结果", expanded=False):
    run_files = list_human_review_run_files()
    if run_files:
        run_labels = [Path(f).name for f in run_files]
        selected_run_label = st.selectbox("加载历史评估结果（可选）", ["不加载"] + run_labels)
        if selected_run_label != "不加载" and st.button("加载历史结果", use_container_width=True):
            selected_run_path = run_files[run_labels.index(selected_run_label)]
            st.session_state.human_review_rows = load_human_review_result_rows(selected_run_path)
            st.session_state.human_review_run_path = selected_run_path
            st.success(f"已加载历史结果：{selected_run_path}")
    else:
        st.info("暂无历史评估结果。")

with st.expander("当前接口配置", expanded=False):
    st.write({
        "模拟模式": mock,
        "接口地址": cfg.get("api_base", ""),
        "裁判模型": cfg.get("judge_model", ""),
        "最大输出长度": cfg.get("judge_max_tokens", 2000),
        "超时秒数": cfg.get("judge_timeout", 120),
        "最大重试次数": cfg.get("judge_max_retries", 3),
        "请求间隔": cfg.get("judge_request_interval", 0),
        "限流重试等待": cfg.get("judge_qps_backoff", 12),
    })


def evaluate_stable(runner: EvalRunner, case, repeats: int, before_call=None) -> EvalResult:
    results = []
    for _ in range(int(repeats)):
        if before_call is not None:
            before_call()
        results.append(runner.evaluate_one(case))
    if len(results) == 1:
        return results[0]
    mid = median([r.score_total for r in results])
    return sorted(results, key=lambda r: abs(r.score_total - mid))[0]


if st.button("开始评估", type="primary", use_container_width=True, disabled=not bool(pairs)):
    run_pairs = pairs[: int(limit)] if limit and limit > 0 else pairs
    config = build_eval_config(cfg, mock=mock)
    errs = config.validate()
    if errs:
        st.error("配置错误：\n" + "\n".join([f"- {e}" for e in errs]))
        st.stop()

    runner = EvalRunner(
        config=config,
        task_type=TaskType.USER_MD,
        prompt_file=selected_prompt_file,
        judge_prompt_version=judge_prompt_version,
    )

    interval = float(config.judge_request_interval or 0.0) if not mock else 0.0
    if int(max_workers) > 1 and not mock:
        interval = max(interval, float(config.judge_qps_backoff or 0.0))
    rate_lock = Lock()
    next_call_at = {"value": time.monotonic()}

    def wait_for_rate_limit() -> None:
        if interval <= 0:
            return
        with rate_lock:
            now = time.monotonic()
            wait_seconds = max(0.0, next_call_at["value"] - now)
            next_call_at["value"] = max(now, next_call_at["value"]) + interval
        if wait_seconds > 0:
            time.sleep(wait_seconds)

    if hasattr(runner.judge_client, "rate_limit_wait_callback"):
        runner.judge_client.rate_limit_wait_callback = wait_for_rate_limit

    judge_model_key = config.judge_model or "mock"
    judge_prompt_text = load_prompt(selected_prompt_file)
    judge_prompt_hash = stable_hash(judge_prompt_text)
    config_hash = eval_config_fingerprint(config)
    run_id = make_human_review_run_id()
    run_path = get_human_review_run_path(run_id)
    cache = {}
    if use_disk_cache:
        cache.update(load_human_review_cache())
    if use_cache:
        cache.update(st.session_state.human_review_cache)

    progress = st.progress(0.0)
    status = st.empty()
    rows_by_index = {}
    cache_hits = 0
    pending = []
    for idx, pair in enumerate(run_pairs):
        key = pair_cache_key(
            pair,
            judge_model_key,
            judge_prompt_version,
            int(repeat_count),
            judge_prompt_hash=judge_prompt_hash,
            config_hash=config_hash,
        )
        if (use_cache or use_disk_cache) and key in cache:
            result1, result2 = cache[key]
            row = build_pair_row(
                pair,
                result1,
                result2,
                margin,
                from_cache=True,
                cache_key=key,
                run_id=run_id,
                judge_prompt_hash=judge_prompt_hash,
                config_hash=config_hash,
                judge_prompt_version=judge_prompt_version,
            )
            rows_by_index[idx] = row
            append_human_review_result_row(run_path, row)
            cache_hits += 1
        else:
            pending.append((idx, key, pair))

    completed = cache_hits
    progress.progress(completed / len(run_pairs) if run_pairs else 0)
    status.write(f"缓存命中 {cache_hits} 行，待评估 {len(pending)} 行")

    def evaluate_pair(item):
        idx, key, pair = item
        result1 = evaluate_stable(runner, pair.case_model1, repeat_count, wait_for_rate_limit)
        result2 = evaluate_stable(runner, pair.case_model2, repeat_count, wait_for_rate_limit)
        return idx, key, pair, result1, result2

    if pending:
        with ThreadPoolExecutor(max_workers=int(max_workers)) as executor:
            future_map = {executor.submit(evaluate_pair, item): item for item in pending}
            for future in as_completed(future_map):
                idx, key, pair, result1, result2 = future.result()
                cache[key] = (result1, result2)
                row = build_pair_row(
                    pair,
                    result1,
                    result2,
                    margin,
                    from_cache=False,
                    cache_key=key,
                    run_id=run_id,
                    judge_prompt_hash=judge_prompt_hash,
                    config_hash=config_hash,
                    judge_prompt_version=judge_prompt_version,
                )
                rows_by_index[idx] = row
                append_human_review_result_row(run_path, row)
                completed += 1
                progress.progress(completed / len(run_pairs) if run_pairs else 0)
                status.write(
                    f"已完成 {completed}/{len(run_pairs)} 行；"
                    f"缓存命中 {cache_hits} 行；当前 Excel 行={pair.row_number}"
                )

    rows = [rows_by_index[i] for i in sorted(rows_by_index)]

    st.session_state.human_review_rows = rows
    st.session_state.human_review_run_path = str(run_path)
    st.session_state.human_review_cache = cache
    if use_disk_cache:
        save_human_review_cache(cache)
    st.success(f"评估完成：{len(rows)} 行，缓存命中 {cache_hits} 行，新评估 {len(pending)} 行")
    st.caption(f"本次 run 已边跑边保存：{run_path}")

rows = st.session_state.get("human_review_rows", [])
if rows:
    st.divider()
    st.subheader("评估结果")

    summary = summarize_pair_rows(rows)
    low_rows = low_confidence_rows(rows, margin)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("总行数", summary["total"])
    c2.metric("可对比", summary["comparable"])
    c3.metric("一致率", f"{summary['agreement_rate'] * 100:.1f}%")
    c4.metric("低置信/待复核", len(low_rows))

    run_path_text = st.session_state.get("human_review_run_path", "")
    if run_path_text:
        st.caption(f"当前 run 文件：{run_path_text}")

    result_df = pd.DataFrame(rows)
    left, right = st.columns(2)
    with left:
        st.markdown("**人工 GSB 分布**")
        display_table(result_df["人工GSB"].value_counts(dropna=False).rename_axis("GSB").reset_index(name="数量"))
    with right:
        st.markdown("**自动 GSB 分布**")
        display_table(result_df["自动GSB"].value_counts(dropna=False).rename_axis("GSB").reset_index(name="数量"))

    st.markdown("**混淆矩阵（行=人工GSB，列=自动GSB）**")
    confusion_df = build_gsb_confusion_matrix(result_df)
    st.dataframe(
        confusion_df.style.apply(highlight_confusion_diagonal, axis=None),
        use_container_width=True,
    )

    if "问题类型" in result_df.columns:
        issue_df = (
            result_df.groupby(["问题类型", "是否一致"], dropna=False)
            .size()
            .reset_index(name="数量")
        )
        st.markdown("**按问题类型统计一致情况**")
        display_table(issue_df)

    st.markdown("**详细结果**")
    display_table(result_df)

    st.subheader("低置信复核队列")
    st.caption("包含自动/人工不一致、分差接近阈值、或裁判模型备注中出现不确定表达的样本。")
    low_df = pd.DataFrame(low_rows)
    if low_df.empty:
        st.info("暂无低置信样本。")
    else:
        display_table(low_df)
        st.download_button(
            "下载低置信复核表",
            data=dataframe_to_excel_bytes(low_df),
            file_name=f"human_review_low_confidence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    excel_bytes = dataframe_to_excel_bytes(result_df)
    st.download_button(
        "下载完整评估结果",
        data=excel_bytes,
        file_name=f"human_review_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    mismatches = result_df[result_df["是否一致"] == False]
    st.subheader("不一致样本详情")
    st.caption(f"共 {len(mismatches)} 条不一致")
    for _, row in mismatches.iterrows():
        with st.expander(f"Excel行 {row['row_number']} | 人工 {row['人工GSB']} / 自动 {row['自动GSB']} | {row['问题类型']}", expanded=False):
            st.write(row["自动判断备注"])
            st.markdown("**用户问题**")
            st.write(row["query"])
            st.markdown("**助手回答**")
            st.write(row["answer"])
            st.markdown(f"**{model1_name} 用户画像（USER.md）**")
            st.code(row.get(f"{model1_name}_user.md", ""), language="markdown")
            st.markdown(f"**{model2_name} 用户画像（USER.md）**")
            st.code(row.get(f"{model2_name}_user.md", ""), language="markdown")
            raw1 = str(row.get(f"{model1_name}_raw_response", "") or "")
            raw2 = str(row.get(f"{model2_name}_raw_response", "") or "")
            if raw1 or raw2:
                with st.expander("裁判模型原始响应", expanded=False):
                    if raw1:
                        st.markdown(f"**{model1_name} 原始响应**")
                        st.code(raw1, language="json")
                    if raw2:
                        st.markdown(f"**{model2_name} 原始响应**")
                        st.code(raw2, language="json")
            st.markdown("**人工备注**")
            st.write(row.get("备注", ""))
