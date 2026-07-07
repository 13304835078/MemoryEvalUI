from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.schema import (
    EVALUATABLE_TASK_TYPES,
    TASK_TYPE_LABELS,
    TaskType,
    cases_from_jsonl,
)
from src.extraction.memory_extractor import (
    EXTRACTION_OUTPUT_DIR,
    MemoryExtractionConfig,
    MemoryExtractionRunner,
    load_generation_prompt,
    sanitize_filename,
)
from src.loaders import ExcelLoader, JsonLoader, MdLoader
from src.ui.config_store import build_eval_config, load_config
from src.ui.data_service import (
    list_case_files,
    save_uploaded_file,
    save_cases,
    cases_to_dataframe,
    prepare_long_memory_cases_from_run_output,
    prepare_cases_from_run_output,
)
from src.ui.prompt_editor import (
    get_default_extraction_prompt_file,
    infer_prompt_version,
    list_extraction_prompt_files,
    load_prompt,
    prompt_text_hash,
)


def get_eval_task_choices() -> list[str]:
    return [task.value for task in EVALUATABLE_TASK_TYPES]


st.title("数据输入")

if "cases" not in st.session_state:
    st.session_state.cases = []

if "cases_file" not in st.session_state:
    st.session_state.cases_file = ""

if "task_type" not in st.session_state:
    st.session_state.task_type = "user_md_update"

if "ui_config" not in st.session_state:
    st.session_state.ui_config = load_config()


task_choices = get_eval_task_choices()
st.session_state.task_type = st.selectbox(
    "任务类型",
    task_choices,
    index=task_choices.index(st.session_state.task_type)
    if st.session_state.task_type in task_choices else 0,
    format_func=lambda value: TASK_TYPE_LABELS.get(value, value),
)

if st.session_state.task_type == TaskType.LONG_MEMORY.value:
    mode_options = [
        "选择已有样本文件",
        "上传长期记忆提取结果 Excel",
        "上传通用 Excel / JSON / JSONL",
        "读取 Markdown 样本目录",
    ]
    mode_help = "长期记忆任务只展示 MEMORY.md 相关输入方式；如需重新提取，请到「记忆提取」页选择长期记忆 MEMORY.md。"
else:
    mode_options = [
        "选择已有样本文件",
        "运行 USER.md 记忆提取",
        "上传 run_user.py 输出 Excel",
        "上传通用 Excel / JSON / JSONL",
        "读取 Markdown 样本目录",
    ]
    mode_help = "用户画像任务只展示 USER.md 相关输入方式。"

previous_mode = st.session_state.get("data_input_mode")
if previous_mode not in mode_options:
    st.session_state.data_input_mode = mode_options[0]

mode = st.radio(
    "输入方式",
    mode_options,
    key="data_input_mode",
    help=mode_help,
)

st.divider()

if mode == "选择已有样本文件":
    files = list_case_files()
    if not files:
        st.warning("data/cases/ 下暂无样本文件。")
    else:
        labels = [Path(f).name for f in files]
        selected_label = st.selectbox("选择样本文件", labels)
        selected_path = files[labels.index(selected_label)]

        if st.button("加载样本文件", use_container_width=True):
            cases = cases_from_jsonl(selected_path)
            st.session_state.cases = cases
            st.session_state.cases_file = selected_path
            if cases:
                st.session_state.task_type = cases[0].task_type.value
            st.success(f"已加载 {len(cases)} 条样本：{selected_path}")

elif mode == "运行 USER.md 记忆提取":
    st.info(
        "适用于原始对话 Excel：必须包含「轮次、query、answer、评测人」列。"
        "系统会先按「轮次 == 1」切 session，再按 chunk_size 分块调用提取模型。"
        "输出 Excel 与 run_user.py 结果兼容，可直接继续生成评测样本。"
    )

    cfg = dict(st.session_state.ui_config)

    uploaded = st.file_uploader("上传原始对话 Excel", type=["xlsx", "xls"], key="memory_extract_upload")
    local_excel_path = st.text_input(
        "或输入本地 Excel 路径",
        value="",
        placeholder=r"C:\Users\...\dialogues.xlsx",
        help="本地单人使用时可直接填写路径，绕过浏览器上传接口。",
        key="memory_extract_local_path",
    )
    sheet_name_raw = st.text_input("Sheet 名称或序号，可留空", value="", key="memory_extract_sheet")

    st.subheader("提取提示词")
    extraction_prompt_files = list_extraction_prompt_files()
    default_extraction_prompt = get_default_extraction_prompt_file("user_md_update")
    if default_extraction_prompt and default_extraction_prompt not in extraction_prompt_files:
        extraction_prompt_files = [default_extraction_prompt] + extraction_prompt_files
    prompt_options = ["使用配置页当前编辑文本"] + extraction_prompt_files
    selected_extraction_prompt = st.selectbox(
        "提取提示词来源",
        prompt_options,
        index=0,
        help="推荐先在配置页保存真实线上提取 prompt，再在这里选择对应版本。",
        key="memory_extract_prompt_source",
    )
    local_prompt_path = st.text_input(
        "或输入本地提取 prompt 路径（支持 .md/.yaml/.yml，可留空）",
        value="",
        placeholder=r"C:\Users\...\user_10.1.2.yaml",
        key="memory_extract_prompt_path",
    )

    try:
        if local_prompt_path.strip():
            extraction_prompt_text = load_generation_prompt(local_prompt_path.strip().strip('"'))
            extraction_prompt_version = Path(local_prompt_path.strip().strip('"')).stem
        elif selected_extraction_prompt == "使用配置页当前编辑文本":
            extraction_prompt_text = st.session_state.get("extraction_prompt_text", "")
            extraction_prompt_version = infer_prompt_version(
                st.session_state.get("selected_extraction_prompt_file", "") or default_extraction_prompt
            )
        else:
            extraction_prompt_text = load_prompt(selected_extraction_prompt, prompt_kind="extraction")
            extraction_prompt_version = infer_prompt_version(selected_extraction_prompt)
    except Exception as exc:
        extraction_prompt_text = ""
        extraction_prompt_version = ""
        st.error(f"提取提示词读取失败：{exc}")

    st.text_area(
        "提取提示词预览",
        value=extraction_prompt_text,
        height=180,
        disabled=True,
        key="memory_extract_prompt_preview",
    )
    st.caption(
        f"提取提示词版本：{extraction_prompt_version or '未识别'}；"
        f"Hash：{prompt_text_hash(extraction_prompt_text)[:12] if extraction_prompt_text else '空'}"
    )

    st.subheader("运行参数")
    c1, c2 = st.columns(2)
    with c1:
        extract_model = st.text_input(
            "提取模型名",
            value=cfg.get("judge_model", "") or "AGENT-GLM5-PERF",
            key="memory_extract_model",
        )
        chunk_size = st.number_input(
            "chunk_size",
            min_value=1,
            max_value=200,
            value=10,
            step=1,
            key="memory_extract_chunk_size",
        )
        reviewer_filter = st.text_input(
            "评测人筛选（可选，多个用逗号分隔）",
            value="",
            key="memory_extract_reviewer_filter",
        )
        max_tokens = st.number_input(
            "最大输出长度",
            min_value=1000,
            max_value=100000,
            value=50000,
            step=1000,
            key="memory_extract_max_tokens",
        )
    with c2:
        request_interval = st.number_input(
            "请求间隔秒数",
            min_value=0.0,
            max_value=120.0,
            value=float(cfg.get("judge_request_interval", 10.0) or 10.0),
            step=0.5,
            key="memory_extract_interval",
        )
        max_attempts = st.number_input(
            "最大尝试次数（含首次）",
            min_value=1,
            max_value=11,
            value=max(1, int(cfg.get("judge_max_retries", 3) or 3)),
            step=1,
            help="例如设置为 3 表示最多请求 3 次：首次 1 次，失败后最多再尝试 2 次。",
            key="memory_extract_retries",
        )
        retry_sleep = st.number_input(
            "失败后重试等待秒数",
            min_value=0.0,
            max_value=180.0,
            value=float(cfg.get("judge_qps_backoff", 15.0) or 15.0),
            step=1.0,
            key="memory_extract_retry_sleep",
        )
        timeout = st.number_input(
            "单次请求超时秒数",
            min_value=10,
            max_value=600,
            value=int(cfg.get("judge_timeout", 120) or 120),
            step=10,
            key="memory_extract_timeout",
        )
        extraction_concurrency = st.number_input(
            "提取并发数",
            min_value=1,
            max_value=100,
            value=min(100, max(1, int(cfg.get("judge_concurrency", 1) or 1))),
            step=1,
            help="不同评测人之间可并发；同一评测人内部仍串行，避免 USER.md 继承关系错乱。",
            key="memory_extract_concurrency",
        )

    think_cols = st.columns(2)
    with think_cols[0]:
        send_enable_thinking = st.checkbox(
            "发送 enable_thinking 字段",
            value=True,
            key="memory_extract_send_thinking",
        )
    with think_cols[1]:
        enable_thinking = st.checkbox(
            "enable_thinking=true",
            value=True,
            disabled=not send_enable_thinking,
            key="memory_extract_thinking",
        )

    auto_make_cases = st.checkbox("提取完成后自动生成评测样本", value=True, key="memory_extract_auto_cases")
    model_name_for_case = st.text_input("生成 case 使用的模型名", value=extract_model, key="memory_extract_case_model")
    prompt_version_for_case = st.text_input(
        "生成 case 使用的提示词版本",
        value=extraction_prompt_version or "unknown",
        key="memory_extract_case_prompt_version",
    )

    input_ready = uploaded is not None or bool(local_excel_path.strip())
    if input_ready and st.button("开始运行 USER.md 记忆提取", type="primary", use_container_width=True):
        try:
            if not extraction_prompt_text.strip():
                raise ValueError("提取提示词为空，请先选择或填写提取 prompt。")
            if uploaded is not None:
                input_path = save_uploaded_file(uploaded, suffix=Path(uploaded.name).suffix)
            else:
                input_path = local_excel_path.strip().strip('"')
                if not Path(input_path).is_file():
                    raise FileNotFoundError(f"本地文件不存在：{input_path}")

            sheet_arg: str | int | None = 0
            if sheet_name_raw.strip():
                sheet_arg = int(sheet_name_raw) if sheet_name_raw.strip().isdigit() else sheet_name_raw.strip()

            eval_config = build_eval_config(cfg, mock=False)
            errs = eval_config.validate()
            if errs:
                raise ValueError("接口配置错误：\n" + "\n".join([f"- {e}" for e in errs]))

            extraction_config = MemoryExtractionConfig.from_eval_config(
                eval_config,
                model=extract_model,
                max_tokens=int(max_tokens),
                request_interval=float(request_interval),
                max_retries=max(0, int(max_attempts) - 1),
                retry_sleep=float(retry_sleep),
                enable_thinking=bool(enable_thinking),
                timeout=int(timeout),
            )
            extraction_config.send_enable_thinking = bool(send_enable_thinking)
            extraction_config.concurrency = int(extraction_concurrency)

            EXTRACTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output_name = (
                f"memory_extract_{sanitize_filename(extract_model)}_"
                f"{sanitize_filename(extraction_prompt_version or 'prompt')}_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            )
            output_path = EXTRACTION_OUTPUT_DIR / output_name

            progress = st.progress(0.0)
            status_box = st.empty()

            def on_progress(done: int, total: int, message: str) -> None:
                progress.progress(done / total if total else 0.0)
                status_box.write(message)

            runner = MemoryExtractionRunner(
                config=extraction_config,
                prompt_text=extraction_prompt_text,
            )
            stats = runner.process_excel(
                input_path,
                output_path,
                sheet_name=sheet_arg,
                reviewer_filter=reviewer_filter.strip() or None,
                chunk_size=int(chunk_size),
                progress_callback=on_progress,
            )
            progress.progress(1.0)
            st.session_state.memory_extraction_output_path = str(output_path)
            st.session_state.memory_extraction_stats = stats
            st.success(f"记忆提取完成：{output_path}")
            st.write({
                "sessions": stats.get("sessions"),
                "chunks": stats.get("chunks"),
                "api_calls": stats.get("api_calls"),
                "concurrency": stats.get("concurrency"),
                "status_counts": stats.get("status_counts"),
            })

            output_df = pd.read_excel(output_path).fillna("")
            st.dataframe(output_df.head(50), use_container_width=True, hide_index=True)
            st.download_button(
                "下载提取结果 Excel",
                data=output_path.read_bytes(),
                file_name=output_path.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

            if auto_make_cases:
                cases, missed_cases, convert_stats = prepare_cases_from_run_output(
                    output_path,
                    model=model_name_for_case,
                    prompt_version=prompt_version_for_case,
                    chunk_size=int(chunk_size),
                    return_missed=True,
                )
                out_name = (
                    f"{sanitize_filename(model_name_for_case)}_"
                    f"{sanitize_filename(prompt_version_for_case)}_user_md_cases_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
                )
                out_path = save_cases(cases, out_name)
                missed_out_path = ""
                if missed_cases:
                    missed_out_name = (
                        f"{sanitize_filename(model_name_for_case)}_"
                        f"{sanitize_filename(prompt_version_for_case)}_user_md_missed_cases_"
                        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
                    )
                    missed_out_path = save_cases(missed_cases, missed_out_name)

                st.session_state.cases = cases
                st.session_state.cases_file = out_path
                st.session_state.missed_cases = missed_cases
                st.session_state.missed_cases_file = missed_out_path
                st.session_state.case_convert_stats = convert_stats
                st.success(f"已自动生成评测样本：{len(cases)} 条 → {out_path}")
                if missed_out_path:
                    st.warning(f"发现 {len(missed_cases)} 个漏抽分块，已另存为：{missed_out_path}")
        except Exception as e:
            st.error(f"记忆提取失败：{e}")

elif mode == "上传 run_user.py 输出 Excel":
    st.info("适用于包含会话ID、轮次、用户问题、助手回答、评测人、user.md、result、reasoning 列的 run_user.py 输出 Excel。")

    uploaded = st.file_uploader("上传 Excel", type=["xlsx", "xls"])
    local_excel_path = st.text_input(
        "或输入本地 Excel 路径",
        value="",
        placeholder=r"C:\Users\...\xxx.xlsx",
        help="本地单人使用时可直接填写路径，绕过浏览器上传接口。",
    )
    chunk_size = st.number_input(
        "分块大小",
        min_value=1,
        max_value=200,
        value=10,
        step=1,
        help="按轮次==1切会话后，每个会话内按此大小分块；最后不足分块大小也会单独处理。",
    )
    model_name = st.text_input("被评测模型名", value="unknown")
    prompt_version = st.text_input("生成提示词版本", value="unknown")

    input_ready = uploaded is not None or bool(local_excel_path.strip())
    if input_ready and st.button("转换为用户画像评测样本", use_container_width=True):
        try:
            if uploaded is not None:
                path = save_uploaded_file(uploaded, suffix=Path(uploaded.name).suffix)
            else:
                path = local_excel_path.strip().strip('"')
                if not Path(path).is_file():
                    raise FileNotFoundError(f"本地文件不存在：{path}")

            cases, missed_cases, convert_stats = prepare_cases_from_run_output(
                path,
                model=model_name,
                prompt_version=prompt_version,
                chunk_size=int(chunk_size),
                return_missed=True,
            )
            out_name = f"{model_name}_{prompt_version}_user_md_cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            out_path = save_cases(cases, out_name)
            missed_out_path = ""
            if missed_cases:
                missed_out_name = f"{model_name}_{prompt_version}_user_md_missed_cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
                missed_out_path = save_cases(missed_cases, missed_out_name)

            st.session_state.cases = cases
            st.session_state.cases_file = out_path
            st.session_state.missed_cases = missed_cases
            st.session_state.missed_cases_file = missed_out_path
            st.session_state.case_convert_stats = convert_stats
            st.success(f"转换完成：{len(cases)} 条完整样本 → {out_path}")
            if missed_out_path:
                st.warning(f"发现 {len(missed_cases)} 个漏抽分块，已另存为：{missed_out_path}")
            st.caption(
                f"总分块：{convert_stats.get('total_chunks', 0)} | "
                f"完整样本：{convert_stats.get('generated_cases', 0)} | "
                f"漏抽样本：{convert_stats.get('missed_cases', 0)}"
            )
            skipped = convert_stats.get("skipped_chunk_details", [])
            if skipped:
                with st.expander("查看漏抽分块", expanded=True):
                    st.dataframe(pd.DataFrame(skipped), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"转换失败：{e}")

elif mode == "上传长期记忆提取结果 Excel":
    st.info(
        "适用于长期记忆提取程序输出。系统按「轮次 == 1」切 session、按分块大小读取末行结果，"
        "并兼容 `MEMORY.md` 与 `生成的MEMORY.md正文` 两种结果列名。评测人变化时，旧 MEMORY.md 从空开始。"
    )

    uploaded = st.file_uploader(
        "上传长期记忆结果 Excel",
        type=["xlsx", "xls"],
        key="long_memory_result_upload",
    )
    local_excel_path = st.text_input(
        "或输入本地 Excel 路径",
        value="",
        placeholder=r"C:\Users\...\memory_result.xlsx",
        key="long_memory_result_path",
    )
    chunk_size = st.number_input(
        "分块大小",
        min_value=1,
        max_value=200,
        value=10,
        step=1,
        help="必须与生成该 Excel 时使用的 chunk_size 一致；当前提取程序默认为 10。",
        key="long_memory_chunk_size",
    )
    model_name = st.text_input(
        "被评测模型名",
        value="unknown",
        key="long_memory_model_name",
    )
    prompt_version = st.text_input(
        "长期记忆提取提示词版本",
        value=infer_prompt_version(get_default_extraction_prompt_file(TaskType.LONG_MEMORY.value)),
        key="long_memory_prompt_version",
    )

    input_ready = uploaded is not None or bool(local_excel_path.strip())
    if input_ready and st.button(
        "转换为长期记忆评测样本",
        type="primary",
        use_container_width=True,
        key="convert_long_memory_cases",
    ):
        try:
            if uploaded is not None:
                path = save_uploaded_file(uploaded, suffix=Path(uploaded.name).suffix)
            else:
                path = local_excel_path.strip().strip('"')
                if not Path(path).is_file():
                    raise FileNotFoundError(f"本地文件不存在：{path}")

            cases, missed_cases, convert_stats = prepare_long_memory_cases_from_run_output(
                path,
                model=model_name,
                prompt_version=prompt_version,
                chunk_size=int(chunk_size),
                return_missed=True,
            )
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_name = (
                f"{sanitize_filename(model_name)}_{sanitize_filename(prompt_version)}_"
                f"long_memory_cases_{timestamp}.jsonl"
            )
            out_path = save_cases(cases, out_name)

            missed_out_path = ""
            if missed_cases:
                missed_out_name = (
                    f"{sanitize_filename(model_name)}_{sanitize_filename(prompt_version)}_"
                    f"long_memory_missed_cases_{timestamp}.jsonl"
                )
                missed_out_path = save_cases(missed_cases, missed_out_name)

            st.session_state.task_type = TaskType.LONG_MEMORY.value
            st.session_state.cases = cases
            st.session_state.cases_file = out_path
            st.session_state.missed_cases = missed_cases
            st.session_state.missed_cases_file = missed_out_path
            st.session_state.case_convert_stats = convert_stats

            st.success(f"转换完成：{len(cases)} 条长期记忆样本 → {out_path}")
            st.caption(
                f"总分块：{convert_stats.get('total_chunks', 0)} | "
                f"完整样本：{convert_stats.get('generated_cases', 0)} | "
                f"漏抽样本：{convert_stats.get('missed_cases', 0)}"
            )
            if missed_out_path:
                st.warning(f"发现 {len(missed_cases)} 个漏抽分块，已另存为：{missed_out_path}")
        except Exception as e:
            st.error(f"长期记忆结果转换失败：{e}")

elif mode == "上传通用 Excel / JSON / JSONL":
    uploaded = st.file_uploader("上传文件", type=["xlsx", "xls", "json", "jsonl"])
    sheet_name = st.text_input("Excel sheet 名称，可留空", value="")

    if uploaded and st.button("加载并转换", use_container_width=True):
        path = save_uploaded_file(uploaded, suffix=Path(uploaded.name).suffix)
        ext = Path(path).suffix.lower()
        task_type = TaskType(st.session_state.task_type)

        try:
            if ext in {".xlsx", ".xls"}:
                loader = ExcelLoader(task_type=task_type, sheet_name=sheet_name)
            elif ext in {".json", ".jsonl"}:
                loader = JsonLoader(task_type=task_type)
            else:
                raise ValueError(f"不支持的文件格式：{ext}")

            cases = loader.load(path)
            out_name = f"cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            out_path = save_cases(cases, out_name)

            st.session_state.cases = cases
            st.session_state.cases_file = out_path
            st.success(f"转换完成：{len(cases)} 条样本 → {out_path}")
        except Exception as e:
            st.error(f"加载失败：{e}")

elif mode == "读取 Markdown 样本目录":
    document_name = "MEMORY.md" if st.session_state.task_type == TaskType.LONG_MEMORY.value else "USER.md"
    st.info(
        f"读取 Markdown 样本目录会生成 {document_name} 评测样本。"
        "目录需包含 old_user.md / dialogue.md / new_user.md。"
        "长期记忆任务也复用这三个文件名，内容请放 MEMORY.md。"
    )
    dir_path = st.text_input("Markdown 样本目录路径")

    if dir_path and st.button("读取 Markdown 样本", use_container_width=True):
        try:
            loader = MdLoader(task_type=TaskType(st.session_state.task_type))
            cases = loader.load(dir_path)
            out_name = f"md_cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            out_path = save_cases(cases, out_name)

            st.session_state.cases = cases
            st.session_state.cases_file = out_path
            st.success(f"读取完成：{len(cases)} 条样本 → {out_path}")
        except Exception as e:
            st.error(f"读取失败：{e}")


st.divider()

st.subheader("当前样本预览")

cases = st.session_state.get("cases", [])
if cases:
    st.caption(f"当前样本文件：{st.session_state.get('cases_file', '')}")
    st.caption(f"共 {len(cases)} 条")
    convert_stats = st.session_state.get("case_convert_stats")
    if convert_stats:
        st.caption(
            f"总分块：{convert_stats.get('total_chunks', 0)} | "
            f"完整样本：{convert_stats.get('generated_cases', 0)} | "
            f"漏抽样本：{convert_stats.get('missed_cases', 0)}"
        )
    st.dataframe(cases_to_dataframe(cases).head(50), use_container_width=True, hide_index=True)

    missed_cases = st.session_state.get("missed_cases", [])
    if missed_cases:
        st.subheader("漏抽样本预览")
        st.caption(f"漏抽样本文件：{st.session_state.get('missed_cases_file', '')}")
        st.dataframe(cases_to_dataframe(missed_cases).head(50), use_container_width=True, hide_index=True)
else:
    st.info("尚未加载样本。")
