from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.user_identity import require_page_identity
require_page_identity()

from src.schema import (
    EVALUATABLE_TASK_TYPES,
    TASK_TYPE_LABELS,
    TaskType,
    cases_from_jsonl,
)
from src.extraction.memory_extractor import sanitize_filename
from src.loaders import ExcelLoader, JsonLoader, MdLoader
from src.ui.config_store import load_config
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
)
from src.ui.next_actions import NextAction, render_next_actions
from src.ui.theme import render_page_header
from src.ui.workspace_context import render_workspace_context, summarize_values


def get_eval_task_choices() -> list[str]:
    return [task.value for task in EVALUATABLE_TASK_TYPES]


render_page_header(
    "评测数据",
    "导入已有提取结果、标准 case 或通用样本，准备进入绝对评测。",
    category="评测工作流",
)

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
        "上传 USER.md 提取结果 Excel",
        "上传通用 Excel / JSON / JSONL",
        "读取 Markdown 样本目录",
    ]
    mode_help = "用户画像任务只接收已有 USER.md 提取结果或标准评测样本；原始对话请到「记忆提取」页处理。"

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

with st.expander("输入是原始对话 Excel？", expanded=False):
    st.write("本页不运行记忆提取。只有原始对话时，请先到「记忆提取」生成 USER.md 或 MEMORY.md。")
    if st.button("前往记忆提取", icon=":material/memory:", key="open_memory_extraction"):
        st.switch_page("pages/10_记忆提取.py")

if mode == "选择已有样本文件":
    files = list_case_files()
    if not files:
        st.warning("data/cases/ 下暂无样本文件。")
    else:
        labels = [Path(f).name for f in files]
        selected_label = st.selectbox("选择样本文件", labels)
        selected_path = files[labels.index(selected_label)]

        if st.button("加载样本文件", width="stretch"):
            cases = cases_from_jsonl(selected_path)
            st.session_state.cases = cases
            st.session_state.cases_file = selected_path
            if cases:
                st.session_state.task_type = cases[0].task_type.value
            st.success(f"已加载 {len(cases)} 条样本：{selected_path}")

elif mode == "上传 USER.md 提取结果 Excel":
    st.info("适用于 run_user.py 或同类程序生成的 Excel，通常包含会话ID、轮次、用户问题、助手回答、评测人、user.md、result、reasoning 列。")

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
    if input_ready and st.button("转换为用户画像评测样本", width="stretch"):
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
                    st.dataframe(pd.DataFrame(skipped), width="stretch", hide_index=True)
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
        width="stretch",
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

    if uploaded and st.button("加载并转换", width="stretch"):
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

    if dir_path and st.button("读取 Markdown 样本", width="stretch"):
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
    st.dataframe(cases_to_dataframe(cases).head(50), width="stretch", hide_index=True)

    missed_cases = st.session_state.get("missed_cases", [])
    if missed_cases:
        st.subheader("漏抽样本预览")
        st.caption(f"漏抽样本文件：{st.session_state.get('missed_cases_file', '')}")
        st.dataframe(cases_to_dataframe(missed_cases).head(50), width="stretch", hide_index=True)
    render_workspace_context(
        task_type=st.session_state.task_type,
        case_count=len(cases),
        cases_file=st.session_state.get("cases_file", ""),
        model_name=summarize_values(case.model_name for case in cases),
        extraction_prompt=summarize_values(case.prompt_version for case in cases),
    )
    render_next_actions([
        NextAction("pages/3_执行评测.py", "进入执行评测", ":material/play_circle:", "使用当前样本启动绝对评测"),
    ])
else:
    st.info("尚未加载样本。")
