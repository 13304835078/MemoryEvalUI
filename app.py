from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
BRAND_DIR = PROJECT_ROOT / "assets" / "brand"
HUAWEI_LOGO_PATH = BRAND_DIR / "huawei_logo.png"
HUAWEI_ICON_PATH = BRAND_DIR / "huawei_icon.svg"

from src.build_info import format_build_label, get_build_info
from src.ui.theme import render_page_header, workflow_html
from src.ui.user_identity import render_identity_sidebar, require_user_identity


st.set_page_config(
    page_title="记忆评测工作台",
    page_icon=str(HUAWEI_ICON_PATH) if HUAWEI_ICON_PATH.exists() else "M",
    layout="wide",
    initial_sidebar_state="auto",
)
if HUAWEI_LOGO_PATH.exists():
    st.logo(
        str(HUAWEI_LOGO_PATH),
        size="large",
        icon_image=str(HUAWEI_ICON_PATH) if HUAWEI_ICON_PATH.exists() else None,
    )

identity = require_user_identity()
render_identity_sidebar(identity)

# Import task modules only after the current Streamlit session has activated its
# workspace. Runtime paths remain contextual for concurrent user sessions.
from src.ui.task_indicator import render_sidebar_task_indicator


def render_home() -> None:
    build_info = get_build_info()
    render_page_header(
        "记忆评测工作台",
        "按数据起点选择提取或评测路径，再统一进入证据分析与提示词迭代。",
        category="系统总览",
    )
    st.markdown(
        f'<span class="me-version-pill">当前版本 {format_build_label(build_info)}</span>',
        unsafe_allow_html=True,
    )

    st.markdown("## 选择你的起点")
    st.markdown("### 从原始对话开始")
    st.caption("输入是包含轮次、query、answer、评测人的原始对话 Excel。")
    st.markdown(
        workflow_html(
            [
                ("配置", "设置提取模型、裁判模型和两类提示词"),
                ("记忆提取", "按 session 和 chunk 生成 USER.md 或 MEMORY.md"),
                ("生成 case", "提取完成后自动转换为标准评测样本"),
                ("评测复核", "执行 Judge 评分并查看结果与证据"),
            ]
        ),
        unsafe_allow_html=True,
    )

    st.markdown("### 从已有提取结果开始")
    st.caption("输入已经包含 USER.md、MEMORY.md，或本身就是标准 case。")
    st.markdown(
        workflow_html(
            [
                ("配置", "确认裁判模型与对应提示词版本"),
                ("评测数据", "导入已有提取结果或标准 case"),
                ("执行评测", "后台评分并保留结构化诊断"),
                ("结果复核", "分析质量、稳定性和失败样本"),
            ]
        ),
        unsafe_allow_html=True,
    )

    st.markdown("## 常用入口")
    quick_cols = st.columns(4)
    quick_links = [
        ("pages/1_配置.py", "配置", ":material/settings:", "设置接口、模型与提示词版本"),
        ("pages/10_记忆提取.py", "记忆提取", ":material/memory:", "原始对话生成记忆与评测 case"),
        ("pages/2_数据输入.py", "评测数据", ":material/dataset:", "已有提取结果或 case 的评测入口"),
        ("pages/6_任务中心.py", "任务中心", ":material/task_alt:", "集中查看后台运行状态"),
    ]
    for col, (page, label, icon, description) in zip(quick_cols, quick_links):
        with col:
            st.page_link(page, label=label, icon=icon, width="stretch")
            st.caption(description)

    left, right = st.columns(2)
    with left:
        st.markdown("### 评测与复核")
        st.page_link("pages/3_执行评测.py", label="执行评测", icon=":material/play_circle:")
        st.page_link("pages/4_结果总览.py", label="结果总览", icon=":material/analytics:")
        st.page_link("pages/5_样本详情.py", label="样本详情与人工复核", icon=":material/fact_check:")
    with right:
        st.markdown("### 优化实验")
        st.page_link("pages/7_提示词改进建议.py", label="提示词改进建议", icon=":material/edit_note:")
        st.page_link("pages/8_裁判提示词AB对比.py", label="裁判提示词 A/B 对比", icon=":material/compare_arrows:")
        st.page_link("pages/11_提取提示词AB对比.py", label="提取提示词 A/B 对比", icon=":material/difference:")
        st.page_link("pages/9_闭环实验.py", label="闭环实验", icon=":material/autorenew:")

    with st.expander("首次使用与运行说明", expanded=False):
        st.markdown(
            """
1. 先在“配置”页测试模型连接，并确认裁判提示词与提取提示词版本。
2. 原始对话先走“记忆提取”；已有提取结果或 case 走“评测数据”。两条路径都会衔接“执行评测”。
3. 长任务切换页面后仍会后台运行，可在“任务中心”查看状态、调整可变参数或请求终止。
4. 提示词建议默认生成候选版本，不覆盖当前文件；进入下一轮前应先复核差异。
5. USER.md 与 MEMORY.md 是两类独立任务，导入、提取、评测和提示词版本需保持一致。
"""
        )

    with st.expander("版本与构建信息", expanded=False):
        st.json(build_info)


navigation = st.navigation(
    {
        "基础设置": [
            st.Page(render_home, title="总览", icon=":material/dashboard:", default=True),
            st.Page("pages/1_配置.py", title="配置", icon=":material/settings:"),
        ],
        "提取工作流": [
            st.Page("pages/10_记忆提取.py", title="记忆提取", icon=":material/memory:"),
        ],
        "评测工作流": [
            st.Page("pages/2_数据输入.py", title="评测数据", icon=":material/dataset:"),
            st.Page("pages/3_执行评测.py", title="执行评测", icon=":material/play_circle:"),
            st.Page("pages/4_结果总览.py", title="结果总览", icon=":material/analytics:"),
            st.Page("pages/5_样本详情.py", title="样本复核", icon=":material/fact_check:"),
        ],
        "优化实验": [
            st.Page("pages/7_提示词改进建议.py", title="提示词改进", icon=":material/edit_note:"),
            st.Page("pages/8_裁判提示词AB对比.py", title="裁判 A/B", icon=":material/compare_arrows:"),
            st.Page("pages/11_提取提示词AB对比.py", title="提取 A/B", icon=":material/difference:"),
            st.Page("pages/9_闭环实验.py", title="闭环实验", icon=":material/autorenew:"),
        ],
        "运行管理": [
            st.Page("pages/6_任务中心.py", title="任务中心", icon=":material/task_alt:"),
        ],
    }
)
render_sidebar_task_indicator()
navigation.run()
