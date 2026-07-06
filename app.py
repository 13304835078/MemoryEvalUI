from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.build_info import format_build_label, get_build_info


st.set_page_config(
    page_title="记忆评测工具",
    page_icon="ME",
    layout="wide",
)

st.title("记忆评测工具")
build_info = get_build_info()
st.caption(f"版本：{format_build_label(build_info)}")

with st.expander("版本和构建信息", expanded=False):
    st.json(build_info)

st.markdown(
    """
这是一个本地记忆提取、自动评测、人工复核和提示词闭环实验工具。

## 推荐主流程

1. **配置**  
   配置接口地址、模型名、token、温度、top_p、enable_thinking、并发、重试和限流等待；管理裁判提示词和提取提示词版本。

2. **数据输入**  
   支持运行 USER.md 记忆提取、上传 USER.md 提取结果、上传长期记忆 MEMORY.md 提取结果，以及上传通用样本文件。长期记忆结果兼容 `MEMORY.md` 和 `生成的MEMORY.md正文` 两种列名。

3. **执行评测**  
   对生成的 case 做单模型绝对评测。评测会后台运行，切换页面后进度不会归零；重新回到页面可继续查看当前任务状态。

4. **结果总览**  
   查看 USER.md 或 MEMORY.md 的平均分、维度分、错误标签、失败样本、稳定性对比和历史结果对比。适合判断两次运行是否只是总分接近，还是具体错误也一致。

5. **样本详情**  
   查看单条样本的对话、旧/新 USER.md 或 MEMORY.md、comment、error_tags、规则引用、证据引用、输出引用和得分；可做人工复核。

## 扩展功能

6. **人工审核结果评估**  
   用于早期 GSB 双模型对比实验：上传人工审核 Excel，分别评分两个模型结果，再按固定分差规则判 G/S/B，并和人工 GSB 做对比。当前主线已暂时放弃 GSB，这个入口保留用于历史对照。

7. **提示词改进建议**  
   基于普通绝对评测结果或人工审核结果，让模型生成裁判提示词/提取提示词的修改建议。无人工审核时建议只作为候选版本，不应直接覆盖线上提示词。

8. **裁判提示词 AB 对比**  
   对同一批人工审核样本，用两个裁判提示词分别评估，比较一致率、分布和差异，适合验证新裁判提示词是否更贴近人工判断。

9. **闭环实验**  
   自动串联：记忆提取 → 生成 case → 执行绝对评测 → 生成候选提取提示词 → 下一轮记忆提取。支持 USER.md 和 MEMORY.md 两类任务；闭环后台运行，切换页面后仍会继续。

10. **记忆提取**
   单独运行 USER.md 或 MEMORY.md 提取，不进入闭环。支持后台进度、终止请求、结果下载、核心列预览和单个 chunk 详情。

## 使用建议

- 首次使用先在 **配置** 页测试连接。
- 新数据建议先小样本跑通，再扩大并发和样本量。
- 如果使用闭环实验，先设 `1-2` 轮和少量 case，确认候选提示词没有跑偏后再放大。
- 所有生成的新提示词默认另存为新版本，不会覆盖原文件。
"""
)

st.info("请从左侧页面开始。")
