from __future__ import annotations

from html import escape

import streamlit as st


_GLOBAL_STYLES = r"""
<style>
:root {
  color-scheme: light dark;
  --me-bg: light-dark(#f7f7f8, #0e1117);
  --me-surface: light-dark(#ffffff, #171b24);
  --me-surface-subtle: light-dark(#f1f2f3, #141820);
  --me-text: light-dark(#202124, #f4f4f5);
  --me-muted: light-dark(#666a73, #a8adb7);
  --me-line: light-dark(#dfe1e5, #303640);
  --me-accent: light-dark(#c7000b, #f04a5e);
  --me-accent-hover: light-dark(#a8000a, #ff6675);
  --me-accent-soft: light-dark(#fbeaec, #3b171d);
  --me-accent-wash: light-dark(rgba(199, 0, 11, 0.06), rgba(240, 74, 94, 0.12));
  --me-support: light-dark(#176b68, #55b8b2);
  --me-warm: light-dark(#9a6700, #f0b45d);
  --me-danger: light-dark(#b42318, #ff6b6b);
  --me-radius: 6px;
}

.stApp {
  color: var(--me-text);
  letter-spacing: 0;
}

.stApp,
[data-testid="stAppViewContainer"] {
  background: var(--me-bg);
}

[data-testid="stMainBlockContainer"] {
  width: min(100%, 1440px);
  padding-top: 0.7rem;
  padding-bottom: 3rem;
}

[data-testid="stHeader"] {
  height: 2.75rem;
  min-height: 2.75rem;
  background: transparent;
}

[data-testid="stSidebar"] {
  background: var(--me-surface-subtle);
  border-right: 1px solid var(--me-line);
}

[data-testid="stSidebarContent"] {
  padding-top: 0.45rem;
}

[data-testid="stSidebarLogo"] {
  box-sizing: content-box;
  padding: 0.45rem 0.65rem;
  border: 1px solid light-dark(#e2e3e6, #404650);
  border-radius: var(--me-radius);
  background: #ffffff;
  object-fit: contain;
}

[data-testid="stSidebarNav"] {
  padding-top: 0.15rem;
}

[data-testid="stSidebarNavLink"] {
  min-height: 2.45rem;
  border-radius: var(--me-radius);
  color: var(--me-text);
  font-weight: 500;
}

[data-testid="stSidebarNavLink"]:hover {
  background: var(--me-accent-wash);
  color: var(--me-accent);
}

[data-testid="stSidebarNavLink"][aria-current="page"] {
  color: var(--me-accent);
  background: var(--me-accent-soft);
  box-shadow: inset 3px 0 0 var(--me-accent);
  font-weight: 650;
}

[data-testid="stSidebarNavSeparator"] {
  color: var(--me-muted);
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0;
  margin-top: 0.9rem;
}

[data-testid="stAppDeployButton"] {
  display: none;
}

.me-page-header {
  position: relative;
  margin: 0 0 1.35rem;
  padding: 0.15rem 0 1rem 1rem;
  border-bottom: 1px solid var(--me-line);
}

.me-page-header::before {
  content: "";
  position: absolute;
  top: 0.28rem;
  bottom: 1rem;
  left: 0;
  width: 3px;
  border-radius: 2px;
  background: var(--me-accent);
}

.me-page-header::after {
  content: "";
  position: absolute;
  bottom: -1px;
  left: 1rem;
  width: 2.8rem;
  height: 2px;
  background: var(--me-accent);
}

.me-page-kicker {
  margin: 0 0 0.28rem;
  color: var(--me-accent);
  font-size: 0.72rem;
  font-weight: 700;
  line-height: 1.2;
}

.me-page-title {
  margin: 0 !important;
  color: var(--me-text);
  font-size: 1.85rem !important;
  font-weight: 720 !important;
  line-height: 1.22 !important;
  letter-spacing: 0 !important;
}

.me-page-subtitle {
  max-width: 56rem;
  margin: 0.38rem 0 0;
  color: var(--me-muted);
  font-size: 0.92rem;
  line-height: 1.55;
}

.me-version-pill {
  display: inline-flex;
  align-items: center;
  min-height: 1.65rem;
  margin-top: 0.65rem;
  padding: 0.18rem 0.55rem;
  border: 1px solid var(--me-line);
  border-radius: 999px;
  background: var(--me-surface);
  color: var(--me-muted);
  font-size: 0.75rem;
  font-weight: 600;
}

.me-context-bar {
  margin: 0 0 1.25rem;
  border-top: 1px solid var(--me-line);
  border-bottom: 1px solid var(--me-line);
  background: var(--me-surface);
}

.me-context-heading {
  padding: 0.55rem 0.8rem 0.3rem;
  color: var(--me-accent);
  font-size: 0.72rem;
  font-weight: 700;
}

.me-context-grid {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
}

.me-context-item {
  min-width: 0;
  padding: 0.4rem 0.8rem 0.72rem;
}

.me-context-item + .me-context-item {
  border-left: 1px solid var(--me-line);
}

.me-context-label,
.me-context-value {
  display: block;
}

.me-context-label {
  margin-bottom: 0.18rem;
  color: var(--me-muted);
  font-size: 0.7rem;
}

.me-context-value {
  overflow: hidden;
  color: var(--me-text);
  font-size: 0.82rem;
  font-weight: 650;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.me-flow {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  margin: 0.25rem 0 1.65rem;
  border-top: 1px solid var(--me-line);
  border-bottom: 1px solid var(--me-line);
  background: var(--me-surface);
}

.me-flow-step {
  position: relative;
  min-height: 5.15rem;
  padding: 1rem 1rem 0.9rem 3.35rem;
}

.me-flow-step + .me-flow-step {
  border-left: 1px solid var(--me-line);
}

.me-flow-number {
  position: absolute;
  top: 1rem;
  left: 1rem;
  display: grid;
  width: 1.75rem;
  height: 1.75rem;
  place-items: center;
  border-radius: 50%;
  background: var(--me-accent-soft);
  color: var(--me-accent);
  font-size: 0.78rem;
  font-weight: 750;
}

.me-flow-title {
  margin: 0 0 0.18rem;
  color: var(--me-text);
  font-size: 0.92rem;
  font-weight: 700;
}

.me-flow-text {
  margin: 0;
  color: var(--me-muted);
  font-size: 0.79rem;
  line-height: 1.45;
}

h1, h2, h3 {
  color: var(--me-text);
  letter-spacing: 0 !important;
}

h2 {
  margin-top: 1.55rem !important;
  font-size: 1.28rem !important;
  line-height: 1.35 !important;
}

h3 {
  margin-top: 1.2rem !important;
  font-size: 1.03rem !important;
  line-height: 1.4 !important;
}

p, li, label {
  line-height: 1.55;
}

[data-testid="stHorizontalBlock"] {
  gap: 0.85rem;
}

div.stButton > button,
div.stDownloadButton > button,
[data-testid="stPageLink"] a {
  min-height: 2.45rem;
  border-radius: var(--me-radius);
  font-weight: 620;
}

[data-testid="stBaseButton-primary"] {
  border-color: var(--me-accent);
  background: var(--me-accent);
}

[data-testid="stBaseButton-primary"]:hover {
  border-color: var(--me-accent-hover);
  background: var(--me-accent-hover);
}

[data-testid="stBaseButton-secondary"],
[data-testid="stDownloadButton"] button,
[data-testid="stPageLink"] a {
  border-color: var(--me-line);
  background: var(--me-surface);
  color: var(--me-text);
}

[data-testid="stBaseButton-secondary"]:hover,
[data-testid="stDownloadButton"] button:hover,
[data-testid="stPageLink"] a:hover {
  border-color: var(--me-accent);
  color: var(--me-accent);
}

[data-baseweb="input"] > div,
[data-baseweb="select"] > div,
[data-baseweb="textarea"] > div,
[data-testid="stFileUploaderDropzone"] {
  border-radius: var(--me-radius) !important;
  border-color: var(--me-line) !important;
  background: var(--me-surface) !important;
}

[data-baseweb="input"] > div:focus-within,
[data-baseweb="select"] > div:focus-within,
[data-baseweb="textarea"] > div:focus-within {
  border-color: var(--me-accent) !important;
  box-shadow: 0 0 0 1px var(--me-accent) !important;
}

[data-testid="stMetric"] {
  min-height: 5.5rem;
  padding: 0.8rem 0.9rem;
  border: 1px solid var(--me-line);
  border-radius: var(--me-radius);
  background: var(--me-surface);
}

[data-testid="stMetricLabel"] {
  color: var(--me-muted);
}

[data-testid="stMetricValue"] {
  color: var(--me-text);
}

[data-testid="stExpander"] {
  overflow: hidden;
  border: 1px solid var(--me-line);
  border-radius: var(--me-radius);
  background: var(--me-surface);
}

[data-testid="stExpander"] details summary:hover {
  color: var(--me-accent);
}

[data-testid="stAlertContainer"] {
  border-radius: var(--me-radius);
}

[data-testid="stDataFrame"],
[data-testid="stTable"] {
  overflow: hidden;
  border: 1px solid var(--me-line);
  border-radius: var(--me-radius);
  background: var(--me-surface);
}

[data-testid="stProgress"] > div > div > div > div {
  background: var(--me-accent);
}

[data-testid="stTabs"] [role="tablist"] {
  gap: 1.15rem;
  border-bottom: 1px solid var(--me-line);
}

[data-testid="stTabs"] [role="tab"] {
  min-height: 2.6rem;
  padding-right: 0;
  padding-left: 0;
  border-radius: 0;
  color: var(--me-muted);
}

[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
  color: var(--me-accent);
}

[data-testid="stChatMessage"] {
  border: 1px solid var(--me-line);
  border-radius: var(--me-radius);
  background: var(--me-surface);
}

hr {
  border-color: var(--me-line) !important;
}

@media (max-width: 900px) {
  [data-testid="stMainBlockContainer"] {
    padding-top: 1rem;
    padding-right: 1rem;
    padding-left: 1rem;
  }

  .me-flow {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .me-context-grid {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }

  .me-context-item:nth-child(4) {
    border-left: 0;
  }

  .me-context-item:nth-child(n + 4) {
    border-top: 1px solid var(--me-line);
  }

  .me-flow-step:nth-child(3) {
    border-left: 0;
    border-top: 1px solid var(--me-line);
  }

  .me-flow-step:nth-child(4) {
    border-top: 1px solid var(--me-line);
  }
}

@media (max-width: 640px) {
  [data-testid="stMainBlockContainer"] {
    padding-top: 1.8rem;
  }

  .me-page-header {
    padding-top: 1.25rem;
    padding-left: 0.8rem;
  }

  .me-page-title {
    font-size: 1.55rem !important;
  }

  .me-flow {
    grid-template-columns: 1fr;
  }

  .me-context-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .me-context-item:nth-child(odd) {
    border-left: 0;
  }

  .me-context-item:nth-child(n + 3) {
    border-top: 1px solid var(--me-line);
  }

  .me-flow-step + .me-flow-step,
  .me-flow-step:nth-child(3),
  .me-flow-step:nth-child(4) {
    border-top: 1px solid var(--me-line);
    border-left: 0;
  }
}
</style>
"""


def apply_global_styles() -> None:
    st.markdown(_GLOBAL_STYLES, unsafe_allow_html=True)


def page_header_html(title: str, subtitle: str = "", category: str = "记忆评测工作台") -> str:
    subtitle_html = (
        f'<p class="me-page-subtitle">{escape(subtitle)}</p>'
        if subtitle
        else ""
    )
    return (
        '<header class="me-page-header">'
        f'<p class="me-page-kicker">{escape(category)}</p>'
        f'<h1 class="me-page-title">{escape(title)}</h1>'
        f"{subtitle_html}"
        "</header>"
    )


def render_page_header(title: str, subtitle: str = "", category: str = "记忆评测工作台") -> None:
    apply_global_styles()
    st.markdown(page_header_html(title, subtitle, category), unsafe_allow_html=True)


def workflow_html(steps: list[tuple[str, str]] | None = None) -> str:
    workflow_steps = steps or [
        ("配置", "连接模型并选择裁判与提取提示词"),
        ("准备数据", "导入样本或独立运行记忆提取"),
        ("执行评测", "后台评分并保留结构化诊断证据"),
        ("分析迭代", "对比结果、复核样本并改进提示词"),
    ]
    body = "".join(
        '<div class="me-flow-step">'
        f'<span class="me-flow-number">{number}</span>'
        f'<p class="me-flow-title">{escape(title)}</p>'
        f'<p class="me-flow-text">{escape(text)}</p>'
        "</div>"
        for number, (title, text) in enumerate(workflow_steps, start=1)
    )
    return f'<div class="me-flow">{body}</div>'
