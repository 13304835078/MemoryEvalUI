from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


@dataclass(frozen=True)
class NextAction:
    page: str
    label: str
    icon: str
    help_text: str = ""


def render_next_actions(actions: list[NextAction], *, title: str = "下一步") -> None:
    if not actions:
        return
    st.markdown(f"### {title}")
    cols = st.columns(min(3, len(actions)))
    for index, action in enumerate(actions):
        with cols[index % len(cols)]:
            st.page_link(
                action.page,
                label=action.label,
                icon=action.icon,
                help=action.help_text or None,
                width="stretch",
            )
