from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from src.persistence import atomic_write_text
from src.runtime_paths import APP_HOME, activate_workspace, ensure_writable_layout
from src.ui.state_io import state_file_lock


IDENTITY_SESSION_KEY = "memory_eval_identity"
USER_REGISTRY_DIR = APP_HOME / "system" / "users"


def _hide_sidebar_navigation() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stSidebarNav"] { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@dataclass(frozen=True)
class UserIdentity:
    workspace_id: str
    display_name: str
    masked_work_id: str


def _normalize_work_id(value: str) -> str:
    work_id = str(value or "").strip().upper()
    if not re.fullmatch(r"[A-Z0-9._-]{2,64}", work_id):
        raise ValueError("工号需为 2-64 位字母、数字、点、下划线或连字符")
    return work_id


def _normalize_name(value: str) -> str:
    name = " ".join(str(value or "").strip().split())
    if not 1 <= len(name) <= 40 or any(ord(char) < 32 for char in name):
        raise ValueError("姓名需为 1-40 个可见字符")
    return name


def _mask_work_id(work_id: str) -> str:
    if len(work_id) <= 4:
        return work_id[0] + "*" * max(1, len(work_id) - 1)
    return f"{work_id[:2]}{'*' * (len(work_id) - 4)}{work_id[-2:]}"


def register_or_validate_identity(work_id: str, display_name: str) -> UserIdentity:
    normalized_id = _normalize_work_id(work_id)
    normalized_name = _normalize_name(display_name)
    work_id_hash = hashlib.sha256(normalized_id.encode("utf-8")).hexdigest()
    workspace_id = f"user_{work_id_hash[:24]}"
    profile_path = USER_REGISTRY_DIR / f"{work_id_hash}.json"
    USER_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

    with state_file_lock(profile_path):
        if profile_path.exists():
            try:
                profile = json.loads(profile_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise ValueError(f"该工号的身份记录损坏，请联系管理员处理：{exc}") from exc
            stored_name = str(profile.get("display_name") or "")
            if stored_name.casefold() != normalized_name.casefold():
                raise ValueError("该工号已绑定其他姓名；如为录入错误，请联系管理员核对身份记录")
            workspace_id = str(profile.get("workspace_id") or workspace_id)
        else:
            atomic_write_text(
                profile_path,
                json.dumps(
                    {
                        "version": 1,
                        "work_id_hash": work_id_hash,
                        "workspace_id": workspace_id,
                        "display_name": normalized_name,
                        "masked_work_id": _mask_work_id(normalized_id),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )

    activate_workspace(workspace_id)
    ensure_writable_layout()
    return UserIdentity(workspace_id, normalized_name, _mask_work_id(normalized_id))


def current_identity() -> UserIdentity | None:
    value = st.session_state.get(IDENTITY_SESSION_KEY)
    if not isinstance(value, dict):
        return None
    try:
        return UserIdentity(
            workspace_id=str(value["workspace_id"]),
            display_name=str(value["display_name"]),
            masked_work_id=str(value["masked_work_id"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _test_identity_bypass() -> UserIdentity | None:
    if os.environ.get("MEMORY_EVAL_TEST_BYPASS_IDENTITY", "").strip() != "1":
        return None
    activate_workspace("")
    return UserIdentity("", "页面测试", "TEST")


def require_user_identity() -> UserIdentity:
    bypass = _test_identity_bypass()
    if bypass is not None:
        return bypass
    identity = current_identity()
    if identity is not None:
        activate_workspace(identity.workspace_id)
        ensure_writable_layout()
        return identity

    _hide_sidebar_navigation()
    st.markdown("## 进入记忆评测工作台")
    st.caption("请输入工号和姓名以进入个人工作区。配置、提示词、上传文件、任务和结果将按工号隔离。")
    with st.form("memory_eval_identity_form", border=True):
        work_id = st.text_input("工号", max_chars=64, autocomplete="off")
        display_name = st.text_input("姓名", max_chars=40, autocomplete="off")
        submitted = st.form_submit_button("进入系统", type="primary", width="stretch")
    st.info(
        "该步骤用于工作区识别和数据隔离，不是密码认证。公司 VM 部署仍需通过 VPN、反向代理或统一身份认证限制访问。"
    )
    if submitted:
        try:
            identity = register_or_validate_identity(work_id, display_name)
            st.session_state[IDENTITY_SESSION_KEY] = {
                "workspace_id": identity.workspace_id,
                "display_name": identity.display_name,
                "masked_work_id": identity.masked_work_id,
            }
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    st.stop()
    raise RuntimeError("unreachable")


def require_page_identity() -> UserIdentity:
    bypass = _test_identity_bypass()
    if bypass is not None:
        return bypass
    identity = current_identity()
    if identity is None:
        _hide_sidebar_navigation()
        st.error("当前页面没有已验证的使用者工作区，请从系统入口输入工号和姓名后再访问。")
        st.page_link("app.py", label="返回系统入口", icon=":material/login:")
        st.stop()
        raise RuntimeError("unreachable")
    activate_workspace(identity.workspace_id)
    ensure_writable_layout()
    return identity


def render_identity_sidebar(identity: UserIdentity) -> None:
    with st.sidebar.container(border=True):
        st.markdown(f"**{identity.display_name}**")
        st.caption(f"工号 {identity.masked_work_id} · 独立工作区")
        if st.button("退出当前工作区", width="stretch", key="logout_identity"):
            st.session_state.clear()
            activate_workspace("")
            st.rerun()
