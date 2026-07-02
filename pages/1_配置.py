from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.schema import TaskType, DIMENSION_WEIGHTS
from src.eval.judge_client import RealJudgeClient
from src.ui.config_store import load_config, save_config, build_eval_config, mask_token
from src.ui.prompt_editor import (
    list_prompt_files,
    list_extraction_prompt_files,
    get_default_prompt_file,
    get_default_extraction_prompt_file,
    load_prompt,
    save_prompt_version,
    infer_prompt_version,
    load_rubric,
)


def get_eval_task_choices() -> list[str]:
    return [t.value for t in TaskType if t.value != "raw_dialogue"]


st.title("配置")

if "ui_config" not in st.session_state:
    st.session_state.ui_config = load_config()

if "task_type" not in st.session_state:
    st.session_state.task_type = "user_md_update"

if "selected_prompt_file" not in st.session_state:
    st.session_state.selected_prompt_file = get_default_prompt_file(st.session_state.task_type)

if "judge_prompt_text" not in st.session_state:
    st.session_state.judge_prompt_text = load_prompt(st.session_state.selected_prompt_file)

if "selected_extraction_prompt_file" not in st.session_state:
    st.session_state.selected_extraction_prompt_file = get_default_extraction_prompt_file(st.session_state.task_type)

if "extraction_prompt_text" not in st.session_state:
    st.session_state.extraction_prompt_text = load_prompt(
        st.session_state.selected_extraction_prompt_file,
        prompt_kind="extraction",
    )


col_api, col_prompt = st.columns([1, 1])

with col_api:
    st.subheader("接口配置")

    cfg = dict(st.session_state.ui_config)

    api_presets = {
        "自定义": "",
        "测试环境 DS": "http://10.32.214.120:8080/service/ds_diversion/llm/v1/chat/completions",
        "现网贵安 DS": "http://10.52.218.139:6655/service/ds_diversion/llm/v1/chat/completions",
    }
    current_api_base = cfg.get("api_base", "")
    preset_names = list(api_presets.keys())
    preset_index = next(
        (i for i, name in enumerate(preset_names) if api_presets[name] and api_presets[name] == current_api_base),
        0,
    )
    selected_preset = st.selectbox("接口预设", preset_names, index=preset_index)
    if selected_preset != "自定义":
        cfg["api_base"] = api_presets[selected_preset]

    cfg["api_base"] = st.text_input(
        "接口地址",
        value=cfg.get("api_base", ""),
        placeholder="https://api.xxx.com/v1 或 https://api.xxx.com/v1/chat/completions",
    )
    cfg["judge_model"] = st.text_input(
        "裁判模型名",
        value=cfg.get("judge_model", ""),
        placeholder="例如 AGENT-GLM5-PERF / gpt-4o-mini",
    )

    cfg["judge_auth_type"] = "bearer"
    cfg["api_token"] = st.text_input(
        "接口令牌",
        value=cfg.get("api_token", ""),
        type="password",
        placeholder="可填 sk-xxx，也可填 Bearer sk-xxx",
    )

    with st.expander("核心生成参数", expanded=True):
        cfg["judge_max_tokens"] = st.number_input(
            "最大输出长度",
            min_value=256,
            max_value=20000,
            value=int(cfg.get("judge_max_tokens", 2000)),
            step=256,
        )
        cfg["judge_temperature"] = st.number_input(
            "温度",
            min_value=0.0,
            max_value=2.0,
            value=float(cfg.get("judge_temperature", 0.0)),
            step=0.1,
            help="裁判模型推荐设为 0，降低随机性。",
        )
        cfg["judge_top_p"] = st.number_input(
            "top_p",
            min_value=0.0,
            max_value=1.0,
            value=float(cfg.get("judge_top_p", 1.0)),
            step=0.05,
        )
        top_k_value = cfg.get("judge_top_k", None)
        cfg["judge_top_k"] = st.number_input(
            "top_k（0 表示不发送）",
            min_value=0,
            max_value=1000,
            value=int(top_k_value or 0),
            step=1,
        )
        if cfg["judge_top_k"] == 0:
            cfg["judge_top_k"] = None

        c_think1, c_think2 = st.columns(2)
        with c_think1:
            cfg["judge_send_enable_thinking"] = st.checkbox(
                "发送 enable_thinking 字段",
                value=bool(cfg.get("judge_send_enable_thinking", True)),
                help="AGENT-DEEPSEEK-*-THINKING 这类模型通常不需要发送该字段。",
            )
        with c_think2:
            cfg["judge_enable_thinking"] = st.checkbox(
                "enable_thinking=true",
                value=bool(cfg.get("judge_enable_thinking", False)),
                disabled=not bool(cfg.get("judge_send_enable_thinking", True)),
            )

    cfg["judge_stop"] = []
    cfg["judge_stream"] = False
    cfg["judge_stream_include_usage"] = True

    with st.expander("核心运行参数", expanded=True):
        cfg["judge_timeout"] = st.number_input(
            "超时秒数",
            min_value=10,
            max_value=600,
            value=int(cfg.get("judge_timeout", 120)),
            step=10,
        )
        cfg["judge_max_retries"] = st.number_input(
            "最大重试次数",
            min_value=1,
            max_value=10,
            value=int(cfg.get("judge_max_retries", 3)),
            step=1,
        )
        cfg["judge_request_interval"] = st.number_input(
            "请求间隔秒数",
            min_value=0.0,
            max_value=60.0,
            value=float(cfg.get("judge_request_interval", 10.5)),
            step=0.5,
            help="内部接口如果限制 0.10 QPS，建议设置为 10.5 到 12 秒。",
        )
        cfg["judge_concurrency"] = st.number_input(
            "并发数",
            min_value=1,
            max_value=100,
            value=min(100, max(1, int(cfg.get("judge_concurrency", 1) or 1))),
            step=1,
            help="并发只增加同时等待/重试的请求数；每次请求启动仍会遵守上面的请求间隔。",
        )
        cfg["judge_qps_backoff"] = st.number_input(
            "限流重试等待秒数",
            min_value=1.0,
            max_value=120.0,
            value=float(cfg.get("judge_qps_backoff", 12.0)),
            step=1.0,
            help="遇到 QPS limit exceeded 时等待多久再重试。",
        )

    cfg["judge_bearer_header_name"] = "Authorization"
    cfg["judge_call_from"] = ""
    cfg["judge_session_id"] = ""
    cfg["judge_interaction_id"] = None
    cfg["judge_send_skip_special_tokens"] = True
    cfg["judge_skip_special_tokens"] = False
    cfg["judge_moderation_action"] = ""
    cfg["judge_prompt_cache_location"] = "none"
    cfg["judge_prompt_cache_id"] = ""
    cfg["judge_extra_body_json"] = "{}"
    cfg["judge_custom_headers_json"] = "{}"

    cfg["mock"] = st.checkbox("默认使用模拟模式", value=bool(cfg.get("mock", True)))

    c1, c2 = st.columns(2)
    with c1:
        if st.button("保存接口配置", use_container_width=True):
            st.session_state.ui_config = cfg
            save_config(cfg)
            st.success("已保存到 config/local_config.json")

    with c2:
        if st.button("测试连接", use_container_width=True):
            test_cfg = build_eval_config(cfg, mock=False)
            errs = test_cfg.validate()
            if errs:
                st.error("配置不完整：\n" + "\n".join([f"- {e}" for e in errs]))
            else:
                try:
                    client = RealJudgeClient(test_cfg)
                    if hasattr(client, "test_connection"):
                        ok, msg = client.test_connection()
                    else:
                        parsed, raw = client.judge(
                            "你是测试助手，请严格输出 JSON。",
                            '{"score_total":5,"scores":{"correctness":5,"coverage":5,"update_logic":5,"memory_boundary":5,"conciseness":5,"format":5},"comment":"ok","error_tags":[],"fatal_error":false}',
                        )
                        ok = parsed is not None
                        msg = "连接成功" if ok else str(raw)
                    if ok:
                        st.success(msg)
                    else:
                        st.error("连接失败")
                        st.code(msg, language="text")
                except Exception as e:
                    st.error(f"连接失败：{e}")

    if cfg.get("api_token"):
        st.caption(f"当前 token：{mask_token(cfg.get('api_token', ''))}")


with col_prompt:
    st.subheader("提示词配置")

    task_choices = get_eval_task_choices()
    task_type = st.selectbox(
        "任务类型",
        task_choices,
        index=task_choices.index(st.session_state.task_type)
        if st.session_state.task_type in task_choices else 0,
    )

    if task_type != st.session_state.task_type:
        st.session_state.task_type = task_type
        st.session_state.selected_prompt_file = get_default_prompt_file(task_type)
        st.session_state.judge_prompt_text = load_prompt(st.session_state.selected_prompt_file)
        st.session_state.selected_extraction_prompt_file = get_default_extraction_prompt_file(task_type)
        st.session_state.extraction_prompt_text = load_prompt(
            st.session_state.selected_extraction_prompt_file,
            prompt_kind="extraction",
        )
        st.rerun()

    tab_judge, tab_extract = st.tabs(["裁判提示词", "提取提示词"])

    with tab_judge:
        prompt_files = list_prompt_files()
        default_prompt = get_default_prompt_file(task_type)

        if default_prompt and default_prompt not in prompt_files:
            prompt_files = [default_prompt] + prompt_files

        selected_prompt = st.selectbox(
            "裁判提示词文件",
            prompt_files,
            index=prompt_files.index(st.session_state.selected_prompt_file)
            if st.session_state.selected_prompt_file in prompt_files else 0,
        )

        if selected_prompt != st.session_state.selected_prompt_file:
            st.session_state.selected_prompt_file = selected_prompt
            st.session_state.judge_prompt_text = load_prompt(selected_prompt)
            st.rerun()

        c1, c2 = st.columns(2)
        with c1:
            if st.button("重新加载裁判提示词", use_container_width=True):
                st.session_state.judge_prompt_text = load_prompt(st.session_state.selected_prompt_file)
                st.rerun()

        with c2:
            st.caption(f"当前版本：{infer_prompt_version(st.session_state.selected_prompt_file)}")

        st.session_state.judge_prompt_text = st.text_area(
            "裁判提示词内容",
            value=st.session_state.judge_prompt_text,
            height=360,
        )

        version_name = st.text_input(
            "保存为新版本文件名",
            value=f"judge_{task_type}_custom.md",
            key="judge_prompt_save_name",
        )

        if st.button("保存为新裁判提示词版本", use_container_width=True):
            saved = save_prompt_version(task_type, st.session_state.judge_prompt_text, version_name)
            st.session_state.selected_prompt_file = saved
            st.success(f"已保存：prompts/judge/{saved}")

    with tab_extract:
        extraction_files = list_extraction_prompt_files()
        default_extraction_prompt = get_default_extraction_prompt_file(task_type)

        if default_extraction_prompt and default_extraction_prompt not in extraction_files:
            extraction_files = [default_extraction_prompt] + extraction_files

        if not extraction_files:
            extraction_files = [""]

        selected_extraction_prompt = st.selectbox(
            "提取提示词文件",
            extraction_files,
            index=extraction_files.index(st.session_state.selected_extraction_prompt_file)
            if st.session_state.selected_extraction_prompt_file in extraction_files else 0,
            help="这里放生成 USER.md 时使用的提取 prompt。执行评测时它只作为规则来源，不作为事实来源。",
        )

        if selected_extraction_prompt != st.session_state.selected_extraction_prompt_file:
            st.session_state.selected_extraction_prompt_file = selected_extraction_prompt
            st.session_state.extraction_prompt_text = load_prompt(
                selected_extraction_prompt,
                prompt_kind="extraction",
            )
            st.rerun()

        c1, c2 = st.columns(2)
        with c1:
            if st.button("重新加载提取提示词", use_container_width=True):
                st.session_state.extraction_prompt_text = load_prompt(
                    st.session_state.selected_extraction_prompt_file,
                    prompt_kind="extraction",
                )
                st.rerun()

        with c2:
            st.caption(f"当前版本：{infer_prompt_version(st.session_state.selected_extraction_prompt_file)}")

        st.session_state.extraction_prompt_text = st.text_area(
            "提取提示词内容",
            value=st.session_state.extraction_prompt_text,
            height=360,
            help="建议把真实线上提取 prompt 粘贴到这里，再另存为版本；裁判会引用这里的规则判断 USER.md 是否稳定。",
        )

        extraction_version_name = st.text_input(
            "保存为新版本文件名",
            value=f"extract_{task_type}_custom.md",
            key="extraction_prompt_save_name",
        )

        if st.button("保存为新提取提示词版本", use_container_width=True):
            saved = save_prompt_version(
                task_type,
                st.session_state.extraction_prompt_text,
                extraction_version_name,
                prompt_kind="extraction",
            )
            st.session_state.selected_extraction_prompt_file = saved
            st.success(f"已保存：prompts/generation/{saved}")


st.subheader("当前评分标准和权重")

rubric = load_rubric(st.session_state.task_type)
weights = DIMENSION_WEIGHTS.get(st.session_state.task_type, {})

with st.expander("维度权重", expanded=True):
    if weights:
        st.json(weights)
    else:
        st.info("当前任务暂无权重配置。")

with st.expander("评分标准内容", expanded=False):
    st.markdown(rubric or "未找到 rubric 文件。")
