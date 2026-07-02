from abc import ABC, abstractmethod
from typing import Optional


COLUMN_ALIASES = {
    "case_id": [
        "case_id", "case id", "用例id", "case标识", "case", "case_no",
        "编号", "case number",
    ],
    "session_id": [
        "session_id", "session", "对话id", "会话id", "会话",
    ],
    "old_memory": [
        "old_memory", "旧user.md", "旧画像", "user.md_old", "原始user.md",
        "旧memory", "old user", "old_user_md", "原始记忆",
    ],
    "dialogue": [
        "dialogue", "对话记录", "对话内容", "对话",
    ],
    "candidate_output": [
        "candidate_output", "新user.md", "候选输出", "new_user.md",
        "new_user_md", "模型输出", "generated", "生成结果",
    ],
    "reference_output": [
        "reference_output", "参考答案", "gold输出", "标注答案",
        "reference", "ground_truth", "gold", "人工标注",
    ],
    "model_name": [
        "model_name", "模型", "模型名称", "model",
    ],
    "prompt_version": [
        "prompt_version", "prompt版本", "版本", "version", "prompt",
    ],
    "reasoning": [
        "reasoning", "reason", "推理", "思考过程", "分析过程", "模型reasoning",
    ],
}

DIALOGUE_SPLIT_COLS = {
    "query": ["query", "user", "提问", "用户", "user_input", "问题"],
    "answer": ["answer", "assistant", "回答", "助手", "assistant_output", "回复"],
}


def fuzzy_match(target: str, candidates: list[str]) -> Optional[str]:
    target = target.strip().lower()
    for key, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in target:
                return key
    return None


def fuzzy_match_dialogue(target: str) -> Optional[str]:
    target = target.strip().lower()
    for key, aliases in DIALOGUE_SPLIT_COLS.items():
        for alias in aliases:
            if alias in target:
                return key
    return None


class LoadError(Exception):
    pass


class InvalidColumnError(LoadError):
    def __init__(self, missing: list[str], actual_columns: list[str]):
        self.missing = missing
        self.actual_columns = actual_columns
        msg = (
            f"无法识别以下必填列: {missing}\n"
            f"实际列名: {actual_columns}\n"
            f"请确保列名包含以下关键词之一..."
        )
        super().__init__(msg)


class BaseLoader(ABC):
    @abstractmethod
    def load(self, path: str) -> list:
        ...
