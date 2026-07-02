import os
import re

from ..schema import Case, DialogueTurn, TaskType
from .base_loader import LoadError, BaseLoader


DIALOGUE_LINE_RE = re.compile(r'^-\s*(user|assistant)\s*[:：]\s*(.*)', re.IGNORECASE)


class MdLoader(BaseLoader):
    def __init__(self, task_type: TaskType):
        self.task_type = task_type

    def load(self, dir_path: str) -> list[Case]:
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"目录不存在: {dir_path}")

        old_path = os.path.join(dir_path, "old_user.md")
        dialogue_path = os.path.join(dir_path, "dialogue.md")
        new_path = os.path.join(dir_path, "new_user.md")

        missing = []
        for p, name in [(old_path, "old_user.md"), (dialogue_path, "dialogue.md"), (new_path, "new_user.md")]:
            if not os.path.isfile(p):
                missing.append(name)
        if missing:
            raise FileNotFoundError(f"目录 {dir_path} 中缺少文件: {missing}")

        with open(old_path, "r", encoding="utf-8") as f:
            old_memory = f.read().strip()

        with open(new_path, "r", encoding="utf-8") as f:
            candidate_output = f.read().strip()

        dialogue = self._parse_dialogue(dialogue_path)

        case_id = os.path.basename(dir_path)
        return [Case(
            case_id=case_id,
            task_type=self.task_type,
            session_id=case_id,
            old_memory=old_memory if old_memory else None,
            dialogue=dialogue,
            candidate_output=candidate_output if candidate_output else None,
            metadata={"source_dir": dir_path},
        )]

    def _parse_dialogue(self, path: str) -> list[DialogueTurn]:
        turns = []
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                m = DIALOGUE_LINE_RE.match(line)
                if m:
                    role = m.group(1).lower()
                    content = m.group(2).strip()
                    turns.append(DialogueTurn(role=role, content=content))
                else:
                    raise LoadError(
                        f"无法解析 {os.path.basename(path)} 第 {lineno} 行: {line[:80]}\n"
                        f"期望格式: - user: 内容 或 - assistant: 内容"
                    )
        return turns
