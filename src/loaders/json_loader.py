import os
import json
from typing import Optional

from ..schema import Case, TaskType
from .base_loader import BaseLoader, LoadError


class JsonLoader(BaseLoader):
    def __init__(self, task_type: Optional[TaskType] = None):
        self.task_type = task_type

    def load(self, path: str) -> list[Case]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件不存在: {path}")

        ext = os.path.splitext(path)[1].lower()

        if ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                items = data
            else:
                items = [data]
        elif ext == ".jsonl":
            items = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    items.append(json.loads(line))
        else:
            raise LoadError(f"不支持的文件格式: {ext}，请使用 .json 或 .jsonl")

        cases = []
        for item in items:
            if "task_type" not in item and self.task_type:
                item["task_type"] = self.task_type.value
            case = Case.from_dict(item)
            cases.append(case)

        return cases
