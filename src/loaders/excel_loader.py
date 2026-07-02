import os
import pandas as pd
from typing import Optional

from ..schema import Case, DialogueTurn, TaskType
from .base_loader import (
    BaseLoader, COLUMN_ALIASES, DIALOGUE_SPLIT_COLS,
    fuzzy_match, fuzzy_match_dialogue, InvalidColumnError,
)


class ExcelLoader(BaseLoader):
    def __init__(self, task_type: TaskType, sheet_name: str = "", chunk_size: Optional[int] = None):
        self.task_type = task_type
        self.sheet_name = sheet_name
        self.chunk_size = chunk_size

    def load(self, path: str) -> list[Case]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件不存在: {path}")

        if self.sheet_name:
            df = pd.read_excel(path, sheet_name=self.sheet_name)
        else:
            df = pd.read_excel(path)

        df = df.fillna("")
        cols_lower = [str(c).strip().lower() for c in df.columns]

        column_map = {}
        for i, col in enumerate(df.columns):
            matched = fuzzy_match(str(col), COLUMN_ALIASES)
            if matched:
                column_map[matched] = col
            else:
                d_match = fuzzy_match_dialogue(str(col))
                if d_match:
                    column_map[d_match] = col

        has_query = "query" in column_map and "answer" in column_map

        required = ["case_id"]
        missing = [r for r in required if r not in column_map]
        if not missing:
            pass
        elif "session_id" not in column_map:
            pass
        else:
            pass

        cases = []
        dialogue_buffer: list[DialogueTurn] = []

        for idx, row in df.iterrows():
            row_dict = row.to_dict()

            if has_query:
                q_col = column_map.get("query")
                a_col = column_map.get("answer")
                q_val = str(row_dict.get(q_col, "")) if q_col else ""
                a_val = str(row_dict.get(a_col, "")) if a_col else ""
                if q_val:
                    dialogue_buffer.append(DialogueTurn(role="user", content=q_val))
                if a_val:
                    dialogue_buffer.append(DialogueTurn(role="assistant", content=a_val))

            if self.chunk_size is None:
                case = self._build_case(idx, row_dict, column_map, [DialogueTurn(role="user", content=q_val), DialogueTurn(role="assistant", content=a_val)] if has_query else [])
                cases.append(case)
            elif len(dialogue_buffer) >= self.chunk_size * 2:
                case = self._build_case(idx, row_dict, column_map, dialogue_buffer.copy())
                cases.append(case)
                dialogue_buffer.clear()

        if self.chunk_size is not None and dialogue_buffer:
            case = self._build_case(len(df) - 1, df.iloc[-1].to_dict(), column_map, dialogue_buffer.copy())
            cases.append(case)

        missing_req = [r for r in required if r not in column_map]
        if missing_req and not cases:
            raise InvalidColumnError(missing_req, list(df.columns))

        return cases

    def _build_case(
        self,
        row_idx: int,
        row: dict,
        column_map: dict[str, str],
        dialogue: list[DialogueTurn],
    ) -> Case:
        def get_val(key: str, default: str = ""):
            col = column_map.get(key)
            return str(row.get(col, default)) if col else default

        case_id = get_val("case_id") or f"case_{row_idx}"
        session_id = get_val("session_id") or f"session_{row_idx}"

        old_memory = get_val("old_memory")
        candidate_output = get_val("candidate_output")
        reference_output = get_val("reference_output")
        model_name = get_val("model_name") or "unknown"
        prompt_version = get_val("prompt_version") or "unknown"
        reasoning = get_val("reasoning")

        return Case(
            case_id=case_id,
            task_type=self.task_type,
            session_id=session_id,
            old_memory=old_memory if old_memory else None,
            dialogue=dialogue,
            candidate_output=candidate_output if candidate_output else None,
            reference_output=reference_output if reference_output else None,
            model_name=model_name,
            prompt_version=prompt_version,
            metadata={
                "source_file": "",
                "row_index_start": row_idx - len(dialogue) // 2 + 1,
                "row_index_end": row_idx + 1,
                "reasoning": reasoning,
            },
        )
