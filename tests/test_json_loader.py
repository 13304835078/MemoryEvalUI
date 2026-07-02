import sys, os, tempfile, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.schema import TaskType
from src.loaders.json_loader import JsonLoader


def test_json_loader():
    data = [
        {
            "case_id": "j1",
            "task_type": "user_md_update",
            "session_id": "s1",
            "old_memory": "old",
            "dialogue": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
            "candidate_output": "new",
            "model_name": "test",
            "prompt_version": "v1",
        },
        {
            "case_id": "j2",
            "task_type": "day_memory",
            "session_id": "s2",
        },
    ]
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False, encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        tmp = f.name
    try:
        loader = JsonLoader()
        cases = loader.load(tmp)
        assert len(cases) == 2
        assert cases[0].case_id == "j1"
        assert cases[0].task_type == TaskType.USER_MD
        assert cases[0].dialogue[0].content == "hi"
        assert cases[1].case_id == "j2"
        assert cases[1].task_type == TaskType.DAY_MEMORY
    finally:
        os.unlink(tmp)


def test_jsonl_loader():
    lines = [
        '{"case_id": "l1", "task_type": "summary", "session_id": "s1"}',
        '{"case_id": "l2", "task_type": "long_memory", "session_id": "s2", "old_memory": "content"}',
        "",
    ]
    with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False, encoding="utf-8") as f:
        f.write("\n".join(lines))
        tmp = f.name
    try:
        loader = JsonLoader()
        cases = loader.load(tmp)
        assert len(cases) == 2
        assert cases[0].case_id == "l1"
        assert cases[1].case_id == "l2"
        assert cases[1].old_memory == "content"
    finally:
        os.unlink(tmp)
