import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.schema import TaskType
from src.loaders.md_loader import MdLoader


def test_md_loader():
    tmp_dir = tempfile.mkdtemp()
    try:
        with open(os.path.join(tmp_dir, "old_user.md"), "w", encoding="utf-8") as f:
            f.write("- 姓名: 张三\n- 职业: 工程师\n")

        with open(os.path.join(tmp_dir, "dialogue.md"), "w", encoding="utf-8") as f:
            f.write("- user: 我最近在学 Rust\n")
            f.write("- assistant: Rust 是个很好的选择！\n")
            f.write("- user: 有什么推荐的学习资源吗\n")
            f.write("- assistant: 推荐 The Rust Book\n")

        with open(os.path.join(tmp_dir, "new_user.md"), "w", encoding="utf-8") as f:
            f.write("- 姓名: 张三\n- 职业: 工程师\n- 学习: Rust\n")

        loader = MdLoader(TaskType.USER_MD)
        cases = loader.load(tmp_dir)
        assert len(cases) == 1
        case = cases[0]

        assert case.old_memory == "- 姓名: 张三\n- 职业: 工程师"
        assert case.candidate_output == "- 姓名: 张三\n- 职业: 工程师\n- 学习: Rust"
        assert len(case.dialogue) == 4
        assert case.dialogue[0].role == "user"
        assert case.dialogue[0].content == "我最近在学 Rust"
        assert case.dialogue[1].role == "assistant"
        assert case.dialogue[1].content == "Rust 是个很好的选择！"
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
