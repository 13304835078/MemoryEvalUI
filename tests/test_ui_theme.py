from __future__ import annotations

from pathlib import Path

from src.ui.theme import page_header_html, workflow_html


def test_page_header_html_escapes_content() -> None:
    rendered = page_header_html(
        "标题 <script>",
        "说明 & 补充",
        category="测试 > 页面",
    )

    assert "标题 &lt;script&gt;" in rendered
    assert "说明 &amp; 补充" in rendered
    assert "测试 &gt; 页面" in rendered
    assert "<script>" not in rendered


def test_workflow_html_contains_four_ordered_steps() -> None:
    rendered = workflow_html()

    assert rendered.count('class="me-flow-step"') == 4
    assert rendered.index("配置") < rendered.index("准备数据")
    assert rendered.index("准备数据") < rendered.index("执行评测")
    assert rendered.index("执行评测") < rendered.index("分析迭代")


def test_workflow_html_supports_distinct_input_paths() -> None:
    rendered = workflow_html(
        [
            ("记忆提取", "原始对话入口"),
            ("评测数据", "已有结果入口"),
        ]
    )

    assert rendered.count('class="me-flow-step"') == 2
    assert "记忆提取" in rendered
    assert "评测数据" in rendered


def test_official_brand_assets_are_present_and_bundled() -> None:
    project_root = Path(__file__).parents[1]

    assert (project_root / "assets" / "brand" / "huawei_logo.png").is_file()
    assert (project_root / "assets" / "brand" / "huawei_icon.svg").is_file()
    assert '("assets", "assets")' in (project_root / "MemoryEvalUI.spec").read_text(encoding="utf-8")
