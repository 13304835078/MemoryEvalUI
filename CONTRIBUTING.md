# 协作开发约定

## 分支

- `main` 保持可运行。
- 功能分支使用 `feature/<简短名称>`。
- 修复分支使用 `fix/<简短名称>`。
- 不直接向 `main` 推送功能修改。

## 开发流程

1. 从最新 `main` 创建分支。
2. 只提交当前任务相关文件。
3. 修改行为时补充或更新测试。
4. 提交 Pull Request，并说明变更、原因和验证方式。
5. 至少一名协作者 Review 后合并。

## 提交前检查

```powershell
python -m pytest -q
python -m py_compile app.py run.py run_streamlit.py
```

## 安全要求

- 不提交 `config/local_config.json`、`.env` 或真实 API Token。
- 不提交上传文件、评测结果、日志和闭环运行状态。
- 示例配置只能使用空值或明确的占位符。
- 提交前检查 `git diff --cached`。

## 提示词版本

- 不直接覆盖稳定提示词。
- 新提示词使用新文件名，并在 PR 中说明来源证据和验证结果。
- 闭环自动生成的候选提示词必须经过人工复核后再作为正式版本使用。
