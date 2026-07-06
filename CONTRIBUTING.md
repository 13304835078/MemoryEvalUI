# 协作开发约定

## 分支

- `main` 保持可运行、可打包。
- 功能分支建议使用 `feature/<简短名称>`。
- 修复分支建议使用 `fix/<简短名称>`。
- 不直接向 `main` 推送未经验证的大改动；多人协作时优先 Pull Request。

## 开发流程

1. 从最新 `main` 创建分支。
2. 只提交当前任务相关文件。
3. 修改行为时补充或更新测试。
4. 修改功能说明、页面流程或数据格式时同步更新文档。
5. 提交 Pull Request，并说明变更、原因、影响范围和验证方式。
6. 至少一名协作者 Review 后合并。

## 提交前检查

基础检查：

```powershell
python -m pytest -q
python scripts/smoke_pages.py
```

涉及启动、打包、运行路径或资源文件时，额外执行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_release.ps1
```

提交前查看暂存区：

```powershell
git diff --cached
git status -sb
```

## 安全要求

- 不提交 `config/local_config.json`、`.env` 或真实 API Token。
- 不提交上传文件、评测结果、日志、闭环运行状态和本地缓存。
- 示例配置只能使用空值或明确占位符。
- 真实业务 Excel 不进仓库；需要复现问题时，用脱敏小样本或构造测试数据。
- 远端仓库如果包含内网接口说明或业务规则，建议保持 Private。

## 提示词版本

- 不直接覆盖稳定提示词。
- 裁判提示词放在 `prompts/judge/`。
- 提取提示词放在 `prompts/generation/`。
- USER.md 提取提示词通常是 Markdown 文本。
- MEMORY.md 提取提示词使用 YAML，包含 create/update 两类模板。
- 新提示词使用新文件名，并在 PR 中说明来源证据、预期影响和验证结果。
- 闭环自动生成的候选提示词必须经过人工复核后再作为正式版本使用。

## USER.md 与 MEMORY.md 边界

- USER.md 用于长期稳定画像，例如身份、稳定偏好、长期交互偏好、关系人等。
- MEMORY.md 用于长期事项、计划、目标、约束、待跟踪状态和持续更新的信息。
- 修改评测或提取逻辑时，需要确认没有把 USER.md 与 MEMORY.md 的边界混淆。

## 后台任务和结果文件

- 执行评测、记忆提取、闭环实验都有后台状态文件，切换页面后应能继续展示进度。
- 结果写入应尽量使用安全写入，避免崩溃时损坏已有结果。
- 评测结果优先保存 JSONL；导出的 CSV/Excel 只用于查看和流转，不是完整可恢复的原始结果。
