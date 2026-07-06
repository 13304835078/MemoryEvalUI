# MemoryEvalUI

MemoryEvalUI 是一个面向 USER.md 用户画像与 MEMORY.md 长期记忆的本地评测工具，覆盖数据导入、记忆提取、绝对评测、结果分析、人工复核、提示词改进和闭环实验。

## 主要能力

- 运行 USER.md 用户画像提取，并将提取结果转换为标准评测 case。
- 运行 MEMORY.md 长期记忆提取，并将提取结果转换为 `long_memory` 评测 case。
- 上传已有 USER.md / MEMORY.md 提取结果 Excel，生成可复用的 case 文件。
- 分别管理裁判提示词与提取提示词：`prompts/judge/` 存放 Judge Prompt，`prompts/generation/` 存放提取 Prompt。
- 调用 Judge 模型进行绝对评分，输出维度分、总分、comment、error_tags、diagnostics、rule_refs、evidence_refs 和 output_refs。
- 在结果总览中查看分数分布、错误标签、稳定性对比、历史结果对比和导出结果。
- 在样本详情中查看单条 case 的对话、旧/新 USER.md 或 MEMORY.md、得分、评语和引用证据。
- 基于评测证据生成受约束的提示词改进建议，优先做增量修改，避免提示词无限膨胀。
- 串联“提取 -> 生成 case -> 评测 -> 生成候选提取提示词 -> 下一轮提取”的闭环实验。
- 保留历史 GSB 双模型人工审核结果评估入口，用于旧实验对照。

## 环境要求

- Windows 或 Linux
- Python 3.11

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux/macOS 激活命令：

```bash
source .venv/bin/activate
```

## 配置

复制配置模板：

```powershell
Copy-Item config/local_config.example.json config/local_config.json
```

在 `config/local_config.json` 中填写真实 API 地址、Token 和模型名称。该文件已被 Git 忽略，禁止提交真实 Token。

常用配置项包括：

- `api_base`：chat/completions 接口地址。
- `api_token`：接口鉴权 Token，Bearer 接口需要写完整 `Bearer xxx`。
- `judge_model`：评测模型名称。
- `judge_temperature`：建议评测场景使用 `0`。
- `judge_top_p`：通常保持 `1.0`，不要随意调低。
- `judge_send_enable_thinking` / `judge_enable_thinking`：控制是否发送 thinking 字段以及字段值。
- `judge_request_interval` / `judge_qps_backoff`：请求间隔与限流重试等待。
- `mock`：本地模拟模式，不调用真实模型。

## 启动

```powershell
streamlit run app.py
```

也可以运行命令行入口：

```powershell
python run.py
```

访问地址默认是：

```text
http://localhost:8501
```

## 常用流程

### USER.md 绝对评测

1. 在“配置”页填写 API，选择 `user_md_update`、USER.md 裁判提示词和 USER.md 提取提示词。
2. 在“数据输入”页上传 run_user.py 输出 Excel，或在“记忆提取”页直接运行 USER.md 提取。
3. 在“执行评测”页选择同一任务类型和提示词，启动评测。
4. 在“结果总览”和“样本详情”查看统计、失败原因和引用证据。

### MEMORY.md 绝对评测

1. 在“配置”页选择 `long_memory`，使用 `judge_long_memory_v1.md` 和 `extract_long_memory_v1.yaml`。
2. 在“记忆提取”页选择“长期记忆 MEMORY.md”，上传原始对话 Excel 并运行提取。
3. 提取完成后可自动生成 long_memory case，再进入“执行评测”页评分。
4. 也可以在“数据输入”页上传已有长期记忆提取结果 Excel，结果列支持 `MEMORY.md` 或 `生成的MEMORY.md正文`。

### 闭环实验

“闭环实验”会自动执行：

```text
记忆提取 -> 生成 case -> 执行绝对评测 -> 生成候选提取提示词 -> 下一轮记忆提取
```

支持 USER.md 和 MEMORY.md。建议先用 Mock 模式或少量 case 跑 1-2 轮，确认候选提示词没有跑偏后再扩大规模。闭环生成的候选提示词会另存为新文件，不会覆盖原始提示词。

## 页面说明

- `配置`：API、模型参数、并发、重试、限流、裁判提示词、提取提示词。
- `数据输入`：加载已有 case，上传 USER.md / MEMORY.md 提取结果，上传通用 Excel/JSONL/Markdown。
- `执行评测`：运行普通绝对评测，支持后台进度和断点续跑。
- `结果总览`：查看统计、稳定性对比、历史结果对比、CSV/Excel 导出。
- `样本详情`：查看单条样本、得分、comment、error_tags、规则引用和证据引用。
- `人工审核结果评估`：历史 GSB 双模型对比入口，当前主线暂不使用。
- `提示词改进建议`：基于绝对评测或 GSB 证据生成候选提示词修改建议。
- `裁判提示词AB对比`：固定 case 和模型配置，只比较两个 Judge Prompt。
- `闭环实验`：自动多轮提取、评测和提示词改进。
- `记忆提取`：单独运行 USER.md 或 MEMORY.md 提取，不进入闭环。

## 目录

- `pages/`：Streamlit 页面。
- `src/extraction/`：USER.md / MEMORY.md 记忆提取。
- `src/eval/`：Judge 调用、评测流程、结果校验、稳定性统计。
- `src/loop/`：闭环实验。
- `src/ui/`：任务状态、配置、页面支持、提示词改进。
- `src/loaders/`：Excel、JSONL、Markdown case 加载。
- `prompts/judge/`：裁判提示词。
- `prompts/generation/`：提取提示词。
- `rules/`：评分规则。
- `scripts/`：命令行工具、打包脚本和页面冒烟测试。
- `tests/`：回归测试。

## 测试

```powershell
python -m pytest -q
python scripts/smoke_pages.py
```

## 打包

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_release.ps1
```

脚本会先运行测试和页面冒烟测试，再构建 `dist/MemoryEvalUI/` 和 `dist/MemoryEvalUI.zip`。如果冒烟测试失败，发布应停止排查，不应继续分发。

## 协作

请从最新 `main` 创建功能分支，通过 Pull Request 合并。具体约定见 [CONTRIBUTING.md](CONTRIBUTING.md)。

项目包含内网接口说明和可能的业务规则，远端仓库建议设置为 Private。不要提交真实配置、上传数据、评测结果、日志或 API Token。
