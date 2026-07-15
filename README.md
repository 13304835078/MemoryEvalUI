# MemoryEvalUI

MemoryEvalUI 是一个面向 USER.md 用户画像与 MEMORY.md 长期记忆的本地评测工具，覆盖数据导入、记忆提取、绝对评测、结果分析、人工复核、提示词改进和闭环实验。

## 主要能力

- 运行 USER.md 用户画像提取，并将提取结果转换为标准评测 case。
- 运行 MEMORY.md 长期记忆提取，并将提取结果转换为 `long_memory` 评测 case。
- 上传已有 USER.md / MEMORY.md 提取结果 Excel，生成可复用的 case 文件。
- 分别管理裁判提示词与提取提示词：`prompts/judge/` 存放 Judge Prompt，`prompts/generation/` 存放提取 Prompt。
- 调用 Judge 模型进行绝对评分，输出维度分、总分、comment、error_tags、diagnostics、rule_refs、evidence_refs、output_refs 和可选 reasoning_refs。
- 将 API/网络/JSON 解析失败标记为“未评分”，与 Judge 判定的严重质量错误分离，避免运行失败按 0 分污染统计。
- 在结果总览中分别查看运行有效性、提取覆盖率、条件质量分、端到端分数、稳定性对比和历史结果对比。
- 在样本详情中查看单条 case 的对话、旧/新 USER.md 或 MEMORY.md、得分、评语和引用证据。
- 基于评测证据后台生成受约束的提示词改进建议，优先做增量修改，避免提示词无限膨胀。
- 通过任务中心集中查看执行评测、记忆提取、闭环实验、提示词建议和裁判提示词 A/B 对比的后台任务状态。
- 多个后台任务可同时运行；同一 API/Token 会共用全局请求启动间隔，降低不同功能叠加超 QPS 的风险。
- 提供 Discovery / Validation / Locked Test 隔离的可信闭环，候选只有通过 Validation 门槛后才会进入下一轮。
- 通过“工号 + 姓名”划分独立工作区，隔离每位使用者的配置、提示词、上传文件、任务和结果。
- 长任务由独立后台进程执行，Streamlit 页面重跑或切换页面不会终止任务。

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

首次进入需填写工号和姓名。该步骤用于工作区识别和数据隔离，不是密码认证；公司 VM 仍应通过 VPN、反向代理或统一身份认证限制访问。

## 常用流程

### USER.md 绝对评测

1. 在“配置”页填写 API，选择 `user_md_update`、USER.md 裁判提示词和 USER.md 提取提示词。
2. 如果输入是原始对话 Excel，在“记忆提取”页运行 USER.md 提取并自动生成 case；如果已经有 run_user.py 输出，则在“评测数据”页直接转换。
3. 在“执行评测”页选择同一任务类型和提示词，启动评测。
4. 在“结果总览”和“样本详情”先确认运行完整，再查看条件分、端到端分数和引用证据。

Judge 的 `evidence_refs` 只能引用旧文档或用户对话；候选文档由 `output_refs` 引用，模型 reasoning 只能放在 `reasoning_refs` 中做过程诊断，不能作为用户事实来源。结果会记录 Judge Prompt Hash、评分协议版本、权重版本和评分配置 Hash，便于判断两次分数是否可直接比较。

### MEMORY.md 绝对评测

1. 在“配置”页选择 `long_memory`，使用 `judge_long_memory_v1.md` 和 `extract_long_memory_v1.yaml`。
2. 在“记忆提取”页选择“长期记忆 MEMORY.md”，上传原始对话 Excel 并运行提取。
3. 提取完成后可自动生成 long_memory case，再进入“执行评测”页评分。
4. 也可以在“评测数据”页上传已有长期记忆提取结果 Excel，结果列支持 `MEMORY.md` 或 `生成的MEMORY.md正文`。

记忆提取正文解析兼容 JSON、`# Output` / `# 输出`、`*输出*`、文档分隔线和结构化 Markdown；解析会保留原始章节结构，并把解析方式、置信度和告警写入结果。未可靠识别正文边界但存在非空原始输出时，会保留为“待复核”的低置信候选并可生成复核 case，但不会继承到后续 chunk；后续 chunk 继续使用最近一次可靠正文，避免一次解析异常污染整条时序链。只有 API 失败、任务终止或输出确实为空时才进入漏抽 case。

新版提取结果将处理状态拆分为 `call_status`、`parse_status`、`case_status`，并分别保存 `raw_output`、`parsed_document`、`effective_document` 和 `inheritance_source`。原有 `status`、`result`、`user.md`、`MEMORY.md` 列继续保留，因此旧数据和现有闭环流程仍可使用。USER.md 与 MEMORY.md 的列映射、输入格式和继承策略统一由任务 Profile 管理。

### 闭环实验

推荐的“可信闭环”会自动执行：

```text
固定按评测人的完整跨-session历史切分
  -> Discovery：提取/评测/生成候选草稿
  -> Validation：当前版本与候选版本对比并执行替换门槛
  -> 通过后晋升候选并进入下一轮
  -> Locked Test：迭代结束后只读取一次，生成最终报告
```

支持 USER.md 和 MEMORY.md。可信模式结构上至少需要 3 位不同评测人；默认统计门槛要求 Validation 至少 2 位，因此实际至少需要 4 位。切分器会按当前门槛预留评测人。同一评测人的全部 session 只会进入一个集合，并保持原始时间顺序。Judge 模型、裁判提示词、解码参数和初始提取规则在一次运行内冻结；候选提取 Prompt 只负责生成候选输出，不能同时改写自己的评分规则。Prompt Advisor 只能看到 Discovery，不能看到 Validation 或 Locked Test。未验证候选只保存在轮次目录，通过 Validation 后才另存到 `prompts/generation/`。旧的单集合流程保留为“探索兼容模式”，只适合快速实验。

Validation 默认检查同一 case 的配对分差、端到端分数、提取覆盖率、单样本退化率、关键错误标签和提示词增长比例。系统按评测人/时序簇做确定性 Bootstrap；默认至少需要 8 个配对 case、2 个独立簇，且 95% 置信区间下界高于 0 才允许晋升。任一侧存在提取接口失败或 Judge 运行失败时，比较标记为不完整并禁止自动替换。

执行评测的断点续跑会校验样本正文、reasoning、模型接口、裁判/提取提示词、评分协议和解码参数组成的完整指纹；只有完全一致的旧结果才会复用。接口支持时可开启 Prompt Cache，减少重复固定提示词的服务端计算，但它不会缩短请求上下文本身。

## 页面说明

- `配置`：API、模型参数、并发、重试、限流、裁判提示词、提取提示词。
- `记忆提取`：接收原始对话 Excel，运行 USER.md 或 MEMORY.md 提取，并可自动生成评测 case。
- `评测数据`：加载已有 case，上传已有 USER.md / MEMORY.md 提取结果或通用 Excel/JSONL/Markdown；不调用提取模型。
- `执行评测`：运行普通绝对评测，支持独立后台进程、后台进度和完整指纹断点续跑。
- `结果总览`：分层查看运行失败、提取完整度和质量分，并做评分配置感知的稳定性对比、历史对比和导出。
- `样本详情`：查看单条样本、得分、comment、error_tags、规则引用和证据引用。
- `任务中心`：集中查看后台任务状态，可对后台任务发停止请求；已发出的单次 API 调用会在返回后停止后续任务。
- `提示词改进建议`：基于普通绝对评测证据后台生成候选提示词修改建议。
- `裁判提示词AB对比`：固定 case 和模型配置，只比较两个 Judge Prompt，支持后台运行。
- `闭环实验`：可信模式执行 Discovery/Validation/Locked Test；探索模式保留旧的单集合多轮流程。

## 目录

- `pages/`：Streamlit 页面。
- `src/extraction/`：USER.md / MEMORY.md 记忆提取。
- `src/eval/`：Judge 调用、评测流程、结果校验、稳定性统计。
- `src/loop/`：闭环编排、评测人完整时序切分、Validation 统计门槛和可信协议。
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
