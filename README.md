# MemoryEvalUI

MemoryEvalUI 是一个面向 USER.md 记忆提取的本地评测工具，覆盖数据导入、记忆提取、绝对评测、结果分析、人工复核、提示词改进和闭环实验。

## 主要能力

- 从 Excel 数据运行 USER.md 记忆提取
- 生成并管理绝对评测 case
- 调用 Judge 模型执行结构化评分
- 查看维度分、错误标签、规则引用和稳定性对比
- 基于评测证据生成增量提示词修改建议
- 串联“提取、评测、改进、再提取”的闭环实验
- 保留历史 GSB 双模型评估入口

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

在 `config/local_config.json` 中填写实际 API 地址、Token 和模型名称。该文件已被 Git 忽略，禁止提交真实 Token。

## 启动

```powershell
streamlit run app.py
```

也可以运行命令行入口：

```powershell
python run.py
```

## 测试

```powershell
python -m pytest -q
```

## 目录

- `pages/`：Streamlit 页面
- `src/extraction/`：记忆提取
- `src/eval/`：Judge 调用与评测
- `src/loop/`：闭环实验
- `src/ui/`：任务状态、配置和页面支持
- `prompts/`：裁判与提取提示词
- `rules/`：评分规则
- `tests/`：回归测试

## 协作

请从 `main` 创建功能分支，通过 Pull Request 合并。具体约定见 [CONTRIBUTING.md](CONTRIBUTING.md)。

项目包含内网接口说明，远端仓库建议设置为 Private。
