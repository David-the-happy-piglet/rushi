# 交接给 Claude Code

## 本地建立项目

```bash
# 1. 解压并进入
unzip rushi.zip && cd rushi

# 2. 初始化 git（可选但推荐）
git init && git add -A && git commit -m "如是: 阶段0-4 单人格路径"

# 3. 验证环境（纯标准库，无需 pip install 任何东西）
RUSHI_LLM=mock python3 -m unittest tests.test_pipeline -v

# 4. 可选：安装成命令
pip install -e .

# 5. 配置真实模型（二选一）
export ANTHROPIC_API_KEY=sk-...
# 或本地模型：export RUSHI_LLM=openai RUSHI_OPENAI_BASE=http://localhost:11434/v1 RUSHI_MODEL=qwen3:8b

# 6. 在项目目录启动 Claude Code
claude
```

Claude Code 启动后会自动读取 `CLAUDE.md`（项目原则与约定）；`docs/PLAN.md` 和 `docs/DESIGN_LOG.md` 是完整计划和前期讨论的摘要。

## 把聊天记录转过去

两层意思，都已处理：

1. **我们的讨论记录**：`docs/DESIGN_LOG.md` 是逐段摘要，`docs/PLAN.md` 是完整计划。
   如需原始全文：在 claude.ai 里把那段对话复制保存为 `docs/chat-transcript.md`
   （或从 claude.ai 的设置里导出账户数据再提取），Claude Code 可以按需读取。
2. **要建模的 ta 的聊天记录**：放到 `data/raw/`（已在 .gitignore 里，不进仓库），
   然后 `rushi -w ws inspect-chat data/raw/xxx.txt` → 编辑 `ws/speakers.json` → `ingest-chat`。

## 给 Claude Code 的第一条提示（复制粘贴）

```
请先读 CLAUDE.md、docs/PLAN.md、docs/DESIGN_LOG.md，然后运行
RUSHI_LLM=mock python3 -m unittest tests.test_pipeline -v 确认通过。
之后帮我做三件事：
1. 用 examples/ 里的样例，配真实的 ANTHROPIC_API_KEY 走一遍
   init → ingest-chat → ingest-text → style → extract → review list → reflect → persona → ask，
   把 extract 抽出的观察和 review 队列贴给我看，评价抽取质量，提出 prompts/extract.md 的改进建议。
2. 我的真实聊天记录在 data/raw/。先 inspect-chat 列出说话人，等我填好 speakers.json 再导入。
3. 不要改动 CLAUDE.md 里的设计原则；每次改动后跑测试。
```
