# 如是 (rushi)

> 「如果是ta的话，也一定会这么做的。」
> 如是我闻 —— 记下你所听到的，如其所是。

把一个人留在你心里的那部分，整理成带时间线的记忆图谱；然后可以和「记忆中的ta」聊天，也可以问「如果是ta，会怎么做」。

**零依赖**：纯 Python 标准库（SQLite + urllib）。有 `ANTHROPIC_API_KEY` 就用 Claude；也可以接 Ollama/vLLM 等本地模型；没有任何 key 时用内置的 `mock` 模型跑通流程。

## 安装

```bash
cd rushi
pip install -e .          # 或者不安装，直接 python -m rushi.cli ...
export ANTHROPIC_API_KEY=sk-...        # 用 Claude
# 或者本地模型：export RUSHI_LLM=openai RUSHI_OPENAI_BASE=http://localhost:11434/v1 RUSHI_MODEL=qwen3:8b
```

也可以把这些写进配置文件：`cp .env.example .env` 后填写（`.env` 已 git-ignore；也可放在 `~/.rushi/.env`）。支持任何 OpenAI 兼容接口，如 MiniMax、DeepSeek。

## 数据与代码隔离

所有人格数据（`persona.db`、`speakers.json`、沙盒、导出、原始聊天记录）都放在**仓库之外**的 `$RUSHI_HOME`（默认 `~/.rushi`）：

```
~/.rushi/
  raw/                     原始聊天记录 / 叙事材料
  workspaces/<名字>/       一个人格一个 workspace：persona.db, speakers.json, sandboxes/, exports/
```

- `-w 名字` → `$RUSHI_HOME/workspaces/名字`；`-w ./路径` 或绝对路径则按原样使用。
- CLI **拒绝**在代码仓库内建 workspace（确需时设 `RUSHI_ALLOW_IN_REPO=1`）。
- `.gitignore` 兜底屏蔽 `*.db`、`speakers.json`（`examples/` 除外）、`exports/` 等。
- 仓库里只有 `examples/` 中虚构的样例人物「阿泽」，用于测试。

## 阶段 0–4：建立单一人格的完整路径

```bash
rushi -w ta init                                 # 0. 建 workspace（~/.rushi/workspaces/ta），生成 speakers.json 模板
rushi -w ta inspect-chat chat.txt                # 看聊天记录里有哪些说话人
#    编辑 ~/.rushi/workspaces/ta/speakers.json：target = ta 的名字；others = 其他人及其与 ta 的关系；aliases = 脱敏别名
rushi -w ta ingest-chat chat.txt                 # 1. 导入聊天（切会话、合并连发、脱敏、标记哪边是 ta）
rushi -w ta ingest-text about.md --reporter user # 1. 导入第三人称描述 / 访谈 / 日记（见格式说明）
rushi -w ta add-fact "他其实很怕狗" --dimension I --year 2018   # 随时补一条事实
rushi -w ta style                                # 2. 风格卡：统计 + 规则（版本化）
rushi -w ta extract                              # 3. 逐块抽取 → 带证据的观察、决策案例、图谱
rushi -w ta coverage                             # 哪些维度证据薄
rushi -w ta review list                          # 人工审核队列（敏感/未落地/矛盾 优先）
rushi -w ta review confirmed clm_xxx             #    confirmed | rejected | edited --text "改写后的说法"
rushi -w ta reflect                              # 3. 反思：跨观察综合 + 登记矛盾（从不删除）
rushi -w ta persona                              # 3. 核心人格卡（常驻上下文，默认排除敏感维度）
rushi -w ta chat --relation close_friend --period 2018-2020     # 4. 以 ta 的身份聊天
rushi -w ta ask "外地 offer 薪资高40% 但妈妈身体不好。选项: 接受/拒绝/拖着"   # 4. 如果是 ta 会怎么做
rushi -w ta export-graph                         # 导出 graph.json → 用 viewer/index.html 打开浏览
rushi -w ta eval blind -n 20                     # 风格盲测表；填好后 rushi eval score
rushi -w ta eval backtest                        # 决策回测（藏起已知决定看能否预测）
```

试跑（离线）：`RUSHI_LLM=mock python -m unittest tests.test_pipeline -v`，或用 `examples/` 里的样例走一遍上面的命令。

## 叙事材料的格式

```
--- year: 2016 ; relation: close_friend ; reporter: user ---
那年他刚搬去上海……

--- reporter: third:老张 ---
老张说他压力大的时候会消失两天……

--- reporter: self ---
我这个人吧，从小就不太会表达……
```

`reporter` 决定证据类型：`self` 可以是 self_report/behavior；`user` 和 `third:*` 只能是 other_report（代码强制，模型声明无效）。

## 设计要点（和计划一一对应）

| 原则 | 代码里的落实 |
|---|---|
| 分清哪边是 ta | `speakers.json` 显式指定，不猜；未知说话人会被提示 |
| 证据优先 | 每条 claim 必须带 quote；quote 会和原文比对（`extract.ground`），对不上就标 `ungrounded`、置信度减半、进审核队列 |
| 时间是一等公民 | claim/edge 都有 `valid_from/valid_to`（ta 人生里何时成立）+ `recorded_at`（我们何时得知）；变化=关闭旧边，不删除 |
| 允许矛盾 | `contradictions` 表 + claim 上的 `contradicts`；反思阶段专门找矛盾；人格卡要求「一方面…另一方面…」 |
| 人在变 | 风格卡按年份分统计；chat/ask 可指定 `--period` |
| 不美化 | 抽取/反思/人格卡/决策四个 prompt 都明确要求保留不讨喜的特质；结构化决策路径只按证据权重算 |
| 网状图谱 | 五层：原文 → 实体/事实 → 观察 → 特质 → 主题；特质之间有跨维度关系（导致/表现为/权衡/互相强化）；参考 Graphiti 的 episodic/entity/community 结构 |
| 推导可追溯 | 每个特质记录 `based_on` + `reasoning` + 推理类型（归纳/因果/对比）；viewer 点特质即可一路展开到原文 |
| 实体消解 | 「妈妈/我妈/母亲」合并为同一节点并保留别名；同一决定在不同材料里只记一个案例 |
| 置信度 | 由代码按证据数、原文段数、证据类型、时期、是否有矛盾计算；单段材料最多 moderate |
| 敏感维度 | J（童年/形成性经历）默认不进聊天与决策，只在后台可见（`--include-sensitive` 才开） |
| 三层记忆 | ① 常驻人格卡+风格规则 ② 每轮检索原话/记忆/观察（char-bigram BM25，可插入 embedding） ③ 模型可回复 `[search: …]` 主动查 |
| 小说不污染人格 | `sandbox create` 复制快照到独立 db；沙盒内 base 以只读打开，任何写入直接报错；没有自动回写，`promote()` 需手动且标记 `origin:fiction` |
| 评估 | 盲测（真实 vs 生成，50% = 分不出）；决策回测（准确率、给真实选择的平均概率、随机基线） |

## 目录

```
rushi/
├── rushi/
│   ├── schema.py      数据模型 + 17 个人格维度
│   ├── store.py       SQLite（从不删除；readonly 模式）
│   ├── ingest/chat.py 聊天解析（csv/json/txt）、会话与连发切分、说话人映射
│   ├── ingest/narrative.py 叙事材料切块、脱敏、reporter 规则
│   ├── llm.py         Anthropic / OpenAI 兼容(本地) / Mock
│   ├── extract.py     逐块抽取 + 证据落地校验 + 图谱写入
│   ├── style.py       风格统计 + 风格卡
│   ├── retrieve.py    离线混合检索（可插入 embedding，RRF 融合）
│   ├── persona.py     反思、矛盾、人格卡、审核队列、覆盖率
│   ├── chat.py        以 ta 聊天（三层记忆）
│   ├── decide.py      如果是 ta 会怎么做（结构化 + 叙事双路径）
│   ├── sandbox.py     小说沙盒（单向隔离）
│   ├── evaluate.py    盲测、回测
│   └── cli.py
├── prompts/           所有 prompt 都是可编辑的 .md
├── viewer/index.html  纯本地的图谱/观察浏览器
├── examples/          样例聊天、叙事、speakers.json
└── tests/test_pipeline.py
```

## 下一步（没做的）

- 向量检索：实现 `retrieve.Embedder`（如 bge-m3），`Retriever(store, embedder=...)` 即可混合检索
- 微信等平台的专用解析器（现在是通用 csv/json/txt）
- 后台 Web 界面：目前是 CLI + 静态查看器
- 结构化决策路径里的价值权重目前由 claim 强度×置信度推导，还没有联合分析式的主动提问来校准
- 多人格沙盒目前只有「ta + 一个描述出来的角色」的最小场景循环，导演层/叙事层只有雏形
