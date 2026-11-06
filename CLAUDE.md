# CLAUDE.md — 如是 (rushi)

## 这是什么
一个「人格模拟器」：把一个人留在你心里的部分整理成**带时间线的记忆图谱**，然后
1. 以 ta 的语气聊天（风格引擎），
2. 回答「如果是 ta，会怎么做」（人格/决策引擎），
3. 把人格放进隔离的沙盒里演绎小说（不回写人格）。

灵感：《葬送的芙莉莲》——「如果是欣梅尔的话，也一定会这么做的」。
名字「如是」= 「如果是ta」+ 「如是我闻」（记下所听到的，如其所是）。

完整计划见 `docs/PLAN.md`；前期所有讨论和决策的摘要见 `docs/DESIGN_LOG.md`。**改架构前先读这两份。**

## 不可妥协的设计原则（改代码时必须守住）
1. **证据优先**：每条 claim 必须带可追溯的 quote；`extract.ground()` 校验不过 → 标 `ungrounded`、置信度减半、进审核队列。没有证据的推断不写入。
2. **时间是一等公民**：`valid_from/valid_to`（在 ta 人生里何时成立）+ `recorded_at`（何时得知）。信息变化 = 关闭旧边/旧 claim，**从不删除**。
3. **允许矛盾**：矛盾是数据，不是错误。进 `contradictions` 表，人格卡写成「一方面…另一方面…」，不调和。
4. **不美化**：所有 prompt 都要求保留不讨喜的特质；结构化决策路径只按证据权重算。
5. **分清哪边是 ta**：`speakers.json` 显式指定，永远不猜。`reporter` 决定证据类型：`user`/`third:*` 的材料永远不能变成 `self_report`（代码强制）。
6. **敏感维度**（J 童年/形成性经历）默认只在后台可见，不进聊天与决策。
7. **沙盒单向隔离**：小说沙盒是独立 db；沙盒内 base 以只读打开，写入即报错；没有自动回写。
8. **输出概率不输出断言**：决策引擎给分布 + 依据 + 不确定性 + 两条路径的一致度。

## 代码约定
- Python ≥ 3.10，**零第三方依赖**（标准库 + SQLite + urllib）。新依赖必须是可选的、带 fallback。
- 一个 workspace = 一个人格；`workspace/persona.db` 是核心资产。
- LLM 通过 `rushi/llm.py` 抽象：`anthropic` | `openai`（Ollama/vLLM 等本地）| `mock`（离线，只测管道）。所有 prompt 在 `prompts/*.md`，用首行的大写标记（如 `EXTRACT_CLAIMS`）路由；MockLLM 靠这个标记返回假数据。
- 记忆三层：常驻人格卡+风格规则 → 每轮检索（`retrieve.py`，char-bigram BM25，可插 `Embedder`，RRF 融合）→ 模型回复 `[search: …]` 主动查。
- 测试：`RUSHI_LLM=mock python -m unittest tests.test_pipeline -v`。改动后必须通过；新增功能要加进这个端到端测试。
- 人格维度在 `schema.DIMENSIONS`（A–Q）；加维度要同步 `viewer/index.html` 的 `DIM` 和 `prompts/extract.md`。

## 常用命令
```
rushi -w ws init | inspect-chat f | ingest-chat f | ingest-text f --reporter user|self|third:X
rushi -w ws add-fact "…" --dimension C --year 2018
rushi -w ws style | extract | coverage | review list | review confirmed <id> | reflect | persona
rushi -w ws chat --relation close_friend --period 2018-2020
rushi -w ws ask "情境…选项: A/B/C"
rushi -w ws export-graph   → viewer/index.html
rushi -w ws eval blind|score|backtest
rushi -w ws sandbox create --name x | sandbox scene --name x --scene … --other … --story
```

## 当前状态与下一步
已完成阶段 0–4 的单人格路径（可离线跑通）。未完成：
- `retrieve.Embedder` 的真实实现（bge-m3 等）
- 微信等平台的专用导出解析器
- 后台 Web 界面（现为 CLI + 静态 viewer）
- 联合分析式的价值权重主动校准（`decide._value_weights` 现在只是强度×置信度）
- 多人格沙盒的导演层 / 叙事层（现为最小两人场景循环）
- 用真实数据跑 `extract → review list` 评估抽取质量，并据此迭代 `prompts/extract.md`

## 伦理边界（产品层面）
纪念而非替代；界面要持续标明「这是模拟」；被模拟者的材料只在本地处理、脱敏后才发给云端模型；面向公众发布需遵守中国《人工智能拟人化互动服务管理暂行办法》与加州 SB 243（见 DESIGN_LOG）。
