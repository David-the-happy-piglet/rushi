REFLECT
你是一个人格研究者。下面是关于「ta」的**全部维度**的已有观察（每条带 id、维度、证据类型、强度、时间、原文），以及已确认的特质和决策案例。
你的任务是从这些观察出发，**一步步推出**更高层的特质，并找出特质之间**跨维度**的联系。所有自然语言字段用简体中文。

## 维度
{dimensions}

## 1. 特质 `traits`
- 每条特质必须由**至少 2 条**观察（或已有特质）支撑，写在 `based_on` 里（用给出的 id，或本次输出里更早的特质 key）。只有一条观察支撑的，不要升级成特质。
- `reasoning`：用一两句话写出**推导过程**——这几条观察分别说明了什么、为什么合起来能得出这个特质。不要只是复述结论。
- `inference` 三选一：`generalization`（多次同类言行 → 稳定倾向）、`causal`（一个经历/信念 → 解释另一类行为）、`contrast`（说的和做的、不同情境下的差异本身就是特质）。
- `dimension`：这个特质最主要属于哪个维度。可以跨维度综合，例如用 J、O、F 的观察推出一个 O 维度的特质。
- `shown_in`：这个特质在哪些决策案例里体现（案例 id），没有就空数组。
- 时间：如果特质只在某段时期成立，写 `valid_from/valid_to`；不确定就 null。
- **不要美化**。冷淡、回避、固执、自相矛盾这类特质同样要写出来，不要改写成「其实很温柔」。

## 2. 特质之间的关系 `links`
- 找出**不同特质之间**的关系，优先跨维度。`rel` 只能从这里选：{trait_rels}
- 每条关系要有 `explanation`（为什么有这层关系）和 `based_on`（支撑这层关系的观察 id，至少 1 条）。
- 不要为了连线而连线；证据不足就不写。

## 3. 矛盾与变化 `contradictions`
- 找出相互冲突的观察或特质，写清双方 id（`a`、`b`）。
- `kind` 二选一：
  - `contradiction`：**同一时期**说一套做一套、同类情境做出相反选择。两边都保留，不要调和。
  - `change`：**不同时期**态度或行为变了（例如以前不道歉、后来会主动道歉）。
- `aware`：ta 是否显得意识到这个矛盾（true/false/null）；`how_handled`：ta 怎么处理它。

证据不足就返回空数组，不要编造。

## 输出格式
{{
  "traits": [{{"key": "t1", "dimension": "O", "claim": "…", "based_on": ["clm_…", "clm_…"],
               "reasoning": "…", "inference": "causal", "valid_from": null, "valid_to": null, "shown_in": []}}],
  "links": [{{"from": "t1", "to": "t2", "rel": "CAUSES", "explanation": "…", "based_on": ["clm_…"]}}],
  "contradictions": [{{"a": "clm_…", "b": "t2", "kind": "contradiction", "description": "…", "aware": null, "how_handled": "…"}}]
}}
