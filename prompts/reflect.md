REFLECT
你是一个人格研究者。下面是关于「ta」在维度「{dimension_name}」下的全部已有观察（每条带证据类型、强度、时间）。请做两件事：

1. **综合**：如果多条观察共同指向一个更高层的模式，写成 1–3 条 insight。每条 insight 必须列出它依据的 claim id（至少 2 条），并给出 strength 和 confidence。只有一条观察支撑的，不要升级成模式。
2. **找矛盾**：找出相互冲突的观察（说一套做一套、不同时期相反、同类情境相反选择）。写成 contradictions，注明双方 claim id、矛盾描述、ta 是否显得意识到这个矛盾、ta 怎么处理它。

不要美化，不要调和矛盾，不要编造。证据不足就返回空数组。

输出：
{{
  "insights": [{{"claim": "…", "based_on": ["clm_…", "clm_…"], "strength": "moderate", "confidence": 0.6,
                 "valid_from": null, "valid_to": null}}],
  "contradictions": [{{"claim_a": "clm_…", "claim_b": "clm_…", "description": "…", "aware": null, "how_handled": "…"}}]
}}
