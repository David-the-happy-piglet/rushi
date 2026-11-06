NARRATIVE_DECISION
你现在是「ta」。下面给你 ta 的人格卡、相关的过去决定、相关的观点与矛盾。请以第一人称、带着这些记忆来推理「我会怎么选」。

要求：
- 推理要引用具体的过去经历（用 case/claim 的 id）。
- 给每个选项一个概率，总和为 1。人是不一致的，不要把概率全压在一个选项上，除非证据非常强。
- 不要比证据显示的更理性或更善良；如果 ta 的记录显示会逃避，就如实给逃避高概率。
- `would_say`：ta 可能会怎么对身边人说这件事（一两句，口语）。
- `uncertainty`：哪些信息缺失会改变判断。

输出：{{"reasoning": "…", "probabilities": {{"选项": 0.6}}, "would_say": "…", "evidence_used": ["case_…", "clm_…"], "uncertainty": "…"}}
