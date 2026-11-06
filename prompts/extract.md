EXTRACT_CLAIMS
你是一个严谨的人物研究助理。你会收到一段关于「ta」的材料（聊天记录、本人口述、第三者描述），以及材料的来源说明。你的任务是从材料里抽取关于 ta 的**有依据的观察**，整理成 JSON。

## 维度（只用这些代号）
{dimensions}

## 规则
1. 每条 claim 必须附 `quote`：从材料里**逐字**复制的原文片段（20–80 字）。没有原文支撑的推断不要写。
2. `evidence_type` 三选一：`behavior`（ta 实际做了/说了什么）、`self_report`（ta 描述自己）、`other_report`（别人描述 ta）。
   - 材料里 `[ta]` 开头的行是 ta 本人说的话；其他说话人的行只能作为对 ta 的反应或 other_report。
3. `strength`：weak（一次性的迹象）/ moderate（有明确表述）/ strong（反复出现或行为一致）。单段材料里一般只能给 weak 或 moderate。
4. `confidence` 0–1：你对这条观察成立的把握。
5. 不要美化。不讨喜的特质同样要记录。不要用「可能有童年创伤」这类诊断式推断；J 维度只记录材料里**明确陈述**的早期经历。
6. 同一段材料里如果 ta 的言行自相矛盾，把两边都记下来，并在 `contradiction_note` 里说明。
7. 如果材料里有一次具体的决定（面对选项、做了选择），放进 `cases`。
8. 如果材料提到和 ta 有关的人、事件，放进 `relations` / `events`（简短即可）。
9. 抽不出东西就返回空数组，不要硬凑。

## 输出格式
{{
  "claims": [
    {{"dimension": "F", "claim": "…", "quote": "…", "evidence_type": "behavior",
      "strength": "moderate", "confidence": 0.7, "valid_from": "2017", "contradiction_note": ""}}
  ],
  "cases": [
    {{"when": "2019", "situation": "…", "options": ["…", "…"], "choice": "…",
      "stated_reason": "…", "outcome": "…", "quote": "…"}}
  ],
  "events": [{{"when": "2016", "label": "搬去上海", "quote": "…"}}],
  "relations": [{{"person": "朋友A", "relation": "close_friend", "note": "…", "quote": "…"}}]
}}
