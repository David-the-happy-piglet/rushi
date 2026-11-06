"""Chat as ta. Three memory layers:
1. always-on core persona card + style rules (artifacts)
2. retrieved examples (voice) and memories (facts) for the current message
3. an explicit memory search the model can request by replying `[search: ...]`
   (a one-step tool loop, simple enough to work with any model)
"""
from __future__ import annotations

import re
from typing import Optional

from .extract import load_prompt
from .llm import LLM
from .retrieve import Retriever
from .schema import SENSITIVE_DIMENSIONS
from .store import Store

SEARCH_RE = re.compile(r"^\s*\[search:\s*(.+?)\]\s*$", re.S)


class ChatSession:
    def __init__(self, store: Store, llm: LLM, relation: str = "unknown",
                 period: Optional[tuple[int, int]] = None, retriever: Optional[Retriever] = None):
        self.store, self.llm, self.relation, self.period = store, llm, relation, period
        self.retriever = retriever or Retriever(store)
        card = store.artifact("persona_card") or {}
        style = store.artifact("style_card") or {}
        self.persona_card = card.get("markdown", "（尚未生成人格卡；运行 rushi persona）")
        self.style_rules = style.get("rules", "（尚未生成风格规则；运行 rushi style）")
        self.history: list[tuple[str, str]] = []

    def _context(self, message: str) -> tuple[str, str]:
        q = message + " " + " ".join(m for _, m in self.history[-2:])
        examples = self.retriever.examples(q, relation=self.relation, year_range=self.period, k=8)
        mem_docs = self.retriever.knowledge(q, exclude=SENSITIVE_DIMENSIONS, year_range=self.period, k=6)
        memories = [d.payload for d in mem_docs] + self.retriever.memories(q, year_range=self.period, k=3)
        return "\n\n".join(examples) or "（无）", "\n".join(memories) or "（无）"

    def reply(self, message: str, allow_search: bool = True) -> str:
        examples, memories = self._context(message)
        period = f"{self.period[0]}–{self.period[1]}" if self.period else "最近有证据的时期"
        history = "\n".join(f"[{'ta' if r == 'ta' else '对方'}] {m}" for r, m in self.history[-8:])
        prompt = (load_prompt("chat")
                  .replace("{relation}", self.relation).replace("{period}", period)
                  .replace("{persona_card}", self.persona_card).replace("{style_rules}", self.style_rules)
                  .replace("{examples}", examples).replace("{memories}", memories)
                  .replace("{message}", (history + "\n" if history else "") + f"[对方] {message}"))
        tool_note = ("\n\n如果对方提到一件你需要查记忆才能回答的事，先只回复一行 `[search: 关键词]`，"
                     "你会收到检索结果后再正式回复。" if allow_search else "")
        out = self.llm.complete(prompt + tool_note, "请以 ta 的方式回复。", max_tokens=400, temperature=0.8)
        m = SEARCH_RE.match(out)
        if m and allow_search:
            extra = self.retriever.memories(m.group(1), k=4) + [d.payload for d in self.retriever.knowledge(m.group(1), exclude=SENSITIVE_DIMENSIONS, k=4)]
            prompt2 = prompt.replace("{memories}", "\n".join(extra) or "（没有找到相关记忆，按 ta 的方式含糊带过）")
            out = self.llm.complete(prompt2, "请以 ta 的方式回复。", max_tokens=400, temperature=0.8)
        out = out.strip()
        self.history.append(("other", message))
        self.history.append(("ta", out))
        return out
