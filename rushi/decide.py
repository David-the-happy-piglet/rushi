"""Decision engine — 如果是 ta，会怎么做.

1. parse the situation into options
2. retrieve: similar past decisions, relevant claims (values, patterns, contradictions)
3. structured path: value weights × LLM-judged impacts → softmax probabilities (transparent)
4. narrative path: first-person reasoning with retrieved memories → probabilities (flexible)
5. combine; disagreement between paths is reported as uncertainty, not hidden
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

from .extract import load_prompt
from .llm import LLM
from .retrieve import Retriever
from .schema import SENSITIVE_DIMENSIONS
from .store import Store

VALUE_DIMENSIONS = {"C", "E", "H", "N"}   # values, morality, money/risk, fears & desires


@dataclass
class Decision:
    situation: str
    options: list[str]
    structured: dict[str, float]
    narrative: dict[str, float]
    combined: dict[str, float]
    reasoning: str
    would_say: str
    evidence: list[str]
    uncertainty: str
    agreement: float               # 1 = both paths agree fully
    values_used: list[tuple[str, float]] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"情境：{self.situation}", ""]
        lines.append("各选项概率（综合 / 结构化 / 叙事）：")
        for o in self.options:
            lines.append(f"  - {o}: {self.combined[o]:.0%}  ({self.structured.get(o, 0):.0%} / {self.narrative.get(o, 0):.0%})")
        lines += ["", f"两条推理路径一致度：{self.agreement:.0%}" + ("  ← 分歧较大，谨慎对待" if self.agreement < 0.6 else "")]
        if self.values_used:
            lines += ["", "结构化路径用到的价值权重："] + [f"  - {v} ({w:+.2f})" for v, w in self.values_used[:8]]
        lines += ["", "第一人称推理：", self.reasoning, "", f"ta 可能会说：{self.would_say}"]
        if self.evidence:
            lines += ["", "引用的证据：" + ", ".join(self.evidence[:10])]
        lines += ["", f"不确定性：{self.uncertainty}"]
        return "\n".join(lines)


def _softmax(scores: dict[str, float], temp: float = 1.0) -> dict[str, float]:
    if not scores:
        return {}
    m = max(scores.values())
    exps = {k: math.exp((v - m) / temp) for k, v in scores.items()}
    z = sum(exps.values()) or 1.0
    return {k: v / z for k, v in exps.items()}


def _value_weights(store: Store, retriever: Retriever, query: str, k: int = 8) -> list[tuple[str, float]]:
    """Derive value weights from claims: strength × confidence, boosted when user-confirmed."""
    docs = retriever.knowledge(query, dimensions=VALUE_DIMENSIONS, k=k * 2)
    weights = []
    smap = {"weak": 0.5, "moderate": 0.8, "strong": 1.0}
    for d in docs:
        if d.meta.get("kind") != "claim":
            continue
        c = store.get_claim(d.id)
        if not c:
            continue
        w = smap.get(c.strength, 0.5) * c.confidence * (1.2 if c.user_review == "confirmed" else 1.0)
        weights.append((c.claim, round(w, 2)))
    return weights[:k]


def decide(store: Store, llm: LLM, situation: str, period: Optional[tuple[int, int]] = None,
           retriever: Optional[Retriever] = None, include_sensitive: bool = False) -> Decision:
    r = retriever or Retriever(store)
    parsed = llm.complete_json(load_prompt("decide_parse"), f"<situation>\n{situation}\n</situation>")
    options = [str(o) for o in (parsed.get("options") or []) if str(o).strip()] or ["做", "不做"]
    query = situation + " " + " ".join(parsed.get("values_touched") or [])

    exclude = set() if include_sensitive else set(SENSITIVE_DIMENSIONS)
    knowledge = r.knowledge(query, exclude=exclude, year_range=period, k=12)
    cases = [d for d in knowledge if d.meta.get("kind") == "case"]
    claims = [d for d in knowledge if d.meta.get("kind") == "claim"]
    contradictions = store.contradictions()
    con_text = "\n".join(f"- {c.description}" for c in contradictions[:5]) or "- 无"

    # ---- structured path
    values = _value_weights(store, r, query)
    structured: dict[str, float] = {}
    if values:
        user = ("<options>\n" + "\n".join(f"- {o}" for o in options) + "\n</options>\n<values>\n"
                + "\n".join(f"- {v} (权重 {w})" for v, w in values) + "\n</values>")
        try:
            imp = llm.complete_json(load_prompt("decide_score"), user).get("impacts", {})
            scores = {}
            for o in options:
                row = imp.get(o, {}) if isinstance(imp, dict) else {}
                scores[o] = sum(float(row.get(v, 0.0) or 0.0) * w for v, w in values)
            structured = _softmax(scores, temp=max(0.5, sum(w for _, w in values) / 3))
        except Exception as e:  # noqa: BLE001
            store.log("decide_struct_error", str(e))
    if not structured:
        structured = {o: 1 / len(options) for o in options}

    # ---- narrative path
    card = (store.artifact("persona_card") or {}).get("markdown", "（无人格卡）")
    user = (f"# 人格卡\n{card}\n\n# 相关的过去决定\n" + ("\n".join(d.payload for d in cases) or "- 无")
            + "\n\n# 相关观察\n" + ("\n".join(f"{d.id} {d.payload}" for d in claims) or "- 无")
            + "\n\n# 图谱推导链（特质 ← 观察 ← 原文）\n"
            + ("\n".join(r.graph_context(query, include_sensitive=include_sensitive, year_range=period)) or "- 无")
            + f"\n\n# 已登记的矛盾\n{con_text}\n\n# 情境\n{situation}\n\n<options>\n"
            + "\n".join(f"- {o}" for o in options) + "\n</options>")
    try:
        nar = llm.complete_json(load_prompt("decide_narrative"), user)
    except Exception as e:  # noqa: BLE001
        store.log("decide_narr_error", str(e))
        nar = {}
    probs = nar.get("probabilities") or {}
    narrative = {o: float(probs.get(o, 0.0) or 0.0) for o in options}
    z = sum(narrative.values())
    narrative = {o: (v / z if z else 1 / len(options)) for o, v in narrative.items()}

    # ---- combine: weight narrative higher when there is real case evidence, structured when values are rich
    w_n = 0.6 if cases else 0.5
    combined = {o: w_n * narrative[o] + (1 - w_n) * structured[o] for o in options}
    z = sum(combined.values()) or 1.0
    combined = {o: v / z for o, v in combined.items()}
    agreement = 1 - 0.5 * sum(abs(narrative[o] - structured[o]) for o in options)

    dec = Decision(situation=situation, options=options, structured=structured, narrative=narrative,
                   combined=combined, reasoning=str(nar.get("reasoning", "")), would_say=str(nar.get("would_say", "")),
                   evidence=[str(e) for e in (nar.get("evidence_used") or [])] + [d.id for d in cases[:3]],
                   uncertainty=str(nar.get("uncertainty", "")), agreement=round(agreement, 2), values_used=values)
    store.log("decide", situation[:80])
    store.commit()
    return dec
