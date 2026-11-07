"""Confidence is computed by code, not taken from the model.

LLMs report 0.8–0.95 for almost everything. Here confidence comes from things we can count:
strength label, evidence type, whether the quote is grounded, and — for traits — how many
independent episodes / voices / periods support it. The model's own number is kept only as a
minor input (observations) and recorded for audit.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

STRENGTH_BASE = {"weak": 0.4, "moderate": 0.6, "strong": 0.75}
EVIDENCE_FACTOR = {"behavior": 1.0, "self_report": 0.9, "other_report": 0.8}


def score_observation(strength: str, evidence_type: str, grounded: bool,
                      llm_confidence: Optional[float] = None) -> float:
    base = STRENGTH_BASE.get(strength, 0.4) * EVIDENCE_FACTOR.get(evidence_type, 0.8)
    if llm_confidence is not None:
        base = 0.75 * base + 0.25 * min(max(float(llm_confidence), 0.0), 0.9)
    if not grounded:
        base *= 0.5
    return round(min(base, 0.9), 2)


def _year(s) -> Optional[str]:
    m = re.match(r"(\d{4})", str(s or ""))
    return m.group(1) if m else None


def score_trait(supports: Iterable, contradicted: bool = False) -> tuple[str, float, dict]:
    """supports: Claim objects (observations or sub-traits). Returns (strength, confidence, breakdown)."""
    supports = list(supports)
    episodes, voices, periods, confs = set(), set(), set(), []
    for c in supports:
        confs.append(c.confidence)
        for e in c.evidence:
            episodes.add(e.ref_id)
            voices.add(e.evidence_type)
        y = _year(c.valid_from)
        if y:
            periods.add(y)
    miss = 1.0
    for x in confs:
        miss *= 1 - 0.5 * x                       # discounted noisy-OR
    conf = 1 - miss
    n_ep = len(episodes)
    cap = 0.55 if n_ep <= 1 else 0.75 if n_ep == 2 else 0.9
    conf = min(conf, cap)
    if len(voices) >= 2:
        conf = min(conf + 0.05, 0.9)              # independent kinds of evidence agree
    if contradicted:
        conf *= 0.85
    if n_ep >= 3 and (len(periods) >= 2 or len(voices) >= 2):
        strength = "strong"
    elif n_ep >= 2:
        strength = "moderate"
    else:
        strength = "weak"
    breakdown = {"n_supports": len(supports), "n_episodes": n_ep, "evidence_types": sorted(voices),
                 "periods": sorted(periods), "contradicted": contradicted, "cap": cap}
    return strength, round(conf, 2), breakdown
