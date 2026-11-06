"""Persona engine: from grounded claims to reflections, contradictions and the core card.

- reflect(): per dimension, synthesise higher-level insights (kind="reflection") that cite ≥2 claims,
  and register contradictions. Never deletes anything.
- build_persona_card(): compress confirmed / high-confidence knowledge into a short card that lives
  in the chat context. Sensitive dimensions are excluded unless include_sensitive=True.
- review helpers for the human-in-the-loop queue.
"""
from __future__ import annotations

from typing import Optional

from .extract import load_prompt
from .llm import LLM
from .schema import DIMENSIONS, SENSITIVE_DIMENSIONS, Claim, Contradiction, Evidence, stable_id
from .store import Store


def _claim_line(c: Claim) -> str:
    ev = c.evidence[0] if c.evidence else None
    et = ev.evidence_type if ev else "?"
    q = f"「{ev.quote[:60]}」" if ev and ev.quote else ""
    return f"- {c.id} [{c.strength}/{et}/conf {c.confidence}/{c.valid_from or '?'}–{c.valid_to or ''}/{c.user_review}] {c.claim} {q}"


def reflect(store: Store, llm: LLM, dimensions: Optional[list[str]] = None, min_claims: int = 3) -> dict:
    """Run the reflection prompt for each dimension with enough material."""
    out = {"insights": 0, "contradictions": 0}
    for dim in dimensions or list(DIMENSIONS):
        claims = [c for c in store.claims(dimension=dim) if c.kind != "reflection"]
        if len(claims) < min_claims:
            continue
        system = load_prompt("reflect").replace("{dimension_name}", f"{dim} {DIMENSIONS[dim]}")
        user = "\n".join(_claim_line(c) for c in claims)
        try:
            data = llm.complete_json(system, user)
        except Exception as e:  # noqa: BLE001
            store.log("reflect_error", f"{dim}: {e}")
            continue
        ids = {c.id for c in claims}
        for ins in data.get("insights", []) or []:
            based = [b for b in (ins.get("based_on") or []) if b in ids]
            if len(based) < 2 or not ins.get("claim"):
                continue
            ev = []
            for b in based:
                src = store.get_claim(b)
                if src and src.evidence:
                    ev.append(src.evidence[0])
            claim = Claim(id=stable_id("clm", "reflect", dim, str(ins["claim"])[:60]), dimension=dim,
                          claim=str(ins["claim"]), strength=ins.get("strength", "moderate"),
                          confidence=float(ins.get("confidence", 0.6)), valid_from=ins.get("valid_from"),
                          valid_to=ins.get("valid_to"), evidence=ev, kind="reflection", tags=["based_on:" + ",".join(based)])
            store.add_claim(claim)
            out["insights"] += 1
        for con in data.get("contradictions", []) or []:
            a, b = con.get("claim_a"), con.get("claim_b")
            if a in ids and b in ids and a != b:
                store.add_contradiction(Contradiction(id=stable_id("con", a, b), claim_a=a, claim_b=b,
                                                      description=str(con.get("description", "")),
                                                      aware=con.get("aware"), how_handled=str(con.get("how_handled", ""))))
                for cid in (a, b):
                    c = store.get_claim(cid)
                    if c and cid not in c.contradicts:
                        c.contradicts.append(b if cid == a else a)
                        store.update_claim(c)
                out["contradictions"] += 1
    store.log("reflect", str(out))
    store.commit()
    return out


def knowledge_lines(store: Store, include_sensitive: bool = False, min_confidence: float = 0.5,
                    pending_ok: bool = True) -> list[str]:
    lines = []
    for dim in DIMENSIONS:
        if dim in SENSITIVE_DIMENSIONS and not include_sensitive:
            continue
        for c in store.claims(dimension=dim):
            if c.user_review == "rejected":
                continue
            if c.user_review == "pending" and (not pending_ok or c.confidence < min_confidence):
                continue
            lines.append(f"- [{dim}] {c.claim} （{c.strength}, conf {c.confidence}, {c.valid_from or '?'}{'–' + c.valid_to if c.valid_to else ''}, {c.user_review}）")
    return lines


def build_persona_card(store: Store, llm: LLM, include_sensitive: bool = False) -> dict:
    lines = knowledge_lines(store, include_sensitive=include_sensitive)
    cons = store.contradictions()
    con_text = "\n".join(f"- {c.description}（aware={c.aware}；处理方式：{c.how_handled}）" for c in cons) or "- 暂无登记"
    user = "## 观察（按维度）\n" + ("\n".join(lines) or "- （暂无）") + "\n\n## 矛盾登记\n" + con_text
    card_md = llm.complete(load_prompt("persona_card"), user, max_tokens=2500)
    card = {"markdown": card_md.strip(), "n_claims": len(lines), "n_contradictions": len(cons),
            "include_sensitive": include_sensitive}
    v = store.save_artifact("persona_card", card)
    store.log("persona_card", f"v{v}")
    store.commit()
    card["version"] = v
    return card


# ----------------------------------------------------------------- review
def review_queue(store: Store, limit: int = 30) -> list[Claim]:
    q = store.claims(review="pending")
    # sensitive + ungrounded + contradiction-tagged first, then low confidence
    q.sort(key=lambda c: (c.dimension not in SENSITIVE_DIMENSIONS, "ungrounded" not in c.tags,
                          "contradiction" not in c.tags, c.confidence))
    return q[:limit]


def set_review(store: Store, claim_id: str, state: str, edited_text: Optional[str] = None) -> Claim:
    c = store.get_claim(claim_id)
    if not c:
        raise KeyError(claim_id)
    if state == "edited" and edited_text:
        c.claim = edited_text
    c.user_review = state
    if state == "confirmed":
        c.confidence = max(c.confidence, 0.8)
    store.update_claim(c)
    store.log("review", f"{claim_id} -> {state}")
    store.commit()
    return c


def coverage(store: Store) -> list[tuple[str, str, int, int, float]]:
    """(dim, name, n_claims, n_confirmed, mean_confidence) — shows where evidence is thin."""
    out = []
    for dim, name in DIMENSIONS.items():
        cl = store.claims(dimension=dim)
        conf = [c for c in cl if c.user_review == "confirmed"]
        mean = round(sum(c.confidence for c in cl) / len(cl), 2) if cl else 0.0
        out.append((dim, name, len(cl), len(conf), mean))
    return out
