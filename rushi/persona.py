"""Persona engine: from grounded claims to reflections, contradictions and the core card.

- reflect(): across all dimensions, derive traits (kind="reflection") from ≥2 observations each, with an
  explicit derivation (based_on + reasoning + inference type), link traits to each other (CAUSES,
  MANIFESTS_AS, TRADES_OFF …), register contradictions/changes, then cluster traits into themes.
  Confidence is computed by `scoring`, not taken from the model. Never deletes anything.
- build_persona_card(): compress confirmed / high-confidence knowledge into a short card that lives
  in the chat context. Sensitive dimensions are excluded unless include_sensitive=True.
- review helpers for the human-in-the-loop queue.
"""
from __future__ import annotations

from typing import Optional

from . import scoring
from .extract import load_prompt, node_id_for_claim, register_contradiction
from .llm import LLM
from .schema import (DIMENSIONS, INFERENCE_TYPES, SENSITIVE_DIMENSIONS, TRAIT_RELS, Claim, Edge, Node, new_id,
                     now_iso, stable_id)
from .store import Store


def _claim_line(c: Claim) -> str:
    ev = c.evidence[0] if c.evidence else None
    et = ev.evidence_type if ev else "?"
    q = f"「{ev.quote[:60]}」" if ev and ev.quote else ""
    return f"- {c.id} [{c.dimension}/{c.strength}/{et}/{c.valid_from or '?'}–{c.valid_to or ''}/{c.user_review}] {c.claim} {q}"


MAX_REFLECT_INPUT = 150


def _is_sensitive(c: Claim) -> bool:
    return c.dimension in SENSITIVE_DIMENSIONS or "sensitive" in c.tags


def _expire_edges(store: Store, node_id: str, rels: set[str], run_id: str) -> None:
    for e in store.edges(from_id=node_id) + store.edges(to_id=node_id):
        if e.rel in rels and e.expired_at is None:
            e.expired_at, e.invalidated_by = now_iso(), run_id
            store.add_edge(e)


def reflect(store: Store, llm: LLM, dimensions: Optional[list[str]] = None, min_claims: int = 3,
            themes: bool = True) -> dict:
    """Cross-dimension reflection: observations → traits (with derivation) → trait links → themes.

    Each run supersedes the previous run's unconfirmed traits (kept, marked superseded_by=<run>,
    their edges expired) so re-running never piles up duplicates and never deletes anything.
    User-confirmed traits are carried over and can be linked to."""
    out = {"traits": 0, "links": 0, "contradictions": 0, "changes": 0, "themes": 0}
    obs = [c for c in store.claims() if c.kind != "reflection" and (not dimensions or c.dimension in dimensions)]
    if len(obs) < min_claims:
        return out
    if len(obs) > MAX_REFLECT_INPUT:
        store.log("reflect_truncated", f"{len(obs)} observations, using top {MAX_REFLECT_INPUT} by confidence")
        obs = sorted(obs, key=lambda c: -c.confidence)[:MAX_REFLECT_INPUT]
    kept = [c for c in store.claims() if c.kind == "reflection" and c.user_review in ("confirmed", "edited")]
    cases = store.cases()

    dims = "\n".join(f"{k}: {v}" for k, v in DIMENSIONS.items())
    rels = "、".join(f"{k}({v})" for k, v in TRAIT_RELS.items() if k not in ("CONTRADICTS", "CHANGED_TO"))
    system = load_prompt("reflect").replace("{dimensions}", dims).replace("{trait_rels}", rels)
    user = ("## 观察\n" + "\n".join(_claim_line(c) for c in obs)
            + "\n\n## 已确认的特质\n" + ("\n".join(_claim_line(c) for c in kept) or "- 无")
            + "\n\n## 决策案例\n" + ("\n".join(f"- {k.id} [{k.when or '?'}] {k.situation} → {k.choice}（理由：{k.stated_reason}）"
                                           for k in cases) or "- 无"))
    try:
        data = llm.complete_json(system, user, max_tokens=12000)
    except Exception as e:  # noqa: BLE001
        store.log("reflect_error", str(e))
        return out
    if not isinstance(data, dict):
        return out

    run_id = new_id("run")
    for old in store.claims():
        if old.kind == "reflection" and old.user_review not in ("confirmed", "edited") and old.superseded_by is None:
            old.superseded_by = run_id
            store.update_claim(old)
            _expire_edges(store, f"trt:{old.id}", {"DERIVED_FROM", "SHOWN_IN", "MEMBER_OF", "CONTRADICTS",
                                                  "CHANGED_TO", *TRAIT_RELS}, run_id)

    by_id: dict[str, Claim] = {c.id: c for c in obs + kept}
    keymap: dict[str, Claim] = {}
    case_nodes = {k.id: k.entity_id for k in cases if k.entity_id}

    def resolve_ref(x) -> Optional[Claim]:
        x = str(x)
        return keymap.get(x) or by_id.get(x)

    for t in data.get("traits", []) or []:
        if not isinstance(t, dict) or not t.get("claim"):
            continue
        dim = str(t.get("dimension", "")).strip().upper()[:1]
        if dim not in DIMENSIONS:
            continue
        supports = [c for c in (resolve_ref(b) for b in (t.get("based_on") or [])) if c]
        supports = list({c.id: c for c in supports}.values())
        if len(supports) < 2:
            continue
        strength, conf, breakdown = scoring.score_trait(supports)
        evidence, seen = [], set()
        for c in supports:
            for e in c.evidence:
                if (e.ref_id, e.quote) not in seen:
                    seen.add((e.ref_id, e.quote)); evidence.append(e)
        inference = t.get("inference") if t.get("inference") in INFERENCE_TYPES else "generalization"
        trait = Claim(id=stable_id("clm", "trait", run_id, dim, str(t["claim"])[:60]), dimension=dim,
                      claim=str(t["claim"]).strip(), strength=strength, confidence=conf,
                      valid_from=t.get("valid_from"), valid_to=t.get("valid_to"), evidence=evidence,
                      kind="reflection",
                      tags=["based_on:" + ",".join(c.id for c in supports)]
                           + (["sensitive"] if dim in SENSITIVE_DIMENSIONS or any(_is_sensitive(c) for c in supports) else []),
                      derivation={"based_on": [c.id for c in supports], "reasoning": str(t.get("reasoning", "")).strip(),
                                  "inference": inference, "score": breakdown, "run": run_id,
                                  "dimensions": sorted({c.dimension for c in supports} | {dim})})
        store.add_claim(trait)
        keymap[str(t.get("key") or trait.id)] = trait
        by_id[trait.id] = trait
        nid = f"trt:{trait.id}"
        store.upsert_node(Node(id=nid, type="Trait", label=trait.claim[:80],
                               attrs={"dimension": dim, "claim_id": trait.id, "run": run_id}))
        for c in supports:
            _edge(store, nid, "DERIVED_FROM", node_id_for_claim(c), claim_id=trait.id,
                  attrs={"inference": inference})
        for k in t.get("shown_in") or []:
            if str(k) in case_nodes:
                _edge(store, nid, "SHOWN_IN", case_nodes[str(k)], claim_id=trait.id)
        out["traits"] += 1

    for l in data.get("links", []) or []:
        if not isinstance(l, dict):
            continue
        a, b, rel = resolve_ref(l.get("from")), resolve_ref(l.get("to")), str(l.get("rel", "")).upper()
        if not a or not b or a.id == b.id or rel not in TRAIT_RELS or rel in ("CONTRADICTS", "CHANGED_TO"):
            continue
        cites = [str(x) for x in (l.get("based_on") or []) if str(x) in by_id]
        if not cites:
            continue
        _edge(store, node_id_for_claim(a), rel, node_id_for_claim(b), confidence=round(min(a.confidence, b.confidence), 2),
              attrs={"explanation": str(l.get("explanation", "")), "based_on": cites, "run": run_id,
                     "cross_dimension": a.dimension != b.dimension})
        out["links"] += 1

    contradicted: set[str] = set()
    for con in data.get("contradictions", []) or []:
        if not isinstance(con, dict):
            continue
        a, b = resolve_ref(con.get("a") or con.get("claim_a")), resolve_ref(con.get("b") or con.get("claim_b"))
        if not a or not b or a.id == b.id:
            continue
        kind = "change" if con.get("kind") == "change" else "contradiction"
        if register_contradiction(store, a.id, b.id, str(con.get("description", "")), kind=kind,
                                  aware=con.get("aware"), how_handled=str(con.get("how_handled", ""))):
            out["changes" if kind == "change" else "contradictions"] += 1
            if kind == "contradiction":
                contradicted |= {a.id, b.id}

    for c in keymap.values():          # contradicted traits lose some confidence
        if c.id in contradicted:
            c = store.get_claim(c.id)
            sup = [by_id[x] for x in c.derivation.get("based_on", []) if x in by_id]
            c.strength, c.confidence, c.derivation["score"] = scoring.score_trait(sup, contradicted=True)
            store.update_claim(c)

    store.log("reflect", str(out))
    store.commit()
    if themes:
        from .themes import build_themes
        out["themes"] = build_themes(store, llm, run_id=run_id)["themes"]
    return out


def _edge(store: Store, frm: str, rel: str, to: str, **kw) -> None:
    store.add_edge(Edge(id=stable_id("edg", frm, rel, to), from_id=frm, rel=rel, to_id=to, **kw))


def knowledge_lines(store: Store, include_sensitive: bool = False, min_confidence: float = 0.5,
                    pending_ok: bool = True) -> list[str]:
    lines = []
    for dim in DIMENSIONS:
        if dim in SENSITIVE_DIMENSIONS and not include_sensitive:
            continue
        for c in store.claims(dimension=dim):
            if c.user_review == "rejected" or ("sensitive" in c.tags and not include_sensitive):
                continue
            if c.user_review == "pending" and (not pending_ok or c.confidence < min_confidence):
                continue
            lines.append(f"- [{dim}] {c.claim} （{c.strength}, conf {c.confidence}, {c.valid_from or '?'}{'–' + c.valid_to if c.valid_to else ''}, {c.user_review}）")
    return lines


def trait_link_lines(store: Store, include_sensitive: bool = False) -> list[str]:
    active = {c.id: c for c in store.claims()}
    out = []
    for e in store.edges(active_only=True):
        if e.rel not in TRAIT_RELS or e.rel in ("CONTRADICTS", "CHANGED_TO"):
            continue
        a, b = active.get(e.from_id.split(":", 1)[-1]), active.get(e.to_id.split(":", 1)[-1])
        if not a or not b or (not include_sensitive and (_is_sensitive(a) or _is_sensitive(b))):
            continue
        out.append(f"- 「{a.claim}」 —{TRAIT_RELS[e.rel]}→ 「{b.claim}」：{e.attrs.get('explanation', '')}")
    return out


def build_persona_card(store: Store, llm: LLM, include_sensitive: bool = False) -> dict:
    lines = knowledge_lines(store, include_sensitive=include_sensitive)
    active = {c.id: c for c in store.claims()}
    ok = lambda cid: cid in active and (include_sensitive or not _is_sensitive(active[cid]))
    cons = [c for c in store.contradictions() if ok(c.claim_a) and ok(c.claim_b)]
    con_text = "\n".join(f"- [{'变化' if c.kind == 'change' else '矛盾'}] {c.description}（aware={c.aware}；处理方式：{c.how_handled}）"
                         for c in cons) or "- 暂无登记"
    user = ("## 观察与特质（按维度）\n" + ("\n".join(lines) or "- （暂无）")
            + "\n\n## 特质之间的关系\n" + ("\n".join(trait_link_lines(store, include_sensitive)) or "- 暂无")
            + "\n\n## 主题\n" + ("\n".join(f"- {t['name']}：{t['summary']}" for t in
                                              (store.artifact("themes") or {}).get("themes", [])
                                              if include_sensitive or not t.get("sensitive")) or "- 暂无")
            + "\n\n## 矛盾与变化登记\n" + con_text)
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
