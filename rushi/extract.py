"""Per-chunk extraction → a small layered graph per chunk, merged into the persona graph.

    L0 Episode ──MENTIONS──▶ L1 entities ──facts──▶ L1 entities
    L0 Episode ──SUPPORTS(quote)──▶ L2 Observation ──ABOUT──▶ L1 entities

Grounding: every claim's quote is checked against the chunk text (exact or fuzzy).
Ungrounded quotes are kept but flagged (`grounded=False`) and their confidence is halved;
they land in the review queue. This is the main defence against hallucinated traits.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from . import resolve, scoring
from .llm import LLM
from .resolve import TARGET_ID
from .schema import (DIMENSIONS, ENTITY_TYPES, FACT_RELS, FACT_TYPE_MAP, SENSITIVE_DIMENSIONS, Chunk, Claim,
                     Contradiction, DecisionCase, Edge, Evidence, Node, Source, new_id, stable_id)
from .store import Store

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(name: str) -> str:
    return (PROMPTS / f"{name}.md").read_text(encoding="utf-8")


def _norm(s: str) -> str:
    return re.sub(r"[\s，。！？、,.!?；;:：\"'“”‘’()（）\[\]【】…~～-]", "", s or "")


def ground(quote: str, text: str) -> bool:
    """Exact → whitespace/punct-insensitive → 70% char-bigram overlap."""
    if not quote:
        return False
    if quote in text:
        return True
    nq, nt = _norm(quote), _norm(text)
    if nq and nq in nt:
        return True
    grams = {nq[i:i + 2] for i in range(len(nq) - 1)}
    if not grams:
        return False
    hit = sum(1 for g in grams if g in nt)
    return hit / len(grams) >= 0.7


def _evidence_type(chunk: Chunk, declared: Optional[str]) -> str:
    """Reporter overrides: a user describing the person can never produce self_report."""
    declared = declared if declared in ("behavior", "self_report", "other_report") else None
    if chunk.reporter.startswith("third") or chunk.reporter == "user":
        return "other_report" if declared != "behavior" else "behavior"
    return declared or "self_report"


def _episode(store: Store, chunk: Chunk) -> str:
    """L0: one Episode node per chunk."""
    nid = f"ep:{chunk.id}"
    if not store.get_node(nid):
        kind = "chat" if chunk.text.lstrip().startswith("[") else chunk.meta.get("kind", "narrative")
        store.upsert_node(Node(id=nid, type="Episode", label=chunk.text.replace("\n", " ")[:40],
                               attrs={"chunk_id": chunk.id, "source_id": chunk.source_id, "reporter": chunk.reporter,
                                      "when": chunk.valid_from, "kind": kind, "excerpt": chunk.text[:300]}))
    return nid


def _structural(store: Store, frm: str, rel: str, to: str, **kw) -> None:
    store.add_edge(Edge(id=stable_id("edg", frm, rel, to), from_id=frm, rel=rel, to_id=to, **kw))


def add_fact_edge(store: Store, frm: str, rel: str, to: str, fact: str, quote: str, grounded: bool,
                  chunk: Chunk, when: Optional[str], evidence_type: str) -> Optional[str]:
    """L1 entity–entity fact. Dedup per (from, rel, to): a repeat adds evidence instead of a new edge."""
    a, b = store.get_node(frm), store.get_node(to)
    if not a or not b or frm == to:
        return None
    raw = str(rel or "").upper().strip()
    allowed = FACT_TYPE_MAP.get((a.type, b.type), {"RELATED_TO"})
    rel = raw if raw in allowed else "RELATED_TO"
    eid = stable_id("fct", frm, rel, to)
    ev = {"chunk_id": chunk.id, "quote": quote, "grounded": grounded, "fact": fact, "evidence_type": evidence_type}
    e = store.get_edge(eid)
    if e:
        if all(x.get("chunk_id") != chunk.id for x in e.attrs.get("evidence", [])):
            e.attrs.setdefault("evidence", []).append(ev)
            e.source_ids = sorted(set(e.source_ids) | {chunk.source_id})
            e.confidence = round(min(0.9, e.confidence + 0.1), 2)
            store.add_edge(e)
        return eid
    store.add_edge(Edge(id=eid, from_id=frm, rel=rel, to_id=to, valid_from=when or chunk.valid_from,
                        confidence=0.6 if grounded else 0.3, evidence_type=evidence_type,
                        source_ids=[chunk.source_id],
                        attrs={"fact": fact, "predicate": raw if raw != rel else "", "layer": 1, "evidence": [ev]}))
    return eid


def _strength(x) -> str:
    # a single chunk can never justify "strong" — that is reserved for cross-episode traits
    return "moderate" if x == "strong" else (x if x in ("weak", "moderate") else "weak")


def extract_chunk(store: Store, llm: LLM, chunk: Chunk, target_node_id: str = TARGET_ID) -> dict:
    dims = "\n".join(f"{k}: {v}" for k, v in DIMENSIONS.items())
    rels = "、".join(f"{k}({v})" for k, v in FACT_RELS.items())
    system = load_prompt("extract").replace("{dimensions}", dims).replace("{fact_rels}", rels)
    user = (f"<source>来源类型: reporter={chunk.reporter}; 时间提示: {chunk.valid_from or '未知'}; "
            f"对话关系: {chunk.relation}</source>\n<material>\n{chunk.text}\n</material>")
    empty = {"claims": 0, "cases": 0, "ungrounded": 0, "entities": 0, "facts": 0}
    try:
        data = llm.complete_json(system, user, max_tokens=6000)
    except Exception as e:  # noqa: BLE001
        store.log("extract_error", f"{chunk.id}: {e}")
        return empty
    if not isinstance(data, dict):
        return empty

    ep = _episode(store, chunk)
    chunk_type = _evidence_type(chunk, None)

    # ---- L1 entities (resolved against the existing graph)
    ents = resolve.clean_entities(data.get("entities"))
    keymap = resolve.resolve_entities(store, llm, ents, chunk.id)
    for nid in set(keymap.values()):
        _structural(store, ep, "MENTIONS", nid, source_ids=[chunk.source_id], attrs={"layer": 0})

    # ---- L1 facts
    fact_ids: list[Optional[str]] = []
    for f in data.get("facts", []) or []:
        if not isinstance(f, dict):
            fact_ids.append(None); continue
        frm, to = keymap.get(str(f.get("from"))), keymap.get(str(f.get("to")))
        quote = str(f.get("quote", "")).strip()
        fact_ids.append(add_fact_edge(store, frm, f.get("rel"), to, str(f.get("fact", "")).strip(), quote,
                                      ground(quote, chunk.text), chunk, f.get("when"), chunk_type) if frm and to else None)

    # ---- L2 observations
    n_claims = n_ungrounded = 0
    local_claims: list[Optional[Claim]] = []
    for c in data.get("claims", []) or []:
        dim = str((c or {}).get("dimension", "")).strip().upper()[:1]
        if not isinstance(c, dict) or dim not in DIMENSIONS or not c.get("claim"):
            local_claims.append(None); continue
        quote = str(c.get("quote", "")).strip()
        grounded = ground(quote, chunk.text)
        if not grounded:
            n_ungrounded += 1
        etype = _evidence_type(chunk, c.get("evidence_type"))
        strength = _strength(c.get("strength"))
        try:
            llm_conf = float(c.get("confidence"))
        except (TypeError, ValueError):
            llm_conf = None
        ev = Evidence(source_id=chunk.source_id, ref_id=chunk.id, quote=quote, evidence_type=etype,
                      ts=chunk.valid_from, grounded=grounded)
        about = sorted({keymap[k] for k in (c.get("about") or []) if k in keymap} | {target_node_id})
        facts = [fact_ids[i] for i in (c.get("facts") or []) if isinstance(i, int) and 0 <= i < len(fact_ids) and fact_ids[i]]
        claim = Claim(
            id=stable_id("clm", chunk.id, dim, str(c["claim"])[:60]),
            dimension=dim, claim=str(c["claim"]).strip(), strength=strength,
            confidence=scoring.score_observation(strength, etype, grounded, llm_conf),
            valid_from=c.get("valid_from") or chunk.valid_from, valid_to=c.get("valid_to") or chunk.valid_to,
            evidence=[ev], user_review="pending",
            tags=(["ungrounded"] if not grounded else []) + (["contradiction"] if c.get("contradiction_note") else [])
                 + (["sensitive"] if dim in SENSITIVE_DIMENSIONS else []),
            derivation={"about": about, "facts": facts, "llm_confidence": llm_conf,
                        "contradiction_note": c.get("contradiction_note") or ""},
        )
        store.add_claim(claim)
        local_claims.append(claim)
        n_claims += 1
        obs = f"obs:{claim.id}"
        store.upsert_node(Node(id=obs, type="Observation", label=claim.claim[:80],
                               attrs={"dimension": dim, "claim_id": claim.id}))
        _structural(store, ep, "SUPPORTS", obs, valid_from=claim.valid_from, evidence_type=etype,
                    source_ids=[chunk.source_id], claim_id=claim.id, confidence=claim.confidence,
                    attrs={"quote": quote, "grounded": grounded, "layer": 0})
        for nid in about:
            _structural(store, obs, "ABOUT", nid, claim_id=claim.id, attrs={"layer": 2})

    # within-chunk contradictions → first-class records + edges (both sides stay valid)
    for i, c in enumerate(data.get("claims", []) or []):
        if i >= len(local_claims) or not local_claims[i] or not isinstance(c, dict):
            continue
        for j in c.get("contradicts") or []:
            if isinstance(j, int) and 0 <= j < len(local_claims) and local_claims[j] and j != i:
                register_contradiction(store, local_claims[j].id, local_claims[i].id,
                                       str(c.get("contradiction_note", "")), kind="contradiction")

    # ---- decision cases, attached to (and deduplicated by) their Decision node
    n_cases = 0
    for c in data.get("cases", []) or []:
        if not isinstance(c, dict) or not c.get("situation"):
            continue
        dec = keymap.get(str(c.get("decision", "")))
        quote = str(c.get("quote", "")).strip()
        grounded = ground(quote, chunk.text)
        if not dec:      # model forgot the Decision entity: make one from the choice
            ent = {"key": "_dec", "name": str(c.get("choice") or c["situation"])[:30], "type": "Decision",
                   "aliases": [], "when": c.get("when")}
            dec = resolve.resolve_entities(store, llm, [ent], chunk.id)["_dec"]
            _structural(store, ep, "MENTIONS", dec, source_ids=[chunk.source_id], attrs={"layer": 0})
        cid = stable_id("case", dec)
        prev = next((k for k in store.cases() if k.id == cid), None)
        if prev:
            prev.source_ids = sorted(set(prev.source_ids) | {chunk.source_id})
            prev.quotes = [*prev.quotes, quote] if quote and quote not in prev.quotes else prev.quotes
            for fld in ("stated_reason", "outcome", "reflection"):
                if len(str(c.get(fld, ""))) > len(getattr(prev, fld)):
                    setattr(prev, fld, str(c.get(fld)))
            prev.confidence = round(min(0.9, prev.confidence + 0.1), 2)
            store.add_case(prev)
        else:
            store.add_case(DecisionCase(
                id=cid, when=c.get("when") or chunk.valid_from, situation=str(c["situation"]).strip(),
                options=[str(o) for o in (c.get("options") or [])], choice=str(c.get("choice", "")),
                stated_reason=str(c.get("stated_reason", "")), outcome=str(c.get("outcome", "")),
                reflection=str(c.get("reflection", "")), source_ids=[chunk.source_id],
                relation_context=chunk.relation, confidence=0.6 if grounded else 0.3,
                entity_id=dec, quotes=[quote] if quote else []))
            add_fact_edge(store, target_node_id, "DECIDED", dec, str(c.get("choice", "")), quote, grounded,
                          chunk, c.get("when"), chunk_type)
        n_cases += 1

    store.mark_extracted(chunk.id)
    store.commit()
    return {"claims": n_claims, "cases": n_cases, "ungrounded": n_ungrounded,
            "entities": len(set(keymap.values())) - 1, "facts": len([f for f in fact_ids if f])}


def register_contradiction(store: Store, a: str, b: str, description: str, kind: str = "contradiction",
                           aware=None, how_handled: str = "") -> Optional[Contradiction]:
    """Contradictions are data. `contradiction`: both sides stay valid, linked by CONTRADICTS.
    `change`: the earlier claim is closed (valid_to) and linked to the later one by CHANGED_TO — never deleted."""
    ca, cb = store.get_claim(a), store.get_claim(b)
    if not ca or not cb or a == b:
        return None
    con = Contradiction(id=stable_id("con", *sorted([a, b])), claim_a=a, claim_b=b, description=description,
                        aware=aware, how_handled=how_handled, kind=kind if kind in ("contradiction", "change") else "contradiction")
    store.add_contradiction(con)
    na, nb = node_id_for_claim(ca), node_id_for_claim(cb)
    if con.kind == "change":
        early, late = (ca, cb) if str(ca.valid_from or "") <= str(cb.valid_from or "") else (cb, ca)
        if not early.valid_to and late.valid_from:
            early.valid_to = late.valid_from
            store.update_claim(early)
        _structural(store, node_id_for_claim(early), "CHANGED_TO", node_id_for_claim(late),
                    attrs={"description": description, "contradiction_id": con.id})
    else:
        for c, other in ((ca, b), (cb, a)):
            if other not in c.contradicts:
                c.contradicts.append(other)
                store.update_claim(c)
        _structural(store, na, "CONTRADICTS", nb, attrs={"description": description, "contradiction_id": con.id,
                                                         "aware": aware, "how_handled": how_handled})
    return con


def node_id_for_claim(c: Claim) -> str:
    return f"trt:{c.id}" if c.kind == "reflection" else f"obs:{c.id}"


def extract_all(store: Store, llm: LLM, limit: Optional[int] = None) -> dict:
    store.upsert_node(Node(id=TARGET_ID, type="Person", label="ta", attrs={"is_target": True, "aliases": []}))
    totals = {"chunks": 0, "claims": 0, "cases": 0, "ungrounded": 0, "entities": 0, "facts": 0}
    for chunk in store.chunks(only_unextracted=True)[: limit or None]:
        r = extract_chunk(store, llm, chunk)
        totals["chunks"] += 1
        for k in r:
            totals[k] += r[k]
    totals["entities"] = len([n for t in ENTITY_TYPES for n in store.nodes(t)]) - 1
    store.log("extract", str(totals))
    store.commit()
    return totals


def add_user_fact(store: Store, text: str, dimension: str, valid_from: Optional[str] = None,
                  reporter: str = "user") -> Claim:
    """The user states something directly ('他其实很怕狗'). Stored as a user_stated claim with the
    statement itself as evidence — reviewable like everything else."""
    src = Source(id=new_id("src"), kind="user_note", origin="cli", reporter=reporter, note=text[:80])
    store.add_source(src)
    chunk = Chunk(id=stable_id("chk", src.id, text[:40]), source_id=src.id, text=text, reporter=reporter,
                  valid_from=valid_from)
    store.add_chunk(chunk)
    store.mark_extracted(chunk.id)
    claim = Claim(id=stable_id("clm", chunk.id, dimension, text[:60]), dimension=dimension, claim=text,
                  strength="moderate", confidence=0.8, valid_from=valid_from, kind="user_stated",
                  user_review="confirmed",
                  evidence=[Evidence(source_id=src.id, ref_id=chunk.id, quote=text,
                                     evidence_type="other_report" if reporter != "self" else "self_report")],
                  tags=["sensitive"] if dimension in SENSITIVE_DIMENSIONS else [],
                  derivation={"about": [TARGET_ID], "facts": []})
    store.add_claim(claim)
    ep = _episode(store, chunk)
    obs = f"obs:{claim.id}"
    store.upsert_node(Node(id=obs, type="Observation", label=text[:80], attrs={"dimension": dimension, "claim_id": claim.id}))
    _structural(store, ep, "SUPPORTS", obs, claim_id=claim.id, source_ids=[src.id], attrs={"quote": text, "grounded": True})
    _structural(store, obs, "ABOUT", TARGET_ID, claim_id=claim.id)
    store.commit()
    return claim
