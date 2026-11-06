"""Per-chunk extraction → grounded claims, decision cases, graph nodes/edges.

Grounding: every claim's quote is checked against the chunk text (exact or fuzzy).
Ungrounded quotes are kept but flagged (`grounded=False`) and their confidence is halved;
they land in the review queue. This is the main defence against hallucinated traits.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from .llm import LLM
from .schema import (DIMENSIONS, Chunk, Claim, DecisionCase, Edge, Evidence, Node, Source,
                     new_id, stable_id)
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


def extract_chunk(store: Store, llm: LLM, chunk: Chunk, target_node_id: str = "person:target") -> dict:
    dims = "\n".join(f"{k}: {v}" for k, v in DIMENSIONS.items())
    system = load_prompt("extract").replace("{dimensions}", dims)
    user = (f"<source>来源类型: reporter={chunk.reporter}; 时间提示: {chunk.valid_from or '未知'}; "
            f"对话关系: {chunk.relation}</source>\n<material>\n{chunk.text}\n</material>")
    try:
        data = llm.complete_json(system, user)
    except Exception as e:  # noqa: BLE001
        store.log("extract_error", f"{chunk.id}: {e}")
        return {"claims": 0, "cases": 0, "ungrounded": 0}
    if not isinstance(data, dict):
        return {"claims": 0, "cases": 0, "ungrounded": 0}

    n_claims = n_ungrounded = 0
    for c in data.get("claims", []) or []:
        dim = str(c.get("dimension", "")).strip().upper()[:1]
        if dim not in DIMENSIONS or not c.get("claim"):
            continue
        quote = str(c.get("quote", "")).strip()
        grounded = ground(quote, chunk.text)
        conf = float(c.get("confidence", 0.5) or 0.5)
        if not grounded:
            conf *= 0.5
            n_ungrounded += 1
        ev = Evidence(source_id=chunk.source_id, ref_id=chunk.id, quote=quote,
                      evidence_type=_evidence_type(chunk, c.get("evidence_type")),
                      ts=chunk.valid_from, grounded=grounded)
        claim = Claim(
            id=stable_id("clm", chunk.id, dim, str(c["claim"])[:60]),
            dimension=dim, claim=str(c["claim"]).strip(),
            strength=c.get("strength", "weak") if c.get("strength") in ("weak", "moderate", "strong") else "weak",
            confidence=round(min(max(conf, 0.0), 1.0), 2),
            valid_from=c.get("valid_from") or chunk.valid_from, valid_to=c.get("valid_to") or chunk.valid_to,
            evidence=[ev],
            user_review="pending",
            tags=(["ungrounded"] if not grounded else []) + (["contradiction"] if c.get("contradiction_note") else []),
        )
        store.add_claim(claim)
        n_claims += 1
        # graph: Person -HAS_OBSERVATION-> Claim-as-node (lightweight; richer typing comes from reflection)
        node = Node(id=f"obs:{claim.id}", type="Observation", label=claim.claim[:80],
                    attrs={"dimension": dim, "claim_id": claim.id})
        store.upsert_node(node)
        store.add_edge(Edge(id=stable_id("edg", target_node_id, "OBSERVED", node.id), from_id=target_node_id,
                            rel="OBSERVED", to_id=node.id, valid_from=claim.valid_from, valid_to=claim.valid_to,
                            confidence=claim.confidence, evidence_type=ev.evidence_type,
                            source_ids=[chunk.source_id], claim_id=claim.id))

    n_cases = 0
    for c in data.get("cases", []) or []:
        if not c.get("situation"):
            continue
        case = DecisionCase(
            id=stable_id("case", chunk.id, str(c["situation"])[:60]),
            when=c.get("when") or chunk.valid_from, situation=str(c["situation"]).strip(),
            options=[str(o) for o in (c.get("options") or [])], choice=str(c.get("choice", "")),
            stated_reason=str(c.get("stated_reason", "")), outcome=str(c.get("outcome", "")),
            reflection=str(c.get("reflection", "")), source_ids=[chunk.source_id],
            relation_context=chunk.relation,
            confidence=0.6 if ground(str(c.get("quote", "")), chunk.text) else 0.3,
        )
        store.add_case(case)
        n_cases += 1

    for ev_ in data.get("events", []) or []:
        if not ev_.get("label"):
            continue
        node = Node(id=stable_id("evt", str(ev_["label"])), type="Event", label=str(ev_["label"]),
                    attrs={"when": ev_.get("when") or chunk.valid_from})
        store.upsert_node(node)
        store.add_edge(Edge(id=stable_id("edg", target_node_id, "EXPERIENCED", node.id), from_id=target_node_id,
                            rel="EXPERIENCED", to_id=node.id, valid_from=ev_.get("when") or chunk.valid_from,
                            source_ids=[chunk.source_id], evidence_type=_evidence_type(chunk, None)))

    for r in data.get("relations", []) or []:
        if not r.get("person"):
            continue
        node = Node(id=stable_id("per", str(r["person"])), type="Person", label=str(r["person"]),
                    attrs={"relation": r.get("relation", "unknown")})
        store.upsert_node(node)
        store.add_edge(Edge(id=stable_id("edg", target_node_id, "RELATES_TO", node.id, str(r.get("note", ""))[:30]),
                            from_id=target_node_id, rel="RELATES_TO", to_id=node.id, valid_from=chunk.valid_from,
                            source_ids=[chunk.source_id], attrs={"note": r.get("note", ""), "relation": r.get("relation")}))

    store.mark_extracted(chunk.id)
    store.commit()
    return {"claims": n_claims, "cases": n_cases, "ungrounded": n_ungrounded}


def extract_all(store: Store, llm: LLM, limit: Optional[int] = None) -> dict:
    store.upsert_node(Node(id="person:target", type="Person", label="ta", attrs={"is_target": True}))
    totals = {"chunks": 0, "claims": 0, "cases": 0, "ungrounded": 0}
    for chunk in store.chunks(only_unextracted=True)[: limit or None]:
        r = extract_chunk(store, llm, chunk)
        totals["chunks"] += 1
        for k in ("claims", "cases", "ungrounded"):
            totals[k] += r[k]
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
                                     evidence_type="other_report" if reporter != "self" else "self_report")])
    store.add_claim(claim)
    store.commit()
    return claim
