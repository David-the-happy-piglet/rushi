"""L4 themes: cluster traits with label propagation (Graphiti §2.3), then let the LLM name them.

Trait–trait affinity:
    explicit trait link (CAUSES, TRADES_OFF, CONTRADICTS …)   1.0
    each shared supporting observation                        0.5
    each shared non-target entity in their observations       0.25
Label propagation is deterministic here (sorted order, ties → keep / smallest label).
A theme needs ≥ 2 traits. Old themes are kept but marked superseded; their MEMBER_OF edges expire.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from .extract import load_prompt
from .llm import LLM
from .resolve import TARGET_ID
from .schema import SENSITIVE_DIMENSIONS, TRAIT_RELS, Edge, Node, now_iso, stable_id
from .store import Store


def _affinity(store: Store, traits: dict) -> dict[str, dict[str, float]]:
    w: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    nid = {f"trt:{c.id}": c for c in traits.values()}
    for e in store.edges(active_only=True):
        if (e.rel in TRAIT_RELS) and e.from_id in nid and e.to_id in nid:
            w[e.from_id][e.to_id] += 1.0
            w[e.to_id][e.from_id] += 1.0
    supports = {n: set(c.derivation.get("based_on", [])) for n, c in nid.items()}
    ents: dict[str, set[str]] = {}
    for n, sup in supports.items():
        es = set()
        for cid in sup:
            for e in store.edges(from_id=f"obs:{cid}", rel="ABOUT"):
                if e.to_id != TARGET_ID:
                    es.add(e.to_id)
        ents[n] = es
    keys = sorted(nid)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            x = 0.5 * len(supports[a] & supports[b]) + 0.25 * len(ents[a] & ents[b])
            if x:
                w[a][b] += x
                w[b][a] += x
    return w


def label_propagation(nodes: list[str], w: dict[str, dict[str, float]], iters: int = 20) -> dict[str, str]:
    label = {n: n for n in nodes}
    for _ in range(iters):
        changed = False
        for n in sorted(nodes):
            score: dict[str, float] = defaultdict(float)
            for m, x in w.get(n, {}).items():
                score[label[m]] += x
            if not score:
                continue
            best = max(score.values())
            top = sorted(l for l, v in score.items() if v == best)
            new = label[n] if label[n] in top else top[0]
            if new != label[n]:
                label[n], changed = new, True
        if not changed:
            break
    return label


def build_themes(store: Store, llm: LLM, run_id: Optional[str] = None) -> dict:
    traits = {c.id: c for c in store.claims() if c.kind == "reflection"}
    for old in store.nodes("Theme"):
        if not old.attrs.get("superseded"):
            old.attrs["superseded"] = run_id or now_iso()
            store.upsert_node(old)
            for e in store.edges(to_id=old.id, rel="MEMBER_OF"):
                if e.expired_at is None:
                    e.expired_at = now_iso()
                    store.add_edge(e)
    if len(traits) < 2:
        store.save_artifact("themes", {"themes": []})
        store.commit()
        return {"themes": 0}
    w = _affinity(store, traits)
    nodes = [f"trt:{cid}" for cid in traits]
    label = label_propagation(nodes, w)
    groups: dict[str, list[str]] = defaultdict(list)
    for n, l in label.items():
        groups[l].append(n)
    groups_l = [sorted(g) for g in groups.values() if len(g) >= 2]
    if not groups_l:
        store.save_artifact("themes", {"themes": []})
        store.commit()
        return {"themes": 0}

    user = "\n\n".join(f"## 组 {i}\n" + "\n".join(f"- [{traits[n[4:]].dimension}] {traits[n[4:]].claim}" for n in g)
                       for i, g in enumerate(groups_l))
    try:
        named = {int(t.get("group")): t for t in (llm.complete_json(load_prompt("themes"), user).get("themes") or [])
                 if isinstance(t, dict) and str(t.get("group", "")).lstrip("-").isdigit()}
    except Exception as e:  # noqa: BLE001
        store.log("themes_error", str(e))
        named = {}

    out = []
    for i, g in enumerate(groups_l):
        members = [traits[n[4:]] for n in g]
        t = named.get(i, {})
        name = str(t.get("name") or members[0].claim[:16])
        dims = sorted({d for c in members for d in c.derivation.get("dimensions", [c.dimension])})
        sensitive = any("sensitive" in c.tags or c.dimension in SENSITIVE_DIMENSIONS for c in members)
        tid = stable_id("thm", *g)
        store.upsert_node(Node(id=tid, type="Theme", label=name,
                               attrs={"summary": str(t.get("summary", "")), "dimensions": dims, "run": run_id,
                                      "members": [c.id for c in members], "sensitive": sensitive}))
        for n in g:
            store.add_edge(Edge(id=stable_id("edg", n, "MEMBER_OF", tid), from_id=n, rel="MEMBER_OF", to_id=tid,
                                attrs={"run": run_id}))
        out.append({"id": tid, "name": name, "summary": str(t.get("summary", "")), "dimensions": dims,
                    "members": [c.id for c in members], "sensitive": sensitive})
    store.save_artifact("themes", {"themes": out, "run": run_id})
    store.log("themes", f"{len(out)} themes")
    store.commit()
    return {"themes": len(out)}
