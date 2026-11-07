"""Entity resolution and dedup (after Graphiti §2.2.1–2.2.2, without embeddings).

1. Exact match on normalised name / aliases → same node.
2. Otherwise gather candidates of a compatible type (all of them when the graph is small, else the
   top-k by char-bigram overlap) and ask the LLM once per chunk: same entity or not?
   Bigram overlap alone cannot see that 妈妈 == 母亲, which is why the small-graph path sends all.
3. Merging never deletes: the surviving node gains the other names as aliases.

Fact edges are deduplicated per (from, rel, to); decision cases per Decision node.
"""
from __future__ import annotations

import re
from typing import Optional

from .llm import LLM
from .schema import ENTITY_TYPES, Node, stable_id
from .store import Store

TARGET_ID = "person:target"
COMPATIBLE = {"Event": {"Event", "Decision"}, "Decision": {"Decision", "Event"}}
MAX_CANDIDATES = 40


# pronouns and generic references would merge unrelated entities ("他" → everyone)
PRONOUNS = {"他", "她", "它", "ta", "我", "你", "您", "他们", "她们", "我们", "对方", "那个人", "这个人", "人家", "自己"}


def usable_alias(a) -> bool:
    a = str(a or "").strip()
    return len(a) >= 2 and a.lower() not in PRONOUNS


def norm_name(s: str) -> str:
    return re.sub(r"[\s·・,，。.!！?？\"'“”‘’()（）\[\]【】<>《》]", "", str(s or "")).lower()


def _grams(s: str) -> set[str]:
    s = norm_name(s)
    return {s[i:i + 2] for i in range(len(s) - 1)} | ({s} if len(s) == 1 else set())


def names_of(n: Node) -> set[str]:
    return {norm_name(n.label)} | {norm_name(a) for a in n.attrs.get("aliases", [])}


def candidates(store: Store, ent: dict, k: int = MAX_CANDIDATES) -> list[Node]:
    types = COMPATIBLE.get(ent["type"], {ent["type"]})
    pool = [n for t in types for n in store.nodes(t) if n.id != TARGET_ID]
    if len(pool) <= k:
        return pool
    q = set().union(*(_grams(x) for x in [ent["name"], *ent.get("aliases", [])]))
    def sim(n: Node) -> float:
        g = set().union(*(_grams(x) for x in [n.label, *n.attrs.get("aliases", [])]))
        return len(q & g) / (len(q | g) or 1)
    return sorted(pool, key=sim, reverse=True)[:k]


def _merge_into(store: Store, node: Node, ent: dict, chunk_id: str) -> Node:
    aliases = set(node.attrs.get("aliases", []))
    for a in [ent["name"], *ent.get("aliases", [])]:
        if a and norm_name(a) != norm_name(node.label):
            aliases.add(a)
    node.attrs["aliases"] = sorted(aliases)
    node.attrs.setdefault("episodes", [])
    if chunk_id not in node.attrs["episodes"]:
        node.attrs["episodes"].append(chunk_id)
    if not node.attrs.get("when") and ent.get("when"):
        node.attrs["when"] = ent["when"]
    store.upsert_node(node)
    return node


def _new_node(store: Store, ent: dict, chunk_id: str) -> Node:
    node = Node(id=stable_id("ent", ent["type"], norm_name(ent["name"])), type=ent["type"], label=ent["name"],
                attrs={"aliases": sorted({a for a in ent.get("aliases", []) if a and a != ent["name"]}),
                       "when": ent.get("when"), "episodes": [chunk_id]})
    existing = store.get_node(node.id)
    return _merge_into(store, existing, ent, chunk_id) if existing else (store.upsert_node(node) or node)


def clean_entities(raw) -> list[dict]:
    out, seen = [], set()
    for e in raw or []:
        if not isinstance(e, dict):
            continue
        key, name, typ = str(e.get("key", "")).strip(), str(e.get("name", "")).strip(), str(e.get("type", "")).strip()
        if not key or not name or key == "ta" or key in seen:
            continue
        typ = typ if typ in ENTITY_TYPES else "Thing"
        seen.add(key)
        out.append({"key": key, "name": name, "type": typ, "when": e.get("when"),
                    "aliases": [str(a).strip() for a in (e.get("aliases") or []) if usable_alias(a)]})
    return out


def resolve_entities(store: Store, llm: LLM, ents: list[dict], chunk_id: str) -> dict[str, str]:
    """Return {local key → node id}. `ta` always maps to the target node."""
    from .extract import load_prompt   # local import: extract imports this module
    mapping: dict[str, str] = {"ta": TARGET_ID}
    pending: list[tuple[dict, list[Node]]] = []
    for ent in ents:
        cands = candidates(store, ent)
        mine = {norm_name(ent["name"])} | {norm_name(a) for a in ent["aliases"]}
        exact = next((n for n in cands if names_of(n) & mine), None)
        if exact:
            mapping[ent["key"]] = _merge_into(store, exact, ent, chunk_id).id
        elif cands:
            pending.append((ent, cands))
        else:
            mapping[ent["key"]] = _new_node(store, ent, chunk_id).id

    if pending:
        lines = ["## New"]
        for ent, _ in pending:
            lines.append(f"- {ent['key']} [{ent['type']}] {ent['name']} 别名={ent['aliases']} 时间={ent.get('when')}")
        lines.append("\n## Existing")
        seen = set()
        for _, cands in pending:
            for n in cands:
                if n.id not in seen:
                    seen.add(n.id)
                    lines.append(f"- {n.id} [{n.type}] {n.label} 别名={n.attrs.get('aliases', [])} 时间={n.attrs.get('when')}")
        try:
            data = llm.complete_json(load_prompt("resolve"), "\n".join(lines), max_tokens=2000)
            matches = {str(m.get("key")): m for m in (data.get("matches") or []) if isinstance(m, dict)}
        except Exception as e:  # noqa: BLE001
            store.log("resolve_error", f"{chunk_id}: {e}")
            matches = {}
        for ent, cands in pending:
            m = matches.get(ent["key"]) or {}
            target = next((n for n in cands if n.id == m.get("match")), None)
            if target:
                if m.get("name") and norm_name(m["name"]) not in names_of(target):
                    ent = {**ent, "aliases": [*ent["aliases"], m["name"]]}
                mapping[ent["key"]] = _merge_into(store, target, ent, chunk_id).id
                store.log("resolve_merge", f"{ent['name']} → {target.id} ({target.label})")
            else:
                mapping[ent["key"]] = _new_node(store, ent, chunk_id).id
    return mapping


def entity_label(store: Store, nid: Optional[str]) -> str:
    n = store.get_node(nid) if nid else None
    return n.label if n else str(nid)
