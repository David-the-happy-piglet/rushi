"""Retrieval. Offline-first: character bigram BM25 (works well for short Chinese messages,
no tokenizer needed). An embedding backend can be plugged in via `Embedder` later; results
of both are merged with reciprocal rank fusion.

Three indexes, each with metadata filters (relation, year range, dimension):
- exchanges: (other's message → ta's reply burst)   → style examples
- chunks:    narrative material                       → memories
- claims/cases: structured knowledge                  → persona reasoning
plus `graph_context()`: BFS over the layered graph from entities/claims the query touches
(Graphiti-style graph search), fused with BM25 by RRF and reranked by node distance.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from .schema import SENSITIVE_DIMENSIONS, TRAIT_RELS, Claim, DecisionCase, Utterance
from .store import Store

_WS = re.compile(r"\s+")


def bigrams(text: str) -> list[str]:
    t = _WS.sub("", text.lower())
    t = re.sub(r"[，。！？、,.!?；;:：\"'“”‘’()（）\[\]【】…~～]", "", t)
    if len(t) < 2:
        return [t] if t else []
    return [t[i:i + 2] for i in range(len(t) - 1)]


@dataclass
class Doc:
    id: str
    text: str            # text used for matching
    payload: str         # text shown to the model
    meta: dict = field(default_factory=dict)


class BM25:
    def __init__(self, docs: Iterable[Doc], k1: float = 1.5, b: float = 0.75):
        self.docs = list(docs)
        self.k1, self.b = k1, b
        self.tf: list[Counter] = []
        self.df: Counter = Counter()
        self.len: list[int] = []
        for d in self.docs:
            grams = bigrams(d.text)
            c = Counter(grams)
            self.tf.append(c)
            self.len.append(len(grams))
            self.df.update(c.keys())
        self.avg = (sum(self.len) / len(self.len)) if self.len else 1.0
        self.N = len(self.docs)

    def search(self, query: str, k: int = 10, where: Optional[Callable[[Doc], bool]] = None) -> list[tuple[Doc, float]]:
        q = Counter(bigrams(query))
        scored = []
        for i, d in enumerate(self.docs):
            if where and not where(d):
                continue
            s = 0.0
            for g in q:
                if g not in self.tf[i]:
                    continue
                idf = math.log(1 + (self.N - self.df[g] + 0.5) / (self.df[g] + 0.5))
                f = self.tf[i][g]
                s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            if s > 0:
                scored.append((d, s))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]


class Embedder:
    """Optional dense backend. Implement `embed(texts) -> list[list[float]]` (e.g. bge-m3 via
    sentence-transformers, or an API) and pass to Retriever(embedder=...)."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


def _cos(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a)) or 1.0
    db = math.sqrt(sum(y * y for y in b)) or 1.0
    return num / (da * db)


class Retriever:
    def __init__(self, store: Store, embedder: Optional[Embedder] = None):
        self.store = store
        self.embedder = embedder
        self.idx_exchanges = BM25(self._exchange_docs())
        self.idx_chunks = BM25(self._chunk_docs())
        self.idx_knowledge = BM25(self._knowledge_docs())
        self._vecs: dict[str, dict[str, list[float]]] = {}
        if embedder:
            for name, idx in (("exchanges", self.idx_exchanges), ("chunks", self.idx_chunks), ("knowledge", self.idx_knowledge)):
                vecs = embedder.embed([d.text for d in idx.docs]) if idx.docs else []
                self._vecs[name] = {d.id: v for d, v in zip(idx.docs, vecs)}

    # ----------------------------------------------------------- doc builders
    def _exchange_docs(self) -> list[Doc]:
        docs = []
        for sid in self.store.sessions():
            utts = self.store.utterances(session_id=sid)
            i = 0
            while i < len(utts):
                if utts[i].is_target:
                    i += 1; continue
                ctx = [utts[i]]
                j = i + 1
                while j < len(utts) and not utts[j].is_target:
                    ctx.append(utts[j]); j += 1
                if j >= len(utts):
                    break
                burst, b = [], utts[j].burst_id
                while j < len(utts) and utts[j].is_target and utts[j].burst_id == b:
                    burst.append(utts[j].text); j += 1
                ctx_text = "\n".join(u.text for u in ctx[-2:])
                reply = " / ".join(burst)
                if reply.strip():
                    u0 = ctx[-1]
                    docs.append(Doc(id=utts[j - 1].id, text=ctx_text, meta={"relation": u0.relation, "year": u0.year},
                                    payload=f"[{u0.speaker.split(':',1)[-1]}] {ctx_text}\n[ta] {reply}"))
                i = j
        return docs

    def _chunk_docs(self) -> list[Doc]:
        return [Doc(id=c.id, text=c.text, payload=c.text,
                    meta={"relation": c.relation, "year": _year(c.valid_from), "reporter": c.reporter,
                          "kind": c.meta.get("kind", "narrative")})
                for c in self.store.chunks()]

    def _knowledge_docs(self) -> list[Doc]:
        docs = []
        for c in self.store.claims():
            quotes = "；".join(e.quote for e in c.evidence[:2])
            docs.append(Doc(id=c.id, text=f"{c.claim} {quotes}", meta={"dimension": c.dimension, "year": _year(c.valid_from),
                                                                    "kind": "claim", "confidence": c.confidence,
                                                                    "sensitive": "sensitive" in c.tags},
                            payload=f"[{c.dimension}·{c.strength}·conf {c.confidence}·{c.valid_from or '?'}] {c.claim}（证据: {quotes[:80]}）"))
        for k in self.store.cases():
            docs.append(Doc(id=k.id, text=f"{k.situation} {k.stated_reason} {' '.join(k.options)}",
                            meta={"kind": "case", "year": _year(k.when)},
                            payload=f"[决定·{k.when or '?'}] 情境: {k.situation} | 选项: {'/'.join(k.options)} | 选择: {k.choice} | 理由: {k.stated_reason} | 结果: {k.outcome}"))
        return docs

    # ----------------------------------------------------------------- search
    def _hybrid(self, name: str, index: BM25, query: str, k: int, where) -> list[Doc]:
        lex = index.search(query, k=k * 2, where=where)
        if not self.embedder or name not in self._vecs:
            return [d for d, _ in lex[:k]]
        qv = self.embedder.embed([query])[0]
        dense = sorted(((d, _cos(qv, self._vecs[name][d.id])) for d in index.docs if (not where or where(d))),
                       key=lambda x: -x[1])[: k * 2]
        rank: dict[str, float] = defaultdict(float)
        docs: dict[str, Doc] = {}
        for r, (d, _) in enumerate(lex):
            rank[d.id] += 1 / (60 + r); docs[d.id] = d
        for r, (d, _) in enumerate(dense):
            rank[d.id] += 1 / (60 + r); docs[d.id] = d
        return [docs[i] for i, _ in sorted(rank.items(), key=lambda x: -x[1])[:k]]

    def examples(self, query: str, relation: Optional[str] = None, year_range: Optional[tuple[int, int]] = None,
                 k: int = 8) -> list[str]:
        where = _where(relation=relation, year_range=year_range)
        return [d.payload for d in self._hybrid("exchanges", self.idx_exchanges, query, k, where)]

    def memories(self, query: str, year_range: Optional[tuple[int, int]] = None, k: int = 5,
                 exclude_sensitive: bool = True) -> list[str]:
        where = _where(year_range=year_range)
        return [d.payload for d in self._hybrid("chunks", self.idx_chunks, query, k, where)]

    def knowledge(self, query: str, dimensions: Optional[set[str]] = None, exclude: Optional[set[str]] = None,
                  year_range: Optional[tuple[int, int]] = None, k: int = 10) -> list[Doc]:
        def where(d: Doc):
            dim = d.meta.get("dimension")
            if dimensions and dim and dim not in dimensions:
                return False
            if exclude and dim in exclude:
                return False
            if exclude and d.meta.get("sensitive") and exclude & SENSITIVE_DIMENSIONS:
                return False
            return _where(year_range=year_range)(d)
        return self._hybrid("knowledge", self.idx_knowledge, query, k, where)


    # ------------------------------------------------------------------ graph
    def _adjacency(self) -> dict[str, list[tuple[str, str]]]:
        if not hasattr(self, "_adj"):
            adj: dict[str, list[tuple[str, str]]] = defaultdict(list)
            for e in self.store.edges(active_only=True):
                adj[e.from_id].append((e.to_id, e.rel))
                adj[e.to_id].append((e.from_id, e.rel))
            self._adj = adj
        return self._adj

    def graph_context(self, query: str, k: int = 6, hops: int = 3, include_sensitive: bool = False,
                      year_range: Optional[tuple[int, int]] = None) -> list[str]:
        """Derivation chains relevant to the query: trait ← observations ← quotes, found by BFS from
        (a) entities named in the query and (b) BM25 hits, fused with RRF, reranked by graph distance."""
        from .resolve import TARGET_ID, norm_name
        adj = self._adjacency()
        nq = norm_name(query)
        seeds: dict[str, int] = {}
        for n in [n for t in ("Person", "Event", "Decision", "Place", "Thing") for n in self.store.nodes(t)]:
            if n.id == TARGET_ID:
                continue
            names = [n.label, *n.attrs.get("aliases", [])]
            if any(len(norm_name(x)) >= 2 and norm_name(x) in nq for x in names):
                seeds[n.id] = 0
        exclude = None if include_sensitive else set(SENSITIVE_DIMENSIONS)
        lex = self.knowledge(query, exclude=exclude, year_range=year_range, k=k * 2)
        for d in lex:
            if d.meta.get("kind") == "claim":
                c = self.store.get_claim(d.id)
                if c:
                    seeds[(f"trt:{c.id}" if c.kind == "reflection" else f"obs:{c.id}")] = 0
        dist: dict[str, int] = dict(seeds)
        frontier = list(seeds)
        for h in range(1, hops + 1):
            nxt = []
            for u in frontier:
                for v, _ in adj.get(u, []):
                    if v not in dist and v != TARGET_ID:      # the target touches everything: never traverse it
                        dist[v] = h
                        nxt.append(v)
            frontier = nxt
        rank: dict[str, float] = defaultdict(float)
        for r_, d in enumerate(lex):
            rank[d.id] += 1 / (60 + r_)
        for nid, dd in dist.items():
            if nid.startswith(("trt:", "obs:")):
                rank[nid.split(":", 1)[1]] += 1 / (60 + 5 * dd) + (0.01 if nid.startswith("trt:") else 0)
        out = []
        for cid, _ in sorted(rank.items(), key=lambda x: -x[1]):
            c = self.store.get_claim(cid)
            if not c or not c.active or (not include_sensitive and ("sensitive" in c.tags or c.dimension in SENSITIVE_DIMENSIONS)):
                continue
            if year_range and _year(c.valid_from) and not (year_range[0] <= _year(c.valid_from) <= year_range[1]):
                continue
            out.append(render_chain(self.store, c, include_sensitive))
            if len(out) >= k:
                break
        return out


def render_chain(store: Store, c: Claim, include_sensitive: bool = False) -> str:
    """One-paragraph derivation chain for prompts: trait → reasoning → supporting observations (quotes)."""
    if c.kind != "reflection":
        q = c.evidence[0].quote[:60] if c.evidence else ""
        return f"[观察·{c.dimension}·conf {c.confidence}] {c.claim}（「{q}」）"
    lines = [f"[特质·{c.dimension}·{c.strength}·conf {c.confidence}] {c.claim}",
             f"  推导（{c.derivation.get('inference', '')}）：{c.derivation.get('reasoning', '')}"]
    for sid in c.derivation.get("based_on", [])[:4]:
        s = store.get_claim(sid)
        if s and (include_sensitive or ("sensitive" not in s.tags and s.dimension not in SENSITIVE_DIMENSIONS)):
            q = s.evidence[0].quote[:50] if s.evidence else ""
            lines.append(f"  ← {s.claim}（「{q}」）")
    for e in store.edges(from_id=f"trt:{c.id}", active_only=True):
        if e.rel in TRAIT_RELS and e.rel not in ("CONTRADICTS", "CHANGED_TO"):
            o = store.get_claim(e.to_id.split(":", 1)[-1])
            if o and o.active and (include_sensitive or "sensitive" not in o.tags):
                lines.append(f"  —{TRAIT_RELS[e.rel]}→ {o.claim}")
    return "\n".join(lines)


def _year(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    m = re.match(r"(\d{4})", str(s))
    return int(m.group(1)) if m else None


def _where(relation: Optional[str] = None, year_range: Optional[tuple[int, int]] = None):
    def f(d: Doc) -> bool:
        if relation and d.meta.get("relation") not in (relation, None, "unknown"):
            return False
        y = d.meta.get("year")
        if year_range and y is not None and not (year_range[0] <= y <= year_range[1]):
            return False
        return True
    return f
