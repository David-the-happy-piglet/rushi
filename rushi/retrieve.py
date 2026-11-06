"""Retrieval. Offline-first: character bigram BM25 (works well for short Chinese messages,
no tokenizer needed). An embedding backend can be plugged in via `Embedder` later; results
of both are merged with reciprocal rank fusion.

Three indexes, each with metadata filters (relation, year range, dimension):
- exchanges: (other's message → ta's reply burst)   → style examples
- chunks:    narrative material                       → memories
- claims/cases: structured knowledge                  → persona reasoning
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from .schema import Claim, DecisionCase, Utterance
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
                                                                    "kind": "claim", "confidence": c.confidence},
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
            return _where(year_range=year_range)(d)
        return self._hybrid("knowledge", self.idx_knowledge, query, k, where)


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
