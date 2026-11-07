"""SQLite-backed memory store. Stdlib only.

One store = one persona. Records are stored as JSON blobs with a few indexed columns,
which keeps the schema flexible while the project is young.

Rules:
- Nothing is ever deleted. Claims get `superseded_by` or `user_review = rejected`;
  edges get `valid_to` / `invalidated_by`.
- Opening a store with `readonly=True` makes every write raise. The fiction sandbox
  opens the base persona this way (see sandbox.py).
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Optional

from .schema import (Chunk, Claim, Contradiction, DecisionCase, Edge, Node, Source, Utterance,
                     _known, claim_from_dict, now_iso)

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources   (id TEXT PRIMARY KEY, kind TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS utterances(id TEXT PRIMARY KEY, source_id TEXT, session_id TEXT,
                                      is_target INTEGER, relation TEXT, year INTEGER, ts TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS chunks    (id TEXT PRIMARY KEY, source_id TEXT, reporter TEXT, data TEXT,
                                      extracted INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS claims    (id TEXT PRIMARY KEY, dimension TEXT, user_review TEXT,
                                      kind TEXT, confidence REAL, data TEXT);
CREATE TABLE IF NOT EXISTS nodes     (id TEXT PRIMARY KEY, type TEXT, label TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS edges     (id TEXT PRIMARY KEY, from_id TEXT, rel TEXT, to_id TEXT,
                                      valid_to TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS cases     (id TEXT PRIMARY KEY, data TEXT);
CREATE TABLE IF NOT EXISTS contradictions (id TEXT PRIMARY KEY, data TEXT);
CREATE TABLE IF NOT EXISTS artifacts (key TEXT, version INTEGER, created_at TEXT, data TEXT,
                                      PRIMARY KEY (key, version));
CREATE TABLE IF NOT EXISTS log       (id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT, op TEXT, detail TEXT);
CREATE INDEX IF NOT EXISTS idx_utt_session ON utterances(session_id);
CREATE INDEX IF NOT EXISTS idx_claims_dim ON claims(dimension);
CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_id);
CREATE INDEX IF NOT EXISTS idx_edges_to ON edges(to_id);
CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type);
"""


class ReadOnlyError(RuntimeError):
    pass


class Store:
    def __init__(self, path: str | Path, readonly: bool = False):
        self.path = Path(path)
        self.readonly = readonly
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if readonly:
            uri = f"file:{self.path}?mode=ro"
            self.conn = sqlite3.connect(uri, uri=True)
        else:
            self.conn = sqlite3.connect(self.path)
            self.conn.executescript(SCHEMA)
        self.conn.row_factory = sqlite3.Row

    # ------------------------------------------------------------------ util
    def _guard(self):
        if self.readonly:
            raise ReadOnlyError(f"store {self.path} is read-only")

    def _exec(self, sql: str, params: Iterable = ()):
        self._guard()
        self.conn.execute(sql, tuple(params))

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.close()

    def log(self, op: str, detail: str = ""):
        self._exec("INSERT INTO log(at, op, detail) VALUES (?,?,?)", (now_iso(), op, detail))

    # --------------------------------------------------------------- sources
    def add_source(self, s: Source):
        self._exec("INSERT OR REPLACE INTO sources VALUES (?,?,?)", (s.id, s.kind, json.dumps(asdict(s), ensure_ascii=False)))

    def sources(self) -> list[Source]:
        return [Source(**json.loads(r["data"])) for r in self.conn.execute("SELECT data FROM sources")]

    # ------------------------------------------------------------ utterances
    def add_utterances(self, utts: Iterable[Utterance]):
        self._guard()
        rows = [(u.id, u.source_id, u.session_id, int(u.is_target), u.relation, u.year, u.ts,
                 json.dumps(asdict(u), ensure_ascii=False)) for u in utts]
        self.conn.executemany("INSERT OR REPLACE INTO utterances VALUES (?,?,?,?,?,?,?,?)", rows)

    def utterances(self, is_target: Optional[bool] = None, relation: Optional[str] = None,
                   session_id: Optional[str] = None) -> list[Utterance]:
        sql, params = "SELECT data FROM utterances WHERE 1=1", []
        if is_target is not None:
            sql += " AND is_target=?"; params.append(int(is_target))
        if relation:
            sql += " AND relation=?"; params.append(relation)
        if session_id:
            sql += " AND session_id=?"; params.append(session_id)
        sql += " ORDER BY ts, id"
        return [Utterance(**json.loads(r["data"])) for r in self.conn.execute(sql, params)]

    def sessions(self) -> list[str]:
        return [r[0] for r in self.conn.execute("SELECT DISTINCT session_id FROM utterances ORDER BY session_id")]

    # ---------------------------------------------------------------- chunks
    def add_chunk(self, c: Chunk):
        self._exec("INSERT OR REPLACE INTO chunks(id, source_id, reporter, data, extracted) VALUES (?,?,?,?,0)",
                   (c.id, c.source_id, c.reporter, json.dumps(asdict(c), ensure_ascii=False)))

    def chunks(self, only_unextracted: bool = False) -> list[Chunk]:
        sql = "SELECT data FROM chunks" + (" WHERE extracted=0" if only_unextracted else "")
        return [Chunk(**json.loads(r["data"])) for r in self.conn.execute(sql)]

    def get_chunk(self, cid: str) -> Optional[Chunk]:
        r = self.conn.execute("SELECT data FROM chunks WHERE id=?", (cid,)).fetchone()
        return Chunk(**json.loads(r["data"])) if r else None

    def mark_extracted(self, cid: str):
        self._exec("UPDATE chunks SET extracted=1 WHERE id=?", (cid,))

    # ---------------------------------------------------------------- claims
    def add_claim(self, c: Claim):
        self._exec("INSERT OR REPLACE INTO claims VALUES (?,?,?,?,?,?)",
                   (c.id, c.dimension, c.user_review, c.kind, c.confidence, json.dumps(asdict(c), ensure_ascii=False)))

    def update_claim(self, c: Claim):
        self.add_claim(c)

    def get_claim(self, cid: str) -> Optional[Claim]:
        r = self.conn.execute("SELECT data FROM claims WHERE id=?", (cid,)).fetchone()
        return claim_from_dict(json.loads(r["data"])) if r else None

    def claims(self, dimension: Optional[str] = None, active_only: bool = True,
               min_confidence: float = 0.0, review: Optional[str] = None) -> list[Claim]:
        sql, params = "SELECT data FROM claims WHERE confidence>=?", [min_confidence]
        if dimension:
            sql += " AND dimension=?"; params.append(dimension)
        if review:
            sql += " AND user_review=?"; params.append(review)
        out = [claim_from_dict(json.loads(r["data"])) for r in self.conn.execute(sql, params)]
        if active_only:
            out = [c for c in out if c.active]
        return out

    # ------------------------------------------------------------------ graph
    def upsert_node(self, n: Node):
        self._exec("INSERT OR REPLACE INTO nodes VALUES (?,?,?,?)",
                   (n.id, n.type, n.label, json.dumps(asdict(n), ensure_ascii=False)))

    def nodes(self, type_: Optional[str] = None) -> list[Node]:
        sql, params = "SELECT data FROM nodes", []
        if type_:
            sql += " WHERE type=?"; params.append(type_)
        return [Node(**json.loads(r["data"])) for r in self.conn.execute(sql, params)]

    def get_node(self, nid: str) -> Optional[Node]:
        r = self.conn.execute("SELECT data FROM nodes WHERE id=?", (nid,)).fetchone()
        return Node(**json.loads(r["data"])) if r else None

    def get_edge(self, eid: str) -> Optional[Edge]:
        r = self.conn.execute("SELECT data FROM edges WHERE id=?", (eid,)).fetchone()
        return Edge(**_known(Edge, json.loads(r["data"]))) if r else None

    def add_edge(self, e: Edge):
        self._exec("INSERT OR REPLACE INTO edges VALUES (?,?,?,?,?,?)",
                   (e.id, e.from_id, e.rel, e.to_id, e.valid_to, json.dumps(asdict(e), ensure_ascii=False)))

    def edges(self, from_id: Optional[str] = None, rel: Optional[str] = None,
              active_only: bool = False, to_id: Optional[str] = None) -> list[Edge]:
        sql, params = "SELECT data FROM edges WHERE 1=1", []
        if from_id:
            sql += " AND from_id=?"; params.append(from_id)
        if to_id:
            sql += " AND to_id=?"; params.append(to_id)
        if rel:
            sql += " AND rel=?"; params.append(rel)
        out = [Edge(**_known(Edge, json.loads(r["data"]))) for r in self.conn.execute(sql, params)]
        if active_only:
            out = [e for e in out if e.invalidated_by is None and e.expired_at is None]
        return out

    # ------------------------------------------------------------------ cases
    def add_case(self, c: DecisionCase):
        self._exec("INSERT OR REPLACE INTO cases VALUES (?,?)", (c.id, json.dumps(asdict(c), ensure_ascii=False)))

    def cases(self) -> list[DecisionCase]:
        return [DecisionCase(**_known(DecisionCase, json.loads(r["data"]))) for r in self.conn.execute("SELECT data FROM cases")]

    # --------------------------------------------------------- contradictions
    def add_contradiction(self, c: Contradiction):
        self._exec("INSERT OR REPLACE INTO contradictions VALUES (?,?)", (c.id, json.dumps(asdict(c), ensure_ascii=False)))

    def contradictions(self) -> list[Contradiction]:
        return [Contradiction(**_known(Contradiction, json.loads(r["data"])))
                for r in self.conn.execute("SELECT data FROM contradictions")]

    # -------------------------------------------------------------- artifacts
    def save_artifact(self, key: str, data: Any) -> int:
        """Versioned blobs: persona cards, style cards, reports. Returns the new version."""
        self._guard()
        cur = self.conn.execute("SELECT COALESCE(MAX(version),0) FROM artifacts WHERE key=?", (key,))
        version = cur.fetchone()[0] + 1
        self.conn.execute("INSERT INTO artifacts VALUES (?,?,?,?)",
                          (key, version, now_iso(), json.dumps(data, ensure_ascii=False)))
        return version

    def artifact(self, key: str, version: Optional[int] = None) -> Optional[Any]:
        if version is None:
            r = self.conn.execute("SELECT data FROM artifacts WHERE key=? ORDER BY version DESC LIMIT 1", (key,)).fetchone()
        else:
            r = self.conn.execute("SELECT data FROM artifacts WHERE key=? AND version=?", (key, version)).fetchone()
        return json.loads(r["data"]) if r else None

    def artifact_versions(self, key: str) -> list[tuple[int, str]]:
        return [(r[0], r[1]) for r in self.conn.execute(
            "SELECT version, created_at FROM artifacts WHERE key=? ORDER BY version", (key,))]

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict[str, int]:
        q = lambda t: self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        return {t: q(t) for t in ("sources", "utterances", "chunks", "claims", "nodes", "edges", "cases", "contradictions")}
