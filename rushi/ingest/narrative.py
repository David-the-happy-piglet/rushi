"""Narrative input: third-person descriptions, interviews, diaries, user notes.

Any free text about the person becomes Chunks. The *reporter* matters:
    self          the person wrote/said it (diary, their own interview answers)
    user          the app user describing them from memory
    third:<alias> someone else's account relayed by the user

Optional front-matter per block lets the user attach time and relation hints:

    --- year: 2016 ; relation: close_friend ; reporter: user ---
    那年他刚搬去上海……

Blocks are separated by blank lines + the `---` header, or, without headers, by paragraphs.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from ..schema import Chunk, stable_id

HEADER_RE = re.compile(r"^---\s*(?P<body>.*?)\s*---\s*$")
PHONE_RE = re.compile(r"(?<!\d)(\+?86[- ]?)?1[3-9]\d{9}(?!\d)|(?<!\d)\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}(?!\d)")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
IDNUM_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
URL_RE = re.compile(r"https?://\S+")


def anonymize(text: str, name_map: Optional[dict[str, str]] = None) -> str:
    """Regex scrubbing + explicit name replacement. Conservative on purpose."""
    text = IDNUM_RE.sub("[证件号]", text)
    text = PHONE_RE.sub("[电话]", text)
    text = EMAIL_RE.sub("[邮箱]", text)
    text = URL_RE.sub("[链接]", text)
    if name_map:
        for name in sorted(name_map, key=len, reverse=True):   # longest first
            if name:
                text = text.replace(name, name_map[name])
    return text


def _parse_header(body: str) -> dict:
    out = {}
    for part in re.split(r"[;；,，]\s*", body):
        if ":" in part or "：" in part:
            k, v = re.split(r"[:：]", part, 1)
            out[k.strip().lower()] = v.strip()
    return out


def split_blocks(text: str) -> list[tuple[dict, str]]:
    """Return list of (meta, body)."""
    lines = text.splitlines()
    blocks: list[tuple[dict, list[str]]] = []
    current_meta: dict = {}
    current: list[str] = []
    saw_header = False
    for line in lines:
        m = HEADER_RE.match(line.strip())
        if m:
            saw_header = True
            if current and "".join(current).strip():
                blocks.append((current_meta, current))
            current_meta, current = _parse_header(m.group("body")), []
        else:
            current.append(line)
    if current and "".join(current).strip():
        blocks.append((current_meta, current))
    if saw_header:
        return [(meta, "\n".join(body).strip()) for meta, body in blocks]
    # no headers → split by paragraphs, merge short ones to ~300-800 chars
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    merged: list[str] = []
    for p in paras:
        if merged and len(merged[-1]) + len(p) < 500:
            merged[-1] += "\n" + p
        else:
            merged.append(p)
    return [({}, p) for p in merged]


def to_chunks(text: str, source_id: str, default_reporter: str = "user",
              name_map: Optional[dict[str, str]] = None, max_len: int = 800) -> list[Chunk]:
    chunks: list[Chunk] = []
    for i, (meta, body) in enumerate(split_blocks(text)):
        body = anonymize(body, name_map)
        pieces = [body[j:j + max_len] for j in range(0, len(body), max_len)] or [body]
        for k, piece in enumerate(pieces):
            year = meta.get("year") or meta.get("时间") or meta.get("time")
            chunks.append(Chunk(
                id=stable_id("chk", source_id, str(i), str(k), piece[:40]),
                source_id=source_id,
                text=piece,
                reporter=meta.get("reporter", default_reporter),
                valid_from=year,
                valid_to=meta.get("until"),
                relation=meta.get("relation", "unknown"),
                meta={"block": i, "part": k},
            ))
    return chunks


def chat_sessions_to_chunks(store, source_id: str, max_len: int = 800) -> list[Chunk]:
    """Render chat sessions as text chunks so the extractor can mine them for claims too.
    Target lines are marked so the extractor knows whose words are whose."""
    chunks = []
    for sid in store.sessions():
        utts = store.utterances(session_id=sid)
        if not utts or not any(u.is_target for u in utts):
            continue
        lines = [f"[{'ta' if u.is_target else u.speaker.split(':',1)[-1]}] {u.text}" for u in utts]
        text = "\n".join(lines)
        year = next((u.year for u in utts if u.year), None)
        relation = utts[0].relation
        for k in range(0, len(text), max_len):
            piece = text[k:k + max_len]
            chunks.append(Chunk(
                id=stable_id("chk", sid, str(k)),
                source_id=source_id,
                text=piece,
                reporter="self",
                valid_from=str(year) if year else None,
                relation=relation,
                meta={"session_id": sid, "kind": "chat"},
            ))
    return chunks
