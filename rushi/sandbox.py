"""Fiction sandbox. The persona goes in; nothing comes back out.

Isolation model
- A sandbox is its own SQLite file under <workspace>/sandboxes/<name>.db.
- Creating it copies a *snapshot* (persona card, style card, active claims, cases) from the base
  persona. The copy is tagged with the base artifact versions it came from.
- The base store is opened READ-ONLY inside sandbox runs; any accidental write raises.
- Events, invented facts and improvised lines live only in the sandbox db.
- There is no automatic promotion. `promote()` exists for the user to copy a *specific* item back
  deliberately, and it records that the item is fictional in origin.

This is phase-5 scaffolding; the runner is deliberately minimal.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .llm import LLM
from .schema import Claim, Evidence, new_id, now_iso
from .store import ReadOnlyError, Store


def create_sandbox(workspace: Path, base_store_path: Path, name: str) -> Path:
    base = Store(base_store_path, readonly=True)
    sb_path = workspace / "sandboxes" / f"{name}.db"
    if sb_path.exists():
        raise FileExistsError(sb_path)
    sb = Store(sb_path)
    # snapshot artifacts
    for key in ("persona_card", "style_card"):
        art = base.artifact(key)
        if art is not None:
            sb.save_artifact(key, {**art, "snapshot_of": str(base_store_path), "snapshot_version": base.artifact_versions(key)[-1][0]})
    for c in base.claims():
        sb.add_claim(c)
    for k in base.cases():
        sb.add_case(k)
    for n in base.nodes():
        sb.upsert_node(n)
    for e in base.edges():
        sb.add_edge(e)
    for con in base.contradictions():
        sb.add_contradiction(con)
    # copy utterances too so style retrieval works inside the sandbox
    sb.add_utterances(base.utterances())
    for s in base.sources():
        sb.add_source(s)
    for ch in base.chunks():
        sb.add_chunk(ch); sb.mark_extracted(ch.id)
    sb.save_artifact("sandbox_meta", {"name": name, "created_at": now_iso(), "base": str(base_store_path),
                                      "fiction": True, "events": []})
    sb.log("sandbox_create", name)
    sb.commit(); sb.close(); base.close()
    return sb_path


def run_scene(sb_path: Path, llm: LLM, scene: str, other_character: str, ticks: int = 6,
              relation: str = "unknown") -> list[dict]:
    """Minimal two-party scene: ta (from snapshot) vs a described other character.
    Returns the event log; also appends it to the sandbox's own artifacts."""
    from .chat import ChatSession
    from .retrieve import Retriever

    sb = Store(sb_path)
    sess = ChatSession(sb, llm, relation=relation, retriever=Retriever(sb))
    events = [{"tick": 0, "who": "narrator", "text": scene}]
    other_line = llm.complete(
        f"你扮演这个角色：{other_character}\n场景：{scene}\n用一两句口语开场，不要解释。", "开始。", max_tokens=150, temperature=0.9).strip()
    for t in range(1, ticks + 1):
        events.append({"tick": t, "who": "other", "text": other_line})
        ta_line = sess.reply(other_line, allow_search=False)
        events.append({"tick": t, "who": "ta", "text": ta_line})
        other_line = llm.complete(
            f"你扮演这个角色：{other_character}\n场景：{scene}\n对方刚说：{ta_line}\n用一两句口语回应，推动情节，不要解释。",
            "继续。", max_tokens=150, temperature=0.9).strip()
    meta = sb.artifact("sandbox_meta") or {}
    meta.setdefault("events", []).append({"scene": scene, "other": other_character, "log": events, "at": now_iso()})
    sb.save_artifact("sandbox_meta", meta)
    sb.log("sandbox_scene", scene[:80])
    sb.commit(); sb.close()
    return events


def storyteller(llm: LLM, events: list[dict], style_hint: str = "第三人称，克制，留白") -> str:
    log = "\n".join(f"[{e['who']}] {e['text']}" for e in events)
    return llm.complete(
        "你是叙事层。把下面的事件日志改写成一段小说片段。只用日志里发生的事，不要新增情节；可以调整节奏和视角。"
        f"风格：{style_hint}。", log, max_tokens=1500, temperature=0.7)


def promote(base_store_path: Path, sb_path: Path, claim_text: str, dimension: str) -> Claim:
    """Deliberate, manual copy of ONE fictional observation into the base persona, flagged as fiction.
    This should be rare; the default is that fiction never touches the persona."""
    base = Store(base_store_path)
    claim = Claim(id=new_id("clm"), dimension=dimension, claim=claim_text, strength="weak", confidence=0.3,
                  kind="observation", user_review="confirmed", tags=["origin:fiction", f"sandbox:{sb_path.name}"],
                  evidence=[Evidence(source_id="sandbox", ref_id=sb_path.name, quote=claim_text,
                                     evidence_type="other_report", grounded=False)])
    base.add_claim(claim)
    base.log("promote_from_fiction", f"{sb_path.name}: {claim_text[:60]}")
    base.commit(); base.close()
    return claim


def assert_isolated(base_store_path: Path) -> bool:
    """Sanity check used by tests: a read-only base must refuse writes."""
    base = Store(base_store_path, readonly=True)
    try:
        base.log("should_fail")
    except ReadOnlyError:
        base.close()
        return True
    base.close()
    return False
