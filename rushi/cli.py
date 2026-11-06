"""rushi — 如是.  CLI for building and querying a single persona.

Workspace layout (one persona per workspace):
    <ws>/persona.db           base memory (never written by fiction)
    <ws>/speakers.json        which chat names are ta, and who the others are
    <ws>/sandboxes/*.db       fiction sandboxes
    <ws>/exports/             graph / blind-test / backtest outputs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import evaluate, persona, sandbox, style
from .chat import ChatSession
from .decide import decide
from .extract import add_user_fact, extract_all
from .ingest import chat as chat_ingest
from .ingest.narrative import chat_sessions_to_chunks, to_chunks
from .llm import get_llm
from .schema import DIMENSIONS, Source, new_id, to_dict
from .store import Store


def _ws(args) -> Path:
    ws = Path(args.workspace).expanduser()
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "exports").mkdir(exist_ok=True)
    return ws


def _store(args) -> Store:
    return Store(_ws(args) / "persona.db")


def _period(s):
    if not s:
        return None
    a, _, b = s.partition("-")
    return (int(a), int(b or a))


# ----------------------------------------------------------------- commands
def cmd_init(args):
    ws = _ws(args)
    Store(ws / "persona.db").close()
    sp = ws / "speakers.json"
    if not sp.exists():
        sp.write_text(json.dumps({"target": ["<ta 在聊天记录里的名字>"], "others": {"<对方名字>": "close_friend"},
                                  "default_relation": "unknown", "aliases": {"<对方名字>": "朋友A"}},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"workspace ready: {ws}\n编辑 {sp} 指定哪一边是 ta。可用关系: close_friend, friend, family, partner, colleague, acquaintance, unknown")


def cmd_inspect_chat(args):
    info = chat_ingest.inspect(chat_ingest.read_messages(args.file))
    print(json.dumps(info, ensure_ascii=False, indent=2, default=str))
    print("\n把 ta 的名字填进 speakers.json 的 target，其他人填进 others。")


def cmd_ingest_chat(args):
    ws, store = _ws(args), _store(args)
    smap = chat_ingest.SpeakerMap.load(args.speakers or ws / "speakers.json")
    msgs = chat_ingest.read_messages(args.file)
    src = Source(id=new_id("src"), kind="chat", origin=Path(args.file).name, reporter="self", note=f"{len(msgs)} messages")
    store.add_source(src)
    utts = chat_ingest.to_utterances(msgs, smap, src.id, session_gap_min=args.session_gap)
    # anonymise names inside message text using aliases
    if smap.aliases:
        from .ingest.narrative import anonymize
        for u in utts:
            u.text = anonymize(u.text, smap.aliases)
    store.add_utterances(utts)
    chunks = chat_sessions_to_chunks(store, src.id)
    for c in chunks:
        store.add_chunk(c)
    store.log("ingest_chat", f"{src.origin}: {len(utts)} utterances, {len(chunks)} chunks")
    store.commit()
    n_t = sum(1 for u in utts if u.is_target)
    print(f"ingested {len(utts)} messages ({n_t} by ta) into {len(set(u.session_id for u in utts))} sessions; {len(chunks)} chunks queued for extraction")


def cmd_ingest_text(args):
    store = _store(args)
    text = Path(args.file).read_text(encoding="utf-8")
    src = Source(id=new_id("src"), kind=args.kind, origin=Path(args.file).name, reporter=args.reporter)
    store.add_source(src)
    name_map = {}
    sp = _ws(args) / "speakers.json"
    if sp.exists():
        name_map = json.loads(sp.read_text(encoding="utf-8")).get("aliases", {})
    chunks = to_chunks(text, src.id, default_reporter=args.reporter, name_map=name_map)
    for c in chunks:
        store.add_chunk(c)
    store.log("ingest_text", f"{src.origin}: {len(chunks)} chunks")
    store.commit()
    print(f"ingested {len(chunks)} chunks (reporter={args.reporter})")


def cmd_add_fact(args):
    store = _store(args)
    c = add_user_fact(store, args.text, args.dimension.upper(), valid_from=args.year, reporter=args.reporter)
    print(f"added {c.id} [{c.dimension}] {c.claim}")


def cmd_style(args):
    store, llm = _store(args), get_llm(args.llm)
    card = style.build_style_card(store, llm)
    print(f"style card v{card['version']} ({card['example_count']} examples)\n")
    print(style.render_stats(card["stats"]))
    print("\n--- rules ---\n" + card["rules"])


def cmd_extract(args):
    store, llm = _store(args), get_llm(args.llm)
    print(f"llm = {llm.name}")
    print(json.dumps(extract_all(store, llm, limit=args.limit), ensure_ascii=False))


def cmd_reflect(args):
    store, llm = _store(args), get_llm(args.llm)
    print(json.dumps(persona.reflect(store, llm), ensure_ascii=False))


def cmd_review(args):
    store = _store(args)
    if args.action == "list":
        for c in persona.review_queue(store, limit=args.limit):
            ev = c.evidence[0] if c.evidence else None
            flag = ("敏感 " if c.dimension in ("J",) else "") + ("未落地 " if "ungrounded" in c.tags else "") + ("矛盾 " if "contradiction" in c.tags else "")
            print(f"{c.id}  [{c.dimension} {DIMENSIONS[c.dimension]}] conf={c.confidence} {flag}\n    {c.claim}\n    证据({ev.evidence_type if ev else '?'}): {ev.quote[:70] if ev else ''}\n")
    else:
        c = persona.set_review(store, args.claim_id, args.action, edited_text=args.text)
        print(f"{c.id} -> {c.user_review}")


def cmd_coverage(args):
    for dim, name, n, nc, mean in persona.coverage(_store(args)):
        bar = "█" * min(n, 30)
        print(f"{dim} {name:<16} {n:>3} 条 (已确认 {nc:>2}, 平均置信 {mean:.2f}) {bar}")


def cmd_persona(args):
    store, llm = _store(args), get_llm(args.llm)
    card = persona.build_persona_card(store, llm, include_sensitive=args.include_sensitive)
    print(f"persona card v{card['version']} from {card['n_claims']} claims, {card['n_contradictions']} contradictions\n")
    print(card["markdown"])


def cmd_chat(args):
    store, llm = _store(args), get_llm(args.llm)
    sess = ChatSession(store, llm, relation=args.relation, period=_period(args.period))
    print(f"chatting as ta (relation={args.relation}, llm={llm.name}). 输入 /quit 退出。")
    while True:
        try:
            msg = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if msg in ("/quit", "/q", ""):
            break
        print("ta: " + sess.reply(msg).replace("\n", "\nta: "))


def cmd_ask(args):
    store, llm = _store(args), get_llm(args.llm)
    dec = decide(store, llm, args.situation, period=_period(args.period), include_sensitive=args.include_sensitive)
    print(dec.render())


def cmd_export_graph(args):
    store, ws = _store(args), _ws(args)
    data = {"nodes": [to_dict(n) for n in store.nodes()], "edges": [to_dict(e) for e in store.edges()],
            "claims": [to_dict(c) for c in store.claims(active_only=False)],
            "cases": [to_dict(c) for c in store.cases()], "contradictions": [to_dict(c) for c in store.contradictions()]}
    out = ws / "exports" / "graph.json"
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out} ({len(data['nodes'])} nodes, {len(data['edges'])} edges, {len(data['claims'])} claims)")


def cmd_stats(args):
    store = _store(args)
    print(json.dumps(store.stats(), ensure_ascii=False))
    for key in ("style_card", "persona_card"):
        print(f"{key}: versions {store.artifact_versions(key)}")


def cmd_eval(args):
    store, llm, ws = _store(args), get_llm(args.llm), _ws(args)
    if args.which == "blind":
        out = ws / "exports" / "blind_test.json"
        sheet = evaluate.make_blind_test(store, llm, n=args.n, out_path=out)
        print(f"wrote {out} with {len(sheet['items'])} items. 填好 guess 后运行: rushi eval score")
    elif args.which == "score":
        print(json.dumps(evaluate.score_blind(ws / "exports" / "blind_test.json"), ensure_ascii=False))
    elif args.which == "backtest":
        print(json.dumps(evaluate.backtest(store, llm, max_cases=args.n), ensure_ascii=False, indent=1))


def cmd_sandbox(args):
    ws, llm = _ws(args), get_llm(args.llm)
    base = ws / "persona.db"
    if args.action == "create":
        p = sandbox.create_sandbox(ws, base, args.name)
        print(f"sandbox created: {p}  (base persona is read-only inside the sandbox)")
    elif args.action == "scene":
        p = ws / "sandboxes" / f"{args.name}.db"
        events = sandbox.run_scene(p, llm, args.scene, args.other, ticks=args.ticks, relation=args.relation)
        for e in events:
            print(f"[{e['who']}] {e['text']}")
        if args.story:
            print("\n--- 叙事层 ---\n" + sandbox.storyteller(llm, events))
    elif args.action == "isolation-check":
        print("isolated" if sandbox.assert_isolated(base) else "NOT isolated")


def main(argv=None):
    p = argparse.ArgumentParser(prog="rushi", description="如是 — persona memory & simulation")
    p.add_argument("-w", "--workspace", default="./workspace", help="persona workspace dir")
    p.add_argument("--llm", default=None, help="anthropic | openai | mock (default from env RUSHI_LLM)")
    sp = p.add_subparsers(dest="cmd", required=True)

    sp.add_parser("init").set_defaults(fn=cmd_init)
    a = sp.add_parser("inspect-chat"); a.add_argument("file"); a.set_defaults(fn=cmd_inspect_chat)
    a = sp.add_parser("ingest-chat"); a.add_argument("file"); a.add_argument("--speakers"); a.add_argument("--session-gap", type=int, default=45); a.set_defaults(fn=cmd_ingest_chat)
    a = sp.add_parser("ingest-text"); a.add_argument("file"); a.add_argument("--reporter", default="user", help="self | user | third:<alias>"); a.add_argument("--kind", default="narrative"); a.set_defaults(fn=cmd_ingest_text)
    a = sp.add_parser("add-fact"); a.add_argument("text"); a.add_argument("--dimension", default="C"); a.add_argument("--year"); a.add_argument("--reporter", default="user"); a.set_defaults(fn=cmd_add_fact)
    sp.add_parser("style").set_defaults(fn=cmd_style)
    a = sp.add_parser("extract"); a.add_argument("--limit", type=int); a.set_defaults(fn=cmd_extract)
    sp.add_parser("reflect").set_defaults(fn=cmd_reflect)
    a = sp.add_parser("review"); a.add_argument("action", choices=["list", "confirmed", "rejected", "edited"]); a.add_argument("claim_id", nargs="?"); a.add_argument("--text"); a.add_argument("--limit", type=int, default=20); a.set_defaults(fn=cmd_review)
    sp.add_parser("coverage").set_defaults(fn=cmd_coverage)
    a = sp.add_parser("persona"); a.add_argument("--include-sensitive", action="store_true"); a.set_defaults(fn=cmd_persona)
    a = sp.add_parser("chat"); a.add_argument("--relation", default="close_friend"); a.add_argument("--period", help="e.g. 2018-2020"); a.set_defaults(fn=cmd_chat)
    a = sp.add_parser("ask"); a.add_argument("situation"); a.add_argument("--period"); a.add_argument("--include-sensitive", action="store_true"); a.set_defaults(fn=cmd_ask)
    sp.add_parser("export-graph").set_defaults(fn=cmd_export_graph)
    sp.add_parser("stats").set_defaults(fn=cmd_stats)
    a = sp.add_parser("eval"); a.add_argument("which", choices=["blind", "score", "backtest"]); a.add_argument("-n", type=int, default=20); a.set_defaults(fn=cmd_eval)
    a = sp.add_parser("sandbox"); a.add_argument("action", choices=["create", "scene", "isolation-check"]); a.add_argument("--name", default="draft1")
    a.add_argument("--scene", default=""); a.add_argument("--other", default="一个陌生人"); a.add_argument("--ticks", type=int, default=6); a.add_argument("--relation", default="unknown"); a.add_argument("--story", action="store_true"); a.set_defaults(fn=cmd_sandbox)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
