"""Evaluation.

Blind test (style): sample real exchanges, generate ta's reply with the chat engine, shuffle
real/generated, write a judging sheet. The human marks which one is real; `score_blind` computes
the detection rate (50% = indistinguishable).

Backtest (decisions): hide known decision cases, ask the engine, compare its top choice with the
real choice. Also reports calibration (mean probability assigned to the real choice).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from .chat import ChatSession
from .decide import decide
from .llm import LLM
from .retrieve import Retriever
from .store import Store


def make_blind_test(store: Store, llm: LLM, n: int = 20, seed: int = 7, out_path: Optional[Path] = None) -> dict:
    rng = random.Random(seed)
    retriever = Retriever(store)
    pairs = []
    sessions = store.sessions()
    rng.shuffle(sessions)
    for sid in sessions:
        utts = store.utterances(session_id=sid)
        for i, u in enumerate(utts):
            if u.is_target or i + 1 >= len(utts) or not utts[i + 1].is_target or u.msg_type != "text":
                continue
            real = " / ".join(x.text for x in utts[i + 1:] if x.is_target and x.burst_id == utts[i + 1].burst_id)
            if len(real) < 4:
                continue
            sess = ChatSession(store, llm, relation=u.relation, retriever=retriever)
            # keep the held-out exchange out of the examples by excluding its own session text
            gen = sess.reply(u.text, allow_search=False).replace("\n", " / ")
            if rng.random() < 0.5:
                a, b, answer = real, gen, "A"
            else:
                a, b, answer = gen, real, "B"
            pairs.append({"id": len(pairs) + 1, "relation": u.relation, "context": u.text, "A": a, "B": b, "answer": answer})
            break
        if len(pairs) >= n:
            break
    sheet = {"instructions": "对每一题，判断 A 和 B 哪一个是 ta 真实说的，填入 guess（A/B）。answer 字段请先不要看。",
             "items": [{k: v for k, v in p.items() if k != "answer"} | {"guess": ""} for p in pairs],
             "_answers": {p["id"]: p["answer"] for p in pairs}}
    if out_path:
        out_path.write_text(json.dumps(sheet, ensure_ascii=False, indent=2), encoding="utf-8")
    return sheet


def score_blind(sheet_path: Path) -> dict:
    sheet = json.loads(Path(sheet_path).read_text(encoding="utf-8"))
    answers = {int(k): v for k, v in sheet["_answers"].items()}
    judged = [it for it in sheet["items"] if it.get("guess") in ("A", "B")]
    correct = sum(1 for it in judged if answers.get(it["id"]) == it["guess"])
    rate = correct / len(judged) if judged else 0.0
    return {"judged": len(judged), "correct": correct, "detection_rate": round(rate, 2),
            "verdict": "接近无法分辨" if judged and rate <= 0.6 else ("部分可辨" if rate <= 0.8 else "容易认出")}


def backtest(store: Store, llm: LLM, max_cases: int = 10) -> dict:
    """Leave-one-out over decision cases. Note: the retriever still sees other cases, which is the point."""
    cases = [c for c in store.cases() if c.choice and c.options and c.choice in c.options]
    results = []
    for case in cases[:max_cases]:
        situation = case.situation + "\n选项: " + " / ".join(case.options)
        dec = decide(store, llm, situation)
        probs = {o: dec.combined.get(o, 0.0) for o in case.options}
        top = max(probs, key=probs.get) if probs else None
        results.append({"case": case.id, "real": case.choice, "predicted": top,
                        "p_real": round(probs.get(case.choice, 0.0), 2), "hit": top == case.choice})
    n = len(results)
    return {"n": n, "accuracy": round(sum(r["hit"] for r in results) / n, 2) if n else None,
            "mean_p_real": round(sum(r["p_real"] for r in results) / n, 2) if n else None,
            "baseline_random": round(sum(1 / len(c.options) for c in cases[:max_cases]) / n, 2) if n else None,
            "results": results}
