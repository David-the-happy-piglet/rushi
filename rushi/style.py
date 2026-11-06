"""Style engine: how ta talks.

Quantitative features from the target's own utterances, per relation and per year,
then an LLM turns stats + examples into hard rules. Both are versioned artifacts.
"""
from __future__ import annotations

import re
import statistics
from collections import Counter, defaultdict
from typing import Optional

from .extract import load_prompt
from .llm import LLM
from .schema import Utterance
from .store import Store

LAUGH_RE = re.compile(r"(哈{2,}|嘿{2,}|hh+|xs|笑死|233+|lol|lmao)", re.I)
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]{2,}")
STOP_GRAMS = {"的话", "一个", "什么", "可以", "就是", "这个", "那个", "我们", "你们", "他们", "没有", "不是",
              "然后", "因为", "所以", "但是", "如果", "还是", "现在", "时候", "知道", "觉得", "这样", "那样"}


def _features(utts: list[Utterance]) -> dict:
    texts = [u.text for u in utts if u.msg_type == "text" and u.text.strip()]
    if not texts:
        return {"n": 0}
    lengths = [len(t) for t in texts]
    bursts = Counter(u.burst_id for u in utts)
    burst_sizes = list(bursts.values())
    ends = Counter(t.strip()[-1] for t in texts if t.strip())
    punct = {
        "ends_with_period": sum(1 for t in texts if t.strip().endswith(("。", "."))) / len(texts),
        "exclaim_rate": sum(t.count("!") + t.count("！") for t in texts) / len(texts),
        "question_rate": sum(1 for t in texts if t.strip().endswith(("?", "？"))) / len(texts),
        "ellipsis_rate": sum(1 for t in texts if "…" in t or "..." in t or "。。" in t) / len(texts),
        "tilde_rate": sum(1 for t in texts if "~" in t or "～" in t) / len(texts),
        "space_as_comma": sum(1 for t in texts if re.search(r"[\u4e00-\u9fff] [\u4e00-\u9fff]", t)) / len(texts),
        "no_punct_rate": sum(1 for t in texts if not re.search(r"[，。！？,.!?]", t)) / len(texts),
    }
    laughs = Counter(m.group(0).lower() for t in texts for m in LAUGH_RE.finditer(t))
    sticker_rate = sum(1 for u in utts if u.msg_type in ("sticker", "emoji")) / max(len(utts), 1)
    emoji_rate = sum(1 for t in texts if EMOJI_RE.search(t)) / len(texts)
    latin_rate = sum(1 for t in texts if LATIN_RE.search(t)) / len(texts)
    grams = Counter()
    for t in texts:
        s = CJK_RE.sub(lambda m: m.group(0), t)
        for n in (2, 3):
            for i in range(len(s) - n + 1):
                g = s[i:i + n]
                if CJK_RE.fullmatch(g[0]) and g not in STOP_GRAMS and not re.search(r"[\s，。！？,.!?]", g):
                    grams[g] += 1
    catchphrases = [(g, c) for g, c in grams.most_common(60) if c >= max(3, len(texts) // 50)][:15]
    return {
        "n": len(texts),
        "len_mean": round(statistics.mean(lengths), 1),
        "len_median": statistics.median(lengths),
        "len_p90": sorted(lengths)[int(len(lengths) * 0.9) - 1] if len(lengths) > 1 else lengths[0],
        "short_rate(<=10)": round(sum(1 for L in lengths if L <= 10) / len(lengths), 2),
        "burst_mean": round(statistics.mean(burst_sizes), 2) if burst_sizes else 1,
        "burst_p90": sorted(burst_sizes)[int(len(burst_sizes) * 0.9) - 1] if len(burst_sizes) > 1 else 1,
        "punct": {k: round(v, 2) for k, v in punct.items()},
        "top_endings": ends.most_common(6),
        "laughs": laughs.most_common(5),
        "sticker_rate": round(sticker_rate, 2),
        "emoji_rate": round(emoji_rate, 2),
        "latin_mix_rate": round(latin_rate, 2),
        "catchphrases": catchphrases,
    }


def profile(store: Store) -> dict:
    utts = store.utterances(is_target=True)
    by_rel: dict[str, list[Utterance]] = defaultdict(list)
    by_year: dict[str, list[Utterance]] = defaultdict(list)
    for u in utts:
        by_rel[u.relation].append(u)
        by_year[str(u.year or "unknown")].append(u)
    return {
        "overall": _features(utts),
        "by_relation": {k: _features(v) for k, v in by_rel.items()},
        "by_year": {k: _features(v) for k, v in sorted(by_year.items())},
    }


def pick_examples(store: Store, relation: Optional[str] = None, k: int = 12, min_len: int = 4) -> list[str]:
    """Context→reply pairs that show the person's voice: other's message followed by ta's burst."""
    out = []
    for sid in store.sessions():
        utts = store.utterances(session_id=sid)
        if relation and utts and utts[0].relation != relation:
            continue
        i = 0
        while i < len(utts):
            if utts[i].is_target or utts[i].msg_type != "text":
                i += 1
                continue
            j = i + 1
            while j < len(utts) and not utts[j].is_target:
                j += 1
            if j >= len(utts):
                break
            burst = []
            b = utts[j].burst_id
            while j < len(utts) and utts[j].is_target and utts[j].burst_id == b:
                burst.append(utts[j].text); j += 1
            reply = " / ".join(burst)
            if len(reply) >= min_len:
                out.append(f"[{utts[i].speaker.split(':',1)[-1]}] {utts[i].text}\n[ta] {reply}")
            i = j
    # prefer longer, more characteristic replies but keep variety
    out.sort(key=lambda s: -len(s.split("[ta] ")[-1]))
    return out[: k * 3][::3][:k] if len(out) > k else out


def render_stats(stats: dict) -> str:
    f = stats.get("overall", {})
    if not f or f.get("n", 0) == 0:
        return "（没有 ta 的文本消息）"
    lines = [
        f"消息数 {f['n']}；平均 {f['len_mean']} 字，中位 {f['len_median']} 字，90% 的消息 ≤ {f['len_p90']} 字；≤10 字的占 {f['short_rate(<=10)']}",
        f"连发：平均一次 {f['burst_mean']} 条，90% 不超过 {f['burst_p90']} 条",
        f"标点：句号结尾 {f['punct']['ends_with_period']}；感叹号/条 {f['punct']['exclaim_rate']}；问号结尾 {f['punct']['question_rate']}；"
        f"省略号 {f['punct']['ellipsis_rate']}；波浪号 {f['punct']['tilde_rate']}；用空格代替逗号 {f['punct']['space_as_comma']}；完全无标点 {f['punct']['no_punct_rate']}",
        f"常见结尾字：{f['top_endings']}",
        f"笑的写法：{f['laughs']}",
        f"表情包占比 {f['sticker_rate']}；含 emoji 的消息 {f['emoji_rate']}；中英夹杂 {f['latin_mix_rate']}",
        f"高频词/口头禅：{[g for g, _ in f['catchphrases']]}",
    ]
    for rel, fr in stats.get("by_relation", {}).items():
        if fr.get("n", 0) >= 20:
            lines.append(f"对 {rel}（{fr['n']} 条）：平均 {fr['len_mean']} 字，连发 {fr['burst_mean']}，感叹号 {fr['punct']['exclaim_rate']}，表情包 {fr['sticker_rate']}")
    for yr, fy in stats.get("by_year", {}).items():
        if fy.get("n", 0) >= 20:
            lines.append(f"{yr} 年（{fy['n']} 条）：平均 {fy['len_mean']} 字，笑 {fy['laughs'][:2]}")
    return "\n".join(lines)


def build_style_card(store: Store, llm: LLM) -> dict:
    stats = profile(store)
    examples = pick_examples(store, k=15)
    rules = llm.complete(load_prompt("style_rules"),
                         "## 统计特征\n" + render_stats(stats) + "\n\n## 真实示例\n" + "\n\n".join(examples))
    card = {"stats": stats, "rules": rules.strip(), "example_count": len(examples)}
    v = store.save_artifact("style_card", card)
    store.log("style_card", f"v{v}")
    store.commit()
    card["version"] = v
    return card
