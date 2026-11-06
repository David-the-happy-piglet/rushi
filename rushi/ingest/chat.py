"""Parse chat exports into Utterances.

Supported inputs
- CSV with columns like time/sender/content (names detected loosely, Chinese or English)
- JSON list of objects with the same keys
- Plain text in common export shapes:
    2024-03-12 23:14:05 张三: 内容
    张三 2024-03-12 23:14:05 \n 内容            (WeChat-style two-line)
    [2024/03/12 23:14] 张三：内容

Speaker mapping ("which side is ta") is explicit and never guessed:
    speakers.json
    {
      "target": ["张三", "Me", "我"],            # names that are the person being modelled
      "others": {"李四": "close_friend", "王五": "colleague"},
      "default_relation": "unknown",
      "aliases": {"李四": "朋友A", "王五": "同事B"}  # optional anonymisation map
    }
Use `rushi inspect-chat` to list speakers first.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from ..schema import Utterance, stable_id

TIME_KEYS = ("time", "timestamp", "date", "datetime", "created_at", "时间", "日期", "createtime", "CreateTime")
SENDER_KEYS = ("sender", "from", "name", "user", "speaker", "author", "发送者", "发送人", "昵称", "nickname", "talker", "Sender")
TEXT_KEYS = ("content", "text", "message", "msg", "body", "内容", "消息", "StrContent", "Content")
TYPE_KEYS = ("type", "msg_type", "类型", "Type")

PLACEHOLDERS = {
    "sticker": "[表情包]", "emoji": "[表情包]", "image": "[图片]", "img": "[图片]", "pic": "[图片]",
    "voice": "[语音]", "audio": "[语音]", "video": "[视频]", "link": "[链接]", "url": "[链接]",
    "file": "[文件]", "transfer": "[转账]", "redpacket": "[红包]", "location": "[位置]",
    "表情": "[表情包]", "图片": "[图片]", "语音": "[语音]", "视频": "[视频]", "链接": "[链接]", "文件": "[文件]",
}
SYSTEM_PATTERNS = [r"撤回了一条消息", r"recalled a message", r"加入了群聊", r"邀请.*加入", r"^\[系统消息\]"]


@dataclass
class RawMessage:
    ts: Optional[str]
    sender: str
    text: str
    msg_type: str = "text"


# ---------------------------------------------------------------- time parsing
_TIME_FORMATS = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M",
                 "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y年%m月%d日 %H:%M", "%Y-%m-%d"]


def parse_time(s: Optional[str]) -> Optional[str]:
    if s is None or str(s).strip() == "":
        return None
    s = str(s).strip()
    if re.fullmatch(r"\d{10,13}", s):            # unix seconds / ms
        n = int(s[:10])
        return datetime.utcfromtimestamp(n).strftime("%Y-%m-%dT%H:%M:%S")
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%dT%H:%M:%S")
        except ValueError:
            continue
    return None


def _pick(d: dict, keys: Iterable[str]) -> Optional[str]:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return str(d[k])
    lowered = {str(k).lower(): v for k, v in d.items()}
    for k in keys:
        if k.lower() in lowered and lowered[k.lower()] not in (None, ""):
            return str(lowered[k.lower()])
    return None


# ----------------------------------------------------------------- readers
def read_messages(path: str | Path) -> list[RawMessage]:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return _read_csv(p)
    if suffix == ".json":
        return _read_json(p)
    return _read_txt(p)


def _normalise_type(t: Optional[str], text: str) -> tuple[str, str]:
    """Return (msg_type, text) with non-text content replaced by placeholders."""
    if t:
        key = str(t).strip().lower()
        for k, ph in PLACEHOLDERS.items():
            if k in key:
                return k if k.isascii() else ph.strip("[]"), ph
        if key not in ("text", "1", "文本", "message"):
            return "other", f"[{t}]"
    stripped = text.strip()
    for k, ph in PLACEHOLDERS.items():
        if stripped in (ph, f"[{k}]", f"<{k}>"):
            return k if k.isascii() else "other", ph
    if re.fullmatch(r"https?://\S+", stripped):
        return "link", "[链接]"
    return "text", text


def _read_csv(p: Path) -> list[RawMessage]:
    out = []
    with p.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            sender = _pick(row, SENDER_KEYS)
            text = _pick(row, TEXT_KEYS) or ""
            if sender is None:
                continue
            mt, text = _normalise_type(_pick(row, TYPE_KEYS), text)
            out.append(RawMessage(parse_time(_pick(row, TIME_KEYS)), sender.strip(), text, mt))
    return out


def _read_json(p: Path) -> list[RawMessage]:
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for k in ("messages", "data", "items", "msgs"):
            if k in data:
                data = data[k]
                break
    out = []
    for row in data:
        if not isinstance(row, dict):
            continue
        sender = _pick(row, SENDER_KEYS)
        text = _pick(row, TEXT_KEYS) or ""
        if sender is None:
            continue
        mt, text = _normalise_type(_pick(row, TYPE_KEYS), text)
        out.append(RawMessage(parse_time(_pick(row, TIME_KEYS)), sender.strip(), text, mt))
    return out


_LINE_PATTERNS = [
    # 2024-03-12 23:14:05 张三: 内容   |  2024/03/12 23:14 张三：内容
    re.compile(r"^\[?(?P<ts>\d{4}[-/]\d{1,2}[-/]\d{1,2}[ T]\d{1,2}:\d{2}(?::\d{2})?)\]?\s+(?P<sender>[^:：]{1,40})[:：]\s?(?P<text>.*)$"),
    # 张三 2024-03-12 23:14:05   (text on next line)
    re.compile(r"^(?P<sender>[^\s:：]{1,40})\s+(?P<ts>\d{4}[-/]\d{1,2}[-/]\d{1,2}[ T]\d{1,2}:\d{2}(?::\d{2})?)\s*$"),
    # 张三: 内容   (no timestamp)
    re.compile(r"^(?P<sender>[^\s:：]{1,40})[:：]\s?(?P<text>.+)$"),
]


def _read_txt(p: Path) -> list[RawMessage]:
    out: list[RawMessage] = []
    pending: Optional[RawMessage] = None
    for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        matched = False
        for pat in _LINE_PATTERNS:
            m = pat.match(line)
            if not m:
                continue
            gd = m.groupdict()
            ts = parse_time(gd.get("ts").replace("/", "-")) if gd.get("ts") else None
            if "text" in gd and gd["text"] is not None:
                if pending:
                    out.append(pending); pending = None
                mt, text = _normalise_type(None, gd["text"])
                out.append(RawMessage(ts, gd["sender"].strip(), text, mt))
            else:  # two-line style: header now, text follows
                if pending:
                    out.append(pending)
                pending = RawMessage(ts, gd["sender"].strip(), "", "text")
            matched = True
            break
        if not matched:
            if pending is not None:
                if pending.text == "":
                    mt, text = _normalise_type(None, line)
                    pending.text, pending.msg_type = text, mt
                else:
                    pending.text += "\n" + line
            elif out:
                out[-1].text += "\n" + line        # continuation of previous message
    if pending:
        out.append(pending)
    return out


# ------------------------------------------------------------- inspection
def inspect(messages: list[RawMessage]) -> dict:
    senders = Counter(m.sender for m in messages)
    times = [m.ts for m in messages if m.ts]
    return {
        "messages": len(messages),
        "senders": senders.most_common(),
        "time_range": (min(times), max(times)) if times else None,
        "sample": [(m.ts, m.sender, m.text[:40]) for m in messages[:8]],
    }


# ------------------------------------------------------------- mapping
@dataclass
class SpeakerMap:
    target: set[str]
    others: dict[str, str]
    default_relation: str = "unknown"
    aliases: dict[str, str] = None

    @classmethod
    def load(cls, path: str | Path) -> "SpeakerMap":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(target=set(d.get("target", [])), others=dict(d.get("others", {})),
                   default_relation=d.get("default_relation", "unknown"), aliases=dict(d.get("aliases", {})))

    def is_target(self, sender: str) -> bool:
        return sender in self.target

    def relation(self, sender: str) -> str:
        return self.others.get(sender, self.default_relation)

    def alias(self, sender: str) -> str:
        if self.aliases and sender in self.aliases:
            return self.aliases[sender]
        return sender


def is_system(text: str) -> bool:
    return any(re.search(p, text) for p in SYSTEM_PATTERNS)


def to_utterances(messages: list[RawMessage], smap: SpeakerMap, source_id: str,
                  session_gap_min: int = 45, burst_gap_sec: int = 120) -> list[Utterance]:
    """Attach target/other roles, relation, session ids and burst ids."""
    if not smap.target:
        raise ValueError("speakers.json must list at least one `target` name (which side is ta)")
    unknown = {m.sender for m in messages} - smap.target - set(smap.others)
    if unknown:
        # Not fatal: unknown senders become `other` with the default relation. Surfaced to the user.
        print(f"[ingest] senders not in speakers.json (treated as others/{smap.default_relation}): {sorted(unknown)[:10]}")

    # relation of the conversation = relation of the (first) non-target participant
    others = [m.sender for m in messages if not smap.is_target(m.sender)]
    conv_relation = smap.relation(others[0]) if others else smap.default_relation

    out: list[Utterance] = []
    prev_ts: Optional[datetime] = None
    prev_speaker: Optional[str] = None
    session_idx, burst_idx = 0, 0
    for i, m in enumerate(messages):
        if is_system(m.text) or not m.text.strip():
            continue
        ts = datetime.strptime(m.ts, "%Y-%m-%dT%H:%M:%S") if m.ts else None
        gap = (ts - prev_ts).total_seconds() if (ts and prev_ts) else None
        if gap is None and prev_ts is not None and ts is None:
            gap = 0
        if prev_ts is None or (gap is not None and gap > session_gap_min * 60):
            session_idx += 1
            burst_idx += 1
        elif m.sender != prev_speaker or (gap is not None and gap > burst_gap_sec):
            burst_idx += 1
        is_t = smap.is_target(m.sender)
        out.append(Utterance(
            id=stable_id("utt", source_id, str(i), m.sender, m.text[:50]),
            source_id=source_id,
            session_id=f"{source_id}:s{session_idx:04d}",
            speaker="target" if is_t else f"other:{smap.alias(m.sender)}",
            is_target=is_t,
            ts=m.ts,
            text=m.text,
            msg_type=m.msg_type,
            burst_id=f"{source_id}:b{burst_idx:06d}",
            relation=conv_relation if not is_t else conv_relation,
            year=ts.year if ts else None,
        ))
        prev_ts = ts or prev_ts
        prev_speaker = m.sender
    return out
