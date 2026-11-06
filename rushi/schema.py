"""Core data models.

Design rules encoded here:
- Every claim carries evidence (quotes + source ids). No evidence → it is a guess, not a claim.
- Everything is bi-temporal: `valid_from/valid_to` = when it was true in the person's life,
  `recorded_at` = when we learned it. Superseded facts are never deleted, only closed.
- Contradictions are first-class objects, not errors.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def stable_id(prefix: str, *parts: str) -> str:
    h = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{h}"


# --------------------------------------------------------------------------- #
# Personality dimensions (the "drawers"). Empty drawers stay empty.
# --------------------------------------------------------------------------- #
DIMENSIONS: dict[str, str] = {
    "A": "身份与背景",
    "B": "人生时间线",
    "C": "价值观与优先级",
    "D": "世界观与立场（分议题）",
    "E": "道德判断",
    "F": "人际关系模式",
    "G": "感情观",
    "H": "工作、金钱与风险",
    "I": "兴趣与审美",
    "J": "童年与形成性经历（敏感）",
    "K": "压力与情绪",
    "L": "认知与决策风格",
    "M": "自我认知与人生叙事",
    "N": "恐惧与渴望",
    "O": "幽默与表达",
    "P": "矛盾与两难",
    "Q": "变化轨迹",
}

# Dimensions that never enter chat generation by default; back-office only.
SENSITIVE_DIMENSIONS = {"J"}

EVIDENCE_TYPES = ("behavior", "self_report", "other_report")
REVIEW_STATES = ("pending", "confirmed", "rejected", "edited")


@dataclass
class Source:
    id: str
    kind: str                      # chat | narrative | interview | user_note
    origin: str                    # file name or "cli"
    reporter: str                  # self | user | third:<alias>
    recorded_at: str = field(default_factory=now_iso)
    note: str = ""


@dataclass
class Utterance:
    id: str
    source_id: str
    session_id: str
    speaker: str                   # "target" or an alias like "other:朋友A"
    is_target: bool
    ts: Optional[str]              # ISO timestamp or None
    text: str
    msg_type: str = "text"         # text | sticker | image | voice | link | other
    burst_id: str = ""             # consecutive messages by the same speaker
    relation: str = "unknown"      # relation of the *other* party to the target
    year: Optional[int] = None


@dataclass
class Chunk:
    """A unit of narrative material (interview answer, user description, diary entry)."""
    id: str
    source_id: str
    text: str
    reporter: str                  # self | user | third:<alias>
    valid_from: Optional[str] = None   # time hint for the content ("2016", "2016-03")
    valid_to: Optional[str] = None
    relation: str = "unknown"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Evidence:
    source_id: str
    ref_id: str                    # chunk id or utterance id
    quote: str
    evidence_type: str             # behavior | self_report | other_report
    ts: Optional[str] = None
    grounded: bool = True          # quote verified against the raw material


@dataclass
class Claim:
    id: str
    dimension: str                 # key of DIMENSIONS
    claim: str
    strength: str = "weak"         # weak | moderate | strong
    confidence: float = 0.5
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    recorded_at: str = field(default_factory=now_iso)
    evidence: list[Evidence] = field(default_factory=list)
    counter_evidence: list[Evidence] = field(default_factory=list)
    user_review: str = "pending"
    kind: str = "observation"      # observation | reflection | user_stated
    tags: list[str] = field(default_factory=list)
    superseded_by: Optional[str] = None
    contradicts: list[str] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return self.user_review != "rejected" and self.superseded_by is None


@dataclass
class Node:
    id: str
    type: str                      # Person | Event | View | Value | Trait | Relationship | Place | Topic
    label: str
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    id: str
    from_id: str
    rel: str                       # HOLDS_VIEW | VALUES | HAS_TRAIT | EXPERIENCED | RELATES_TO | ...
    to_id: str
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    recorded_at: str = field(default_factory=now_iso)
    confidence: float = 0.5
    evidence_type: str = "self_report"
    source_ids: list[str] = field(default_factory=list)
    claim_id: Optional[str] = None
    invalidated_by: Optional[str] = None
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionCase:
    id: str
    when: Optional[str]
    situation: str
    options: list[str]
    choice: str
    stated_reason: str = ""
    outcome: str = ""
    reflection: str = ""
    source_ids: list[str] = field(default_factory=list)
    relation_context: str = ""
    confidence: float = 0.5


@dataclass
class Contradiction:
    id: str
    claim_a: str
    claim_b: str
    description: str
    aware: Optional[bool] = None   # does the person seem aware of it?
    how_handled: str = ""
    recorded_at: str = field(default_factory=now_iso)


def to_dict(obj: Any) -> dict:
    return asdict(obj)


def dumps(obj: Any) -> str:
    return json.dumps(asdict(obj), ensure_ascii=False)


def evidence_from_dict(d: dict) -> Evidence:
    return Evidence(**{k: d.get(k) for k in Evidence.__dataclass_fields__ if k in d})


def claim_from_dict(d: dict) -> Claim:
    d = dict(d)
    d["evidence"] = [evidence_from_dict(e) for e in d.get("evidence", [])]
    d["counter_evidence"] = [evidence_from_dict(e) for e in d.get("counter_evidence", [])]
    return Claim(**{k: v for k, v in d.items() if k in Claim.__dataclass_fields__})
