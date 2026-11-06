"""LLM providers.

- AnthropicLLM: calls the Messages API with urllib (no SDK dependency). Needs ANTHROPIC_API_KEY.
- OpenAICompatLLM: any OpenAI-compatible endpoint (Ollama, vLLM, LM Studio) for local models.
- MockLLM: deterministic, offline. Lets the whole pipeline run end-to-end in tests and demos.

Select with env RUSHI_LLM = anthropic | openai | mock (default: mock if no key is set).
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Optional


class LLM:
    name = "base"

    def complete(self, system: str, user: str, max_tokens: int = 2000, temperature: float = 0.3) -> str:
        raise NotImplementedError

    def complete_json(self, system: str, user: str, max_tokens: int = 3000) -> dict | list:
        text = self.complete(system + "\n\n只输出 JSON，不要任何前言、解释或 Markdown 代码块。", user, max_tokens, 0.0)
        return parse_json(text)


def parse_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.S)
        if m:
            return json.loads(m.group(1))
        raise


class AnthropicLLM(LLM):
    name = "anthropic"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model or os.environ.get("RUSHI_MODEL", "claude-sonnet-4-6")
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")

    def complete(self, system, user, max_tokens=2000, temperature=0.3) -> str:
        body = json.dumps({
            "model": self.model, "max_tokens": max_tokens, "temperature": temperature,
            "system": system, "messages": [{"role": "user", "content": user}],
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=body, method="POST",
            headers={"content-type": "application/json", "x-api-key": self.api_key,
                     "anthropic-version": "2023-06-01"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Anthropic API error {e.code}: {e.read().decode('utf-8', 'ignore')[:500]}")
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


class OpenAICompatLLM(LLM):
    """Works with Ollama (http://localhost:11434/v1), vLLM, LM Studio, etc."""
    name = "openai"

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None, api_key: Optional[str] = None):
        self.base_url = (base_url or os.environ.get("RUSHI_OPENAI_BASE", "http://localhost:11434/v1")).rstrip("/")
        self.model = model or os.environ.get("RUSHI_MODEL", "qwen3:8b")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "local")

    def complete(self, system, user, max_tokens=2000, temperature=0.3) -> str:
        body = json.dumps({"model": self.model, "max_tokens": max_tokens, "temperature": temperature,
                           "messages": [{"role": "system", "content": system},
                                        {"role": "user", "content": user}]}).encode("utf-8")
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, method="POST",
                                     headers={"content-type": "application/json",
                                              "authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]


class MockLLM(LLM):
    """Offline stand-in. Produces plausible, *grounded* structured output by simple heuristics,
    so that extraction → claims → persona → chat → decide can be exercised without an API key.
    Quality is intentionally low; it exists to test plumbing."""
    name = "mock"

    KEYWORDS = {
        "C": ["重要", "优先", "宁愿", "在乎", "原则", "值得", "必须", "绝不", "宁可", "最看重"],
        "D": ["政治", "国家", "政府", "社会", "制度", "政策", "公平", "自由", "选举", "历史"],
        "F": ["朋友", "吵架", "冷战", "道歉", "信任", "闺蜜", "兄弟", "同事", "父母", "妈", "爸"],
        "G": ["喜欢", "恋爱", "分手", "结婚", "婚姻", "前任", "表白", "暧昧", "爱情", "男朋友", "女朋友"],
        "H": ["工作", "offer", "辞职", "工资", "钱", "投资", "买房", "风险", "创业", "加班"],
        "I": ["喜欢看", "爱听", "游戏", "电影", "音乐", "旅行", "吃", "咖啡", "书", "动漫"],
        "K": ["压力", "焦虑", "崩溃", "烦", "累", "失眠", "哭", "生气", "难受", "紧张"],
        "L": ["决定", "犹豫", "拖", "纠结", "直觉", "想清楚", "后悔", "冲动", "计划"],
        "M": ["我这个人", "我一直", "我觉得我", "我的人生", "我从小", "一直以来"],
        "N": ["害怕", "怕", "想要", "梦想", "渴望", "担心", "希望"],
        "O": ["哈哈", "笑死", "开玩笑", "搞笑", "无语", "绝了"],
        "J": ["小时候", "童年", "小学", "初中", "那时候家里", "我爸妈当年"],
    }

    def complete(self, system, user, max_tokens=2000, temperature=0.3) -> str:
        if "EXTRACT_CLAIMS" in system:
            return self._extract(user)
        if "PARSE_SITUATION" in system:
            return self._parse_situation(user)
        if "SCORE_OPTIONS" in system:
            return self._score(user)
        if "NARRATIVE_DECISION" in system:
            return self._narrative(user)
        if "PERSONA_CARD" in system:
            return self._persona(user)
        if "REFLECT" in system:
            return json.dumps({"insights": [], "contradictions": []}, ensure_ascii=False)
        if "STYLE_RULES" in system:
            return "- 模仿示例的长度和拆条\n- 不用句号结尾\n- 保持口头禅"
        # chat / sandbox: echo a terse reply built from retrieved examples when available
        m = re.search(r"\[ta\]\s*(.+)", system + "\n" + user)
        return m.group(1).strip().replace(" / ", "\n") if m else "嗯"

    def _extract(self, user: str) -> str:
        m = re.search(r"<material>\s*([\s\S]*?)\s*</material>", user)
        text = m.group(1) if m else user
        sentences = [s.strip() for s in re.split(r"[。！？\n]", text) if len(s.strip()) >= 6]
        claims = []
        for s in sentences:
            body = s.split("]", 1)[-1].strip() if s.startswith("[") else s
            for dim, kws in self.KEYWORDS.items():
                if any(k in body for k in kws):
                    claims.append({"dimension": dim, "claim": body[:60], "quote": body[:80],
                                   "evidence_type": "self_report" if s.startswith("[ta]") or "我" in body else "other_report",
                                   "strength": "weak", "confidence": 0.5})
                    break
        cases = []
        for s in sentences:
            if ("选择" in s or "决定" in s or "拒绝" in s or "接受" in s) and len(s) > 10:
                cases.append({"situation": s[:60], "options": ["做", "不做"], "choice": "不确定",
                              "stated_reason": "", "quote": s[:80], "when": None})
        return json.dumps({"claims": claims[:8], "events": [], "relations": [], "cases": cases[:2]}, ensure_ascii=False)

    def _parse_situation(self, user: str) -> str:
        m = re.search(r"<situation>\s*([\s\S]*?)\s*</situation>", user)
        text = m.group(1) if m else user
        opts = re.findall(r"(?:选项|option)\s*[:：]?\s*(.+)", text)
        if opts:
            options = [o.strip() for o in re.split(r"[/、,，|]", opts[0]) if o.strip()]
        else:
            options = ["接受", "拒绝", "拖着不决定"]
        return json.dumps({"summary": text[:80], "options": options,
                           "people_involved": [], "values_touched": []}, ensure_ascii=False)

    def _score(self, user: str) -> str:
        opts = re.findall(r"- (.+)", user.split("<options>")[-1].split("</options>")[0]) if "<options>" in user else []
        vals = re.findall(r"- (.+)", user.split("<values>")[-1].split("</values>")[0]) if "<values>" in user else []
        matrix = {o: {v.split(" (")[0]: 0.0 for v in vals} for o in opts}
        return json.dumps({"impacts": matrix}, ensure_ascii=False)

    def _narrative(self, user: str) -> str:
        opts = re.findall(r"- (.+)", user.split("<options>")[-1].split("</options>")[0]) if "<options>" in user else ["A"]
        n = max(len(opts), 1)
        return json.dumps({"reasoning": "（mock）以第一人称推理略。", "probabilities": {o: round(1 / n, 2) for o in opts},
                           "would_say": "我再想想", "evidence_used": [], "uncertainty": "证据不足"}, ensure_ascii=False)

    def _persona(self, user: str) -> str:
        lines = re.findall(r"^- \[(\w)\] (.+)$", user, flags=re.M)
        card = ["# 人格卡（mock 生成）", "", "## 核心特质"]
        card += [f"- {t}" for _, t in lines[:6]] or ["- （证据不足）"]
        card += ["", "## 不确定/缺证据", "- 大部分维度"]
        return "\n".join(card)


def get_llm(kind: Optional[str] = None) -> LLM:
    kind = (kind or os.environ.get("RUSHI_LLM") or ("anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "mock")).lower()
    if kind == "anthropic":
        return AnthropicLLM()
    if kind in ("openai", "local", "ollama"):
        return OpenAICompatLLM()
    return MockLLM()
