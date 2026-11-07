"""Tiny .env loader (stdlib only).

Looks for, in order: ./.env (repo / cwd), then $RUSHI_HOME/.env (default ~/.rushi/.env).
Real environment variables always win; earlier files win over later ones.
Both files are git-ignored / outside the repo — never commit API keys.
"""
from __future__ import annotations

import os
from pathlib import Path

_loaded = False


def _parse(path: Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip().removeprefix("export ").strip()
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]
        elif " #" in v:
            v = v.split(" #", 1)[0].rstrip()
        if k:
            out[k] = v
    return out


def load_env() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    home = Path(os.environ.get("RUSHI_HOME", "~/.rushi")).expanduser()
    for p in (Path.cwd() / ".env", home / ".env"):
        if p.is_file():
            for k, v in _parse(p).items():
                if v and k not in os.environ:
                    os.environ[k] = v
