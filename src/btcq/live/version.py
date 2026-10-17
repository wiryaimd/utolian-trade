"""A short id of the installed btcq code, so the dashboard can show which version the bot and the dashboard run."""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def code_id() -> str:
    """First 8 hex digits of a SHA-256 over every .py and .html file of the btcq package (path and content)."""
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.suffix in (".py", ".html") and "__pycache__" not in p.parts:
            h.update(p.relative_to(root).as_posix().encode())
            h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:8]
