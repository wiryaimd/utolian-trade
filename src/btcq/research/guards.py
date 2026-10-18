"""Pre-registration guards shared by rounds from 7 on: file hashes, committed logs, registration sections.

They are the Round 6 patterns lifted out of round6.py (which keeps its own copies unchanged)."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

from btcq.config import ROOT


class GitError(SystemExit):
    """git could not answer: every guard that needs it fails closed."""


def git(*args, strict: bool = False) -> str:
    """git output decoded as UTF-8. strict: a failing git raises GitError instead of returning ""."""
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                              check=True).stdout.strip()
    except Exception as e:  # noqa: BLE001
        if strict:
            raise GitError(f"git {' '.join(args[:2])} failed: {type(e).__name__}") from e
        return ""


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha_text(p: Path) -> str:
    """SHA256 of a text file with CRLF normalised (git may rewrite line endings on checkout)."""
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sha_str(s: str) -> str:
    return hashlib.sha256(s.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def code_sha(files: list[str]) -> str:
    """SHA256 over source files (paths relative to ROOT, line endings normalised)."""
    h = hashlib.sha256()
    for f in files:
        h.update(f.encode())
        h.update((ROOT / f).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def require_committed(paths) -> None:
    """Every path exists, is tracked by git and has no uncommitted change."""
    rels = [rel(p) for p in paths]
    missing = [r for r in rels if not (ROOT / r).exists()]
    if missing:
        raise SystemExit(f"missing: {missing}")
    untracked = [r for r in rels if not git("ls-files", "--error-unmatch", "--", r)]
    if untracked or git("status", "--porcelain", "--", *rels, strict=True):
        raise SystemExit(f"commit these first (pre-registration): {untracked or rels}")


def clean_src() -> None:
    if git("status", "--porcelain", "--", "src", strict=True):
        raise SystemExit("src/ has uncommitted changes: commit the code before a pre-registered run")


def section(text: str, heading: str) -> str:
    """The '## heading...' section of a markdown text, up to the next '## ' heading."""
    if text.startswith(heading):
        i = 0
    else:
        i = text.find("\n" + heading)
        if i < 0:
            return ""
        i += 1
    j = text.find("\n## ", i + 1)
    return text[i:] if j < 0 else text[i:j]


def committed_section(prereg: Path, heading: str) -> str:
    return section(git("show", f"HEAD:{rel(prereg)}", strict=True).replace("\r\n", "\n"), heading)


def require_section(prereg: Path, heading: str, must_contain=()) -> str:
    """The committed registration must hold the section, with every required phrase and no {PLACEHOLDER}."""
    require_committed([prereg])
    sec = committed_section(prereg, heading)
    if not sec:
        raise SystemExit(f"{rel(prereg)} has no committed section '{heading}'")
    missing = [m for m in must_contain if m not in sec]
    if missing:
        raise SystemExit(f"the committed section '{heading}' lacks: {missing}")
    if re.search(r"\{[A-Z_]+\}", sec):
        raise SystemExit(f"the committed section '{heading}' still has placeholders")
    return sec


def committed_versions(path: Path) -> list[str]:
    """Every committed version of a file, on any branch: a record once committed can't be erased by a checkout,
    a forward commit that restores an older file, or a branch cut before it."""
    try:
        r = rel(path)
    except ValueError:
        return []
    revs = git("log", "--all", "--format=%H", "--", r).split()
    return [t for t in (git("show", f"{c}:{r}") for c in revs) if t]


def records(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                out.append({"raw": line})
    return out


def all_records(path: Path, *, committed_only: bool = False) -> list[dict]:
    """The distinct records of a JSONL log over its committed versions plus, unless committed_only, the working
    copy (in first-seen order: oldest commit first, the working copy last)."""
    texts = committed_versions(path)[::-1]
    if not committed_only and Path(path).exists():
        texts.append(Path(path).read_text(encoding="utf-8"))
    seen, out = set(), []
    for t in texts:
        for r in records(t):
            key = json.dumps(r, sort_keys=True)
            if key not in seen:
                seen.add(key)
                out.append(r)
    return out


def log_records(path: Path, kind: str, *, committed_only: bool = False, **match) -> list[dict]:
    """Records of one kind (and matching fields) in a JSONL log, committed versions and working copy."""
    return [r for r in all_records(path, committed_only=committed_only)
            if r.get("kind") == kind and all(r.get(k) == v for k, v in match.items())]
