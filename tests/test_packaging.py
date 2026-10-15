"""Every source file must be committable: an over-broad ignore rule once dropped src/btcq/data."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("git") is None or not (ROOT / ".git").exists(), reason="needs a git checkout")
def test_no_source_file_is_gitignored():
    files = [str(p.relative_to(ROOT)) for p in (ROOT / "src" / "btcq").rglob("*")
             if p.is_file() and "__pycache__" not in p.parts]
    assert files
    r = subprocess.run(["git", "check-ignore", "--stdin"], input="\n".join(files), cwd=ROOT,
                       capture_output=True, text=True)
    ignored = [f for f in r.stdout.splitlines() if f.strip()]
    assert not ignored, f"source files ignored by .gitignore: {ignored}"
