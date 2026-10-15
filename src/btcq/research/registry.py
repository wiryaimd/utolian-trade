"""Global trial registry: every configuration ever backtested counts toward multiple testing.

A trial is keyed by (strategy, strategy version, params, period). Re-running the
identical key replaces the row; changing strategy code must bump its version, which
then counts as new trials.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from btcq.config import RESEARCH_DIR

PATH = RESEARCH_DIR / "trials.csv"
COLS = ["strategy", "version", "params", "period", "sr", "T", "skew", "kurt", "trades",
        "sharpe_ann", "utc"]


def load() -> pd.DataFrame:
    if PATH.exists():
        return pd.read_csv(PATH)
    return pd.DataFrame(columns=COLS)


def record(rows: list[dict]) -> None:
    df = load()
    new = pd.DataFrame(rows)
    new["params"] = new["params"].apply(lambda p: p if isinstance(p, str) else json.dumps(p, sort_keys=True))
    new["utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    key = ["strategy", "version", "params", "period"]
    if len(df):
        merged = df.merge(new[key], on=key, how="left", indicator=True)
        df = df[(merged["_merge"] == "left_only").to_numpy()]
    out = pd.concat([df, new[COLS]], ignore_index=True) if len(df) else new[COLS]
    PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(PATH, index=False)


NEFF_LOG = RESEARCH_DIR / "neff_history.jsonl"


def ratcheted_effective_trials(pr_now: float, n_series: int, note: str = "") -> float:
    """Effective N used for the DSR: the max participation ratio ever recorded (Amendment 2).

    The eigenvalue participation ratio falls when near-duplicate trials are added (one
    cluster dominates the spectrum), which would make the multiple-testing correction
    more lenient the more variants are tried. Ratcheting keeps it non-decreasing.
    """
    hist = [json.loads(x) for x in NEFF_LOG.read_text().splitlines()] if NEFF_LOG.exists() else []
    if not hist or hist[-1]["n_series"] != n_series:
        hist.append({"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "n_series": n_series, "pr": pr_now, "note": note})
        NEFF_LOG.parent.mkdir(parents=True, exist_ok=True)
        NEFF_LOG.write_text("\n".join(json.dumps(h) for h in hist) + "\n")
    return max(h["pr"] for h in hist)


def multiple_testing_inputs(period: str = "dev") -> tuple[int, float]:
    """(number of trials, cross-sectional variance of per-period Sharpe) over all trials."""
    df = load()
    df = df[df["period"] == period]
    if len(df) < 2:
        return max(len(df), 1), 0.0
    return len(df), float(np.var(df["sr"].astype(float), ddof=1))
