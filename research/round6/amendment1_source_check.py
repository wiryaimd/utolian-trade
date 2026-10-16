"""Data-quality check of candidate Yahoo futures for the energy markets (dates, nulls, equal closes, dev-only moves).
Raw files go to a scratch folder; no holdout price or return is printed."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd

from btcq.data import external as E
from btcq.research import round6 as R

OUT = Path(r"C:\Users\adnya\AppData\Local\Temp\claude\F--trade2\44ccba62-de87-4374-a0f3-14dec46d3a01\scratchpad\round6\futures_raw")
OUT.mkdir(parents=True, exist_ok=True)
frames = {m: E._verified_frame(m) for m in E.MARKETS}
ref = R.reference_dates(frames)
closed_us = R.nyse_holidays(1993, 2026) | R.KNOWN_CLOSURES
cands = [("CLUSDT", "CL=F", "2000-08-23"), ("BZUSDT", "BZ=F", "2007-07-30"), ("NATGASUSDT", "NG=F", "2000-08-30"),
         ("COPPERUSDT", "HG=F", "2000-08-30")]
rows = []
for m, tk, first in cands:
    src = E.Source(m, "yahoo", tk, first)
    p = OUT / f"{src.stem}.json"
    if not p.exists():
        p.write_bytes(E.fetch(src.url))
    raw = p.read_bytes()
    now = datetime.now(timezone.utc)
    df, info = E.parse_yahoo(raw, now)
    df, info2 = E.clean(df, m)
    # nulls by date (dates only)
    res = json.loads(raw)["chart"]["result"][0]
    from zoneinfo import ZoneInfo
    tz = ZoneInfo(res["meta"]["exchangeTimezoneName"])
    nd = []
    for t, c in zip(res["timestamp"], res["indicators"]["quote"][0]["close"]):
        if c is None:
            loc = datetime.fromtimestamp(t, tz)
            d = loc.date() if loc.hour < 17 else E._next_weekday(loc.date())
            nd.append(d.isoformat())
    nulls_on_ref = sum(1 for d in nd if d not in closed_us)
    f = pd.DataFrame({"ts": E.date_ms(df["date"]), "close": df["close"].to_numpy(), "adjclose": np.nan})
    rep = R.market_report(m, f, {**info, **info2, "nulls_dropped": info["nulls_dropped"]}, ref)
    rep |= {"ticker": tk, "first_expected": first, "nulls_on_ref_days": nulls_on_ref,
            "null_share_on_ref": nulls_on_ref / max(len(f) + len(nd), 1),
            "nonpositive_dropped": info2["nonpositive_dropped"], "nonpositive_unexpected": info2["nonpositive_unexpected"]}
    rows.append(rep)
df = R.red_flags(pd.DataFrame(rows))
cols = ["market", "ticker", "rows", "first_date", "last_date", "missing_ref_days", "longest_missing_run", "max_weekday_gap",
        "worst_year_missing_share", "extra_days_vs_ref", "longest_equal_run", "nulls_dropped", "null_share",
        "nulls_on_ref_days", "null_share_on_ref", "nonpositive_dropped", "nonpositive_unexpected", "dev_abs_lr_gt_20pct",
        "dev_max_abs_lr", "dev_reverting_spikes", "dev_spike_dates", "red_flag_reasons"]
with pd.option_context("display.width", 250, "display.max_columns", 50, "display.max_colwidth", 60):
    print(df[cols].T.to_string())
