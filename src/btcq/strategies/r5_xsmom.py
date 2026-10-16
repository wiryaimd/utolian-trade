"""Round 5 XS: dollar-neutral cross-sectional momentum among the 5 majors (perps only).

On each decision bar (weekly: the Sunday bar, which closes Monday 00:00 UTC; daily: every bar, with a
7-day minimum hold) rank the eligible coins by their L-day log return (ties by symbol), go long the
top k and short the bottom k, k = n // 2. Weights within a side are inverse 30-day volatility, and each
side is 0.5x the account's gross at entry. Each leg has a disaster stop at close x (1 -/+ 4 x MAD30),
the 30-day mean absolute daily log return, which also defines R. A continuing leg keeps its size; a coin
moving from long to short reverses; a stopped-out leg re-enters at the next decision if still on its side.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from btcq.backtest.engine import Signals
from btcq.strategies.base import Strategy

CONFIGS = [{"mode": "weekly", "L": 7}, {"mode": "weekly", "L": 14}, {"mode": "weekly", "L": 21},
           {"mode": "weekly", "L": 28}, {"mode": "daily", "L": 14, "min_hold": 7},
           {"mode": "daily", "L": 21, "min_hold": 7}]
PRIMARY = 1
GROSS_SIDE = 0.5
SLOTS = 5
VOL_DAYS = 30
STOP_K = 4.0
MIN_CLOSES = 31


def panel(closes: dict[str, pd.Series]) -> pd.DataFrame:
    """Calendar (day-open ms) x coin daily closes; NaN before a coin's perp exists."""
    df = pd.concat(closes, axis=1).sort_index()
    cal = np.arange(int(df.index[0]), int(df.index[-1]) + 1, 86_400_000, dtype=np.int64)
    return df.reindex(cal)


def features(px: pd.DataFrame, L: int):
    lr = np.log(px).diff()
    score = np.log(px / px.shift(L))
    sigma = lr.rolling(VOL_DAYS, min_periods=VOL_DAYS).std(ddof=0)
    mad = lr.abs().rolling(VOL_DAYS, min_periods=VOL_DAYS).mean()
    return score, sigma, mad


def eligible(px: pd.DataFrame, score: pd.DataFrame, sigma: pd.DataFrame) -> pd.DataFrame:
    return (px.notna().cumsum() >= MIN_CLOSES) & score.notna() & (sigma > 0)


def ranks(score: pd.DataFrame, elig: pd.DataFrame) -> pd.DataFrame:
    """1 = highest score among eligible coins (ties broken by symbol); 0 = not eligible."""
    out = pd.DataFrame(0, index=score.index, columns=score.columns, dtype=np.int64)
    cols = list(score.columns)
    for i, (sc, el) in enumerate(zip(score.to_numpy(), elig.to_numpy())):
        idx = [j for j in range(len(cols)) if el[j]]
        idx.sort(key=lambda j: (-sc[j], cols[j]))
        for r, j in enumerate(idx, 1):
            out.iat[i, j] = r
    return out


def sides(rk: pd.DataFrame) -> pd.DataFrame:
    """+1 for the top k, -1 for the bottom k, 0 otherwise; k = n // 2 eligible coins."""
    n = (rk > 0).sum(axis=1).to_numpy()[:, None]
    k = n // 2
    r = rk.to_numpy()
    s = np.where((r > 0) & (r <= k), 1, np.where((r > 0) & (r > n - k), -1, 0))
    return pd.DataFrame(s.astype(np.int8), index=rk.index, columns=rk.columns)


def permute_ranks(rk: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Null: on each date, the eligible coins' ranks are randomly permuted among them."""
    r = rk.to_numpy().copy()
    for i in range(r.shape[0]):
        idx = np.flatnonzero(r[i] > 0)
        if len(idx) > 1:
            r[i, idx] = r[i, rng.permutation(idx)]
    return pd.DataFrame(r, index=rk.index, columns=rk.columns)


def hysteresis(desired: pd.DataFrame, decision: np.ndarray, min_hold: int) -> pd.DataFrame:
    """Book-level minimum hold: the whole long/short book is re-formed only when the target book changed
    and at least min_hold days have passed since the last re-formation (so it stays k long / k short)."""
    d = desired.to_numpy()
    out = np.zeros_like(d)
    cur = np.zeros(d.shape[1], dtype=d.dtype)
    held = 0
    for i in range(d.shape[0]):
        if decision[i]:
            if not cur.any():
                if d[i].any():
                    cur, held = d[i].copy(), 0
            else:
                held += 1
                if (d[i] != cur).any() and held >= min_hold:
                    cur, held = d[i].copy(), 0
        out[i] = cur
    return pd.DataFrame(out.astype(np.int8), index=desired.index, columns=desired.columns)


def decision_mask(index: np.ndarray, mode: str) -> np.ndarray:
    if mode == "weekly":
        return pd.to_datetime(index, unit="ms", utc=True).dayofweek.to_numpy() == 6   # Sunday bar -> Monday
    return np.ones(len(index), dtype=bool)


def leg_plan(px: pd.DataFrame, cfg: dict, ranks_override: pd.DataFrame | None = None) -> dict:
    """Per-coin legs on the calendar: side, size (multiple of a 1/SLOTS coin sleeve) and stop, defined on
    decision bars only. Returns {"side","size","stop","decision","ranks","weight"} frames."""
    score, sigma, mad = features(px, int(cfg["L"]))
    elig = eligible(px, score, sigma)
    rk = ranks(score, elig) if ranks_override is None else ranks_override
    dec = decision_mask(px.index.to_numpy(), cfg["mode"])
    want = sides(rk)
    side = hysteresis(want, dec, int(cfg.get("min_hold", 0))) if cfg["mode"] == "daily" else want
    inv = (1.0 / sigma).where(side != 0)
    w = inv / inv.where(side == 1).sum(axis=1).to_frame().to_numpy()
    w_short = inv / inv.where(side == -1).sum(axis=1).to_frame().to_numpy()
    weight = w.where(side == 1, w_short.where(side == -1, 0.0)).fillna(0.0)
    size = weight * GROSS_SIDE * SLOTS
    c = px.to_numpy()
    stop = np.where(side.to_numpy() == 1, c * (1 - STOP_K * mad.to_numpy()),
                    np.where(side.to_numpy() == -1, c * (1 + STOP_K * mad.to_numpy()), np.nan))
    return {"side": side, "size": size, "stop": pd.DataFrame(stop, index=px.index, columns=px.columns),
            "decision": dec, "ranks": rk, "weight": weight}


def to_engine(plan: dict, sym: str, bar_ts: np.ndarray) -> Signals:
    """Engine signals for one coin on its perp daily bars (exact timestamp alignment is asserted)."""
    idx = plan["side"].index.to_numpy(np.int64)
    pos = np.searchsorted(idx, bar_ts)
    if not (np.all(pos < len(idx)) and np.array_equal(idx[np.minimum(pos, len(idx) - 1)], bar_ts)):
        raise ValueError(f"{sym}: perp daily bars are not on the panel calendar")
    dec = plan["decision"][pos]
    side = plan["side"][sym].to_numpy()[pos]
    stop = plan["stop"][sym].to_numpy()[pos]
    size = plan["size"][sym].to_numpy()[pos]
    ok = dec & (side != 0) & np.isfinite(stop) & np.isfinite(size) & (size > 0)
    entry = np.where(ok, side, 0).astype(np.int8)
    return Signals("1d", entry, np.where(ok, stop, np.nan), np.full(len(bar_ts), np.nan),
                   exit_long=dec & (side != 1), exit_short=dec & (side != -1),
                   size=np.where(ok, size, np.nan))


class XSMomentum(Strategy):
    """Metadata for the registry and report; XS is a portfolio rule run by btcq.research.round5."""
    code = "XS"
    name = "Cross-sectional momentum among the 5 majors, dollar-neutral"
    tf = "1d"
    version = 1
    holdout_contaminated = False
    hypothesis = ("Relative strength persists over 1-4 weeks among the large-cap perps (post-2020 crypto "
                  "momentum factor), independently of the market's direction.")
    rules = ("Each Monday 00:00 UTC rank the coins by 14-day log return; long the top 2, short the bottom 2, "
             "inverse-vol weighted, 0.5x gross per side; per-leg stop at 4x the 30-day mean absolute daily move.")
    grid = {"cfg": list(range(len(CONFIGS)))}

    def compute(self, data, **params):
        raise NotImplementedError("XS is a portfolio rule; see btcq.research.round5")
