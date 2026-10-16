"""Round 5 D1: daily long-only Donchian-ensemble trend, vol-targeted (after Zarattini, Pagani & Barbon,
"Catching Crypto Trends", 2025).

Per market and lookback L, on daily closes (00:00 UTC):
- flat -> long when close_t > max of the prior L closes (strict inequality);
- long -> flat when close_t < trailing level T, the running max since entry of
  MID_L = (max + min of the last L closes, today included) / 2;
- no position until L+1 closes exist and the vol estimate is ready.
Each (market, L) is one sub-sleeve sized at entry to min(0.25 / sigma, 2.0) x its own capital, where
sigma = std (ddof=0) of the last 90 daily log returns x sqrt(365). Long only; no other stop.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit

from btcq.backtest.engine import Signals
from btcq.config import TF_MS
from btcq.features import ta
from btcq.strategies.base import Strategy

LOOKBACKS = (5, 10, 20, 30, 60, 90, 150, 250, 360)
CONFIGS = {"all9": LOOKBACKS, "fast5": (5, 10, 20, 30, 60), "slow5": (60, 90, 150, 250, 360)}
PRIMARY = "all9"
VOL_TARGET = 0.25
LEV_CAP = 2.0
VOL_DAYS = 90
STOP_SENTINEL = 0.0      # a valid long stop that can never trigger (R is computed from the trailing level)


@njit(cache=True)
def donchian_state(c, L, ready):
    """Causal state machine for one lookback. Returns (state, exit, trail, mid) per day:
    state 1 = long at the close of day t, exit[t] = the long ended at day t's close,
    trail[t] = trailing level while long, mid[t] = MID_L(t)."""
    n = c.shape[0]
    state = np.zeros(n, np.int8)
    exit_ = np.zeros(n, np.bool_)
    trail = np.full(n, np.nan)
    mid = np.full(n, np.nan)
    pos = 0
    T = np.nan
    for t in range(L, n):
        U = c[t - L]
        for j in range(t - L + 1, t):
            if c[j] > U:
                U = c[j]
        hi = c[t]
        lo = c[t]
        for j in range(t - L + 1, t):
            if c[j] > hi:
                hi = c[j]
            if c[j] < lo:
                lo = c[j]
        M = 0.5 * (hi + lo)
        mid[t] = M
        if pos == 0:
            if c[t] > U and ready[t]:
                pos = 1
                T = M
        else:
            if M > T:
                T = M
            if c[t] < T:
                pos = 0
                exit_[t] = True
                T = np.nan
        state[t] = pos
        if pos == 1:
            trail[t] = T
    return state, exit_, trail, mid


def vol_size(c: np.ndarray) -> np.ndarray:
    """min(VOL_TARGET / sigma, LEV_CAP); NaN until VOL_DAYS returns exist."""
    r = np.diff(np.log(c), prepend=np.nan)
    sigma = ta.rolling_std(r, VOL_DAYS) * np.sqrt(365)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.isfinite(sigma) & (sigma > 0), np.minimum(VOL_TARGET / sigma, LEV_CAP), np.nan)


@dataclass
class SubSignal:
    ts: np.ndarray      # day-open ms on the complete calendar of the stitched closes
    state: np.ndarray
    exit: np.ndarray
    trail: np.ndarray
    size: np.ndarray


def sub_signals(closes: pd.Series, L: int) -> SubSignal:
    """closes: stitched daily closes on a complete UTC calendar (index: day-open ms)."""
    c = closes.to_numpy(np.float64)
    size = vol_size(c)
    state, ex, trail, _ = donchian_state(c, int(L), np.isfinite(size))
    return SubSignal(closes.index.to_numpy(np.int64), state, ex, trail, size)


def to_engine(sub: SubSignal, bar_ts: np.ndarray) -> tuple[Signals, np.ndarray, np.ndarray]:
    """Engine signals on the perp market's daily bars (exact timestamp alignment is asserted).
    entry is a level signal (long whenever the state is long), so a position already open at the window
    start is entered at the first decision executed after it. Returns (signals, trail_at_bar, state_at_bar)."""
    pos = np.searchsorted(sub.ts, bar_ts)
    if not (np.all(pos < len(sub.ts)) and np.array_equal(sub.ts[np.minimum(pos, len(sub.ts) - 1)], bar_ts)):
        raise ValueError("perp daily bars are not all on the stitched calendar")
    state = sub.state[pos]
    size = sub.size[pos]
    entry = ((state == 1) & np.isfinite(size)).astype(np.int8)
    n = len(bar_ts)
    # level exit (out whenever the state is flat): identical to the 1->0 flags on gap-free bars, and robust
    # to a perp day missing from the bars
    sig = Signals("1d", entry, np.where(entry == 1, STOP_SENTINEL, np.nan), np.full(n, np.nan),
                  exit_long=(state == 0), size=np.where(entry == 1, size, np.nan))
    return sig, sub.trail[pos], state


def apply_r(trades: pd.DataFrame, trail_at_bar: np.ndarray, floor: float = 1e-3) -> tuple[pd.DataFrame, int]:
    """R from the trailing level at entry (MID_L for a fresh entry): init_risk = qty x (entry - trail).
    A non-positive or tiny distance is floored at floor x entry notional; the count is returned."""
    if not len(trades):
        return trades, 0
    t = trades.copy()
    lvl = trail_at_bar[t["entry_tf"].to_numpy()]
    dist = t["entry_px"].to_numpy() - lvl
    fl = ~np.isfinite(dist) | (dist < floor * t["entry_px"].to_numpy())
    dist = np.where(fl, floor * t["entry_px"].to_numpy(), dist)
    t["init_risk"] = t["qty"].to_numpy() * dist
    t["R"] = t["net"] / t["init_risk"]
    return t, int(fl.sum())


class DonchianEnsemble(Strategy):
    """Metadata for the registry and report; D1 is a portfolio rule run by btcq.research.round5."""
    code = "D1"
    name = "Daily long-only Donchian ensemble, vol-targeted"
    tf = "1d"
    version = 1
    holdout_contaminated = True
    hypothesis = ("Time-series momentum at 1-week to 1-year horizons on the 5 majors, expressed as a "
                  "9-lookback daily ensemble that adjusts exposure about once a day.")
    rules = ("Per market and lookback L in {5,10,20,30,60,90,150,250,360} days: long when the daily close "
             "exceeds the prior L-day high, out when it falls below the running max of the L-day midpoint; "
             "each sub-sleeve sized at entry to 25% annualised vol (cap 2x); long only.")
    grid = {"set": list(CONFIGS)}

    def compute(self, data, **params):
        raise NotImplementedError("D1 is a portfolio rule; see btcq.research.round5")


DAY_MS = TF_MS["1d"]
