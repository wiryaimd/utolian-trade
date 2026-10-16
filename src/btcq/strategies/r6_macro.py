"""Round 6 M1: long/short daily Donchian-ensemble trend on macro markets (Binance TradFi perps).

Per market and lookback L (trading days, the market's own rows):
- long when flat and close > max of the prior L closes; out when close < running max since entry of
  MID_L = (max + min of the last L closes, today included) / 2;
- short is the mirror rule (the long machine run on -close); long and short are never open together.
Each (market, L) sub-sleeve is sized at entry to min(0.25 / sigma, 2.0) x its own capital, where
sigma = std (ddof 0) of the last 90 daily log returns x sqrt(252). Stops are never-triggering sentinels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from btcq.backtest.engine import Signals
from btcq.features import ta
from btcq.strategies.base import Strategy
from btcq.strategies.r5_d1 import CONFIGS, LOOKBACKS, PRIMARY, donchian_state  # noqa: F401 (re-exported)

VOL_TARGET = 0.25
LEV_CAP = 2.0
VOL_DAYS = 90
ANN = 252
LONG_SENTINEL = 0.0
SHORT_SENTINEL = 1e12
TAKER = 5e-4
FUNDING_STRESS_ANN = 0.05
SLIP = {"XAUUSDT": 2e-4, "CLUSDT": 2e-4, "XAGUSDT": 2e-4, "BZUSDT": 2e-4, "QQQUSDT": 2e-4, "EWYUSDT": 2e-4,
        "NATGASUSDT": 3e-4, "SPYUSDT": 3e-4, "COPPERUSDT": 5e-4, "XPTUSDT": 5e-4, "XPDUSDT": 5e-4}


def vol_size_td(c: np.ndarray) -> np.ndarray:
    """min(VOL_TARGET / sigma, LEV_CAP) on trading-day closes; NaN until VOL_DAYS returns exist."""
    r = np.diff(np.log(c), prepend=np.nan)
    sigma = ta.rolling_std(r, VOL_DAYS) * np.sqrt(ANN)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.isfinite(sigma) & (sigma > 0), np.minimum(VOL_TARGET / sigma, LEV_CAP), np.nan)


def ls_state(c: np.ndarray, L: int, ready: np.ndarray):
    """(state +1/0/-1, trailing level, mid) for one lookback."""
    sl, _, tl, mid = donchian_state(c, int(L), ready)
    ss, _, tn, _ = donchian_state(-c, int(L), ready)
    if ((sl == 1) & (ss == 1)).any():
        raise AssertionError("long and short open together")
    state = (sl.astype(np.int8) - ss.astype(np.int8)).astype(np.int8)
    trail = np.where(sl == 1, tl, np.where(ss == 1, -tn, np.nan))
    return state, trail, mid


@dataclass
class LSSub:
    ts: np.ndarray       # trade dates (ms, 00:00 UTC)
    state: np.ndarray
    trail: np.ndarray
    size: np.ndarray


def sub_signals(closes: pd.Series, L: int) -> LSSub:
    c = closes.to_numpy(np.float64)
    size = vol_size_td(c)
    state, trail, _ = ls_state(c, L, np.isfinite(size))
    return LSSub(closes.index.to_numpy(np.int64), state, trail, size)


def engine_signals(state: np.ndarray, size: np.ndarray) -> Signals:
    """Level signals: enter on the state's side whenever flat (size ready), exit whenever the state is not
    on the position's side; a flip is an exit plus an entry at the same fill. Stops are sentinels."""
    state = np.asarray(state, np.int8)
    size = np.asarray(size, np.float64)
    entry = np.where(np.isfinite(size), state, 0).astype(np.int8)
    n = len(state)
    stop = np.where(entry == 1, LONG_SENTINEL, np.where(entry == -1, SHORT_SENTINEL, np.nan))
    return Signals("1d", entry, stop, np.full(n, np.nan), exit_long=(state != 1), exit_short=(state != -1),
                   size=np.where(entry != 0, size, np.nan))


def to_engine(sub: LSSub, bar_ts: np.ndarray) -> tuple[Signals, np.ndarray, np.ndarray]:
    """Signals on the market's daily engine bars. Engine rows are stamped 23:59 UTC of each trade date, so
    the 1d bars are labelled 00:00 of the trade date and must equal the closes' dates exactly."""
    if not np.array_equal(bar_ts, sub.ts):
        raise ValueError("engine bars and closes are not on the same trade dates")
    return engine_signals(sub.state, sub.size), sub.trail, sub.state


def apply_r(trades: pd.DataFrame, trail_at_bar: np.ndarray, floor: float = 1e-3) -> tuple[pd.DataFrame, int]:
    """R from the trailing level at entry: init_risk = qty x (entry - trail) x dir (floored)."""
    if not len(trades):
        return trades, 0
    t = trades.copy()
    lvl = trail_at_bar[t["entry_tf"].to_numpy()]
    px = t["entry_px"].to_numpy()
    dist = (px - lvl) * t["dir"].to_numpy()
    fl = ~np.isfinite(dist) | (dist < floor * px)
    dist = np.where(fl, floor * px, dist)
    t["init_risk"] = t["qty"].to_numpy() * dist
    t["R"] = t["net"] / t["init_risk"]
    return t, int(fl.sum())


class MacroTrend(Strategy):
    """Metadata for the Round 6 registry and report; M1 is a portfolio rule run by btcq.research.round6."""
    code = "M1"
    name = "Macro trend ensemble, long/short (Binance TradFi perps)"
    tf = "1d"
    version = 1
    holdout_contaminated = False
    hypothesis = ("Time-series momentum across asset classes (metals, energy, equity indices): a long/short "
                  "9-lookback daily Donchian ensemble earns a trend premium after Binance-like costs.")
    rules = ("Per market and lookback L in {5,...,360} trading days: long above the prior L-day high, short below "
             "the prior L-day low, out when the close crosses the running extreme of the L-day midpoint; each "
             "sub-sleeve sized at entry to 25% annualised vol (cap 2x); 11 markets x 9 lookbacks.")
    grid = {"set": list(CONFIGS)}

    def compute(self, data, **params):
        raise NotImplementedError("M1 is a portfolio rule; see btcq.research.round6")
