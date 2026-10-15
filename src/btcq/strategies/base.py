"""Strategy interface shared by the backtester and the live bot."""
from __future__ import annotations

import numpy as np
import pandas as pd

from btcq.backtest.engine import Signals
from btcq.backtest.stats import grid_configs
from btcq.data.bardata import BarData


class Strategy:
    code: str = ""
    name: str = ""
    tf: str = "4h"
    version: int = 1
    hypothesis: str = ""
    rules: str = ""
    grid: dict[str, list] = {}
    # Earliest instant the strategy's inputs exist (e.g. real funding history).
    data_start: pd.Timestamp | None = None
    # True when the idea was shaped by an already-seen holdout: its holdout can't pass or fail.
    holdout_contaminated: bool = False
    # Round 3: which sleeves the holdout gate is judged on ("all", or "alts" for S1-derived families).
    holdout_scope: str = "all"

    def configs(self) -> list[dict]:
        return grid_configs(self.grid)

    def compute(self, data: BarData, **params) -> Signals:
        raise NotImplementedError


def finalize(tf: str, long, short, stop_long, stop_short, target_long=None, target_short=None,
             **kw) -> Signals:
    """Combine directional conditions into Signals; bars with NaN inputs never trade."""
    long = np.asarray(long, dtype=bool)
    short = np.asarray(short, dtype=bool)
    stop = np.where(long, stop_long, np.where(short, stop_short, np.nan))
    n = len(long)
    tl = np.full(n, np.nan) if target_long is None else np.asarray(target_long, np.float64)
    ts = np.full(n, np.nan) if target_short is None else np.asarray(target_short, np.float64)
    target = np.where(long, tl, np.where(short, ts, np.nan))
    entry = np.where(long & ~short, 1, np.where(short & ~long, -1, 0)).astype(np.int8)
    entry[np.isnan(stop)] = 0
    return Signals(tf=tf, entry=entry, stop=stop, target=target, **kw)


def r_target(close, stop, r: float):
    """Price r times the stop distance away from close, on the profit side."""
    close = np.asarray(close, np.float64)
    stop = np.asarray(stop, np.float64)
    return close + r * (close - stop)
