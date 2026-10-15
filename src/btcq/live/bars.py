"""Live bar building and signal evaluation.

1m closed klines are aggregated into strategy bars exactly like `store.resample`,
and the strategy's own `compute()` (the code the backtests ran) is evaluated on the
latest window of closed bars.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from btcq.config import MINUTE_MS, TF_MS

COLS = ["ts", "open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume", "n_min"]


@dataclass
class Minute:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    quote_volume: float = 0.0
    taker_buy_volume: float = 0.0


class BarBuilder:
    """Aggregates closed 1m bars into UTC-aligned bars of one timeframe."""

    def __init__(self, tf: str = "4h", history: pd.DataFrame | None = None, window: int = 1500):
        self.tf = tf
        self.ms = TF_MS[tf]
        self.rows: deque[list] = deque(maxlen=window)
        if history is not None and len(history):
            h = history if "n_min" in history else history.assign(n_min=0)
            for r in h[COLS].itertuples(index=False):
                self.rows.append(list(r))
        self.partial: list | None = None

    def close_if_new_bucket(self, ts: int) -> list[list]:
        """Close the partial bar if `ts` starts a later bucket (handles gaps in the 1m stream)."""
        if self.partial is not None and ts // self.ms * self.ms != self.partial[0]:
            return [self._close()]
        return []

    def add(self, m: Minute) -> list[list]:
        """Add one closed minute; returns the bar it completes, if any."""
        b = m.ts // self.ms * self.ms
        p = self.partial
        if p is None:
            self.partial = [b, m.open, m.high, m.low, m.close, m.volume, m.quote_volume,
                            m.taker_buy_volume, 1]
        else:
            p[2] = max(p[2], m.high)
            p[3] = min(p[3], m.low)
            p[4] = m.close
            p[5] += m.volume
            p[6] += m.quote_volume
            p[7] += m.taker_buy_volume
            p[8] += 1
        if m.ts + MINUTE_MS == b + self.ms:
            return [self._close()]
        return []

    def _close(self) -> list:
        bar, self.partial = self.partial, None
        self.rows.append(bar)
        return bar

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(list(self.rows), columns=COLS)

    def state(self) -> dict:
        return {"partial": self.partial}

    def restore(self, state: dict) -> None:
        self.partial = state.get("partial")


class FrameData:
    """Minimal BarData stand-in exposing one timeframe's closed bars to a strategy."""

    def __init__(self, tf: str, frame: pd.DataFrame):
        self.tf = tf
        self._frame = frame
        self.funding_df = pd.DataFrame({"ts": [], "rate": []})

    def bars(self, tf: str) -> pd.DataFrame:
        if tf != self.tf:
            raise ValueError(f"live data only provides {self.tf} bars, strategy asked for {tf}")
        return self._frame

    def metrics(self) -> pd.DataFrame:
        return pd.DataFrame({"ts": []})

    def premium(self) -> pd.DataFrame:
        return pd.DataFrame({"ts": [], "close": []})


class SignalRunner:
    """Evaluates a strategy on the latest closed bars and returns the last bar's decision."""

    def __init__(self, strategy, params: dict):
        self.strategy = strategy
        self.params = params

    def evaluate(self, frame: pd.DataFrame) -> dict:
        sig = self.strategy.compute(FrameData(self.strategy.tf, frame), **self.params)
        entry, stop, target, trail, xl, xs, limit, size = sig.arrays(len(sig.entry))
        if np.isfinite(limit[-1]):
            raise NotImplementedError("limit entries are not supported by the live book")
        return {"entry": int(entry[-1]), "stop": float(stop[-1]), "target": float(target[-1]),
                "trail": float(trail[-1]), "exit_long": bool(xl[-1]), "exit_short": bool(xs[-1]),
                "size": float(size[-1]), "max_bars": int(sig.max_bars)}
