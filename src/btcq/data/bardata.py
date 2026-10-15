"""The data a strategy is allowed to see: bars per timeframe, funding, OI metrics, premium.

Backtests (Market) and the live bot both hand strategies one of these, so the
signal code path is identical in research and production.

Conventions:
- metrics "ts" is the time the snapshot became usable (snapshot time + OI_LAG_MS).
- premium rows are 1h bars labelled by open time; a bar is known only at its close.
"""
from __future__ import annotations

import pandas as pd

from btcq.config import TF_MS
from btcq.data import store


class BarData:
    def __init__(self, m1: pd.DataFrame, funding: pd.DataFrame | None = None,
                 metrics: pd.DataFrame | None = None, premium: pd.DataFrame | None = None):
        self.m1 = m1
        self.funding_df = funding if funding is not None else pd.DataFrame({"ts": [], "rate": []})
        self._metrics = metrics
        self._premium = premium
        self._bars: dict[str, pd.DataFrame] = {}

    def bars(self, tf: str) -> pd.DataFrame:
        if tf not in self._bars:
            self._bars[tf] = store.resample(self.m1, tf)
        return self._bars[tf]

    def metrics(self) -> pd.DataFrame:
        return self._metrics if self._metrics is not None else pd.DataFrame({"ts": []})

    def premium(self) -> pd.DataFrame:
        return self._premium if self._premium is not None else pd.DataFrame({"ts": [], "close": []})

    def truncated(self, cut_ms: int) -> "BarData":
        """Same data with everything not yet known at cut_ms removed (for look-ahead tests)."""
        met, prem = self.metrics(), self.premium()
        return BarData(self.m1[self.m1["ts"] < cut_ms].reset_index(drop=True),
                       self.funding_df[self.funding_df["ts"] < cut_ms].reset_index(drop=True),
                       met[met["ts"] < cut_ms].reset_index(drop=True) if len(met) else None,
                       prem[prem["ts"] + TF_MS["1h"] <= cut_ms].reset_index(drop=True)
                       if len(prem) else None)
