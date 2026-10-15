import numpy as np

from btcq.features import ta
from btcq.features.mtf import align_htf
from btcq.strategies.base import Strategy, finalize


class RangeMeanReversion(Strategy):
    code = "S8"
    name = "Range-regime VWAP mean reversion"
    tf = "15m"
    hypothesis = ("When the higher timeframe is not trending, short-term deviations from VWAP are "
                  "liquidity-driven and revert; trading them as a liquidity provider earns the spread.")
    rules = ("Only when 4h ADX(14) < adx_max. Deviation = (close - rolling 24h VWAP) / VW std. "
             "Long when deviation crosses below -z, short above +z. Target = VWAP at signal time; "
             "stop = stop_mult * distance-to-VWAP beyond entry; 4h time stop. Maker variant posts a "
             "limit at the signal close that must be traded through to fill.")
    grid = {"z": [2.0, 2.5, 3.0], "adx_max": [20, 25], "maker": [False, True],
            "stop_mult": [1.0, 1.5]}

    def compute(self, data, z=2.5, adx_max=20, maker=True, stop_mult=1.0):
        b4 = data.bars("4h")
        ax4 = ta.adx(b4["high"].to_numpy(), b4["low"].to_numpy(), b4["close"].to_numpy(), 14)
        b = data.bars(self.tf)
        h, l, c, v = (b[x].to_numpy() for x in ("high", "low", "close", "volume"))
        ax = align_htf(b["ts"], "15m", b4["ts"], "4h", ax4)
        vwap, sd = ta.rolling_vwap(h, l, c, v, 96)
        dev = (c - vwap) / sd
        calm = ax < adx_max
        long = calm & ta.cross_below(dev, -z)
        short = calm & ta.cross_above(dev, z)
        dist = np.abs(c - vwap)
        kw = {"limit": np.where(long | short, c, np.nan)} if maker else {}
        return finalize(self.tf, long, short, c - stop_mult * dist, c + stop_mult * dist,
                        vwap, vwap, max_bars=16, **kw)
