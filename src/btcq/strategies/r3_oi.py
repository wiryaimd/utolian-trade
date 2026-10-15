"""Round 3 F4/F5: open-interest positioning signals (metrics from 2021-12 for all symbols)."""
import numpy as np
import pandas as pd

from btcq.features import ta
from btcq.features.mtf import align_events
from btcq.strategies.base import Strategy, finalize

OI_START = pd.Timestamp("2021-12-01", tz="UTC")


def _oi(data, bars) -> np.ndarray:
    """Latest OI (contracts) usable at each bar close; metrics ts is already availability time."""
    m = data.metrics()
    if not len(m) or "sum_open_interest" not in m:
        return np.full(len(bars), np.nan)
    return align_events(bars["ts"], "1h", m["ts"].to_numpy(np.int64),
                        m["sum_open_interest"].to_numpy(np.float64))


class OIBreakout(Strategy):
    code = "F4"
    name = "OI-confirmed breakout"
    tf = "1h"
    version = 1
    data_start = OI_START
    hypothesis = ("A breakout backed by rising open interest is new leverage entering in the breakout's "
                  "direction and tends to continue; one on falling OI is mostly covering and fades.")
    rules = ("Long when the 1h close crosses above the prior N-hour high and OI grew over the same N "
             "hours by more than oi_thr standard deviations (0 = any growth); short on the mirror "
             "breakdown (also with rising OI = new shorts). Stop 2*ATR(14); exit by 3*ATR chandelier "
             "trail or after 2N hours.")
    grid = {"N": [12, 24, 48], "oi_thr": [0.0, 1.0], "exit": ["chandelier", "time"]}

    def compute(self, data, N=24, oi_thr=0.0, exit="chandelier"):
        b = data.bars(self.tf)
        h, l, c = b["high"].to_numpy(), b["low"].to_numpy(), b["close"].to_numpy()
        oi = _oi(data, b)
        with np.errstate(divide="ignore", invalid="ignore"):
            doi = oi / ta.shift(oi, N) - 1
            sd = ta.rolling_std(oi / ta.shift(oi, 1) - 1, 720) * np.sqrt(N)
        fresh = doi > oi_thr * sd
        a = ta.atr(h, l, c, 14)
        long = ta.cross_above(c, ta.shift(ta.rolling_max(h, N))) & fresh
        short = ta.cross_below(c, ta.shift(ta.rolling_min(l, N))) & fresh
        kw = {"trail": 3 * a} if exit == "chandelier" else {"max_bars": 2 * N}
        return finalize(self.tf, long, short, c - 2 * a, c + 2 * a, **kw)


class OIFlush(Strategy):
    code = "F5"
    name = "OI-flush reversal"
    tf = "1h"
    version = 1
    data_start = OI_START
    hypothesis = ("A sharp hourly OI drop together with a large price move marks forced liquidations; "
                  "the cascade overshoots and price partially reverts once it is done.")
    rules = ("Trigger: 1h OI change z-score (30 days) <= -z_oi and |1h move| > 1.5 * prior ATR(14). "
             "Fade the move: long after a drop, short after a spike. Stop beyond the bar extreme + "
             "0.5*ATR. Exit after H hours, or at a 50% retracement of the move (with the same time cap).")
    grid = {"z_oi": [2.5, 3.0, 4.0], "H": [4, 12], "exit": ["time", "target"]}

    def compute(self, data, z_oi=3.0, H=4, exit="time"):
        b = data.bars(self.tf)
        h, l, c = b["high"].to_numpy(), b["low"].to_numpy(), b["close"].to_numpy()
        oi = _oi(data, b)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = ta.zscore(oi / ta.shift(oi, 1) - 1, 720)
        a = ta.shift(ta.atr(h, l, c, 14))
        move = c - ta.shift(c)
        flush = (z <= -z_oi) & (np.abs(move) > 1.5 * a)
        long = flush & (move < 0)
        short = flush & (move > 0)
        sl, ss = l - 0.5 * a, h + 0.5 * a
        if exit == "time":
            return finalize(self.tf, long, short, sl, ss, max_bars=H)
        half = 0.5 * np.abs(move)
        return finalize(self.tf, long, short, sl, ss, c + half, c - half, max_bars=H)
