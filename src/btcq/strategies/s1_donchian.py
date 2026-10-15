import numpy as np

from btcq.features import ta
from btcq.strategies.base import Strategy, finalize


class DonchianTrend(Strategy):
    code = "S1"
    name = "Donchian trend / TSMOM"
    tf = "4h"
    hypothesis = ("Time-series momentum: BTC trends persist and returns are fat-tailed, so "
                  "breakouts with a wide trailing stop capture outsized moves.")
    rules = ("Long when the 4h close crosses above the prior N-bar high; short when it crosses "
             "below the prior N-bar low. Initial stop k*ATR(14). Exit by chandelier trail (k*ATR "
             "from the extreme since entry) or by the opposite N/2-bar channel. Reverses on an "
             "opposite breakout.")
    grid = {"n": [20, 30, 40, 55, 80], "k": [2.5, 3.5, 4.5], "exit": ["chandelier", "channel"]}
    long_only = False
    vol_target: float | None = None  # annualised vol target for notional sizing (None = 1% risk at stop)
    vol_lookback = 180               # 4h bars = 30 days

    def compute(self, data, n=40, k=3.5, exit="chandelier"):
        b = data.bars(self.tf)
        h, l, c = b["high"].to_numpy(), b["low"].to_numpy(), b["close"].to_numpy()
        a = ta.atr(h, l, c, 14)
        upper = ta.shift(ta.rolling_max(h, n))
        lower = ta.shift(ta.rolling_min(l, n))
        long = ta.cross_above(c, upper)
        short = np.zeros_like(long) if self.long_only else ta.cross_below(c, lower)
        kw = {}
        if exit == "chandelier":
            kw["trail"] = k * a
        else:
            m = max(n // 2, 5)
            kw["exit_long"] = c < ta.shift(ta.rolling_min(l, m))
            kw["exit_short"] = c > ta.shift(ta.rolling_max(h, m))
        if self.vol_target is not None:
            r = np.diff(np.log(c), prepend=np.nan)
            sigma = ta.rolling_std(r, self.vol_lookback) * np.sqrt(6 * 365)
            ready = np.isfinite(sigma) & (sigma > 0)
            long &= ready
            short &= ready
            kw["size"] = np.where(ready, self.vol_target / np.where(ready, sigma, 1.0), np.nan)
        return finalize(self.tf, long, short, c - k * a, c + k * a, **kw)


class DonchianLong(DonchianTrend):
    code = "S1L"
    name = "Donchian trend, long-only"
    holdout_contaminated = True
    hypothesis = ("Round 2: S1's edge came mostly from longs, and BTC's positive drift makes shorts "
                  "costly. Drop the short side.")
    rules = DonchianTrend.rules.replace("short when it crosses below the prior N-bar low",
                                        "short breakouts are ignored") + " Long-only."
    long_only = True


class DonchianVolTarget(DonchianTrend):
    code = "S1V"
    name = "Donchian trend, vol-targeted"
    holdout_contaminated = True
    hypothesis = ("Round 2: size positions to a constant 30% annualised volatility (30-day realised) "
                  "instead of 1% risk at the stop, smoothing risk across volatility regimes.")
    rules = DonchianTrend.rules + " Notional = equity x 0.30 / sigma_30d (cap 3x); stop still active."
    vol_target = 0.30


class DonchianLongVolTarget(DonchianTrend):
    code = "S1LV"
    name = "Donchian trend, long-only + vol-targeted"
    holdout_contaminated = True
    hypothesis = "Round 2: both changes together (long-only and 30% vol-target sizing)."
    rules = DonchianLong.rules + " Notional = equity x 0.30 / sigma_30d (cap 3x); stop still active."
    long_only = True
    vol_target = 0.30
