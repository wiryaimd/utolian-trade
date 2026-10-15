import numpy as np

from btcq.config import PERP_START
from btcq.features import ta
from btcq.features.mtf import align_events
from btcq.strategies.base import Strategy, finalize, r_target


class FundingContrarian(Strategy):
    code = "S6"
    name = "Funding crowding contrarian"
    tf = "4h"
    data_start = PERP_START
    hypothesis = ("Extreme funding means leveraged traders are crowded on one side; when price is "
                  "also stretched and starts to turn, the crowded side gets squeezed.")
    rules = ("Funding percentile = rank of the latest settled funding vs the last 90 days (270 "
             "settlements). Short when percentile >= p, close is >= stretch*ATR above EMA(20) and "
             "the bar closes red; long on the mirror image. Stop beyond the 3-bar extreme + 0.5*ATR. "
             "Exit at 1.5R/2.5R (3-day time stop) or a pure 2-day time exit.")
    grid = {"p": [0.85, 0.90, 0.95], "stretch": [1.0, 2.0], "exit": ["1.5R", "2.5R", "time"]}

    def compute(self, data, p=0.9, stretch=1.0, exit="1.5R"):
        b = data.bars(self.tf)
        o, h, l, c = (b[k].to_numpy() for k in ("open", "high", "low", "close"))
        f = data.funding_df
        f = f[f["ts"] >= PERP_START.value // 10**6]
        frank = ta.pct_rank(f["rate"].to_numpy(), 270)
        fr = align_events(b["ts"], self.tf, f["ts"].to_numpy(), frank)
        a = ta.atr(h, l, c, 14)
        z = (c - ta.ema(c, 20)) / a
        short = (fr >= p) & (z >= stretch) & (c < o)
        long = (fr <= 1 - p) & (z <= -stretch) & (c > o)
        sl = ta.rolling_min(l, 3) - 0.5 * a
        ss = ta.rolling_max(h, 3) + 0.5 * a
        if exit == "time":
            return finalize(self.tf, long, short, sl, ss, max_bars=12)
        r = float(exit[:-1])
        return finalize(self.tf, long, short, sl, ss, r_target(c, sl, r), r_target(c, ss, r),
                        max_bars=18)
