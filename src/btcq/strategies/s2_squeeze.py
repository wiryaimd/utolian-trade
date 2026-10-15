from btcq.features import ta
from btcq.strategies.base import Strategy, finalize, r_target


class VolSqueeze(Strategy):
    code = "S2"
    name = "Volatility squeeze breakout"
    tf = "4h"
    hypothesis = ("Volatility clusters and mean-reverts: unusually tight Bollinger bandwidth "
                  "precedes expansion, and the first close outside the band picks the direction.")
    rules = ("Squeeze = bandwidth percentile (120 bars) <= q at some point in the prior 6 bars. "
             "Long on a close crossing above the upper band (2 sd), short below the lower band. "
             "Stop at the middle band or 1.5*ATR. Exit at 2R / 3R target or a 3*ATR trail.")
    grid = {"bb_n": [20, 30], "q": [0.1, 0.2, 0.3], "stop": ["mid", "atr"],
            "exit": ["2R", "3R", "trail"]}

    def compute(self, data, bb_n=20, q=0.2, stop="mid", exit="2R"):
        b = data.bars(self.tf)
        h, l, c = b["high"].to_numpy(), b["low"].to_numpy(), b["close"].to_numpy()
        a = ta.atr(h, l, c, 14)
        mid, up, lo, bw = ta.bollinger(c, bb_n, 2.0)
        squeeze = ta.recent(ta.shift(ta.pct_rank(bw, 120)) <= q, 6)
        long = ta.cross_above(c, up) & squeeze
        short = ta.cross_below(c, lo) & squeeze
        if stop == "mid":
            sl, ss = mid, mid
        else:
            sl, ss = c - 1.5 * a, c + 1.5 * a
        kw = {}
        tl = ts = None
        if exit == "trail":
            kw["trail"] = 3.0 * a
        else:
            r = float(exit[:-1])
            tl, ts = r_target(c, sl, r), r_target(c, ss, r)
        return finalize(self.tf, long, short, sl, ss, tl, ts, **kw)
