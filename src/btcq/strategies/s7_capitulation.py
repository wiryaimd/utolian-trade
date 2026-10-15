from btcq.features import ta
from btcq.strategies.base import Strategy, finalize, r_target


class CapitulationReversal(Strategy):
    code = "S7"
    name = "Capitulation / squeeze reversal"
    tf = "15m"
    hypothesis = ("Forced liquidations push price past fair value in a single violent bar; "
                  "liquidity providers who absorb that flow are paid by the subsequent reversion.")
    rules = ("Trigger bar: range > k * prior ATR(96) and volume > m * prior 10-day median volume. "
             "Big red bar -> long (close must sit >= wick of the range above the low); big green "
             "bar -> short (mirror). Stop beyond the bar extreme + 0.25*ATR. Exit at 1R/2R (4h time "
             "stop) or a pure 2h time exit.")
    grid = {"k": [2.5, 3.5], "m": [3.0, 5.0], "wick": [0.0, 0.3], "exit": ["1R", "2R", "time"]}

    def compute(self, data, k=3.0, m=4.0, wick=0.0, exit="1R"):
        b = data.bars(self.tf)
        o, h, l, c, v = (b[x].to_numpy() for x in ("open", "high", "low", "close", "volume"))
        a = ta.shift(ta.atr(h, l, c, 96))
        vmed = ta.shift(ta.rolling_median(v, 960))
        rng = h - l
        big = (rng > k * a) & (v > m * vmed)
        safe = rng > 0
        pos = (c - l) / (rng + (~safe))
        long = big & safe & (c < o) & (pos >= wick)
        short = big & safe & (c > o) & ((1 - pos) >= wick)
        sl = l - 0.25 * a
        ss = h + 0.25 * a
        if exit == "time":
            return finalize(self.tf, long, short, sl, ss, max_bars=8)
        r = float(exit[:-1])
        return finalize(self.tf, long, short, sl, ss, r_target(c, sl, r), r_target(c, ss, r),
                        max_bars=16)
