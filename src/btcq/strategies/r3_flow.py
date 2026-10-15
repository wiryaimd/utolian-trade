"""Round 3 F3: taker order-flow continuation."""
from btcq.config import PERP_START
from btcq.features import ta
from btcq.strategies.base import Strategy, finalize


class TakerFlow(Strategy):
    code = "F3"
    name = "Taker-flow continuation"
    tf = "1h"
    version = 1
    data_start = PERP_START
    hypothesis = ("Signed order flow: when aggressive buyers (or sellers) dominate for hours, informed "
                  "or forced flow tends to keep pushing price the same way over the next hours.")
    rules = ("Net flow = sum over k hours of (2*taker_buy - volume) / volume. Long when its z-score vs "
             "the last 30 days crosses above thr and close > k-hour VWAP; short on the mirror. Stop "
             "2*ATR(14), exit after H hours.")
    grid = {"k": [4, 12, 24], "thr": [1.5, 2.0], "H": [6, 24]}

    def compute(self, data, k=12, thr=2.0, H=24):
        b = data.bars(self.tf)
        h, l, c, v = (b[x].to_numpy() for x in ("high", "low", "close", "volume"))
        tb = b["taker_buy_volume"].to_numpy()
        flow = ta.rolling_sum(2 * tb - v, k) / ta.rolling_sum(v, k)
        z = ta.zscore(flow, 720)
        vw, _ = ta.rolling_vwap(h, l, c, v, k)
        a = ta.atr(h, l, c, 14)
        long = ta.cross_above(z, thr) & (c > vw)
        short = ta.cross_below(z, -thr) & (c < vw)
        return finalize(self.tf, long, short, c - 2 * a, c + 2 * a, max_bars=H)
