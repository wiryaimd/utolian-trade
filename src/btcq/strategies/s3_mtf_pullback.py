import numpy as np

from btcq.features import ta
from btcq.features.mtf import align_htf
from btcq.strategies.base import Strategy, finalize, r_target


class MTFPullback(Strategy):
    code = "S3"
    name = "MTF trend pullback"
    tf = "15m"
    hypothesis = ("Buying dips inside an established higher-timeframe trend: the 4h trend gives "
                  "direction, the 15m oversold pullback gives a low-risk entry point.")
    rules = ("4h uptrend = close > EMA(n), EMA rising over 6 bars, ADX(14) >= adx_min (mirror for "
             "downtrend). On 15m, RSI(14) < rsi_lo within the last 4 bars, then a close above the "
             "previous bar's high -> long. Stop below the 8-bar low minus 0.25*ATR. Target R "
             "multiple, 24h time stop.")
    grid = {"ema": [50, 100, 200], "adx_min": [0, 20], "rsi_lo": [30, 40], "R": [1.5, 2.0, 3.0]}

    def compute(self, data, ema=100, adx_min=20, rsi_lo=30, R=2.0):
        b4 = data.bars("4h")
        h4, l4, c4 = b4["high"].to_numpy(), b4["low"].to_numpy(), b4["close"].to_numpy()
        e = ta.ema(c4, ema)
        slope = e - ta.shift(e, 6)
        ax = ta.adx(h4, l4, c4, 14)
        up4 = ((c4 > e) & (slope > 0) & (ax >= adx_min)).astype(np.float64)
        dn4 = ((c4 < e) & (slope < 0) & (ax >= adx_min)).astype(np.float64)
        b = data.bars(self.tf)
        h, l, c = b["high"].to_numpy(), b["low"].to_numpy(), b["close"].to_numpy()
        up, dn = align_htf(b["ts"], "15m", b4["ts"], "4h", up4, dn4)
        r = ta.rsi(c, 14)
        a = ta.atr(h, l, c, 14)
        long = (up > 0.5) & ta.recent(r < rsi_lo, 4) & (c > ta.shift(h))
        short = (dn > 0.5) & ta.recent(r > 100 - rsi_lo, 4) & (c < ta.shift(l))
        sl = ta.rolling_min(l, 8) - 0.25 * a
        ss = ta.rolling_max(h, 8) + 0.25 * a
        return finalize(self.tf, long, short, sl, ss, r_target(c, sl, R), r_target(c, ss, R),
                        max_bars=96)
