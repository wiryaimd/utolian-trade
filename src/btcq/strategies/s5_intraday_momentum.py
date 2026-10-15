import numpy as np
import pandas as pd

from btcq.config import TF_MS
from btcq.features import ta
from btcq.features.mtf import align_htf
from btcq.strategies.base import Strategy, finalize


class IntradayMomentum(Strategy):
    code = "S5"
    name = "Intraday time-series momentum"
    tf = "15m"
    hypothesis = ("Documented BTC intraday momentum (Shen, Urquhart & Wang 2022): the return early "
                  "in the UTC day predicts the direction of the rest/late part of the day.")
    rules = ("Early return = 00:00 UTC open to the close at t1 hours. If |early| > thr * its 30-day "
             "std, enter in its direction at t2 ('now' = immediately at t1). Stop 1 daily ATR(14) "
             "away, always flat at 24:00 UTC.")
    grid = {"t1": [0.5, 1, 4, 8], "t2": ["now", "16:00", "20:00", "23:30"], "thr": [0.0, 0.5]}

    def compute(self, data, t1=1, t2="now", thr=0.0):
        b = data.bars(self.tf)
        ts = b["ts"].to_numpy(np.int64)
        o, c = b["open"].to_numpy(), b["close"].to_numpy()
        day = ts // TF_MS["1d"]
        mod = (ts % TF_MS["1d"]) // 60_000
        t1_min = int(t1 * 60)
        t2_min = t1_min if t2 == "now" else int(t2[:2]) * 60 + int(t2[3:])
        day_open = pd.Series(np.where(mod == 0, o, np.nan)).groupby(day).transform("max").to_numpy()
        at_t1 = mod == t1_min - 15  # bar whose close is t1
        early = pd.Series(np.where(at_t1, c / day_open - 1, np.nan)).groupby(day) \
            .transform("max").to_numpy()
        per_day = pd.Series(early).groupby(day).first()
        sd = per_day.shift(1).rolling(30, min_periods=20).std().reindex(day).to_numpy()
        sig_bar = mod == t2_min - 15
        strong = np.abs(early) > thr * sd
        long = sig_bar & strong & (early > 0)
        short = sig_bar & strong & (early < 0)
        d = data.bars("1d")
        atr_d = ta.atr(d["high"].to_numpy(), d["low"].to_numpy(), d["close"].to_numpy(), 14)
        ad = align_htf(ts, "15m", d["ts"], "1d", atr_d)
        eod = mod == 23 * 60 + 45
        return finalize(self.tf, long, short, c - ad, c + ad, exit_long=eod, exit_short=eod)
