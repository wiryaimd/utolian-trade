import numpy as np
import pandas as pd

from btcq.config import TF_MS
from btcq.strategies.base import Strategy, finalize, r_target


class SessionBreakout(Strategy):
    code = "S4"
    name = "Asia-range session breakout"
    tf = "15m"
    hypothesis = ("Participation and information flow jump when Europe and the US open; a break "
                  "of the quiet Asia-session range tends to continue intraday.")
    rules = ("Asia range = high/low of 00:00-08:00 UTC. From 08:00 until end_h UTC, the first 15m "
             "close outside the range enters in that direction (max one trade per day). Stop at "
             "range midpoint or the opposite side. Target 1R/2R or none; always flat at 24:00 UTC. "
             "Optional filter: only trade when today's range is narrower than the 20-day median.")
    grid = {"end_h": [16, 20], "stop": ["mid", "opposite"], "target": ["1R", "2R", "eod"],
            "filter": ["none", "narrow"]}

    def compute(self, data, end_h=20, stop="mid", target="1R", filter="none"):
        b = data.bars(self.tf)
        ts = b["ts"].to_numpy(np.int64)
        h, l, c = b["high"].to_numpy(), b["low"].to_numpy(), b["close"].to_numpy()
        day = ts // TF_MS["1d"]
        mod = (ts % TF_MS["1d"]) // 60_000
        asia = mod < 8 * 60
        rng = pd.DataFrame({"day": day[asia], "h": h[asia], "l": l[asia]}).groupby("day") \
            .agg(rh=("h", "max"), rl=("l", "min"), n=("h", "size"))
        rng = rng[rng["n"] >= 28]
        width = rng["rh"] - rng["rl"]
        rng["med"] = width.shift(1).rolling(20, min_periods=10).median()
        rh = rng["rh"].reindex(day).to_numpy()
        rl = rng["rl"].reindex(day).to_numpy()
        med = rng["med"].reindex(day).to_numpy()
        window = (mod >= 8 * 60) & (mod < end_h * 60)
        ok = window & ~np.isnan(rh)
        if filter == "narrow":
            ok &= (rh - rl) <= med
        brk_up = ok & (c > rh)
        brk_dn = ok & (c < rl)
        first = pd.Series(brk_up | brk_dn).groupby(day).cumsum().to_numpy() == 1
        long = brk_up & first
        short = brk_dn & first
        mid = (rh + rl) / 2
        sl, ss = (mid, mid) if stop == "mid" else (rl, rh)
        tl = ts_ = None
        if target != "eod":
            r = float(target[:-1])
            tl, ts_ = r_target(c, sl, r), r_target(c, ss, r)
        eod = mod == 23 * 60 + 45
        return finalize(self.tf, long, short, sl, ss, tl, ts_, exit_long=eod, exit_short=eod)
