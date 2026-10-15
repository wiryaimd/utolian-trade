"""Round 3 F6: perp premium (basis) dislocation."""
import numpy as np

from btcq.config import PERP_START
from btcq.features import ta
from btcq.features.mtf import align_htf
from btcq.strategies.base import Strategy, finalize


class PremiumDislocation(Strategy):
    code = "F6"
    name = "Premium dislocation"
    tf = "1h"
    version = 1
    data_start = PERP_START
    hypothesis = ("When the perp trades unusually rich or cheap to the spot index, arbitrage and "
                  "funding pressure pull it back, mostly through the perp price itself.")
    rules = ("Premium = Binance 1h premium index (perp vs spot index), known at each hour's close. "
             "Short when its 30-day z-score crosses above z (perp rich), long when it crosses below "
             "-z. Stop 2*ATR(14), exit after H hours.")
    grid = {"z": [2.0, 2.5, 3.0], "H": [4, 8, 24]}

    def compute(self, data, z=2.5, H=8):
        b = data.bars(self.tf)
        h, l, c = b["high"].to_numpy(), b["low"].to_numpy(), b["close"].to_numpy()
        p = data.premium()
        if len(p):
            pc = align_htf(b["ts"], "1h", p["ts"].to_numpy(np.int64), "1h",
                           p["close"].to_numpy(np.float64))
        else:
            pc = np.full(len(b), np.nan)
        zz = ta.zscore(pc, 720)
        a = ta.atr(h, l, c, 14)
        short = ta.cross_above(zz, z)
        long = ta.cross_below(zz, -z)
        return finalize(self.tf, long, short, c - 2 * a, c + 2 * a, max_bars=H)
