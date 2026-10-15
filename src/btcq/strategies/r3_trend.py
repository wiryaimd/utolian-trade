"""Round 3 trend families derived from S1 (holdout gate evaluated on the 4 alts only)."""
from btcq.config import PERP_START
from btcq.strategies.s1_donchian import DonchianTrend


class S1Frozen(DonchianTrend):
    code = "F1"
    name = "S1 frozen, 5 markets"
    tf = "4h"
    version = 1
    data_start = PERP_START
    holdout_scope = "alts"
    hypothesis = ("Cross-market replication: if BTC's 4h trend edge is real time-series momentum, the "
                  "exact same rule with the same parameters should also work on other major perps.")
    rules = ("S1 exactly as chosen on BTC dev (4h Donchian n=40, ATR stop k=3.5, opposite 20-bar "
             "channel exit), traded on BTC, ETH, SOL, XRP and BNB as equal-capital sleeves.")
    grid = {"n": [40], "k": [3.5], "exit": ["channel"]}


class Trend1h(DonchianTrend):
    code = "F2"
    name = "1h Donchian trend, 5 markets"
    tf = "1h"
    version = 1
    data_start = PERP_START
    holdout_scope = "alts"
    hypothesis = ("Time-series momentum at a faster horizon: the same breakout logic on 1h bars trades "
                  "roughly 4x as often as the 4h version.")
    rules = ("Long when the 1h close crosses above the prior n-bar high, short below the prior n-bar "
             "low. Initial stop k*ATR(14); exit by chandelier trail or the opposite n/2-bar channel.")
    grid = {"n": [24, 48, 96], "k": [2.5, 3.5], "exit": ["chandelier", "channel"]}


class S1FrozenLong(S1Frozen):
    """Shadow-only variant for the paper bot (not a research trial): F1 with shorts ignored.
    Every research round found shorts near zero R; this book lets forward data test that."""
    code = "F1L"
    name = "F1 long-only (shadow)"
    long_only = True
