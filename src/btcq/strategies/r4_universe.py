"""Round 4: F1 frozen, traded on a point-in-time top-20 universe of USD-M crypto perps."""
import pandas as pd

from btcq.strategies.r3_trend import S1Frozen


class UniverseTrend(S1Frozen):
    code = "U1"
    name = "F1 on the point-in-time top-20 perp universe"
    version = 1
    data_start = pd.Timestamp("2020-04-01", tz="UTC")
    holdout_scope = "gated"
    hypothesis = ("Replication by breadth: if F1's edge is crypto time-series momentum rather than a quirk of "
                  "five majors, the identical rule must work on liquid perps never used in research, and "
                  "trading ~20 of them gives roughly 1-2 trades a day.")
    rules = ("F1 unchanged (4h Donchian n=40 breakout, 3.5*ATR(14) stop, opposite 20-bar channel exit) on "
             "the monthly top-20 USD-M crypto perps by 90-day median quote volume, 0.30% risk per trade, "
             "at most 20 open positions; gated on the members other than BTC/ETH/SOL/XRP/BNB.")
    grid = {"n": [40], "k": [3.5], "exit": ["channel"]}


class UniverseTrendLong(UniverseTrend):
    code = "U1L"
    name = "U1 long-only"
    long_only = True
    hypothesis = "Information only: U1 with short breakouts ignored (every earlier round found shorts near 0R)."
