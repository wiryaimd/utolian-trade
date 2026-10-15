"""Hand-checked engine mechanics on synthetic 1m paths (15m strategy bars)."""
import numpy as np
import pandas as pd
import pytest

from btcq.backtest import engine
from btcq.config import Costs, Risk
from conftest import T0, flat, make_m1, needs_data

C = Costs(taker=0.0005, maker=0.0002, slippage=0.0001)
R = Risk(risk_per_trade=0.01, max_leverage=3.0, initial_equity=10_000.0)


def market(paths, funding=None, synthetic=0.0):
    return engine.Market(make_m1(paths), funding, synthetic_funding=synthetic)


def sig(n_bars, entry_bar, d, stop, target=np.nan, **kw):
    e = np.zeros(n_bars, np.int8)
    e[entry_bar] = d
    s = np.full(n_bars, np.nan)
    s[entry_bar] = stop
    t = np.full(n_bars, np.nan)
    t[entry_bar] = target
    return engine.Signals("15m", e, s, t, **kw)


def test_long_hits_target_with_fees_and_slippage():
    # bar 0 flat at 100; bar 1 opens 100 and trades up through 105 in minute 3.
    paths = flat(100, 15) + [(100, 100.5, 99.5, 100), (100, 101, 99.8, 101), (101, 103, 100.5, 103),
                             (103, 106, 102.8, 105.5)] + flat(105.5, 11)
    mk = market(paths)
    res = engine.run(mk, sig(2, 0, 1, 95.0, 105.0), C, R)
    t = res.trades.iloc[0]
    epx = 100 * 1.0001
    qty = min(0.01 * 10_000 / (epx - 95), 3 * 10_000 / epx)
    xpx = 105 * (1 - 0.0001)
    assert t["reason"] == "target"
    assert t["entry_i"] == 15 and t["exit_i"] == 18
    assert t["entry_px"] == pytest.approx(epx)
    assert t["exit_px"] == pytest.approx(xpx)
    assert t["qty"] == pytest.approx(qty)
    fees = qty * epx * 0.0005 + qty * xpx * 0.0005
    assert t["fees"] == pytest.approx(fees)
    assert t["net"] == pytest.approx(qty * (xpx - epx) - fees)
    assert t["R"] == pytest.approx(t["net"] / (qty * (epx - 95)))
    assert res.equity.iloc[-1] == pytest.approx(10_000 + t["net"])


def test_stop_wins_when_stop_and_target_touch_same_minute():
    paths = flat(100, 15) + [(100, 106, 94, 100)] + flat(100, 14)
    res = engine.run(market(paths), sig(2, 0, 1, 95.0, 105.0), C, R)
    t = res.trades.iloc[0]
    assert t["reason"] == "stop"
    assert t["exit_px"] == pytest.approx(95 * (1 - 0.0001))


def test_gap_through_stop_fills_at_open():
    paths = flat(100, 15) + [(100, 100, 99, 99)] + [(90, 91, 89, 90)] + flat(90, 13)
    res = engine.run(market(paths), sig(2, 0, 1, 95.0), C, R)
    t = res.trades.iloc[0]
    assert t["reason"] == "stop"
    assert t["exit_px"] == pytest.approx(90 * (1 - 0.0001))
    assert t["R"] < -1.5


def test_short_mirror_and_leverage_cap():
    paths = flat(100, 15) + [(100, 100.2, 99.0, 99.0)] + flat(99.0, 14)
    # 0.1% stop -> uncapped size would be 10x; cap holds it at 3x equity.
    res = engine.run(market(paths), sig(2, 0, -1, 100.1), C, R)
    t = res.trades.iloc[0]
    epx = 100 * (1 - 0.0001)
    assert t["dir"] == -1
    assert t["qty"] == pytest.approx(3 * 10_000 / epx)
    assert t["reason"] == "stop"  # high 100.2 >= 100.1


def test_funding_charged_only_while_held():
    # 8h of flat bars spanning the 08:00 settlement.
    n = 9 * 60
    paths = flat(100, n)
    fund = pd.DataFrame({"ts": [T0 + 8 * 3_600_000], "rate": [0.001]})
    mk = market(paths, funding=fund)
    n_bars = n // 15
    s = sig(n_bars, 0, 1, 90.0)
    s.max_bars = n_bars - 3  # exit well after 08:00
    res = engine.run(mk, s, C, R)
    t = res.trades.iloc[0]
    assert t["funding"] == pytest.approx(t["qty"] * 100 * 0.001)
    # Short receives funding.
    res2 = engine.run(mk, sig(n_bars, 0, -1, 110.0, max_bars=n_bars - 3), C, R)
    assert res2.trades.iloc[0]["funding"] == pytest.approx(-res2.trades.iloc[0]["qty"] * 100 * 0.001)


def test_time_stop_exits_at_next_bar_open():
    paths = flat(100, 15 * 6)
    res = engine.run(market(paths), sig(6, 0, 1, 90.0, max_bars=2), C, R)
    t = res.trades.iloc[0]
    assert t["reason"] == "time"
    assert t["entry_i"] == 15 and t["exit_i"] == 45


def test_limit_requires_trade_through():
    # Limit at 99: minute touching exactly 99 must NOT fill; next bar trades to 98.9 and fills.
    paths = flat(100, 15) + [(100, 100, 99.0, 99.5)] + flat(99.5, 14) + flat(100, 15)
    lim = np.full(3, np.nan)
    lim[0] = 99.0
    res = engine.run(market(paths), sig(3, 0, 1, 95.0, limit=lim), C, R)
    assert len(res.trades) == 0  # expired after one bar without trade-through
    paths2 = flat(100, 15) + [(100, 100, 98.9, 99.5)] + flat(99.5, 14)
    res2 = engine.run(market(paths2), sig(2, 0, 1, 95.0, limit=lim[:2]), C, R)
    t = res2.trades.iloc[0]
    assert t["entry_px"] == pytest.approx(99.0)
    assert t["maker"] == 1
    assert t["fees"] == pytest.approx(t["qty"] * 99.0 * 0.0002 + t["qty"] * t["exit_px"] * 0.0005)


def test_trailing_stop_ratchets_on_bar_close():
    # Price rallies 100 -> 110 over bar 1, trail 3 -> stop moves to 107 at bar-1 close, hit in bar 2.
    up = [(100 + i * 10 / 15, 100 + (i + 1) * 10 / 15, 100 + i * 10 / 15, 100 + (i + 1) * 10 / 15)
          for i in range(15)]
    paths = flat(100, 15) + up + [(110, 110, 106, 106)] + flat(106, 14)
    tr = np.full(3, 3.0)
    res = engine.run(market(paths), sig(3, 0, 1, 90.0, trail=tr), C, R)
    t = res.trades.iloc[0]
    assert t["reason"] == "stop"
    assert t["exit_px"] == pytest.approx(107 * (1 - 0.0001))


def test_reverse_on_opposite_signal():
    paths = flat(100, 45)
    e = np.array([1, -1, 0], np.int8)
    s = engine.Signals("15m", e, np.array([90.0, 110.0, np.nan]), np.full(3, np.nan))
    res = engine.run(market(paths), s, C, R)
    assert list(res.trades["reason"]) == ["reverse", "end"]
    assert list(res.trades["dir"]) == [1, -1]


@needs_data
def test_buy_and_hold_matches_btc_return():
    from btcq.data import store
    m1 = store.load_1m("dev")
    m1 = m1[m1["ts"] >= int(pd.Timestamp("2021-01-01", tz="UTC").value // 10**6)]
    mk = engine.Market(m1, store.load_funding("dev"))
    b = mk.bars("4h")
    e = np.zeros(len(b), np.int8)
    e[0] = 1
    s = engine.Signals("4h", e, np.r_[1.0, np.full(len(b) - 1, np.nan)], np.full(len(b), np.nan))
    res = engine.run(mk, s, Costs(taker=0.0005, slippage=0.0), Risk(1.0, 1.0, 10_000))
    t = res.trades.iloc[0]
    expected_gross = (mk.c[-1] / mk.o[t["entry_i"]] - 1) * 10_000
    assert t["gross"] == pytest.approx(expected_gross, rel=1e-9)
    assert t["net"] == pytest.approx(t["gross"] - t["fees"] - t["funding"])
    assert res.equity.iloc[-1] == pytest.approx(10_000 + t["net"], rel=1e-9)


def test_size_multiple_overrides_risk_sizing_and_respects_cap():
    paths = flat(100, 15) + [(100, 100.5, 99.5, 100)] + flat(100, 14)
    size = np.array([0.5, np.nan])
    res = engine.run(market(paths), sig(2, 0, 1, 90.0, size=size), C, R)
    t = res.trades.iloc[0]
    epx = 100 * 1.0001
    assert t["qty"] == pytest.approx(0.5 * 10_000 / epx)  # notional = 0.5x equity, not 1% risk
    assert t["init_risk"] == pytest.approx(t["qty"] * (epx - 90))
    res2 = engine.run(market(paths), sig(2, 0, 1, 90.0, size=np.array([7.0, np.nan])), C, R)
    assert res2.trades.iloc[0]["qty"] == pytest.approx(3 * 10_000 / epx)  # capped at 3x
