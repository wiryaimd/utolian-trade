"""Round-3 portfolio mechanics: sleeve sizing, assembly, combined null, latency."""
import numpy as np
import pandas as pd
import pytest

from btcq.backtest import engine
from btcq.config import Costs
from btcq.research.portfolio import assemble, combine_null, sleeve_risk
from conftest import flat, make_m1

C0 = Costs(taker=0.0, maker=0.0, slippage=0.0)


def _sig(n_bars, d, stop):
    e = np.zeros(n_bars, np.int8)
    e[0] = d
    s = np.full(n_bars, np.nan)
    s[0] = stop
    return engine.Signals("15m", e, s, np.full(n_bars, np.nan))


def test_sleeve_risk_is_one_percent_of_account_and_notional_capped():
    n = 5
    r = sleeve_risk(n)
    assert r.risk_per_trade == pytest.approx(0.05) and r.max_leverage == 3.0
    mk = engine.Market(make_m1(flat(100, 45)), None, synthetic_funding=0.0)
    # Wide stop: risk-limited. Sleeve capital is 1/5 of the account.
    t = engine.run(mk, _sig(3, 1, 90.0), C0, r).trades.iloc[0]
    assert t["init_risk"] / r.initial_equity == pytest.approx(0.05)       # 5% of sleeve
    assert t["init_risk"] / (r.initial_equity * n) == pytest.approx(0.01)  # = 1% of account
    # Tight stop: the 3x sleeve cap binds, so 5 sleeves can never exceed 3x of the account.
    t2 = engine.run(mk, _sig(3, 1, 99.9), C0, r).trades.iloc[0]
    assert t2["qty"] * t2["entry_px"] == pytest.approx(3 * r.initial_equity)


def test_assemble_means_sleeves_and_dilutes_trade_returns():
    idx = pd.date_range("2022-01-01", periods=4, freq="D", tz="UTC")
    a = pd.Series([0.02, 0.0, -0.01, 0.0], index=idx)
    b = pd.Series([np.nan, 0.04, 0.01, 0.0], index=idx)  # sleeve starting a day later
    ta_ = pd.DataFrame({"entry_time": [idx[0]], "ret": [0.02], "R": [1.0]})
    tb_ = pd.DataFrame({"entry_time": [idx[1]], "ret": [0.04], "R": [2.0]})
    res, daily = assemble([(a, ta_), (b, tb_)])
    assert daily.tolist() == pytest.approx([0.02, 0.02, 0.0, 0.0])
    assert res.trades["ret"].tolist() == pytest.approx([0.01, 0.02])
    assert res.equity.iloc[-1] == pytest.approx(10_000 * 1.02 * 1.02)


def test_combine_null_is_trade_weighted():
    out = combine_null([np.array([1.0, 1.0]), np.array([0.0, 4.0])], [3, 1])
    assert out.tolist() == pytest.approx([0.75, 1.75])


def test_latency_delays_market_entry_by_one_minute():
    paths = flat(100, 15) + [(100, 100, 100, 100), (101, 101, 101, 101)] + flat(101, 13)
    mk = engine.Market(make_m1(paths), None, synthetic_funding=0.0)
    t0 = engine.run(mk, _sig(2, 1, 90.0), C0).trades.iloc[0]
    t1 = engine.run(mk, _sig(2, 1, 90.0), C0, latency_min=1).trades.iloc[0]
    assert t0["entry_i"] == 15 and t0["entry_px"] == pytest.approx(100)
    assert t1["entry_i"] == 16 and t1["entry_px"] == pytest.approx(101)
