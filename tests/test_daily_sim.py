"""Round 8 daily simulator: engine parity at lag 1, same-close fills at lag 0, liquidation, calendar returns and the
funding drag. Synthetic paths only."""
import numpy as np
import pandas as pd
import pytest

from btcq.backtest import daily_sim as DS
from btcq.backtest import engine
from btcq.config import Costs, Risk
from btcq.data import external as E
from btcq.strategies import r6_macro as M

T0 = pd.Timestamp("2001-01-02", tz="UTC")


def trade_dates(n, seed=0):
    """Weekdays with a few random holes (holidays)."""
    rng = np.random.default_rng(seed)
    d = pd.bdate_range(T0, periods=int(n * 1.05) + 5, tz="UTC")
    d = d[rng.random(len(d)) > 0.04][:n]
    return d.as_unit("ms").asi8.astype(np.int64)


def random_path(n, seed):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, n)))
    pos = np.zeros(n, np.int8)
    for i in range(1, n):
        pos[i] = pos[i - 1] if rng.random() < 0.6 else rng.choice([-1, 0, 1])
    size = rng.uniform(0.3, 2.6, n)
    size[:7] = np.nan
    pos[:7] = 0
    return c, pos, size


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_lag1_reproduces_engine_run(seed):
    n = 260
    ts = trade_dates(n, seed)
    c, pos, size = random_path(n, seed)
    first = 20
    slip, fee, eq0 = 3e-4, 5e-4, 10_000 / 3
    mk = engine.Market(E.rows_from_closes(pd.Series(c, index=ts)), None, None, synthetic_funding=0.0)
    sig = M.engine_signals(pos, np.where(pos != 0, size, np.nan))
    res = engine.run(mk, sig, Costs(taker=fee, maker=2e-4, slippage=slip, synthetic_funding=0.0),
                     Risk(risk_per_trade=0.01, max_leverage=2.0, initial_equity=eq0),
                     start=pd.Timestamp(int(ts[first]), unit="ms", tz="UTC"), latency_min=0)
    r = DS.simulate(c, c, pos, size, first, n - 1, lag=1, fee=fee, slip=slip, eq0=eq0, liquidate=False)
    et, dt = res.trades, r.trades
    assert len(et) == len(dt) > 10
    for a, b in (("entry_i", "entry_i"), ("exit_i", "exit_i"), ("dir", "dir")):
        np.testing.assert_array_equal(et[a].to_numpy(), dt[b].to_numpy())
    for col in ("entry_px", "exit_px", "qty", "gross", "fees", "net", "eq_before"):
        np.testing.assert_allclose(dt[col].to_numpy(), et[col].to_numpy(), rtol=0, atol=1e-12 * eq0)
    reasons = {"exit_signal": {"exit", "reverse"}, "end": {"end"}}
    assert all(d in reasons[e] for e, d in zip(et["reason"], dt["reason"]))
    eq = res.equity.to_numpy()
    np.testing.assert_allclose(r.equity[first:], eq, rtol=0, atol=1e-12 * eq0)
    assert len(eq) == n - first


def reference_lag0(c, pos, size, first, last, fee, slip, eq0, cap=2.0):
    """An independent same-close account: (marks, list of (entry_i, exit_i, side, net))."""
    cash, p, q, epx, ef, ee, e_i = eq0, 0, 0.0, 0.0, 0.0, 0.0, 0
    marks, trades = np.full(len(c), np.nan), []
    for i in range(first, last + 1):
        tau = 0 if i == last else int(pos[i])
        if p != 0 and tau != p:
            px = c[i] * (1 - slip * p)
            g = q * (px - epx) * p
            f = q * px * fee
            cash += g - f
            trades.append((e_i, i, p, g - ef - f))
            p = 0
        if p == 0 and tau != 0 and np.isfinite(size[i]):
            px = c[i] * (1 + slip * tau)
            q = min(size[i], cap) * cash / px
            p, epx, ef, e_i = tau, px, q * px * fee, i
            cash -= ef
        marks[i] = cash + (q * (c[i] - epx) * p if p else 0.0)
    return marks, trades


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_lag0_fills_at_the_decision_close(seed):
    n = 300
    c, pos, size = random_path(n, seed)
    r = DS.simulate(c, c, pos, size, 15, n - 1, lag=0, fee=5e-4, slip=2e-4, eq0=1000.0, liquidate=False)
    marks, trades = reference_lag0(c, pos, size, 15, n - 1, 5e-4, 2e-4, 1000.0)
    np.testing.assert_allclose(r.equity[15:], marks[15:], rtol=0, atol=1e-9)
    assert [(a, b, s) for a, b, s, _ in trades] == list(zip(r.trades["entry_i"], r.trades["exit_i"], r.trades["dir"]))
    np.testing.assert_allclose(r.trades["net"].to_numpy(), [x[3] for x in trades], rtol=0, atol=1e-9)


def test_same_close_pnl_identity():
    """Long at C0, renewed at C1, exited at C2: every day's equity change is qty x close change, and the fill days
    carry exactly the fees and slippage."""
    c = np.array([100.0, 102.0, 101.0, 103.0, 104.0])
    pos = np.array([1, 1, 0, 0, 0], np.int8)
    size = np.full(5, 1.5)
    fee, slip, eq0 = 5e-4, 3e-4, 1000.0
    r = DS.simulate(c, c, pos, size, 0, 4, lag=0, fee=fee, slip=slip, eq0=eq0)
    px_in = 100 * (1 + slip)
    q = 1.5 * eq0 / px_in
    assert r.trades.loc[0, "qty"] == pytest.approx(q, rel=1e-15)
    assert r.equity[0] == pytest.approx(eq0 - q * px_in * fee + q * (100 - px_in), rel=1e-14)
    assert r.equity[1] - r.equity[0] == pytest.approx(q * 2.0, rel=1e-12)           # renewal: no cost, no resize
    px_out = 101 * (1 - slip)
    assert r.equity[2] - r.equity[1] == pytest.approx(q * (px_out - 102) - q * px_out * fee, rel=1e-12)
    assert r.equity[3] == r.equity[2] == r.equity[4]
    assert len(r.trades) == 1 and r.trades.loc[0, "exit_i"] == 2 and r.trades.loc[0, "reason"] == "exit"
    assert r.fees[0] == pytest.approx(q * px_in * fee) and r.fees[2] == pytest.approx(q * px_out * fee)
    assert r.lev[0] == pytest.approx(q * 100 / r.equity[0]) and r.lev[2] == 0.0


def test_flip_and_the_last_row():
    c = np.array([100.0, 99.0, 101.0, 100.0])
    pos = np.array([1, -1, -1, 1], np.int8)
    r = DS.simulate(c, c, pos, np.ones(4), 0, 3, lag=0, fee=0.0, slip=0.0, eq0=100.0)
    assert list(r.trades["reason"]) == ["reverse", "end"]
    assert list(r.trades["dir"]) == [1, -1]
    assert r.turnover[3] > 0 and r.lev[3] == 0.0                # the last row's long is never opened
    assert r.trades.loc[1, "qty"] == pytest.approx(99.0 / 99.0)  # the short is sized on the cash after the exit


def test_cap_and_liquidation():
    c = np.array([100.0, 100.0, 40.0, 30.0, 50.0])
    pos = np.array([1, 1, 1, 1, 1], np.int8)
    r = DS.simulate(c, c, pos, np.full(5, 5.0), 0, 4, lag=0, fee=0.0, slip=0.0, eq0=100.0)
    assert r.trades.loc[0, "qty"] == pytest.approx(2.0)          # min(size, 2) x cash / price
    assert r.liq_row == 2 and r.trades.loc[0, "reason"] == "liquidation"
    assert r.equity[2] == 0.0 and np.all(r.equity[2:] == 0.0)   # closed at max(equity, 0), flat after
    assert r.trades.loc[0, "net"] == pytest.approx(-100.0)
    c2 = np.array([100.0, 100.0, 96.0, 90.0])
    r2 = DS.simulate(c2, c2, np.ones(4, np.int8), np.full(4, 2.0), 0, 3, lag=0, fee=0.0, slip=0.0, eq0=100.0)
    assert r2.liq_row == -1                                      # notional 192 < 20 x equity 92
    # equity 2 against a notional of 102 (>= 20 x 2) at 51: liquidated at the mark, booked at the equity
    c3 = np.array([100.0, 100.0, 51.0, 80.0])
    r3 = DS.simulate(c3, c3, np.ones(4, np.int8), np.full(4, 2.0), 0, 3, lag=0, fee=0.0, slip=0.0, eq0=100.0)
    assert r3.liq_row == 2 and r3.equity[2] == pytest.approx(2.0) and r3.equity[3] == pytest.approx(2.0)
    assert list(r3.trades["reason"]) == ["liquidation"]


def test_an_exit_below_zero_equity_is_capped_like_a_liquidation():
    c = np.array([100.0, 100.0, 40.0, 41.0])
    r = DS.simulate(c, c, np.array([1, 1, 0, 1], np.int8), np.full(4, 2.0), 0, 3, lag=0, fee=5e-4, slip=0.0,
                    eq0=100.0)
    assert r.liq_row == 2 and list(r.trades["reason"]) == ["liquidation"]
    assert r.equity[2] == 0.0 and r.equity[3] == 0.0 and r.trades.loc[0, "net"] == pytest.approx(-100.0)
    assert r.turnover[3] == 0.0                                  # flat for good: the next signal is not opened


def test_the_end_of_window_close_is_capped_too():
    # entered at 100 (lag 1), marked at 50.06 with equity 0.02 > 0; the exit fee (0.05) leaves the cash < 0
    c = np.array([100.0, 100.0, 100.0, 50.06])
    r = DS.simulate(c, c, np.array([0, 1, 1, 1], np.int8), np.full(4, 2.0), 0, 3, lag=1, fee=5e-4, slip=0.0,
                    eq0=100.0, liq_lev=1e9)
    assert r.equity[3] == 0.0 and r.trades.loc[0, "reason"] == "liquidation" and r.liq_row == 3
    assert r.trades.loc[0, "net"] == pytest.approx(-100.0)


def test_calendar_daily_and_funding_drag():
    ts = pd.DatetimeIndex(["2001-01-04", "2001-01-05", "2001-01-08"], tz="UTC").as_unit("ms").asi8
    cal = pd.date_range("2001-01-03", "2001-01-09", tz="UTC").as_unit("ms")
    eq = np.array([101.0, 102.0, 99.0])
    d = DS.calendar_daily(ts, eq, cal, 100.0)
    np.testing.assert_allclose(d.to_numpy(), [0, 0.01, 102 / 101 - 1, 0, 0, 99 / 102 - 1, 0])
    lev = np.array([1.5, -2.0, 0.0])
    f = DS.funding_drag(ts, lev, cal, 0.05)
    assert f.loc["2001-01-05"] == pytest.approx(1.5 * 0.05 / 365)
    assert f.loc["2001-01-08"] == pytest.approx(2.0 * 0.05 * 3 / 365)       # Friday's short carries the weekend
    assert f.sum() == pytest.approx((1.5 + 6.0) * 0.05 / 365)


def test_inputs_are_checked():
    c = np.ones(5)
    with pytest.raises(ValueError):
        DS.simulate(c, c, np.zeros(5, np.int8), np.ones(5), 3, 2)
    with pytest.raises(ValueError):
        DS.simulate(c, c, np.zeros(5, np.int8), np.ones(5), 0, 4, lag=2)
    bad = c.copy()
    bad[2] = np.nan
    with pytest.raises(ValueError):
        DS.simulate(bad, bad, np.zeros(5, np.int8), np.ones(5), 0, 4)
