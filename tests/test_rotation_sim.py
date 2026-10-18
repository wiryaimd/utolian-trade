"""Round 7 rotation simulator: single-coin parity with engine.run, forced exits, resizes, hard checks and the
minute -> daily columns."""
import numpy as np
import pandas as pd
import pytest

from btcq.backtest import engine, rotation_sim as RS
from btcq.config import Costs, Risk
from btcq.research.round4 import no_trade_gaps

DAY = RS.DAY_MS
COOL = 7 * DAY
T0 = int(pd.Timestamp("2021-03-01", tz="UTC").value // 10**6)


def synth_m1(days: int, seed: int = 1, t0: int = T0, sigma: float = 0.0012, drift: float = 0.0):
    rng = np.random.default_rng(seed)
    n = days * 1440
    ts = t0 + 60_000 * np.arange(n, dtype=np.int64)
    c = 100.0 * np.exp(np.cumsum(rng.normal(drift, sigma, n)))
    o = np.r_[100.0, c[:-1]] * np.exp(rng.normal(0, 0.0002, n))   # opens differ from the previous close
    hi, lo = np.maximum(o, c) * 1.0003, np.minimum(o, c) * 0.9997
    m1 = pd.DataFrame({"ts": ts, "open": o, "high": hi, "low": lo, "close": c, "volume": np.ones(n),
                       "quote_volume": c, "taker_buy_volume": np.full(n, 0.5), "is_perp": True})
    fts = t0 + np.arange(days * 3, dtype=np.int64) * 8 * 3_600_000
    fund = pd.DataFrame({"ts": fts, "rate": rng.normal(0.0001, 0.0004, len(fts))})
    return m1, fund


def one_coin_panel(mk: engine.Market, slip: float, end_ms: int, sym: str = "AAAUSDT"):
    cal = np.arange(mk.ts[0] // DAY * DAY, end_ms, DAY, dtype=np.int64)
    fx, gaps = no_trade_gaps(mk.ts, mk.m1["volume"].to_numpy(), end_ms)
    cols = RS.daily_columns(mk.ts, mk.o, mk.c, mk.m1["volume"].to_numpy(), mk.fund, cal, fx, gaps, COOL)
    n = len(cal)
    days = pd.to_datetime(cal, unit="ms", utc=True).as_unit("ms")
    p = RS.stack({sym: cols}, days, np.ones((n, 1), bool), np.full((n, 1), 5), np.full((n, 1), slip))
    return p, fx


def engine_signals(mk: engine.Market, p: RS.Panel, W: np.ndarray, slip: float):
    """engine.Signals equivalent of fill-day weights W (no resize): the decision at bar d fills on day d + 1."""
    bars = mk.bars("1d")
    bt = bars["ts"].to_numpy(np.int64)
    cal = p.days.asi8
    t_fill = np.searchsorted(cal, bt + DAY)
    nb = len(bt)
    w = np.where(t_fill < len(cal), W[np.minimum(t_fill, len(cal) - 1), 0], 0.0)
    entry = (w > 0).astype(np.int8)
    size = np.where(entry == 1, w * (1 + slip), np.nan)   # the engine sizes on the slipped price
    return engine.Signals("1d", entry, np.where(entry == 1, 0.0, np.nan), np.full(nb, np.nan),
                          exit_long=(w == 0), size=size)


def compare(res: engine.Result, sim: RS.SimResult, p: RS.Panel, mk: engine.Market):
    tr, ep = res.trades, sim.episodes
    assert len(tr) == len(ep) > 0
    day_of = lambda idx: np.searchsorted(p.days.asi8, mk.ts[idx] // DAY * DAY)  # noqa: E731
    np.testing.assert_array_equal(day_of(tr["entry_i"].to_numpy()), ep["entry_day"].to_numpy())
    np.testing.assert_array_equal(day_of(tr["exit_i"].to_numpy()), ep["exit_day"].to_numpy())
    for a, b in (("qty", "qty0"), ("entry_px", "entry_px"), ("exit_px", "exit_px"), ("fees", "fees"),
                 ("funding", "funding"), ("net", "net"), ("gross", "cash_flow"), ("eq_before", "eq_before")):
        np.testing.assert_allclose(tr[a].to_numpy(), ep[b].to_numpy(), rtol=1e-9, atol=1e-9, err_msg=a)
    assert list(tr["reason"].replace({"exit_signal": "rule"})) == list(ep["reason"])
    eq = res.equity.copy()
    eq.index = eq.index.as_unit("ms")
    eq = eq.reindex(sim.equity.index)
    np.testing.assert_allclose(eq.to_numpy(), sim.equity.to_numpy(), rtol=1e-11)


def weights_pattern(n: int, first: int, pattern):
    W = np.zeros((n, 1))
    for a, b, w in pattern:
        W[first + a:first + b, 0] = w
    return W


@pytest.mark.parametrize("w", [1 / 3, 1.0])
def test_single_coin_parity_with_engine_run(w):
    m1, fund = synth_m1(40, seed=3)
    mk = engine.Market(m1, fund, None, synthetic_funding=Costs.synthetic_funding)
    end_ms = int(m1["ts"].iloc[-1]) + 60_000
    slip = 0.0003
    p, fx = one_coin_panel(mk, slip, end_ms)
    first, last = 5, p.shape[0] - 1
    W = weights_pattern(p.shape[0], first, [(0, 4, w), (6, 13, w), (13, 14, 0.0), (15, 30, w), (31, 99, w)])
    cb = np.ones(p.shape, bool)
    sim = RS.simulate(p, W, cb, first, last, taker=0.0005, slip_mult=1.0, penalty=0.01, resize=False)
    sig = engine_signals(mk, p, W, slip)
    res = engine.run(mk, sig, Costs(taker=0.0005, maker=0.0002, slippage=0.0), Risk(1.0, 3.0, 10_000.0),
                     start=p.days[first], latency_min=1, slippage=np.full(len(mk.ts), slip),
                     force_exit=np.zeros(len(mk.ts), bool), force_penalty=0.01)
    assert list(sim.episodes["reason"]) == ["rule", "rule", "rule", "end"]
    compare(res, sim, p, mk)


def test_forced_exit_parity_and_cooldown_block():
    m1, fund = synth_m1(45, seed=5)
    day0 = int(m1["ts"].iloc[0])
    a = day0 + 20 * DAY + 13 * 3_600_000            # stops trading on day 20 at 13:00, resumes 2.5 days later
    b = a + int(2.5 * DAY)
    halt = (m1["ts"] > a) & (m1["ts"] < b)
    m1.loc[halt, "volume"] = 0.0
    m1.loc[halt, ["open", "high", "low", "close"]] = float(m1.loc[m1["ts"] == a, "close"].iloc[0])
    mk = engine.Market(m1, fund, None, synthetic_funding=Costs.synthetic_funding)
    end_ms = int(m1["ts"].iloc[-1]) + 60_000
    p, fx = one_coin_panel(mk, 0.0002, end_ms)
    t_a = p.index_of(pd.Timestamp(a // DAY * DAY, unit="ms", tz="UTC"))
    assert p.fx_post[t_a, 0] and not p.fx_pre[t_a, 0]
    blocked = np.flatnonzero(p.blocked[:, 0])
    # decisions from day 20's close (the halt began at 13:00) until 7 days after trading resumes (day 23 01:00)
    np.testing.assert_array_equal(blocked, np.arange(20, 30))
    first, last = 3, p.shape[0] - 1
    W = np.zeros((p.shape[0], 1))
    W[first:t_a + 1, 0] = 0.5                       # held into the halt -> forced exit on day t_a
    W[blocked + 1, 0] = 0.0
    W[blocked[-1] + 2:, 0] = 0.5
    sim = RS.simulate(p, W, np.ones(p.shape, bool), first, last, taker=0.0005, penalty=0.01, resize=False)
    force = np.zeros(len(mk.ts), bool)
    force[fx] = True
    sig = engine_signals(mk, p, W, 0.0002)
    res = engine.run(mk, sig, Costs(taker=0.0005, maker=0.0002, slippage=0.0), Risk(1.0, 3.0, 10_000.0),
                     start=p.days[first], latency_min=1, slippage=np.full(len(mk.ts), 0.0002), force_exit=force,
                     force_penalty=0.01)
    assert list(sim.episodes["reason"]) == ["delist", "end"]
    compare(res, sim, p, mk)


def test_forced_exit_before_the_fill_minute_is_taken_at_that_minute():
    m1, fund = synth_m1(30, seed=7)
    day0 = int(m1["ts"].iloc[0])
    a = day0 + 15 * DAY                              # the 00:00 minute of day 15 is the last traded one
    m1.loc[(m1["ts"] > a) & (m1["ts"] < a + 2 * DAY), "volume"] = 0.0
    mk = engine.Market(m1, fund, None, synthetic_funding=Costs.synthetic_funding)
    p, fx = one_coin_panel(mk, 0.0002, int(m1["ts"].iloc[-1]) + 60_000)
    assert p.fx_pre[15, 0] and not p.fx_post[15, 0] and p.blocked[14, 0]
    j = int(fx[0])
    assert mk.ts[j] == a
    np.testing.assert_allclose(p.fx_px[15, 0], mk.c[j])
    np.testing.assert_allclose(p.fund_fx[15, 0], mk.o[j] * mk.fund[j])


def two_coin_panel(n=6):
    days = pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC").as_unit("ms")
    z = np.zeros((n, 2))
    f = np.zeros((n, 2), bool)
    close = np.array([[10, 20], [11, 20], [12, 18], [12, 18], [13, 19], [13, 19]], float)[:n]
    fill = np.array([[10, 20], [10.5, 20.5], [11.5, 19], [12, 18], [12.5, 18.5], [13, 19]], float)[:n]
    return RS.Panel(days=days, symbols=["AUSDT", "BUSDT"], close=close, traded=~f, fill_px=fill,
                    fund_pre=z.copy(), fund_post=z.copy(), fx_pre=f.copy(), fx_post=f.copy(), fx_px=np.full((n, 2), np.nan),
                    fund_fx=z.copy(), blocked=f.copy(), days_since_gap=np.full((n, 2), np.inf), member=~f,
                    rank=np.full((n, 2), 5), slip=np.full((n, 2), 0.001))


def test_two_coin_daily_resize_by_hand():
    p = two_coin_panel()
    W = np.zeros((6, 2))
    W[1] = [0.5, 0.5]
    W[2] = [0.5, 0.5]
    W[3] = [0.5, 0.0]
    sim = RS.simulate(p, W, np.ones((6, 2), bool), 1, 4, taker=0.0005, penalty=0.0, resize=True, K=3)
    e0, fee, s = 10_000.0, 0.0005, 0.001
    # day 1: buy both at the fills with slippage
    qa, qb = 0.5 * e0 / 10.5, 0.5 * e0 / 20.5
    cash = e0 - qa * 10.5 * (1 + s) * (1 + fee) - qb * 20.5 * (1 + s) * (1 + fee)
    e1 = cash + qa * 11 + qb * 20
    # day 2: resize both to half of e1 at the fills
    qa2, qb2 = 0.5 * e1 / 11.5, 0.5 * e1 / 19
    for dq, x in ((qa2 - qa, 11.5), (qb2 - qb, 19)):
        px = x * (1 + s) if dq > 0 else x * (1 - s)
        cash -= dq * px + abs(dq) * px * fee
    e2 = cash + qa2 * 12 + qb2 * 18
    # day 3: A resized, B sold
    qa3 = 0.5 * e2 / 12
    for dq, x in ((qa3 - qa2, 12), (-qb2, 18)):
        px = x * (1 + s) if dq > 0 else x * (1 - s)
        cash -= dq * px + abs(dq) * px * fee
    e3 = cash + qa3 * 12
    # day 4: A sold at the fill (W[4] = 0)
    cash += qa3 * 12.5 * (1 - s) * (1 - fee)
    e4 = cash
    np.testing.assert_allclose(sim.equity.to_numpy(), [e1, e2, e3, e4], rtol=1e-12)
    ep = sim.episodes.sort_values(["exit_day", "symbol"])
    assert list(ep["symbol"]) == ["BUSDT", "AUSDT"] and list(ep["trades"]) == [3, 4]
    np.testing.assert_allclose(ep["net"].sum(), e4 - e0, rtol=1e-12)


def test_no_resize_keeps_the_quantity():
    p = two_coin_panel()
    W = np.zeros((6, 2))
    W[1:4, 0] = 0.5
    a = RS.simulate(p, W, np.ones((6, 2), bool), 1, 4, penalty=0.0, resize=False)
    assert a.episodes["trades"].tolist() == [2]
    b = RS.simulate(p, W, np.ones((6, 2), bool), 1, 4, penalty=0.0, resize=True)
    assert b.episodes["trades"].tolist() == [4]


def test_stuck_coin_keeps_its_quantity_and_marks_at_the_last_close():
    p = two_coin_panel()
    p.fill_px[2, 0] = np.nan
    p.close[2, 0] = np.nan                    # no row at all on day 2
    W = np.zeros((6, 2))
    W[1, 0] = 0.5
    W[2, 0] = np.nan                          # stuck: keep
    W[3, 0] = 0.0
    sim = RS.simulate(p, W, np.ones((6, 2), bool), 1, 4, penalty=0.0)
    q = 0.5 * 10_000 / 10.5
    assert sim.equity.iloc[1] == pytest.approx(sim.equity.iloc[0])   # marked at day 1's close again
    assert sim.episodes["exit_day"].tolist() == [3]
    assert sim.episodes["qty0"].iloc[0] == pytest.approx(q)


@pytest.mark.parametrize("case,msg", [
    ("price", "fill without a price"), ("cand", "not a candidate"), ("slots", "more holdings"),
    ("sum", "sum above 1"), ("exit", "without a fill price"), ("equity", "equity at or below zero")])
def test_hard_checks(case, msg):
    p = two_coin_panel()
    W = np.zeros((6, 2))
    cb = np.ones((6, 2), bool)
    K = 3
    if case == "price":
        p.fill_px[2, 1] = np.nan
        W[2, 1] = 0.3
    elif case == "cand":
        cb[2, 0] = False
        W[2, 0] = 0.3
    elif case == "slots":
        W[2] = [0.3, 0.3]
        K = 1
    elif case == "sum":
        W[2] = [0.6, 0.6]
    elif case == "exit":
        W[1, 0] = 0.3
        p.fill_px[2, 0] = np.nan
    elif case == "equity":
        W[1, 0] = 1.0
        p.close[1, 0] = 1e-9                  # the coin is worth nothing at the mark: fees leave equity < 0
        K = 1
    with pytest.raises(RS.SimError, match=msg):
        RS.simulate(p, W, cb, 1, 4, penalty=0.0, K=K)


def test_fee_and_slippage_multipliers_change_costs_only():
    p = two_coin_panel()
    W = np.zeros((6, 2))
    W[1:4] = 0.5
    a = RS.simulate(p, W, np.ones((6, 2), bool), 1, 4, penalty=0.0)
    b = RS.simulate(p, W, np.ones((6, 2), bool), 1, 4, penalty=0.0, slip_mult=2.0, fee_mult=2.0)
    z = RS.simulate(p, W, np.ones((6, 2), bool), 1, 4, penalty=0.0, slip_mult=0.0, fee_mult=0.0)
    assert z.equity.iloc[-1] > a.equity.iloc[-1] > b.equity.iloc[-1]
    assert (z.fees == 0).all() and b.fees.sum() > 1.9 * a.fees.sum()
    assert RS.mean_daily_fast(p, W, np.ones((6, 2), bool), 1, 4, taker=0.0005, slip_mult=1.0, penalty=0.0) == \
        pytest.approx(a.daily.mean(), rel=1e-12)


# ---------------------------------------------------------------- minute -> daily columns

def tiny_coin(rows, fund_rows=(), t0=T0):
    """rows: (minute offset from t0, open, close, volume)."""
    a = np.asarray(rows, float)
    ts = (t0 + a[:, 0] * 60_000).astype(np.int64)
    m1 = pd.DataFrame({"ts": ts, "open": a[:, 1], "high": np.maximum(a[:, 1], a[:, 2]),
                       "low": np.minimum(a[:, 1], a[:, 2]), "close": a[:, 2], "volume": a[:, 3],
                       "quote_volume": a[:, 3], "taker_buy_volume": a[:, 3] / 2, "is_perp": True})
    f = pd.DataFrame({"ts": np.asarray([t0 + m * 60_000 for m, _ in fund_rows], np.int64),
                      "rate": np.asarray([r for _, r in fund_rows], float)})
    mk = engine.Market(m1, f, None, synthetic_funding=0.0)
    cal = np.arange(T0, T0 + 3 * DAY, DAY, dtype=np.int64)
    fx, gaps = no_trade_gaps(mk.ts, m1["volume"].to_numpy(), int(cal[-1] + DAY))
    return RS.daily_columns(mk.ts, mk.o, mk.c, m1["volume"].to_numpy(), mk.fund, cal, fx, gaps, COOL), mk


def test_missing_2359_bar_uses_the_last_row_and_missing_0001_fills_at_the_next_row():
    rows = [(0, 10, 10, 1), (1, 10, 10.1, 1), (1438, 11, 11.5, 1),           # day 0 ends at 23:58
            (1440, 12, 12, 1), (1442, 12.2, 12.3, 1), (2879, 13, 13, 1),       # day 1: no 00:01 row
            (2880, 14, 14, 1), (2890, 14, 14, 1), (4319, 14, 14.4, 1)]
    cols, _ = tiny_coin(rows)
    np.testing.assert_allclose(cols["close"], [11.5, 13, 14.4])
    np.testing.assert_allclose(cols["fill_px"], [10, 12.2, 14])


def test_no_row_after_0001_means_no_fill_that_day():
    rows = [(0, 10, 10, 1), (1, 10, 10, 1), (1440, 11, 11, 1), (2880, 12, 12, 1), (2881, 12, 12, 1),
            (4319, 12, 12, 1)]
    cols, _ = tiny_coin(rows)
    assert np.isnan(cols["fill_px"][1]) and np.isfinite(cols["fill_px"][[0, 2]]).all()


def test_funding_lands_before_or_after_the_fill_minute():
    rows = [(m, 100, 100, 1) for m in range(0, 3 * 1440, 1) if m != 1440 + 0]   # the 00:00 row of day 1 is missing
    fund = [(0, 0.001), (480, 0.002), (1440, 0.003), (1440 + 480, 0.004), (2880, 0.005)]
    cols, mk = tiny_coin(rows, fund)
    # day 0: 00:00 settles before the fill, 08:00 after; day 1: the 00:00 rate lands on the 00:01 fill minute
    np.testing.assert_allclose(cols["fund_pre"], [0.1, 0.3, 0.5])
    np.testing.assert_allclose(cols["fund_post"], [0.2, 0.4, 0.0])


def test_gap_in_lookback_and_cooldown():
    rows = [(m, 100, 100, 1) for m in range(0, 3 * 1440)]
    cols, _ = tiny_coin(rows)
    assert not cols["blocked"].any() and np.isinf(cols["days_since_gap"]).all()
    m1, fund = synth_m1(30, seed=2)
    a = int(m1["ts"].iloc[0]) + 5 * DAY + 600 * 60_000
    m1 = m1[(m1["ts"] <= a) | (m1["ts"] >= a + 2 * DAY)].reset_index(drop=True)   # a 48h hole in the rows
    mk = engine.Market(m1, fund, None, synthetic_funding=0.0)
    p, _ = one_coin_panel(mk, 0.0002, int(m1["ts"].iloc[-1]) + 60_000)
    b_day = 7                                   # trading resumes on day 7 at 10:00
    assert p.blocked[5, 0] and p.blocked[6, 0] and p.blocked[b_day + 6, 0] and not p.blocked[b_day + 7, 0]
    np.testing.assert_allclose(p.days_since_gap[b_day + 7, 0], 7 + 14 / 24)
    assert not p.blocked[4, 0]


def test_stress_doubles_fees_and_slippage_on_forced_exits_but_not_the_penalty():
    p = two_coin_panel()
    p.fx_post[3, 0] = True
    p.fx_px[3, 0] = 11.0
    W = np.zeros((6, 2))
    W[1:4, 0] = 0.5
    for mult in (1.0, 2.0):
        sim = RS.simulate(p, W, np.ones((6, 2), bool), 1, 4, penalty=0.01, slip_mult=mult, fee_mult=mult,
                          resize=False)                       # no resize trade on day 3: only the forced exit
        ep = sim.episodes.iloc[0]
        assert ep["reason"] == "delist"
        assert ep["exit_px"] == pytest.approx(11.0 * (1 - (0.001 * mult + 0.01)), rel=1e-12)
        assert sim.turnover.iloc[2] > 0                       # the forced exit is a trade (day 3 = index 2)


def test_forced_exit_at_0000_parity():
    m1, fund = synth_m1(35, seed=9)
    day0 = int(m1["ts"].iloc[0])
    a = day0 + 20 * DAY                              # the 00:00 minute of day 20 is the last traded one
    halt = (m1["ts"] > a) & (m1["ts"] < a + 2 * DAY)
    m1.loc[halt, "volume"] = 0.0
    mk = engine.Market(m1, fund, None, synthetic_funding=Costs.synthetic_funding)
    p, fx = one_coin_panel(mk, 0.0002, int(m1["ts"].iloc[-1]) + 60_000)
    assert p.fx_pre[20, 0] and p.blocked[19, 0]
    first, last = 3, p.shape[0] - 1
    W = np.zeros((p.shape[0], 1))
    W[first:20, 0] = 0.5
    blocked = np.flatnonzero(p.blocked[:, 0])
    W[blocked[-1] + 2:, 0] = 0.5
    sim = RS.simulate(p, W, np.ones(p.shape, bool), first, last, penalty=0.01, resize=False)
    force = np.zeros(len(mk.ts), bool)
    force[fx] = True
    res = engine.run(mk, engine_signals(mk, p, W, 0.0002), Costs(taker=0.0005, maker=0.0002, slippage=0.0),
                     Risk(1.0, 3.0, 10_000.0), start=p.days[first], latency_min=1,
                     slippage=np.full(len(mk.ts), 0.0002), force_exit=force, force_penalty=0.01)
    assert list(sim.episodes["reason"]) == ["delist", "end"]
    compare(res, sim, p, mk)


def test_a_fill_day_with_only_the_0000_row_is_stuck_not_a_gap():
    from btcq.strategies import r7_rotation as R
    m1, fund = synth_m1(20, seed=4)
    day0 = int(m1["ts"].iloc[0])
    m1 = m1[(m1["ts"] <= day0 + 10 * DAY) | (m1["ts"] >= day0 + 11 * DAY)].reset_index(drop=True)
    mk = engine.Market(m1, fund, None, synthetic_funding=Costs.synthetic_funding)
    p, fx = one_coin_panel(mk, 0.0002, int(m1["ts"].iloc[-1]) + 60_000)
    assert not len(fx) and not p.blocked[:, 0].any()          # exactly 24h without a trade is not a gap
    assert np.isnan(p.fill_px[10, 0]) and np.isfinite(p.close[10, 0])
    sc = np.full(p.shape, np.nan)
    sc[8, 0] = 0.1                                             # bought on day 9 from the day-8 decision
    sel = R.select_path(sc, R.no_fill_next(p), p.blocked, 3, 6, 8, 10)
    assert sel[8, 0] == 1 and sel[9, 0] == R.STUCK and sel[10, 0] == 0
    W = R.to_weights(sel, 3, 8, 10)
    assert W[9, 0] == pytest.approx(1 / 3) and np.isnan(W[10, 0]) and W[11, 0] == 0
    sim = RS.simulate(p, W, np.ones(p.shape, bool), 9, 12, penalty=0.01)
    ep = sim.episodes
    assert ep["entry_day"].tolist() == [9] and ep["exit_day"].tolist() == [11] and ep["trades"].tolist() == [2]
