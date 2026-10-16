"""Round 6: M1 long/short macro trend rule, engine mapping, funding stress, shift null, gates, data report,
registry isolation and guards (synthetic data)."""
import json

import numpy as np
import pandas as pd
import pytest

from btcq.backtest import engine, stats
from btcq.data import external as E
from btcq.data import store
from btcq.research import registry
from btcq.research import round6 as R
from btcq.research import run
from btcq.strategies import r5_d1 as D
from btcq.strategies import r6_macro as M

DAY = 86_400_000


def weekdays(first: str, n: int, seed: int = 0, holiday_p: float = 0.035) -> np.ndarray:
    d = pd.bdate_range(first, periods=int(n * 1.05) + 10, tz="UTC")
    keep = np.random.default_rng(seed).random(len(d)) > holiday_p
    return d[keep][:n].as_unit("ms").asi8


def closes(first: str, n: int, seed: int = 0, drift=None, vol: float = 0.012) -> pd.Series:
    rng = np.random.default_rng(seed)
    mu = np.zeros(n) if drift is None else np.asarray(drift, float)[:n]
    c = 50 * np.exp(np.cumsum(mu + rng.normal(0, vol, n)))
    return pd.Series(c, index=weekdays(first, n, seed + 100))


def window(s: pd.Series, pad_days: int = 3) -> R.Window:
    a = pd.Timestamp(int(s.index[0]), unit="ms", tz="UTC").floor("D")
    b = pd.Timestamp(int(s.index[-1]), unit="ms", tz="UTC").floor("D") + pd.Timedelta(days=pad_days)
    return R.Window("t", "dev", a, b)


def regimes(n: int, seed: int, amp: float = 0.004) -> np.ndarray:
    """Trend regimes of irregular length and random sign (no periodicity a shift could re-align with)."""
    rng = np.random.default_rng(seed)
    lens = rng.integers(20, 80, n // 20 + 2)
    sign = rng.choice([-1.0, 1.0], len(lens))
    return np.concatenate([np.full(k, amp * g) for k, g in zip(lens, sign)])[:n]


def closes_between(first: str, last: str, seed: int, amp: float = 0.004) -> pd.Series:
    """Weekday closes with holidays from first to last; trending regimes (amp 0 = a random walk)."""
    d = pd.bdate_range(first, last, tz="UTC")
    d = d[np.random.default_rng(seed + 100).random(len(d)) > 0.035]
    n = len(d)
    r = regimes(n, seed + 30, amp) + np.random.default_rng(seed).normal(0, 0.012, n)
    return pd.Series(50 * np.exp(np.cumsum(r)), index=d.as_unit("ms").asi8)


# ---------------------------------------------------------------- the rule

@pytest.mark.parametrize("seed,L", [(0, 3), (1, 5), (2, 20), (3, 60)])
def test_ls_state_matches_two_reference_machines(seed, L):
    from test_round5 import reference_state
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, 500)))
    ready = np.ones(500, bool)
    ready[:30] = False
    st, trail, _ = M.ls_state(c, L, ready)
    sl, _, tl = reference_state(list(c), L, ready)
    ss, _, ts = reference_state(list(-c), L, ready)
    assert np.array_equal(st, sl - ss) and not ((sl == 1) & (ss == 1)).any()
    assert (st == 1).any() and (st == -1).any()
    assert np.allclose(trail[st == 1], tl[st == 1]) and np.allclose(trail[st == -1], -ts[st == -1])
    assert np.all(trail[st == 1] <= c[st == 1]) and np.all(trail[st == -1] >= c[st == -1])


def test_ls_state_mirror():
    c = 100 * np.exp(np.cumsum(np.random.default_rng(7).normal(0, 0.02, 400)))
    ready = np.ones(400, bool)
    a, _, _ = M.ls_state(c, 10, ready)
    m, _, _ = M.ls_state(-c, 10, ready)
    assert np.array_equal(m, -a) and (a == 1).any() and (a == -1).any()


def test_vol_size_td_uses_252_nan_until_ready_and_caps():
    rng = np.random.default_rng(3)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))
    s = M.vol_size_td(c)
    assert np.isnan(s[:90]).all() and np.isfinite(s[90:]).all()
    r = np.diff(np.log(c))[100 - 90:100]
    assert np.isclose(s[100], min(0.25 / (r.std(ddof=0) * np.sqrt(252)), 2.0))
    flat = 100 * np.exp(np.cumsum(np.full(200, 1e-5) + rng.normal(0, 1e-5, 200)))
    assert np.nanmax(M.vol_size_td(flat)) == 2.0


def test_flip_is_two_fills_at_the_next_close():
    c = pd.Series([10.0, 11, 12, 13, 12, 11, 10, 9, 9.5, 9.8], index=weekdays("2024-01-01", 10, holiday_p=0))
    m = R.market_from_closes("XAUUSDT", c)
    state = np.array([0, 1, 1, 1, -1, -1, 0, 0, 0, 0], np.int8)
    sig = M.engine_signals(state, np.ones(10))
    res = engine.run(m.mk, sig, R.costs("XAUUSDT"), R.RISK, start=window(c).start, latency_min=0)
    t = res.trades
    assert t[["entry_i", "exit_i", "dir"]].values.tolist() == [[2, 5, 1], [5, 7, -1]]
    slip = M.SLIP["XAUUSDT"]
    assert np.isclose(t["entry_px"].iloc[0], 12 * (1 + slip)) and np.isclose(t["exit_px"].iloc[0], 11 * (1 - slip))
    assert np.isclose(t["entry_px"].iloc[1], 11 * (1 - slip)) and np.isclose(t["exit_px"].iloc[1], 9 * (1 + slip))
    assert t["reason"].tolist() == ["exit_signal", "exit_signal"]


def test_cost_table_and_stress():
    assert set(M.SLIP) == set(E.MARKETS)
    c = R.costs("COPPERUSDT")
    assert (c.taker, c.slippage, c.synthetic_funding) == (5e-4, 5e-4, 0.0)
    c2 = R.costs("COPPERUSDT", 2.0)
    assert np.isclose(c2.taker, 1e-3) and np.isclose(c2.slippage, 1e-3)
    assert R.costs("XAUUSDT").slippage == 2e-4 and R.costs("SPYUSDT").slippage == 3e-4


# ---------------------------------------------------------------- engine vs rule

@pytest.fixture(scope="module")
def mkts():
    out = {}
    for i, s in enumerate(["XAUUSDT", "CLUSDT", "SPYUSDT"]):
        first = ["2008-01-01", "2008-07-01", "2009-03-02"][i]
        out[s] = R.market_from_closes(s, closes_between(first, "2014-12-31", seed=10 + i))
    return out


def _w(mkts):
    a = min(int(m.bar_ts[0]) for m in mkts.values())
    b = max(int(m.bar_ts[-1]) for m in mkts.values())
    return R.Window("t", "dev", pd.Timestamp(a, unit="ms", tz="UTC"), pd.Timestamp(b + 3 * DAY, unit="ms", tz="UTC"))


def test_engine_follows_state_no_stops_and_short_R(mkts):
    w = _w(mkts)
    subs = R.build(mkts)
    assert len(subs) == 3 * 9 and all(not m.mk.fund.any() for m in mkts.values())
    parts, info = R.run_all(subs, mkts, w)
    mm = R.mismatches(subs, parts, mkts, w)
    assert mm == {"state_mismatches": 0, "stop_exits": 0} and not info["ruins"]
    t = pd.concat([tr for _, tr in parts.values()])
    sh = t[t["dir"] == -1]
    assert len(sh) > 10 and (t["dir"] == 1).sum() > 10
    key = ("XAUUSDT", 20)
    tr = parts[key][1]
    lvl = subs[key].trail[tr["entry_tf"].to_numpy()]
    dist = (tr["entry_px"].to_numpy() - lvl) * tr["dir"].to_numpy()
    ok = (tr["dir"] == -1).to_numpy() & (dist > 1e-3 * tr["entry_px"].to_numpy())
    assert ok.sum() > 3 and (lvl[ok] > tr["entry_px"].to_numpy()[ok]).all()
    assert np.allclose(tr["R"].to_numpy()[ok], tr["net"].to_numpy()[ok] / (tr["qty"].to_numpy()[ok] * dist[ok]))
    # fills are at the next trading day's close
    mk = mkts["XAUUSDT"].mk
    assert np.allclose(tr["entry_px"] / mk.c[tr["entry_i"]], 1 + tr["dir"] * M.SLIP["XAUUSDT"])
    assert (tr["entry_i"].to_numpy() == mk._ex[tr["entry_tf"].to_numpy()]).all()


def test_mismatches_catch_a_broken_position(mkts):
    w = _w(mkts)
    subs = R.build(mkts)
    parts, _ = R.run_all(subs, mkts, w)
    key = ("CLUSDT", 10)
    st = subs[key].state.copy()
    i = int(np.flatnonzero(st == 1)[5])
    st[i] = -1
    subs[key].state = st
    assert R.mismatches(subs, parts, mkts, w)["state_mismatches"] >= 1
    assert R.hard_errors({"state_mismatches": 1, "stop_exits": 0}) and not R.hard_errors(
        {"state_mismatches": 0, "stop_exits": 0})


def test_slot_assembly_sum_over_99_weekend_zeros(mkts):
    w = _w(mkts)
    subs = R.build(mkts)
    parts, _ = R.run_all(subs, mkts, w)
    res, daily = R.assemble(parts, w)
    tot = sum(d for d, _ in parts.values()) / 99
    assert np.allclose(daily.to_numpy(), tot.to_numpy())
    wk = daily.index.weekday
    assert (daily[wk >= 5] == 0).all() and (daily[wk < 5] != 0).any()
    _, fast = R.assemble(parts, w, "fast5")
    assert np.allclose(fast, sum(d for (s, L), (d, _) in parts.items() if L in D.CONFIGS["fast5"]) / 55)


def test_monday_return_carries_the_weekend():
    c = pd.Series([10.0, 10, 10, 11, 12], index=weekdays("2024-01-03", 5, holiday_p=0))   # Wed..Tue
    m = R.market_from_closes("SPYUSDT", c)
    w = window(c)
    sig = M.engine_signals(np.array([1, 1, 1, 1, 1], np.int8), np.ones(5))
    res = engine.run(m.mk, sig, R.costs("SPYUSDT", 0.0), R.RISK, start=w.start, latency_min=0)
    d = R.sub_daily(R.liquidate(res, m.mk)[0], w.calendar, R.RISK.initial_equity)
    mon = pd.Timestamp("2024-01-08", tz="UTC")
    assert d[pd.Timestamp("2024-01-06", tz="UTC")] == 0 and d[pd.Timestamp("2024-01-07", tz="UTC")] == 0
    qty = res.trades["qty"].iloc[0]
    assert qty == pytest.approx(1e4 / 10) and d[mon] == pytest.approx(qty * (11 - 10) / 1e4)


# ---------------------------------------------------------------- funding stress

def _one_sub(sym, c, st, size=1.0, L=5):
    m = R.market_from_closes(sym, c)
    n = len(c)
    sub = R.Sub(sym, L, M.engine_signals(st, np.full(n, size)), np.full(n, np.nan), np.asarray(st, np.int8),
                np.full(n, size), 2)
    return {(sym, L): sub}, {sym: m}


@pytest.mark.parametrize("side", [1, -1])
def test_funding_drag_costs_either_side_with_calendar_days(side):
    c = closes("2024-01-01", 60, seed=5)
    st = np.full(60, side, np.int8)
    st[:2] = 0
    subs, mk = _one_sub("XAUUSDT", c, st)
    w = window(c)
    parts, info = R.run_all(subs, mk, w)
    key = ("XAUUSDT", 5)
    drag = R.funding_drag(subs, mk, parts, info["eq"], w)[key]
    cal = R._cal_ms(w)
    lev, pos, on = R.exposure_path(mk["XAUUSDT"], parts[key][1], info["eq"][key], cal)
    slots = pos[on]
    gaps = np.r_[1, np.diff(mk["XAUUSDT"].bar_ts) // DAY]
    e = int(parts[key][1]["entry_i"].iloc[0])
    assert (drag[: slots[e] + 1] == 0).all()                        # nothing until a position is held overnight
    for i in range(e + 1, len(slots)):
        want = abs(lev[slots[i] - 1]) * 0.05 * gaps[i] / 365
        assert drag[slots[i]] == pytest.approx(want) and drag[slots[i]] > 0
    mondays = [i for i in range(e + 1, len(slots)) if gaps[i] == 3]
    assert mondays and all(drag[slots[i]] == pytest.approx(3 * abs(lev[slots[i] - 1]) * 0.05 / 365) for i in mondays)
    assert (drag[~np.isin(np.arange(len(cal)), slots)] == 0).all()   # weekends carry no separate charge


def test_funding_stress_never_stops_a_profitable_long():
    """Review 3: routing the stress funding through engine cash stopped a long, profitable 2x position once the
    funding paid exceeded its cash (equity still large), which faked a liquidation and a one-day jump."""
    d = pd.bdate_range("2000-01-03", "2012-12-31", tz="UTC")
    r = np.full(len(d), 0.29 / 252) + np.random.default_rng(1).normal(0, 0.004, len(d))
    mk = {"SPYUSDT": R.market_from_closes("SPYUSDT", pd.Series(50 * np.cumprod(1 + r), index=d.as_unit("ms").asi8))}
    w = _w(mk)
    ev = R.evaluate(mk, w, with_null=False)
    base, fund = ev["daily"], ev["funding_daily"]
    assert not ev["liquidations"]["base"] and fund.mean() < base.mean()
    assert (base - fund).min() >= 0 and (base - fund).max() < 0.002          # a daily drag, never a jump


# ---------------------------------------------------------------- shift null

def _cal(first, n):
    a = pd.Timestamp(first, tz="UTC")
    return pd.date_range(a, a + pd.Timedelta(days=n - 1), freq="D").as_unit("ms").asi8


def _nin(lev, a, r=None):
    lev = np.asarray(lev, float)
    return R.NullInput(lev, np.zeros(len(lev)) if r is None else np.asarray(r, float), np.zeros(len(lev)), a)


def test_shifted_constant_exposure_stays_constant():
    lev = np.r_[np.zeros(40), np.full(460, 1.5)]
    for k in (30, 31, 77, 150):
        assert np.array_equal(R.shifted_exposure(_nin(lev, 40), k), lev)


def test_shift_keeps_time_in_market_of_a_late_span_and_k7_is_a_week():
    rng = np.random.default_rng(4)
    lev = np.r_[np.zeros(900), np.repeat(rng.choice([-1.2, 0.0, 0.8], 60), 10)]
    x = _nin(lev, 900)
    sh = R.shifted_exposure(x, 200)
    assert np.array_equal(sh[:900], lev[:900]) and np.array_equal(np.sort(sh[900:]), np.sort(lev[900:]))
    sh7 = R.shifted_exposure(x, 7)
    assert np.array_equal(sh7[907:], lev[900:-7]) and np.array_equal(sh7[900:907], lev[-7:])


def test_null_inputs_reproduce_the_engine_exactly(mkts):
    w = _w(mkts)
    subs = R.build(mkts)
    parts, info = R.run_all(subs, mkts, w)
    inp = R.null_inputs(subs, mkts, parts, info["eq"], w)
    cal = R._cal_ms(w)
    for key, x in inp.items():
        d, tr = parts[key]
        mk = mkts[key[0]].mk
        fills = set((mk.ts[np.r_[tr["entry_i"].to_numpy(), tr["exit_i"].to_numpy()]] - E.ROW_OFFSET_MS).tolist())
        fill_slot = np.isin(cal, list(fills))
        assert np.allclose(np.r_[0.0, x.lev[:-1]] * x.r - x.cost, d.to_numpy(), atol=1e-15)
        assert np.abs(x.cost[~fill_slot]).max() < 1e-12            # exposure x return explains every other day
        assert (x.cost[fill_slot] > -1e-12).all() and x.cost[fill_slot].sum() > 0
        assert (x.lev[: x.a] == 0).all() and 0 < np.abs(x.lev).max() < R.LIQ_LEV
    actual = float(R.assemble(parts, w)[1].mean())
    nl = R.shift_null(inp, actual, n_draws=5)                     # k = 0 must reproduce the engine (asserted inside)
    assert nl["actual_mean_daily"] == actual and len(nl["dist"]) == 5


def _null_p(mk, n_draws=30, seed=66):
    w = _w(mk)
    subs = R.build(mk)
    parts, info = R.run_all(subs, mk, w)
    actual = float(R.assemble(parts, w)[1].mean())
    return R.shift_null(R.null_inputs(subs, mk, parts, info["eq"], w), actual, n_draws=n_draws, seed=seed), actual


def test_shift_null_detects_planted_edge_not_a_random_walk(mkts):
    nl, actual = _null_p(mkts)
    assert actual > 0 and nl["p_value"] < 0.05 and nl["s_min"] > 61
    rw = {s: R.market_from_closes(s, closes_between(str(pd.Timestamp(int(m.bar_ts[0]), unit="ms").date()), "2014-12-31",
                                                    seed=90 + i, amp=0.0))
          for i, (s, m) in enumerate(mkts.items())}
    nl2, _ = _null_p(rw)
    assert nl2["p_value"] > 0.05


def test_shift_null_captures_the_same_drift_exposure(mkts):
    """A drift mu adds mu x (sum of exposure held into trade dates) to a statistic. Averaged over the null's
    shifts, that exposure must equal the real run's, so a drift without timing skill moves the actual and the
    null alike (the review found the old engine re-simulation of shifted states captured only ~80% of it)."""
    w = _w(mkts)
    subs = R.build(mkts)
    parts, info = R.run_all(subs, mkts, w)
    inp = R.null_inputs(subs, mkts, parts, info["eq"], w)
    s_min = min(len(x.lev) - x.a for x in inp.values() if x.a < len(x.lev))
    ks = np.random.default_rng(0).integers(30, s_min - 30 + 1, 400)

    def held(k):
        return sum(float(R.shifted_exposure(x, int(k))[:-1][x.r[1:] != 0].sum()) for x in inp.values())

    actual = held(0)
    null = np.array([held(k) for k in ks])
    assert abs(null.mean() / actual - 1) < 0.02, (null.mean(), actual)


# ---------------------------------------------------------------- benchmark, gates, headline

def test_ew_benchmark_averages_available_markets_only():
    a = pd.Series([10.0, 11, 12.1], index=_ms(["2024-01-02", "2024-01-03", "2024-01-04"]))
    b = pd.Series([20.0, 22], index=_ms(["2024-01-02", "2024-01-04"]))      # holiday on 01-03
    mk = {"XAUUSDT": R.market_from_closes("XAUUSDT", a), "SPYUSDT": R.market_from_closes("SPYUSDT", b)}
    w = R.Window("t", "dev", pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-07", tz="UTC"))
    bm = R.ew_benchmark(mk, w)
    assert bm[pd.Timestamp("2024-01-03", tz="UTC")] == pytest.approx(0.1)
    assert bm[pd.Timestamp("2024-01-04", tz="UTC")] == pytest.approx((0.1 + 0.1) / 2)
    assert bm[pd.Timestamp("2024-01-06", tz="UTC")] == 0 and bm[pd.Timestamp("2024-01-02", tz="UTC")] == 0


def _ms(dates):
    return pd.DatetimeIndex(pd.to_datetime(dates)).tz_localize("UTC").as_unit("ms").asi8


def _ev(**kw):
    ev = {"weekly": {"t": 2.0}, "stress_2x_mean_daily": 1e-9, "funding_mean_daily": 1e-9, "year_share": 0.5,
          "markets_positive": 7, "null": {"p_value": 0.0499}, "pbo": 0.5, "plateau": 1.0,
          "alpha_vs_ew": {"psr_alpha": 0.95}}
    for k, v in kw.items():
        if isinstance(v, dict):
            ev[k] = ev[k] | v
        else:
            ev[k] = v
    return ev


def test_dev_gate_edges():
    assert all(R.gates_dev(_ev()).values()) and len(R.gates_dev(_ev())) == 9
    for kw in [{"weekly": {"t": 1.999}}, {"stress_2x_mean_daily": 0.0}, {"funding_mean_daily": -1e-9},
               {"year_share": 0.5001}, {"markets_positive": 6}, {"null": {"p_value": 0.05}}, {"pbo": 0.5001},
               {"plateau": 0.5}, {"alpha_vs_ew": {"psr_alpha": 0.9499}}]:
        assert sum(not v for v in R.gates_dev(_ev(**kw)).values()) == 1, kw


def test_holdout_gates_and_headline_n(monkeypatch, tmp_path):
    idx = pd.date_range("2016-01-01", periods=800, freq="D", tz="UTC").as_unit("ms")
    hd = pd.Series(np.random.default_rng(1).normal(4e-4, 4e-3, 800), index=idx)
    f1 = pd.Series(np.random.default_rng(2).normal(0, 1e-2, 400),
                   index=pd.date_range("2016-06-01", periods=400, freq="D", tz="UTC").as_unit("ms"))
    f1.rename("ret").to_frame().to_parquet(tmp_path / "F1_dev_daily.parquet")
    import btcq.research.round5 as R5
    monkeypatch.setattr(R5, "RESULTS", tmp_path)
    evh = {"daily": hd, "summary": {"sharpe": 1.0}, "markets_positive": 7}
    src = pd.Series(np.random.default_rng(3).normal(4e-4, 4e-3, 1600))
    g = R.gates_holdout(evh, 2.0, src)
    assert list(g) == ["holdout mean>0", "holdout Sharpe>=50% dev", "holdout MDD<=dev boot p95",
                       ">=7/11 markets positive (holdout)", "weekly corr with F1 (2020-23)<=0.3"]
    assert g["holdout Sharpe>=50% dev"] and not R.gates_holdout(evh, 2.01, src)["holdout Sharpe>=50% dev"]
    assert evh["f1"]["days"] == 400 and g["weekly corr with F1 (2020-23)<=0.3"]
    assert R.n_dsr(1.3) == pytest.approx(14.849570528171945 + 3) and R.n_dsr(20.0) == 20.0
    h = R.headline(True, True, hd, R.n_dsr(1.3))
    assert h["n"] == pytest.approx(17.8496, abs=1e-4) and h["REPLICATED"] is True
    assert R.headline(True, False, hd, 17.85)["REPLICATED"] is False


def test_boot_source_window():
    idx = pd.date_range("2011-01-01", "2015-12-31", freq="D", tz="UTC").as_unit("ms")
    s = R.boot_source(pd.Series(1.0, index=idx), "2011-06-22")
    assert s.index[0] == pd.Timestamp("2011-06-22", tz="UTC") and s.index[-1] == idx[-1]


# ---------------------------------------------------------------- data report

def _frame(dates, closes_):
    return pd.DataFrame({"ts": _ms(dates), "close": np.asarray(closes_, float), "adjclose": np.nan})


def _report(m, dates, c, info=None, ref=None, nulls=None):
    ref = _ms(pd.bdate_range("2014-01-01", "2026-10-30")) if ref is None else ref
    return R.red_flags(pd.DataFrame([R.market_report(m, _frame(dates, c), info or {}, ref, nulls)]))


def _clean_series(seed=0, start="2014-01-02", end="2026-10-15"):
    d = pd.bdate_range(start, end)
    c = 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.01, len(d))))
    return d, c


def test_red_flags_clean_and_each_flag(monkeypatch):
    monkeypatch.setitem(E.SOURCES, "SPYUSDT", E.Source("SPYUSDT", "yahoo", "SPY", "2014-01-02"))
    d, c = _clean_series()
    ok = _report("SPYUSDT", d, c)
    assert not ok["red_flag"].iloc[0], ok["red_flag_reasons"].iloc[0]
    cases = {
        "weekend rows": (d.append(pd.DatetimeIndex(["2026-10-17"])), np.r_[c, c[-1] * 1.01]),
        "missing run >3": (d.delete(range(300, 304)), np.delete(c, range(300, 304))),
        ">=5 equal closes": (d, np.r_[c[:500], np.full(5, c[500]), c[505:]]),
        "last date before 2026-09-29": (d[d < "2026-09-20"], c[: (d < "2026-09-20").sum()]),
        "first date >30d late": (d[d >= "2014-03-01"], c[(d < "2014-03-01").sum():]),
    }
    for flag, (dd, cc) in cases.items():
        rep = _report("SPYUSDT", dd, cc)
        assert flag in rep["red_flag_reasons"].iloc[0], (flag, rep["red_flag_reasons"].iloc[0])
    spike = c.copy()
    spike[400] *= 1.3
    rep = _report("SPYUSDT", d, spike)
    assert "reverting spike (Yahoo, dev)" in rep["red_flag_reasons"].iloc[0]
    split = c.copy()
    split[300:] /= 2.0
    rep = _report("SPYUSDT", d, split, info={"split_dates": d[300].strftime("%Y-%m-%d")})
    assert "unadjusted split (dev)" in rep["red_flag_reasons"].iloc[0]
    rep = _report("SPYUSDT", d, c, info={"nonpositive_unexpected": "2015-03-03"})
    assert "non-positive close not allowlisted" in rep["red_flag_reasons"].iloc[0]
    trading = list(d[::80].strftime("%Y-%m-%d"))                    # 42 null rows on reference days: 1.3%
    rep = _report("SPYUSDT", d, c, info={"nulls_dropped": len(trading)}, nulls=trading)
    assert "nulls >1%" in rep["red_flag_reasons"].iloc[0] and rep["nulls_on_ref_days"].iloc[0] == len(trading)
    holidays = [x.strftime("%Y-%m-%d") for x in pd.date_range("2014-01-04", periods=200, freq="W-SAT")]
    rep = _report("SPYUSDT", d, c, info={"nulls_dropped": 200}, nulls=holidays)   # Amendment 1: not reference days
    assert "nulls >1%" not in rep["red_flag_reasons"].iloc[0] and rep["nulls_on_ref_days"].iloc[0] == 0


def test_data_report_ignores_holdout_returns(monkeypatch):
    monkeypatch.setitem(E.SOURCES, "SPYUSDT", E.Source("SPYUSDT", "yahoo", "SPY", "2014-01-02"))
    d, c = _clean_series(seed=5)
    a = _report("SPYUSDT", d, c)
    c2 = c.copy()
    hold = d >= "2016-01-01"
    c2[hold] *= np.exp(np.random.default_rng(9).normal(0, 0.3, hold.sum()))     # wild, but signs/equalities kept
    b = _report("SPYUSDT", d, c2)
    pd.testing.assert_frame_equal(a, b)


def test_eia_spikes_are_reported_not_flagged(monkeypatch):
    monkeypatch.setitem(E.SOURCES, "NATGASUSDT", E.Source("NATGASUSDT", "eia", "RNGWHHD", "2014-01-02"))
    d, c = _clean_series(seed=6)
    c[400] *= 1.5
    rep = _report("NATGASUSDT", d, c)
    assert rep["dev_reverting_spikes"].iloc[0] >= 1 and not rep["red_flag"].iloc[0]


# ---------------------------------------------------------------- registry, guards, power

def test_round6_registry_is_separate(tmp_path, monkeypatch):
    g_trials = registry.PATH.read_bytes() if registry.PATH.exists() else None
    g_neff = registry.NEFF_LOG.read_bytes() if registry.NEFF_LOG.exists() else None
    monkeypatch.setattr(R, "TRIALS", tmp_path / "r6" / "trials.csv")
    monkeypatch.setattr(R, "NEFF_LOG", tmp_path / "r6" / "neff.jsonl")
    ev = {"cfg_daily": {c: pd.Series([0.001, 0.0, -0.001, 0.002]) for c in D.CONFIGS}, "cfg_trades": dict.fromkeys(D.CONFIGS, 3)}
    rows = R._trial_rows(ev, 1)
    assert len({json.dumps(r["params"], sort_keys=True) for r in rows}) == 3
    registry.record(rows, path=R.TRIALS)
    assert registry.multiple_testing_inputs("dev", path=R.TRIALS)[0] == 3
    assert registry.ratcheted_effective_trials(1.4, 3, "t", log=R.NEFF_LOG) == 1.4
    assert (registry.PATH.read_bytes() if registry.PATH.exists() else None) == g_trials
    assert (registry.NEFF_LOG.read_bytes() if registry.NEFF_LOG.exists() else None) == g_neff
    # the default path is resolved at call time
    monkeypatch.setattr(registry, "PATH", tmp_path / "global.csv")
    registry.record(rows)
    assert (tmp_path / "global.csv").exists()


def test_all_trial_returns_never_sees_round6(tmp_path, monkeypatch):
    assert R.R6_RESULTS == run.RESULTS / "round6"
    monkeypatch.setattr(run, "RESULTS", tmp_path)
    monkeypatch.setattr(R, "R6_RESULTS", tmp_path / "round6")
    pd.DataFrame({"S1_0": [0.1, 0.2]}).to_parquet(tmp_path / "S1_trial_returns.parquet")
    (tmp_path / "round6").mkdir()
    pd.DataFrame({"M1_all9": [0.3, 0.1]}).to_parquet(tmp_path / "round6" / "M1_trial_returns.parquet")
    assert list(run.all_trial_returns().columns) == ["S1_0"]
    assert list(R.r6_trial_returns().columns) == ["M1_all9"]


def test_round6_section_search():
    txt = "# P\n## Round 5\n### Power table\nx\n## Round 6 — registered\n### Rule\ny\n## Round 7\n### Power table\n"
    assert "Power table" not in R._r6_section(txt)
    assert "Power table" in R._r6_section(txt.replace("### Rule", "### Power table"))
    assert R._r6_section("## Round 5\n### Power table\n") == ""


def test_download_refuses_without_round6_prereg(tmp_path, monkeypatch):
    p = tmp_path / "PREREGISTRATION.md"
    p.write_text("## Round 5\n### Power table\n", encoding="utf-8")
    monkeypatch.setattr(R, "PREREG", p)
    from btcq.research import round4
    monkeypatch.setattr(round4, "_require_committed", lambda paths: None)
    monkeypatch.setattr(E, "fetch", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no download")))
    monkeypatch.setattr(E, "download_all", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no download")))
    with pytest.raises(SystemExit):
        R.cmd_download()
    with pytest.raises(SystemExit):
        R.cmd_sweep()


@pytest.fixture
def r6_tmp(tmp_path, monkeypatch):
    """Round 6 results, logs and holdout log in a temp dir; committed-file and data checks stubbed."""
    monkeypatch.setattr(R, "RESULTS", tmp_path)
    monkeypatch.setattr(R, "RUN_LOG", tmp_path / "runs.jsonl")
    monkeypatch.setattr(store, "HOLDOUT_LOG", tmp_path / "holdout_log.jsonl")
    monkeypatch.setattr(R, "_require_data_report", lambda: None)
    monkeypatch.setattr(R, "_require_prereg", lambda: None)
    monkeypatch.setattr(R, "_clean_src", lambda: None)
    from btcq.research import round4
    monkeypatch.setattr(round4, "_require_committed", lambda paths: None)
    monkeypatch.setattr(E, "load_dates", lambda m: _ms(["2026-09-30"]))
    runs = tmp_path / "runs.jsonl"                                   # every write to the run log counts as committed
    monkeypatch.setattr(R, "_committed_run_logs", lambda: [runs.read_text()] if runs.exists() else [])
    for name in ("DATA_REPORT", "POWER_CSV", "TRIALS", "NEFF_LOG"):  # never the real Round 6 files
        monkeypatch.setattr(R, name, tmp_path / "r6" / name.lower())
    monkeypatch.setattr(R, "R6_RESULTS", tmp_path / "round6")
    monkeypatch.setattr(E, "MANIFEST", tmp_path / "r6" / "data_manifest.csv")
    for f in ("M1_dev_daily.parquet", "M1_dev_trades.parquet", "F1_dev_daily.parquet"):
        (tmp_path / f).write_bytes(b"x")
    return tmp_path


def _m1(tmp, **kw):
    r = {"dev_pass": True, "full_book_day": "2011-06-22", "version": M.MacroTrend().version,
         "code_sha256": R.code_sha(), "f1_daily_sha256": R._sha_file(tmp / "F1_dev_daily.parquet")} | kw
    (tmp / "M1.json").write_text(json.dumps(r))


def _log(tmp, *recs):
    (tmp / "holdout_log.jsonl").write_text("".join(json.dumps(x) + "\n" for x in recs))


def test_preflight_and_resume_once(r6_tmp, monkeypatch):
    calls = []
    monkeypatch.setattr(E, "unlock_r6", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(store, "unlock_holdout", lambda reason: calls.append(("crypto", reason)))
    _m1(r6_tmp, dev_pass=False)
    probs, resume = R.preflight_holdout()
    assert any("did not pass dev" in p for p in probs) and not resume
    with pytest.raises(SystemExit):
        R.cmd_validate("x")
    _m1(r6_tmp)
    assert R.preflight_holdout() == ([], False)
    _m1(r6_tmp, version=99)
    assert any("not the swept version" in p for p in R.preflight_holdout()[0])
    _m1(r6_tmp, code_sha256="0" * 64)
    assert any("code changed" in p for p in R.preflight_holdout()[0])
    _m1(r6_tmp)
    unlock = {"round": 6, "kind": "unlock", "version": M.MacroTrend().version, "reason": "x [R6: M1 v1]"}
    _log(r6_tmp, unlock)
    assert R.preflight_holdout() == ([], True)
    _log(r6_tmp, unlock | {"version": 99})
    assert any("logged unlock was for version 99" in p for p in R.preflight_holdout()[0])
    _log(r6_tmp, unlock, {"round": 6, "kind": "resume", "version": 1, "reason": "x [R6 resume M1 v1]"})
    assert any("already resumed" in p for p in R.preflight_holdout()[0])
    _log(r6_tmp, {"reason": "legacy [R6: M1]"}, {"reason": "legacy [R6: M1]"})
    assert any("more than one" in p for p in R.preflight_holdout()[0])
    _log(r6_tmp, {"reason": "crypto final holdout"})
    assert R.preflight_holdout() == ([], False)                       # crypto unlocks never count
    monkeypatch.setattr(E, "load_dates", lambda m: _ms(["2026-09-25"]))
    assert any("ends before" in p for p in R.preflight_holdout()[0])
    assert calls == []


def test_sweep_guards(r6_tmp, monkeypatch):
    reached = []
    monkeypatch.setattr(R, "load_markets", lambda w, adjusted=False: reached.append(1) or (_ for _ in ()).throw(
        RuntimeError("past the guards")))
    v = M.MacroTrend().version
    with pytest.raises(SystemExit, match="commit research/round6/runs.jsonl"):
        R.cmd_sweep()                                                 # phase 1: only the start record
    with pytest.raises(RuntimeError):                                 # no results yet: the guards let it run
        R.cmd_sweep()
    _m1(r6_tmp)
    with pytest.raises(SystemExit, match="runs once"):
        R.cmd_sweep()
    with pytest.raises(SystemExit, match="version bump"):
        R.cmd_sweep(rerun=True)
    _m1(r6_tmp, version=v - 1)
    with pytest.raises(RuntimeError):                                 # a bumped rerun is allowed before the holdout
        R.cmd_sweep(rerun=True)
    _m1(r6_tmp, version=v - 1, holdout={"x": 1})
    with pytest.raises(SystemExit, match="closed"):
        R.cmd_sweep(rerun=True)
    _m1(r6_tmp, version=v - 1)
    _log(r6_tmp, {"round": 6, "kind": "unlock", "version": v - 1, "reason": "x [R6: M1 v0]"})
    with pytest.raises(SystemExit, match="closed"):
        R.cmd_sweep(rerun=True)
    (r6_tmp / "holdout_log.jsonl").unlink()
    R._log_run("r6-sweep", version=v, hard_error="2 stop exits")
    with pytest.raises(SystemExit, match="hard error"):
        R.cmd_sweep(rerun=True)
    assert len(reached) == 2


def test_sweep_logs_a_hard_error_and_refuses_the_same_version(r6_tmp, monkeypatch):
    monkeypatch.setattr(R, "load_markets", lambda w, adjusted=False: {})
    monkeypatch.setattr(R, "evaluate", lambda *a, **k: (_ for _ in ()).throw(R.HardError("2 stop exits")))
    with pytest.raises(SystemExit, match="commit"):
        R.cmd_sweep()
    with pytest.raises(SystemExit, match="hard error, logged"):
        R.cmd_sweep()
    logs = [json.loads(x) for x in (r6_tmp / "runs.jsonl").read_text().splitlines()]
    v = M.MacroTrend().version
    assert [x["kind"] for x in logs] == ["r6-sweep-start", "r6-sweep"] and logs[-1]["version"] == v
    assert logs[0]["code_sha256"] == R.code_sha() and logs[-1]["hard_error"] == "2 stop exits"
    with pytest.raises(SystemExit, match="stopped on a hard error"):
        R.cmd_sweep()


def test_an_interrupted_sweep_may_not_rerun_with_different_code(r6_tmp, monkeypatch):
    reached = []
    monkeypatch.setattr(R, "load_markets", lambda w, adjusted=False: reached.append(1) or (_ for _ in ()).throw(
        KeyboardInterrupt()))
    with pytest.raises(SystemExit, match="commit"):
        R.cmd_sweep()                                                 # phase 1
    with pytest.raises(KeyboardInterrupt):
        R.cmd_sweep()                                                 # interrupted: no M1.json
    with pytest.raises(KeyboardInterrupt):
        R.cmd_sweep()                                                 # the same code may retry
    committed = (r6_tmp / "runs.jsonl").read_text()
    monkeypatch.setattr(R, "_committed_run_logs", lambda: ["", committed])     # an older commit; HEAD lost it
    (r6_tmp / "runs.jsonl").unlink()                                  # a 'git checkout' of the working tree
    monkeypatch.setattr(R, "code_sha", lambda: "f" * 64)
    with pytest.raises(SystemExit, match="different code"):
        R.cmd_sweep()                                                 # the committed start record still binds
    assert len(reached) == 2


def test_a_started_version_may_not_rerun_on_different_data(r6_tmp, monkeypatch):
    man = r6_tmp / "r6" / "data_manifest.csv"
    man.parent.mkdir(parents=True, exist_ok=True)
    man.write_bytes(b"market,parquet_sha256\nSPYUSDT,a\n")
    monkeypatch.setattr(R, "load_markets", lambda w, adjusted=False: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(SystemExit, match="commit"):
        R.cmd_sweep()
    with pytest.raises(KeyboardInterrupt):
        R.cmd_sweep()
    man.write_bytes(b"market,parquet_sha256\nSPYUSDT,b\n")            # re-downloaded data
    with pytest.raises(SystemExit, match="different data"):
        R.cmd_sweep()


def test_a_committed_hard_error_survives_a_checkout(r6_tmp, monkeypatch):
    rec = json.dumps({"kind": "r6-sweep", "version": M.MacroTrend().version, "hard_error": "x"})
    monkeypatch.setattr(R, "_committed_run_logs", lambda: [rec + "\n"])
    assert R._hard_error_versions() == {M.MacroTrend().version}
    with pytest.raises(SystemExit, match="stopped on a hard error"):
        R.cmd_sweep()


def test_prereg_guard_refuses_placeholders(tmp_path, monkeypatch):
    p = tmp_path / "PREREGISTRATION.md"
    monkeypatch.setattr(R, "PREREG", p)
    from btcq.research import round4
    monkeypatch.setattr(round4, "_require_committed", lambda paths: None)
    p.write_text("## Round 6 — x\n### Power table\n{POWER_TABLE}\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        R._require_prereg()
    p.write_text("## Round 6 — x\n### Power table\n| constant | 0.5 |\nCode: `{CODE_HASH}`\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        R._require_prereg()
    p.write_text("## Round 6 — x\n### Power table\n| constant | 0.5 |\n## Round 7\n{X}\n", encoding="utf-8")
    R._require_prereg()


def test_preflight_pins_the_manifest_and_f1_inputs(r6_tmp, monkeypatch):
    man = r6_tmp / "data_manifest.csv"
    man.write_bytes(b"market,parquet_sha256\nSPYUSDT,x\n")
    monkeypatch.setattr(E, "MANIFEST", man)
    _m1(r6_tmp, manifest_sha256=R._sha_text(man))
    assert R.preflight_holdout() == ([], False)
    man.write_bytes(man.read_bytes().replace(b"\n", b"\r\n"))       # git line-ending churn is not a change
    assert R.preflight_holdout() == ([], False)
    man.write_bytes(b"market,parquet_sha256\nSPYUSDT,y\n")
    assert any("manifest changed" in p for p in R.preflight_holdout()[0])
    _m1(r6_tmp, manifest_sha256=R._sha_text(man))
    (r6_tmp / "F1_dev_daily.parquet").write_bytes(b"other")
    assert any("F1_dev_daily.parquet changed" in p for p in R.preflight_holdout()[0])


def test_a_resume_needs_an_earlier_unlock(r6_tmp):
    with pytest.raises(PermissionError):
        E.unlock_r6("peek", kind="resume", version=1)
    _m1(r6_tmp)
    _log(r6_tmp, {"round": 6, "kind": "resume", "version": 1, "reason": "x [R6 resume M1 v1]"})
    assert any("no earlier unlock" in p for p in R.preflight_holdout()[0])


def test_validate_records_an_integrity_error_as_a_fail(r6_tmp, monkeypatch):
    _m1(r6_tmp)
    unlocks = []
    monkeypatch.setattr(E, "unlock_r6", lambda reason, kind="unlock", version=None: unlocks.append((kind, version)))
    monkeypatch.setattr(R, "load_markets", lambda w, adjusted=False: {})
    monkeypatch.setattr(R, "evaluate", lambda *a, **k: (_ for _ in ()).throw(R.HardError("3 stop exits")))
    with pytest.raises(SystemExit, match="recorded as a FAIL"):
        R.cmd_validate("x")
    r = json.loads((r6_tmp / "M1.json").read_text())
    assert r["final_pass"] is False and r["headline"]["REPLICATED"] is False and "integrity_error" in r["holdout"]
    assert unlocks == [("unlock", M.MacroTrend().version)]
    assert any("already evaluated" in p for p in R.preflight_holdout()[0])
    with pytest.raises(SystemExit, match="closed"):
        R.cmd_sweep(rerun=True)
    logs = [json.loads(x) for x in (r6_tmp / "runs.jsonl").read_text().splitlines()]
    assert [x["kind"] for x in logs] == ["r6-validate-start", "r6-validate"] and logs[-1]["hard_error"]


def test_hard_error_versions_block_a_rerun(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "RUN_LOG", tmp_path / "runs.jsonl")
    monkeypatch.setattr(R, "_committed_run_logs", lambda: [])
    assert R._hard_error_versions() == set()
    R._log_run("r6-sweep", version=1, hard_error="2 stop exits")
    R._log_run("r6-sweep", version=2, verdict=False)
    assert R._hard_error_versions() == {1}


def test_power_table_td_bounds():
    w = R.ramp_weights()
    assert w[0] < w[-1] == 1.0 and np.all(np.diff(w) >= 0) and 10 < w.sum() / 252 < 18     # 12.4 after Amendment 1
    c = stats.power_table_td(sharpes=(0.0, 0.6, 1.0), n_sims=1500, seed=1)
    rp = stats.power_table_td(sharpes=(0.0, 0.6, 1.0), n_sims=1500, seed=1, ramp=w)
    for t in (c, rp):
        assert t[0]["p_dev_pass"] < 0.06 and t[-1]["p_dev_pass"] > 0.9
        for r in t:
            assert r["p_strict"] <= r["p_replicated"] <= r["p_dev_pass"]
            assert r["p_dev_hurdle_0.75"] <= r["p_dev_hurdle_0.45"] <= r["p_dev_hurdle_0.25"] <= r["p_dev_pass"]
            assert all(r[f"p_dev_hurdle_{h}"] <= r[f"p_funding_{h}"] for h in ("0.25", "0.45", "0.6", "0.75"))
    assert rp[1]["p_dev_pass"] < c[1]["p_dev_pass"] and rp[0]["variant"] == "ramp"
    assert c[0]["min_sharpe_dev"] == pytest.approx(2 / np.sqrt(30)) and rp[0]["min_sharpe_dev"] > 0.45
    assert rp[1]["p_dev_hurdle_0.6"] < rp[1]["p_dev_pass"]            # a hurdle above min_sharpe_dev binds
    assert rp[1]["p_dev_hurdle_0.45"] == rp[1]["p_dev_pass"]          # one below it is implied by the dev gate


def test_evaluate_end_to_end_small(mkts):
    w = _w(mkts)
    ev = R.evaluate(mkts, w, n_draws=3)
    assert ev["hard_checks"]["state_mismatches"] == 0 and ev["hard_checks"]["stop_exits"] == 0
    assert set(R.gates_dev(ev)) and ev["funding_mean_daily"] < ev["mean_daily"]
    assert ev["stress_2x_mean_daily"] < ev["mean_daily"] and 0 < ev["exposure"]["mean_gross"] < 2
    assert set(ev["market_contrib"]) == set(mkts) and len(ev["null"]["dist"]) == 3
    pub = R._public(ev)
    assert "daily" not in pub and "dist" not in pub["null"]
    json.dumps(run._clean(pub))


def test_liquidation_closes_a_ruined_sub_sleeve():
    c = pd.Series(np.r_[np.full(5, 10.0), np.linspace(10.5, 30, 20), np.full(5, 30.0)],
                  index=weekdays("2024-01-01", 30, holiday_p=0))
    m = R.market_from_closes("SPYUSDT", c)
    w = window(c)
    st = np.full(30, -1, np.int8)
    st[0] = 0
    sig = M.engine_signals(st, np.full(30, 2.0))
    res = engine.run(m.mk, sig, R.costs("SPYUSDT", 0.0), R.RISK, start=w.start, latency_min=0)
    assert res.equity.min() < 0                              # unlimited liability without the rule
    eq, j = R.liquidate(res, m.mk)
    assert j is not None and eq.iloc[j] >= 0 and (eq.iloc[j:] == eq.iloc[j]).all()
    qty = res.trades["qty"].iloc[0]
    lev = qty * c.to_numpy()[: j] / res.equity.to_numpy()[: j]
    assert (lev[1:] < R.LIQ_LEV).all() and qty * c.iloc[j] >= R.LIQ_LEV * res.equity.iloc[j]
    d = R.sub_daily(eq, w.calendar, R.RISK.initial_equity)
    assert (d >= -1).all() and (d[d.index > eq.index[j]] == 0).all()
    eq0, j0 = R.liquidate(engine.run(m.mk, M.engine_signals(np.zeros(30, np.int8), np.ones(30)), R.costs("SPYUSDT"),
                                     R.RISK, start=w.start, latency_min=0), m.mk)
    assert j0 is None


# ---------------------------------------------------------------- review follow-ups

def _gap_market(sym="XAUUSDT", gap=-0.55, seed=3, end="2009-12-31", after=None):
    """A calm uptrend (sizes at the 2x cap) with one overnight gap: long sub-sleeves lose > 100% of capital
    (gap -0.55) or are liquidated at 20x with a little equity left (about -0.49)."""
    d = pd.bdate_range("2005-01-03", end, tz="UTC")
    r = np.full(len(d), 0.0015) + np.random.default_rng(seed).normal(0, 0.002, len(d))
    r[900] = gap
    if after is not None:
        r[901] = after
    c = 50 * np.cumprod(1 + r)
    return {sym: R.market_from_closes(sym, pd.Series(c, index=d.as_unit("ms").asi8))}


def test_real_run_liquidations_are_booked_outcomes():
    mk = _gap_market()
    w = _w(mk)
    ev = R.evaluate(mk, w, with_null=False)                      # no hard error: a liquidation is an outcome
    assert ev["liquidations"]["base"] and ev["hard_checks"] == {"state_mismatches": 0, "stop_exits": 0}
    for key, (d, tr) in ev["parts"].items():
        assert d.min() >= -1.0
        if f"{key[0]}/{key[1]}" in ev["liquidations"]["base"]:
            gap_day = pd.Timestamp(int(mk[key[0]].bar_ts[900]), unit="ms", tz="UTC")
            assert (d[d.index > gap_day] == 0).all()                # flat after the liquidation
            last = tr.iloc[-1]
            assert last["reason"] == "liquidation" and last["exit_i"] == 900 and tr["entry_i"].max() <= 900


@pytest.mark.parametrize("gap,after", [(-0.495, 0.01), (-0.495, -0.01), (-0.49, -0.10), (-0.55, None), (-0.68, None)])
def test_a_liquidation_is_closed_at_its_mark(gap, after):
    """Review 3: a 20x liquidation with equity left kept its exposure one more day, which crashed the null's
    decomposition check or put a phantom 20x-60x exposure into every shifted draw."""
    mk = _gap_market(gap=gap, after=after)
    w = _w(mk)
    subs = R.build(mk)
    parts, info = R.run_all(subs, mk, w)
    assert info["liq_row"]
    inp = R.null_inputs(subs, mk, parts, info["eq"], w, info["liq_row"])      # must not raise
    cal = R._cal_ms(w)
    for key, row in info["liq_row"].items():
        m = mk[key[0]].mk
        slot = int(np.searchsorted(cal, m.ts[row] - E.ROW_OFFSET_MS))
        assert (inp[key].lev[slot:] == 0).all()
        tr = parts[key][1]
        last = tr.iloc[-1]
        booked = float(info["eq"][key].iloc[-1])
        assert last["reason"] == "liquidation" and last["exit_i"] == row
        assert last["net"] == pytest.approx(booked - last["eq_before"]) and booked >= 0
        assert last["gross"] == pytest.approx(last["net"] + last["fees"] + last["funding"])
    ev = R.evaluate(mk, w, with_null=False)
    for key, (d, _) in ev["parts"].items():
        if key in info["liq_row"]:
            assert (d - R.funding_drag(subs, mk, ev["parts"], info["eq"], w)[key]).clip(lower=-1).min() >= -1
    assert ev["funding_daily"].min() >= -1
    actual = float(R.assemble(parts, w)[1].mean())
    assert R.shift_null(inp, actual, n_draws=5)["actual_mean_daily"] == actual
    ev = R.evaluate(mk, w, n_draws=5)
    assert ev["null"]["draws"] == 5


def test_a_liquidation_only_in_the_2x_run_is_booked_and_reported(mkts, monkeypatch):
    orig = R.run_all

    def fake(subs, mk, w, mult=1.0, **kw):
        parts, info = orig(subs, mk, w, mult=mult, **kw)
        if mult == 2.0:
            info["ruins"] = ["XAUUSDT/5"]
        return parts, info

    monkeypatch.setattr(R, "run_all", fake)
    ev = R.evaluate(mkts, _w(mkts), with_null=False)
    assert ev["liquidations"]["stress_2x"] == ["XAUUSDT/5"] and not ev["liquidations"]["base"]


def test_null_inputs_refuse_a_wrong_decomposition(mkts):
    w = _w(mkts)
    subs = R.build(mkts)
    parts, info = R.run_all(subs, mkts, w)
    with pytest.raises(AssertionError, match="does not reproduce"):
        R.null_inputs(subs, mkts, parts, {k: v * 1.01 for k, v in info["eq"].items()}, w)


def test_holdout_starts_in_sync_with_the_state_and_funding_across_a_flip():
    c = closes_between("2014-01-01", "2017-12-29", seed=21, amp=0.0)      # no trend strong enough to liquidate
    m = R.market_from_closes("SPYUSDT", c)
    hold = R.Window("h", "full", pd.Timestamp("2016-01-01", tz="UTC"), pd.Timestamp("2018-01-01", tz="UTC"))
    n = len(c)
    st = np.ones(n, np.int8)
    st[:5] = 0
    j = int(np.searchsorted(m.bar_ts, int(pd.Timestamp("2016-06-01", tz="UTC").value // 10**6)))
    st[j:] = -1                                                      # a flip, then short to the end
    sig = M.engine_signals(st, np.ones(n))
    r1 = engine.run(m.mk, sig, R.costs("SPYUSDT"), R.RISK, start=hold.start, latency_min=0)
    t = r1.trades
    start_i = m.mk.index_at(hold.start)
    assert t["entry_i"].iloc[0] == start_i and t["entry_tf"].iloc[0] == start_i - 1 and t["dir"].iloc[0] == 1
    assert t["dir"].tolist() == [1, -1] and t["exit_i"].iloc[0] == t["entry_i"].iloc[1] and t["reason"].iloc[1] == "end"
    subs, mk = _one_sub("SPYUSDT", c, st)
    parts1, info1 = R.run_all(subs, mk, hold)
    assert not info1["liq_row"]
    assert parts1[("SPYUSDT", 5)][1][["entry_i", "exit_i", "dir"]].equals(t[["entry_i", "exit_i", "dir"]])
    drag = R.funding_drag(subs, mk, parts1, info1["eq"], hold)[("SPYUSDT", 5)]
    cal = R._cal_ms(hold)
    held_days = np.flatnonzero(drag > 0)
    first_after = int(np.searchsorted(cal, m.bar_ts[start_i + 1]))
    assert held_days[0] == first_after and held_days[-1] == int(np.searchsorted(cal, m.bar_ts[-1]))
    trade_slots = np.searchsorted(cal, m.bar_ts[start_i + 1:])
    assert len(held_days) == len(trade_slots)                        # every trade date after entry, once (flip too)
    sub = {("SPYUSDT", 20): R.build({"SPYUSDT": m})[("SPYUSDT", 20)]}
    parts, _ = R.run_all(sub, {"SPYUSDT": m}, hold)
    assert R.mismatches(sub, parts, {"SPYUSDT": m}, hold) == {"state_mismatches": 0, "stop_exits": 0}


def test_brent_uk_holidays_and_partial_years_do_not_flag(monkeypatch):
    monkeypatch.setitem(E.SOURCES, "BZUSDT", E.Source("BZUSDT", "eia", "RBRTE", "2004-11-18"))
    uk = R.uk_bank_holidays()
    assert {"2022-06-02", "2022-06-03", "2022-09-19", "2022-12-27", "2023-05-08", "2021-12-28"} <= uk
    assert "2022-12-25" not in uk                                    # weekends are never listed
    d = pd.bdate_range("2004-11-18", "2026-10-15")
    d = d[~d.strftime("%Y-%m-%d").isin(uk)]
    d = d.delete(5)                                                  # one missing day in the 32-day first year
    c = 100 * np.exp(np.cumsum(np.random.default_rng(2).normal(0, 0.01, len(d))))
    ref = _ms(pd.bdate_range("2004-01-01", "2026-10-30"))           # the reference covers the stub year
    rep = _report("BZUSDT", d, c, ref=ref)
    assert not rep["red_flag"].iloc[0], rep["red_flag_reasons"].iloc[0]
    monkeypatch.setattr(R, "MIN_YEAR_REF_DAYS", 0)                   # without the 120-day rule the stub flags
    assert "year missing >3%" in _report("BZUSDT", d, c, ref=ref)["red_flag_reasons"].iloc[0]


def test_year_missing_share_flags_scattered_holes(monkeypatch):
    monkeypatch.setitem(E.SOURCES, "QQQUSDT", E.Source("QQQUSDT", "yahoo", "QQQ", "2014-01-02"))
    d, c = _clean_series()
    yr = np.flatnonzero(d.year == 2018)[::25][:10]                  # 10 isolated days of 2018 (about 4%)
    rep = _report("QQQUSDT", d.delete(yr), np.delete(c, yr))
    reasons = rep["red_flag_reasons"].iloc[0]
    assert "year missing >3%" in reasons and "missing run >3" not in reasons


def test_spy_is_judged_against_its_peer_etfs(monkeypatch):
    monkeypatch.setitem(E.SOURCES, "SPYUSDT", E.Source("SPYUSDT", "yahoo", "SPY", "2014-01-02"))
    d, c = _clean_series()
    holes = np.flatnonzero(d.year == 2018)[::25][:10]
    frames = {"SPYUSDT": _frame(d.delete(holes), np.delete(c, holes)), "XAUUSDT": _frame(d, c), "QQQUSDT": _frame(d, c)}
    ref = R.reference_dates(frames)
    assert len(ref) == len(d)                                         # the peers fill SPY's holes in the reference
    rep = R.red_flags(pd.DataFrame([R.market_report("SPYUSDT", frames["SPYUSDT"], {}, ref)]))
    assert "year missing >3%" in rep["red_flag_reasons"].iloc[0]


def test_spy_holes_before_the_peers_exist_are_found_by_the_nyse_calendar(monkeypatch):
    monkeypatch.setitem(E.SOURCES, "SPYUSDT", E.Source("SPYUSDT", "yahoo", "SPY", "1993-01-29"))
    hol = R.nyse_holidays(1993, 2000) | R.KNOWN_CLOSURES
    d = pd.bdate_range("1993-01-29", "2000-03-31")
    d = d[~d.strftime("%Y-%m-%d").isin(hol)]
    holes = np.flatnonzero(d.year == 1996)[::25][:10]
    c = 100 * np.exp(np.cumsum(np.random.default_rng(3).normal(0, 0.01, len(d))))
    frames = {"SPYUSDT": _frame(d.delete(holes), np.delete(c, holes))}
    ref = R.reference_dates(frames)
    assert len(ref) == len(d)
    rep = R.red_flags(pd.DataFrame([R.market_report("SPYUSDT", frames["SPYUSDT"], {}, ref)]))
    assert "year missing >3%" in rep["red_flag_reasons"].iloc[0]
    clean = R.red_flags(pd.DataFrame([R.market_report("SPYUSDT", _frame(d, c), {}, R.reference_dates(
        {"SPYUSDT": _frame(d, c)}))]))
    assert "year missing" not in clean["red_flag_reasons"].iloc[0]


def test_nyse_holiday_rules():
    h = R.nyse_holidays(1997, 2024)
    assert {"2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27", "2024-06-19", "2024-07-04",
            "2024-09-02", "2024-11-28", "2024-12-25", "2022-12-26", "2021-12-24", "2020-07-03", "2015-07-03",
            "2017-01-02"} <= h
    assert not {"1997-01-20", "2021-12-31", "2021-06-18", "2024-01-02"} & h  # no MLK in 1997, no Saturday NY


def test_reference_itself_and_the_pre_reference_handover_are_gap_checked(monkeypatch):
    monkeypatch.setitem(E.SOURCES, "SPYUSDT", E.Source("SPYUSDT", "yahoo", "SPY", "2014-01-02"))
    d, c = _clean_series()
    hole = d.delete(range(300, 305))                                 # SPY misses 5 days: its own reference
    rep = _report("SPYUSDT", hole, np.delete(c, range(300, 305)), ref=_ms(hole))
    assert "missing run >3" in rep["red_flag_reasons"].iloc[0]
    monkeypatch.setitem(E.SOURCES, "CLUSDT", E.Source("CLUSDT", "eia", "RWTC", "2014-01-02"))
    ref = _ms(d[d >= "2015-01-01"])                                  # the reference starts in 2015
    gap = (d < "2014-12-20") | (d >= "2015-01-01")                   # a hole right before it
    rep = _report("CLUSDT", d[gap], c[gap], ref=ref)
    assert "missing run >3" in rep["red_flag_reasons"].iloc[0]


def test_unadjusted_split_without_an_event_and_a_split_date_without_a_row(monkeypatch):
    monkeypatch.setitem(E.SOURCES, "SPYUSDT", E.Source("SPYUSDT", "yahoo", "SPY", "2014-01-02"))
    d, c = _clean_series()
    split = c.copy()
    split[300:] /= 10.0
    rep = _report("SPYUSDT", d, split)                               # Yahoo listed no split event
    assert "|log return| > ln 1.8 (Yahoo, dev)" in rep["red_flag_reasons"].iloc[0]
    rep = _report("SPYUSDT", d.delete(300), np.delete(c, 300), info={"split_dates": d[300].strftime("%Y-%m-%d")})
    assert "split date without a row (dev)" in rep["red_flag_reasons"].iloc[0]
    monkeypatch.setitem(E.SOURCES, "NATGASUSDT", E.Source("NATGASUSDT", "eia", "RNGWHHD", "2014-01-02"))
    rep = _report("NATGASUSDT", d, split)                           # EIA spot: big moves are reported only
    assert "|log return| > ln 1.8" not in rep["red_flag_reasons"].iloc[0]
