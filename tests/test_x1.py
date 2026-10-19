"""Execution study X1: tick inference, gates, leg accounting, Q3 events and costs, the screen, Stage B arithmetic and
the pre-registration guards. Synthetic data only: no real price is loaded and nothing reaches the network."""
import json
import socket
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from btcq.backtest import engine, queue_sim, stats
from btcq.backtest import maker_engine as ME
from btcq.config import HOLDOUT_START, SLIPPAGE, SYMBOLS, Costs, Risk
from btcq.data import ticks
from btcq.research import guards, registry
from btcq.research import run as research_run
from btcq.research import x1_execution as x1
from btcq.research.portfolio import assemble, sleeve_risk
from btcq.research.run import aligned_daily
from btcq.strategies.r3_trend import S1Frozen
from conftest import flat, make_m1

C = Costs(taker=0.0005, maker=0.0002, slippage=0.0001)
DAY = 86_400_000
MIN = 60_000
HOLD = int(HOLDOUT_START.value // 10**6)
SEC = "## Execution study X1\nregistered text"
STAGE_A = x1.stage_a


def ms(s) -> int:
    t = pd.Timestamp(s)
    return int((t if t.tzinfo else t.tz_localize("UTC")).value // 10**6)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("network access in a test")
    monkeypatch.setattr(ticks, "_http_get", refuse)
    monkeypatch.setattr(ticks, "_list_sizes", refuse)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


# ---------------------------------------------------------------- stats.block_bootstrap_means

def test_block_bootstrap_means_is_deterministic_and_centred():
    x = np.random.default_rng(3).normal(0.001, 0.02, 1461)
    a = stats.block_bootstrap_means(x, 4000, block=30, seed=11)
    assert a.shape == (4000,)
    np.testing.assert_array_equal(a, stats.block_bootstrap_means(x, 4000, block=30, seed=11))
    assert not np.array_equal(a, stats.block_bootstrap_means(x, 4000, block=30, seed=12))
    assert abs(a.mean() - x.mean()) < 4 * a.std() / np.sqrt(len(a))
    # a block bootstrap keeps the autocorrelation: an AR(1) series' mean is more uncertain than the iid SE says
    e = np.random.default_rng(4).normal(0, 1, 2000)
    ar = np.zeros(2000)
    for i in range(1, 2000):
        ar[i] = 0.8 * ar[i - 1] + e[i]
    assert stats.block_bootstrap_means(ar, 2000, block=30).std() > 2 * ar.std() / np.sqrt(len(ar))
    np.testing.assert_allclose(stats.block_bootstrap_means(np.full(50, 0.25), 100, block=7), 0.25, rtol=1e-15)
    assert stats.block_bootstrap_means(x[:5], 10, block=30).shape == (10,)    # block longer than x wraps
    with pytest.raises(ValueError):
        stats.block_bootstrap_means([], 10)


def test_block_bootstrap_means_draws_blocks_of_consecutive_values():
    x = np.arange(10, dtype=float)
    # one block covering x exactly once (wrapping): every draw is a rotation, so every mean equals x's mean
    np.testing.assert_allclose(stats.block_bootstrap_means(x, 50, block=10, seed=1), 4.5)
    m = stats.block_bootstrap_means(x, 200, block=1, seed=1)
    assert len(np.unique(m)) > 20


# ---------------------------------------------------------------- tick sizes

@pytest.mark.parametrize("d, tick", [(0.1000000000036, 0.1), (0.00999999999, 0.01), (0.03, 0.02), (0.04, 0.05),
                                     (0.07, 0.05), (0.08, 0.1), (1e-4, 1e-4), (2.0, 2.0), (500.0, 500.0)])
def test_snap_tick(d, tick):
    assert x1.snap_tick(d) == tick
    assert np.isnan(x1.snap_tick(0.0)) and np.isnan(x1.snap_tick(np.nan))


def grid_m1(month_ticks: dict[str, float], start="2019-12-30", spot_before=None, seed=0, n_per_day=1440):
    """A 1m frame over the given months, prices on each month's grid."""
    rng = np.random.default_rng(seed)
    frames = []
    for mon, tick in month_ticks.items():
        a = pd.Timestamp(mon + "-01", tz="UTC")
        n = 5 * n_per_day                              # five days are enough for the grid
        ts = ms(a) + 60_000 * np.arange(n, dtype=np.int64)
        walk = 100 + np.cumsum(rng.normal(0, 0.05, n))
        c = np.round(walk / tick) * tick
        o = np.r_[c[0], c[:-1]]
        h = np.maximum(o, c) + np.round(rng.uniform(0, 0.2, n) / tick) * tick
        lo = np.minimum(o, c) - np.round(rng.uniform(0, 0.2, n) / tick) * tick
        frames.append(pd.DataFrame({"ts": ts, "open": o, "high": h, "low": lo, "close": c, "is_perp": True}))
    df = pd.concat(frames, ignore_index=True)
    if spot_before is not None:
        df.loc[df["ts"] < ms(spot_before), "is_perp"] = False
    return df


def test_infer_tick_on_a_grid_and_off_it():
    m1 = grid_m1({"2021-03": 0.01})
    inf = x1.infer_tick(m1[["open", "high", "low", "close"]].to_numpy().ravel())
    assert inf["candidate"] == 0.01 and inf["share"] == 1.0
    off = x1.infer_tick(np.array([1.0, 1.03, 1.06, 1.09, 1.12, 1.15]))      # gaps 0.03: snapped to 0.02, not whole
    assert off["candidate"] == 0.02 and off["share"] == 0.0
    assert np.isnan(x1.infer_tick([5.0, 5.0])["candidate"])


def test_tick_table_uses_the_grid_or_falls_back_to_exchange_info():
    # 2020-02 trades on a 0.03 grid: the smallest gap snaps to 0.02, of which the 0.03 gaps are no multiple
    m1 = grid_m1({"2019-12": 0.01, "2020-01": 0.01, "2020-02": 0.03, "2020-03": 0.5, "2024-01": 0.01})
    tt = x1.tick_table(m1, "XUSDT", 0.1)
    assert tt["month"].tolist() == ["2020-01", "2020-02", "2020-03"]     # before START and from 2024 are cut
    r = tt.set_index("month")
    assert r.loc["2020-01", "tick"] == 0.01 and r.loc["2020-01", "source"] == "1m"
    assert r.loc["2020-02", "source"] == "exchangeInfo" and r.loc["2020-02", "tick"] == 0.1
    assert not r.loc["2020-02", "consistent"] and r.loc["2020-02", "multiple_share"] < 0.99
    assert r.loc["2020-03", "tick"] == 0.5 and (r["exchange_info_tick"] == 0.1).all()
    g = m1[(m1["ts"] >= ms("2020-01-01")) & (m1["ts"] < ms("2020-02-01"))]
    assert r.loc["2020-01", "tick_bp"] == pytest.approx(0.01 / np.median(g["close"]) * 1e4)
    spot = x1.tick_table(grid_m1({"2020-01": 0.01, "2020-02": 0.01}, spot_before="2020-02-01"), "BTCUSDT", 0.1)
    assert spot["month"].tolist() == ["2020-02"]                          # spot rows are not the perp grid


def test_the_99pct_rule_tolerates_a_rare_odd_gap_but_not_stray_finer_prices():
    px = np.arange(1000) * 0.1 + 50
    inf = x1.infer_tick(np.r_[px, px[-1] + 100.15])                       # one gap of 1001.5 ticks in 1000
    assert inf["candidate"] == 0.1 and 0.99 <= inf["share"] < 1.0
    few = px[:100]
    inf = x1.infer_tick(np.r_[few, few[:3] + 0.13])                       # 3 stray prices make 6 odd gaps of 102
    assert inf["candidate"] == 0.02 and inf["share"] < 0.99


def test_exchange_info_ticks(tmp_path):
    p = tmp_path / "ei.json"
    p.write_text(json.dumps({"symbols": [
        {"symbol": "BTCUSDT", "filters": [{"filterType": "LOT_SIZE"}, {"filterType": "PRICE_FILTER", "tickSize": "0.10"}]},
        {"symbol": "XRPUSDT", "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.0001"}]}]}), encoding="utf-8")
    assert x1.exchange_info_ticks(p) == {"BTCUSDT": 0.1, "XRPUSDT": 0.0001}
    real = x1.exchange_info_ticks()
    assert set(SYMBOLS) <= set(real) and real["XRPUSDT"] == 0.0001


# ---------------------------------------------------------------- legs and the sanity identity

def leg_case():
    """Long entry maker-filled at minute 15, signal exit maker-filled at 30, a run-away entry at 60 falling back
    at 65 (T = 5), closed by the window end."""
    paths = flat(100, 15)
    paths += [(100, 100.5, 99.8, 100.2)] + flat(100.2, 14)
    paths += [(100.2, 100.4, 100.1, 100.3)] + flat(100.3, 14)
    paths += flat(100.3, 15) + flat(100.3, 5) + flat(101, 25)
    mk = engine.Market(make_m1(paths), None, synthetic_funding=0.0)
    n = len(paths) // 15
    e = np.zeros(n, np.int8)
    s = np.full(n, np.nan)
    e[[0, 3]] = 1
    s[[0, 3]] = 90.0
    xl = np.zeros(n, bool)
    xl[1] = True
    sig = engine.Signals("15m", e, s, np.full(n, np.nan), exit_long=xl)
    return mk, sig


def test_leg_table_from_a_maker_run():
    mk, sig = leg_case()
    r = ME.run(mk, sig, C, Risk(), policy=ME.MakerPolicy(5, "strict"), tick=np.full(len(mk.ts), 0.1))
    tr = r.trades.assign(symbol="BTCUSDT")
    assert tr["reason"].tolist() == ["exit_signal", "end"]
    legs = x1.leg_table(tr, mk.ts, C)
    assert legs["leg"].tolist() == ["entry", "exit", "entry"] and legs["post_i"].tolist() == [15, 30, 60]
    assert legs["fill_i"].tolist() == [15, 30, 65] and legs["maker"].tolist() == [1, 1, 0]
    assert legs["side"].tolist() == [1, -1, 1] and legs["wait_min"].tolist() == [0.0, 0.0, 5.0]
    np.testing.assert_allclose(legs["saving_bp"].iloc[:2], 3.0, atol=1e-9)
    drift = (101 - 100.3) / 100.3 * 1e4
    assert legs["drift_bp"].iloc[2] == pytest.approx(drift) and legs["saving_bp"].iloc[2] == pytest.approx(-drift)
    assert (legs["post_ms"] == mk.ts[legs["post_i"]]).all() and legs["fee"].tolist() == [C.maker, C.maker, C.taker]
    s = x1.leg_summary(legs)
    assert s["legs"] == 3 and s["maker_fills"] == 2 and s["fill_rate"] == pytest.approx(2 / 3)
    assert s["mean_saving_bp"] == pytest.approx((3 + 3 - drift) / 3) and s["mean_drift_unfilled_bp"] == pytest.approx(drift)
    assert s["entry_fill_rate"] == 0.5 and s["exit_fill_rate"] == 1.0
    assert x1.sanity_problems(s, strict=True) == []
    t0 = ME.run(mk, sig, C, Risk(), policy=ME.MakerPolicy(0), tick=np.full(len(mk.ts), 0.1))
    l0 = x1.leg_table(t0.trades.assign(symbol="BTCUSDT"), mk.ts, C)
    assert len(l0) == 3 and (l0["saving_bp"] == 0).all() and (l0["wait_min"] == 0).all() and (l0["maker"] == 0).all()
    assert x1.sanity_problems(x1.leg_summary(l0), strict=True) == []


def test_leg_table_skips_stop_exits_and_flags_missing_references():
    mk, sig = leg_case()
    r = ME.run(mk, sig, C, Risk(), policy=ME.MakerPolicy(5, "strict"), tick=np.full(len(mk.ts), 0.1))
    tr = r.trades.assign(symbol="BTCUSDT")
    tr.loc[0, "reason"] = "stop"
    assert x1.leg_table(tr, mk.ts, C)["leg"].tolist() == ["entry", "entry"]
    tr.loc[1, "entry_ref_px"] = np.nan
    with pytest.raises(x1.HardError, match="reference price"):
        x1.leg_table(tr, mk.ts, C)
    assert x1.leg_table(tr.iloc[:0], mk.ts, C).empty


def test_leg_summary_counts_the_orders_a_post_and_cancel_run_left_without_a_leg():
    mk, sig = leg_case()
    tick = np.full(len(mk.ts), 0.1)
    r = ME.run(mk, sig, C, Risk(), policy=ME.MakerPolicy(5, "strict", False), tick=tick)
    legs = x1.leg_table(r.trades.assign(symbol="BTCUSDT"), mk.ts, C)
    assert r.info["orders_posted"] == 3 and r.info["entries_missed"] == 1 and len(legs) == 2
    s = x1.leg_summary(legs, r.info)
    # the executed legs all filled at 3 bp, but one of the three posted orders was cancelled: fill rate 2/3
    assert s["leg_fill_rate"] == 1.0 and s["mean_saving_bp"] == pytest.approx(3.0)
    assert s["fill_rate"] == pytest.approx(2 / 3) and s["entry_fill_rate"] == 0.5 and s["exit_fill_rate"] == 1.0
    assert s["orders_posted"] == 3 and s["entry_orders"] == 2 and s["exit_orders"] == 1
    assert s["orders_without_leg"] == 1 and s["entries_missed"] == 1 and s["exits_cancelled"] == 0
    assert x1.sanity_problems(s, strict=True) == []
    assert x1.leg_summary(legs)["fill_rate"] == 1.0                     # without info: the executed legs only
    t0 = ME.run(mk, sig, C, Risk(), policy=ME.MakerPolicy(0), tick=tick)
    s0 = x1.leg_summary(x1.leg_table(t0.trades.assign(symbol="BTCUSDT"), mk.ts, C), t0.info)
    assert s0["orders_posted"] == 0 and np.isnan(s0["fill_rate"]) and s0["leg_fill_rate"] == 0.0


def test_run_problems_reconcile_fees_and_order_counts_with_the_engine():
    mk, sig = leg_case()
    tick = np.full(len(mk.ts), 0.1)
    for pol in (ME.MakerPolicy(5, "strict"), ME.MakerPolicy(5, "optimistic", False), ME.MakerPolicy(60, "central"),
                ME.MakerPolicy(0)):
        r = ME.run(mk, sig, C, Risk(), policy=pol, tick=tick)
        tr = r.trades.assign(symbol="BTCUSDT")
        assert x1.run_problems(tr, x1.leg_table(tr, mk.ts, C), r.info, C, posts=pol.T > 0) == [], pol
    r = ME.run(mk, sig, C, Risk(), policy=ME.MakerPolicy(5, "strict"), tick=tick)
    tr = r.trades.assign(symbol="BTCUSDT")
    legs = x1.leg_table(tr, mk.ts, C)
    assert any("fees" in p for p in x1.run_problems(tr.assign(fees=tr["fees"] * 2), legs, r.info, C, True))
    # maker entries charged the taker fee: the 3 bp check reads only the flags, the fee reconciliation sees it
    mf = tr["maker"] == 1
    taxed = tr.copy()
    taxed.loc[mf, "fees"] += (C.taker - C.maker) * taxed.loc[mf, "qty"] * taxed.loc[mf, "entry_px"]
    assert x1.sanity_problems(x1.leg_summary(x1.leg_table(taxed, mk.ts, C)), strict=True) == []
    assert any("fees" in p for p in x1.run_problems(taxed, legs, r.info, C, True))
    assert any("order outcomes" in p for p in x1.run_problems(tr, legs, r.info | {"orders_posted": 4}, C, True))
    moved = r.info | {"entry_maker_fills": 2, "entry_fallbacks": 0}       # same total, different split
    assert any("order counts" in p for p in x1.run_problems(tr, legs, moved, C, True))
    assert any("taker run" in p for p in x1.run_problems(tr, legs, r.info, C, posts=False))
    stop = tr.copy()
    stop.loc[0, "reason"] = "stop"
    assert any("flagged maker" in p for p in x1.run_problems(stop, x1.leg_table(stop, mk.ts, C), r.info, C, True))


def summ(saving, maker):
    legs = pd.DataFrame({"saving_bp": saving, "maker": maker, "leg": "entry", "drift_bp": 0.0, "wait_min": 0.0})
    return x1.leg_summary(legs)


def test_sanity_identity_and_the_strict_fill_saving():
    assert x1.sanity_problems(summ([3.0, 3.0, -10.0], [1, 1, 0]), strict=True) == []
    assert x1.sanity_problems(summ([-1.0, -2.0], [0, 0]), strict=True) == []          # nothing filled
    assert x1.sanity_problems(summ([3.0, 3.0], [1, 1]), strict=True) == []            # everything filled
    probs = x1.sanity_problems(summ([3.5, 3.0, -10.0], [1, 1, 0]), strict=True)        # a strict fill moved the price
    assert len(probs) == 1 and "strict maker fill" in probs[0]
    assert x1.sanity_problems(summ([3.5, 3.0, -10.0], [1, 1, 0]), strict=False) == []
    s = summ([3.0, -10.0], [1, 0])
    s["mean_saving_bp"] += 1e-6
    assert any("identity" in p for p in x1.sanity_problems(s, strict=True))
    assert x1.sanity_problems(summ([], []), strict=True) == []


# ---------------------------------------------------------------- gates

def days():
    return pd.date_range("2020-01-01", "2023-12-31", freq="D", tz="UTC").as_unit("ms")


def test_gate_values_arithmetic():
    rng = np.random.default_rng(5)
    idx = days()
    taker_s = {s: pd.Series(rng.normal(0.001, 0.02, len(idx)), index=idx) for s in SYMBOLS}
    bump = {s: 0.0001 if s != "XRPUSDT" else -0.0002 for s in SYMBOLS}
    maker_s = {s: taker_s[s] + bump[s] for s in SYMBOLS}
    taker = pd.concat(taker_s, axis=1).mean(axis=1)
    maker = pd.concat(maker_s, axis=1).mean(axis=1)
    d = maker - taker
    v = x1.gate_values(taker, maker, {5: taker + 1e-5, 60: taker - 1e-5}, taker_s, maker_s)
    assert v["mean_daily_diff"] == pytest.approx(0.00004) and v["gain_ann"] == pytest.approx(0.00004 * 365)
    assert v["boot_p5"] == pytest.approx(0.00004, rel=1e-6)          # a constant difference has no spread
    assert v["neighbour_mean_daily_diff"] == pytest.approx({5: 1e-5, 60: -1e-5})
    assert v["years_positive"] == 4 and v["coins_positive"] == 4
    assert v["coin_sum_diff"]["XRPUSDT"] == pytest.approx(-0.0002 * len(idx))
    assert v["year_sum_diff"][2020] == pytest.approx(d[d.index.year == 2020].sum())
    assert v["sharpe_taker"] == pytest.approx(taker.mean() / taker.std() * np.sqrt(365))
    eq = np.r_[1.0, np.cumprod(1 + maker.to_numpy())]
    assert v["max_dd_maker"] == pytest.approx(np.max(1 - eq / np.maximum.accumulate(eq)))
    g = x1.gates_from_values(v)
    assert g["(a) gain >= +0.5%/yr"] and g["(b) bootstrap 5% bound > 0"]                    # 1.46%/yr
    assert not g["(c) plateau: T=5 and T=60 gain > 0"] and g["(d) >=3 of 4 years and >=4 of 5 coins positive"]
    with pytest.raises(x1.HardError):
        x1.gate_values(taker, maker.iloc[1:], {}, taker_s, maker_s)


def boundary_values(**kw):
    v = {"gain_ann": x1.MIN_GAIN, "boot_p5": 1e-12, "neighbour_mean_daily_diff": {5: 1e-12, 60: 1e-12},
         "years_positive": x1.MIN_YEARS, "coins_positive": x1.MIN_COINS, "sharpe_taker": 1.2, "max_dd_taker": 0.25}
    v["sharpe_maker"] = v["sharpe_taker"] - x1.SHARPE_TOL
    v["max_dd_maker"] = v["max_dd_taker"] + x1.DD_TOL
    return v | kw


def test_gates_pass_exactly_at_the_thresholds_and_each_fails_alone():
    assert all(x1.gates_from_values(boundary_values()).values())
    b = boundary_values()
    bad = [boundary_values(gain_ann=np.nextafter(x1.MIN_GAIN, 0)), boundary_values(boot_p5=0.0),
           boundary_values(neighbour_mean_daily_diff={5: 1e-12, 60: 0.0}),
           boundary_values(neighbour_mean_daily_diff={60: 1e-12}),
           boundary_values(years_positive=x1.MIN_YEARS - 1), boundary_values(coins_positive=x1.MIN_COINS - 1),
           boundary_values(sharpe_maker=np.nextafter(b["sharpe_maker"], 0)),
           boundary_values(max_dd_maker=np.nextafter(b["max_dd_maker"], 1))]
    for v in bad:
        g = x1.gates_from_values(v)
        assert sum(not x for x in g.values()) == 1, g


def fake_by_policy(diffs: dict[str, float], n_days=None):
    """by_policy entries for the gates: every sleeve of run k is the taker's plus diffs[k] per day."""
    rng = np.random.default_rng(9)
    idx = days()[:n_days] if n_days else days()
    tk = {s: pd.Series(rng.normal(0.0012, 0.02, len(idx)), index=idx) for s in SYMBOLS}
    out = {}
    for k in [x1.TAKER] + [x1.run_key(m, T) for m in x1.MODELS for T in x1.T_GRID]:
        sl = {s: tk[s] + diffs.get(k, 0.0) for s in SYMBOLS}
        out[k] = {"daily": pd.concat(sl, axis=1).mean(axis=1), "sleeves": sl}
    return out


def test_model_gates_strict_pass_and_plateau():
    up = {x1.run_key("strict", T): 0.0001 for T in x1.T_GRID}
    v, g = x1.model_gates(fake_by_policy(up), "strict")
    assert all(g.values()), (g, v)
    assert v["gain_ann"] == pytest.approx(0.0365)
    flat_nb = up | {x1.run_key("strict", 60): 0.0}
    _, g = x1.model_gates(fake_by_policy(flat_nb), "strict")
    assert [k for k, ok in g.items() if not ok] == ["(c) plateau: T=5 and T=60 gain > 0"]
    _, g = x1.model_gates(fake_by_policy(up), "central")                 # central did not improve
    assert not g["(a) gain >= +0.5%/yr"]


# ---------------------------------------------------------------- post-and-cancel information

def test_missed_entries_are_matched_by_symbol_and_bar():
    taker = pd.DataFrame({"symbol": ["BTCUSDT", "BTCUSDT", "ETHUSDT", "ETHUSDT"], "entry_tf": [10, 20, 10, 30],
                          "R": [2.0, -1.0, -0.5, 1.0]})
    run = pd.DataFrame({"R": [1.0, 0.5]})
    m = x1.missed_info(taker, {"BTCUSDT": np.array([20, 99]), "ETHUSDT": np.array([10])}, run)
    assert m["missed_entries"] == 3 and m["matched_to_taker_trades"] == 2 and m["unmatched"] == 1
    assert m["mean_R_missed"] == pytest.approx(-0.75) and m["mean_R_taken"] == pytest.approx(1.5)
    assert m["run_trades"] == 2 and m["run_mean_R"] == 0.75 and m["taker_trades"] == 4
    none = x1.missed_info(taker, {"BTCUSDT": np.array([], np.int64)}, run)
    assert none["missed_entries"] == 0 and np.isnan(none["mean_R_missed"])


# ---------------------------------------------------------------- Q3 events

def hour_bars(close, wick=0.5):
    c = np.asarray(close, float)
    return pd.DataFrame({"close": c, "high": c + wick, "low": c - wick})


def test_momentum_events_need_a_fresh_cross_of_the_prior_24_bar_level():
    c = np.r_[np.full(30, 100.0), np.full(30, 101.0), np.full(10, 99.0)]
    c[10] = 100.4                                       # inside the first 24 bars: no level yet
    idx, side = x1.momentum_events(hour_bars(c))
    assert idx.tolist() == [30, 60] and side.tolist() == [1, -1]
    # a close at the level itself is no breakout; the next one above it is
    idx, side = x1.momentum_events(hour_bars(np.r_[np.full(30, 100.0), [100.0, 100.1]], wick=0.0))
    assert idx.tolist() == [31] and side.tolist() == [1]


def test_reversion_events_use_the_prior_720_bar_sigma():
    r = np.tile([0.001, -0.001], 380)                   # 760 returns, sigma ~ 0.001
    r = np.r_[r, 0.0025, -0.0019, -0.003, 0.0]
    c = 100 * np.exp(np.cumsum(np.r_[0.0, r]))
    idx, side = x1.reversion_events(hour_bars(c))
    assert idx.tolist() == [761, 763] and side.tolist() == [-1, 1]      # bar i carries return r[i - 1]
    short = 100 * np.exp(np.cumsum(np.r_[0.0, np.tile([0.001, -0.001], 300), 0.05]))
    assert len(x1.reversion_events(hour_bars(short))[0]) == 0           # fewer than 720 prior returns


def synth_market(sym="BTCUSDT", start="2023-03-01", n_days=60, seed=1, tick=0.01, funding=True):
    rng = np.random.default_rng(seed)
    n = n_days * 1440
    ts = ms(start) + 60_000 * np.arange(n, dtype=np.int64)
    c = np.round(100 * np.exp(np.cumsum(rng.normal(0, 0.0012, n))) / tick) * tick
    o = np.r_[c[0], c[:-1]]
    w = np.round(np.abs(rng.normal(0, 0.03, (2, n))) / tick) * tick
    m1 = pd.DataFrame({"ts": ts, "open": o, "high": np.maximum(o, c) + w[0], "low": np.minimum(o, c) - w[1],
                       "close": c, "volume": 1.0, "quote_volume": c, "taker_buy_volume": 0.5, "is_perp": True})
    f = pd.DataFrame({"ts": ts[::480], "rate": 0.0001}) if funding else None
    mk = engine.Market(m1, f, synthetic_funding=0.0)
    mk.symbol = sym
    return mk


def test_q3_legs_draw_random_rows_then_sides_and_execute_events_after_the_latency(monkeypatch):
    monkeypatch.setattr(x1, "START", pd.Timestamp("2023-03-05", tz="UTC"))
    monkeypatch.setattr(x1, "Q3_RANDOM_PER_COIN", 50)
    mk = synth_market(n_days=25)
    legs = x1.q3_legs(mk, "BTCUSDT", np.random.default_rng(1102))
    rnd = legs[legs["class"] == "random"]
    elig = np.flatnonzero(mk.ts >= ms("2023-03-05"))
    g = np.random.default_rng(1102)
    rows = elig[g.integers(0, len(elig), 50)]
    np.testing.assert_array_equal(rnd["m0"], rows)
    np.testing.assert_array_equal(rnd["side"], g.integers(0, 2, 50) * 2 - 1)
    mom = legs[legs["class"] == "momentum"]
    idx, side = x1.momentum_events(mk.bars("1h"))
    ex = mk.exec_idx("1h", 1)[idx]
    keep = (ex < len(mk.ts)) & (mk.ts[np.minimum(ex, len(mk.ts) - 1)] >= ms("2023-03-05"))
    np.testing.assert_array_equal(mom["m0"], ex[keep])
    np.testing.assert_array_equal(mom["side"], side[keep])
    assert len(mom) > 5 and (mk.ts[legs["m0"]] >= ms("2023-03-05")).all()
    assert (mk.ts[mom["m0"]] % 3_600_000 == 60_000).all()                 # bar close + 1 minute
    assert set(legs["class"]) == {"random", "momentum"}                   # 25 days: no 720-bar sigma yet


def costs_frame():
    """Six complete random BTC legs on three UTC days with hand-set savings for strict T=15."""
    n = 6
    d0 = ms("2023-06-01")
    df = pd.DataFrame({"class": "random", "symbol": "BTCUSDT", "complete": [True] * 6 + [False],
                       "m0_ms": [d0, d0 + 5, d0 + DAY, d0 + DAY + 9, d0 + 2 * DAY, d0 + 2 * DAY + 1, d0],
                       "slip": SLIPPAGE["BTCUSDT"]})
    for T in x1.T_GRID:
        for m in x1.MODELS:
            p = f"{m}_T{T}"
            df[f"saving_{p}"] = 1.0
            df[f"filled_{p}"] = False
            df[f"wait_{p}"] = float(T)
            for k in x1.MARKOUTS:
                df[f"markout{k}_{p}"] = np.nan
    df["saving_strict_T15"] = [3.0, 3.0, -1.0, 5.0, 2.0, 0.0, 99.0]
    df["filled_strict_T15"] = [True, True, False, False, True, False, True]
    df["wait_strict_T15"] = [1.0, 2.0, 15.0, 15.0, 3.0, 15.0, 0.0]
    df["markout5_strict_T15"] = [1.0, -1.0, np.nan, np.nan, 3.0, np.nan, 50.0]
    assert len(df) == n + 1
    return df


def test_q3_costs_day_clustered_ci_and_c_eff():
    q = x1.q3_costs(costs_frame())
    assert len(q) == len(x1.CLASSES) * (len(SYMBOLS) + 1) * len(x1.T_GRID) * len(x1.MODELS)
    r = q[(q["class"] == "random") & (q["symbol"] == "BTCUSDT") & (q["T"] == 15) & (q["model"] == "strict")].iloc[0]
    assert r["n"] == 6 and r["days"] == 3 and r["mean_saving_bp"] == pytest.approx(2.0)   # the incomplete leg is out
    se = np.std([3.0, 2.0, 1.0], ddof=1) / np.sqrt(3)                                    # per-day means 3, 2, 1
    assert r["se_bp"] == pytest.approx(se)
    assert r["ci_lo"] == pytest.approx(2.0 - 1.96 * se) and r["ci_hi"] == pytest.approx(2.0 + 1.96 * se)
    assert r["fill_rate"] == pytest.approx(0.5) and r["mean_wait_filled_min"] == pytest.approx(2.0)
    assert r["mean_wait_min"] == pytest.approx(51 / 6) and r["markout_bp_5"] == pytest.approx(1.0)
    assert r["taker_cost_bp"] == pytest.approx(6.0) and r["c_eff_bp"] == pytest.approx(4.0)
    a = q[(q["class"] == "random") & (q["symbol"] == "ALL") & (q["T"] == 15) & (q["model"] == "strict")].iloc[0]
    assert a["n"] == 6 and a["c_eff_bp"] == pytest.approx(4.0)
    e = q[(q["class"] == "momentum") & (q["symbol"] == "ALL")]
    assert (e["n"] == 0).all() and e["mean_saving_bp"].isna().all()


def test_day_clustered_se_is_the_se_of_the_leg_mean_with_unequal_days():
    x = np.r_[0.0, np.full(99, 10.0)]                       # 1 leg on day 0, 99 legs on day 1
    g, se = x1._cluster_se(x, np.r_[0, np.ones(99, np.int64)])
    assert g == 2 and se == pytest.approx(0.198)            # not 5.0, the SE of the mean of the two day means
    assert np.isnan(x1._cluster_se(np.array([1.0, 2.0]), np.array([4, 4]))[1])
    df = costs_frame()
    d0 = ms("2023-06-01")
    df["m0_ms"] = [d0, d0 + 1, d0 + 2, d0 + DAY, d0 + 2 * DAY, d0 + 2 * DAY + 1, d0]   # 3, 1 and 2 legs a day
    r = x1.q3_costs(df)
    r = r[(r["class"] == "random") & (r["symbol"] == "BTCUSDT") & (r["T"] == 15) & (r["model"] == "strict")].iloc[0]
    # savings 3, 3, -1 | 5 | 2, 0 around the leg mean 2: day sums of deviations -1, 3, -2
    se = np.sqrt(3 / 2 * (1 + 9 + 4)) / 6
    assert r["mean_saving_bp"] == pytest.approx(2.0) and r["se_bp"] == pytest.approx(se)
    assert r["ci_lo"] == pytest.approx(2.0 - 1.96 * se) and r["ci_hi"] == pytest.approx(2.0 + 1.96 * se)


def test_q3_simulate_matches_simulate_legs_and_keeps_one_leg_set():
    mk = synth_market(n_days=3)
    legs = pd.DataFrame({"class": "random", "symbol": "BTCUSDT", "m0": [100, 2000, len(mk.ts) - 30],
                         "side": [1, -1, 1]})
    tick = np.full(len(mk.ts), 0.01)
    out = x1.q3_simulate(mk, legs, tick, 0.0001)
    r = ME.simulate_legs(mk.o, mk.h, mk.l, mk.c, mk.ts, legs["m0"], legs["side"], tick, 0.0001, 15, "central", True)
    np.testing.assert_array_equal(out["saving_central_T15"], r["saving_bp"])
    np.testing.assert_array_equal(out["filled_central_T15"], r["filled"])
    assert out["complete"].tolist() == [True, True, False] and out["complete_T5"].tolist() == [True, True, False]
    f = x1.q3_legs_file(out)
    assert f["leg_id"].tolist() == [0, 1, 2] and "saving_strict_T60" in f and "markout1_strict_T5" not in f


# ---------------------------------------------------------------- the screen

def screen_trades():
    return pd.DataFrame({"symbol": ["ETHUSDT", "BTCUSDT"], "dir": [1, -1], "qty": [1.0, 1.0],
                         "entry_px": [100.0, 100.0], "exit_px": [100.0, 100.0], "gross": [0.09, 0.06],
                         "maker": [0, 1], "reason": ["exit_signal", "stop"],
                         "entry_time": pd.to_datetime(["2021-03-01", "2019-05-01"], utc=True),
                         "exit_time": pd.to_datetime(["2021-03-02", "2019-05-02"], utc=True)})


def test_screen_arithmetic():
    ceff = {(c, s, m): v for m in x1.MODELS for c, s, v in (("momentum", "ETHUSDT", 4.0), ("random", "ETHUSDT", 3.0),
                                                            ("momentum", "BTCUSDT", 5.0), ("random", "BTCUSDT", 9.0))}
    rows = {r["period"]: r for r in x1.screen_family("F2", screen_trades(), ceff)}
    a, b = rows["all"], rows["2020+"]
    # ETH: 9 bp gross + 1 bp slippage on each taker leg = 11; BTC: maker entry, so 6 + 1 = 7
    assert a["trades"] == 2 and a["gross_bp"] == pytest.approx(9.0) and a["taker_rt_bp"] == pytest.approx(12.0)
    # maker round trip: ETH 4 + 3 (signal exit at the random c_eff), BTC 5 + 6 (stop exit stays taker)
    assert a["maker_rt_bp_strict"] == pytest.approx(9.0) and a["share_nonstop_exits"] == 0.5
    assert not a["execution_sensitive_strict"]                          # gross is not above the maker round trip
    assert b["trades"] == 1 and b["gross_bp"] == pytest.approx(11.0) and b["maker_rt_bp_central"] == pytest.approx(7.0)
    assert b["execution_sensitive_optimistic"]
    s = screen_trades().drop(columns="symbol")                           # S-families: BTC, 1 bp
    r = x1.screen_family("S4", s, ceff)[0]
    assert r["gross_bp"] == pytest.approx(9.0) and r["maker_rt_bp_strict"] == pytest.approx((5 + 9 + 5 + 6) / 2)
    late = screen_trades()
    late.loc[0, "exit_time"] = HOLDOUT_START
    with pytest.raises(x1.HardError, match="holdout"):
        x1.screen_family("F2", late, ceff)


# ---------------------------------------------------------------- guards: start record, run once, pins

def guard_env(monkeypatch, tmp_path):
    """A sandbox for the commands: (run log, stage-A calls, the committed records, the start pins)."""
    monkeypatch.setattr(guards, "require_section", lambda *a, **k: SEC)
    monkeypatch.setattr(guards, "clean_src", lambda: None)
    monkeypatch.setattr(guards, "require_committed", lambda paths: None)
    res = tmp_path / "results"
    res.mkdir(exist_ok=True)
    monkeypatch.setattr(x1, "RESULTS", res)
    monkeypatch.setattr(x1, "X1_DIR", tmp_path / "x1")
    log = tmp_path / "x1" / "runs.jsonl"
    monkeypatch.setattr(x1, "RUN_LOG", log)
    f1 = tmp_path / "f1_daily.parquet"
    pd.DataFrame({"ret": [0.0, 0.01]}, index=days()[:2]).to_parquet(f1)
    monkeypatch.setattr(x1, "F1_DEV_DAILY", f1)
    committed = []
    real = guards.log_records

    def log_records(path, kind, *, committed_only=False, **match):
        mine = [r for r in committed if r.get("kind") == kind and all(r.get(k) == v for k, v in match.items())]
        return mine if committed_only else real(path, kind, committed_only=False, **match) + mine
    monkeypatch.setattr(guards, "log_records", log_records)
    monkeypatch.setattr(guards, "committed_versions", lambda p: [])
    monkeypatch.setattr(registry, "PATH", tmp_path / "trials.csv")
    monkeypatch.setattr(registry, "NEFF_LOG", tmp_path / "neff.jsonl")
    calls = []

    def stage_a():
        calls.append(1)
        raise RuntimeError("ran")
    monkeypatch.setattr(x1, "stage_a", stage_a)
    pins = {"kind": "x1-start", "version": 1, **x1.start_pins(guards.sha_str(SEC))}
    return log, calls, committed, pins


def kinds(log):
    return [json.loads(x)["kind"] for x in log.read_text().splitlines()] if log.exists() else []


def test_start_logs_the_record_once_and_stops_until_it_is_committed(monkeypatch, tmp_path):
    log, calls, committed, pins = guard_env(monkeypatch, tmp_path)
    with pytest.raises(SystemExit, match="commit research/x1/runs.jsonl"):
        x1.cmd_start()
    recs = [json.loads(x) for x in log.read_text().splitlines()]
    assert kinds(log) == ["x1-start"] and all(recs[0][k] == v for k, v in pins.items())
    assert {"code_sha256", "prereg_section_sha256", "f1_daily_sha256", "pyarrow_version", "numba_version"} <= set(recs[0])
    with pytest.raises(SystemExit, match="commit"):
        x1.cmd_start()
    assert kinds(log) == ["x1-start"]
    committed.append(recs[0])
    assert x1.cmd_start()["code_sha256"] == x1.code_sha()


def test_stage_a_refuses_without_a_committed_start_record(monkeypatch, tmp_path):
    log, calls, committed, pins = guard_env(monkeypatch, tmp_path)
    with pytest.raises(SystemExit, match="no committed X1 start record"):
        x1.cmd_stage_a()
    with pytest.raises(SystemExit):
        x1.cmd_start()                                                   # logged, but not committed
    with pytest.raises(SystemExit, match="no committed X1 start record"):
        x1.cmd_stage_a()
    assert not calls and kinds(log) == ["x1-start"]


def test_stage_a_runs_once_and_a_hard_error_is_logged(monkeypatch, tmp_path):
    log, calls, committed, pins = guard_env(monkeypatch, tmp_path)
    committed.append(pins)
    with pytest.raises(SystemExit, match="hard error, logged"):
        x1.cmd_stage_a()
    assert calls == [1] and kinds(log) == ["x1-stage-a-run", "x1-stage-a"]
    assert "RuntimeError: ran" in json.loads(log.read_text().splitlines()[1])["hard_error"]
    with pytest.raises(SystemExit, match="runs once"):
        x1.cmd_stage_a()
    assert calls == [1]
    with pytest.raises(SystemExit, match="hard error: the version is closed"):
        x1.stage_a_record()


def test_stage_a_refuses_a_killed_run_and_changed_pins(monkeypatch, tmp_path):
    log, calls, committed, pins = guard_env(monkeypatch, tmp_path)
    committed.append(pins)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps({"kind": "x1-stage-a-run", "version": 1}) + "\n")       # e.g. killed mid-run
    with pytest.raises(SystemExit, match="runs once"):
        x1.cmd_stage_a()
    log.write_text("")
    for k in ("code_sha256", "prereg_section_sha256", "f1_daily_sha256", "exchange_info_sha256", "numba_version",
              "pyarrow_version"):
        committed[0] = pins | {k: "other"}
        with pytest.raises(SystemExit, match=f"different {k}"):
            x1.cmd_stage_a()
        with pytest.raises(SystemExit, match=f"different {k}"):
            x1.cmd_start()
    assert not calls and kinds(log) == []
    committed[0] = pins
    x1.F1_DEV_DAILY.write_bytes(x1.F1_DEV_DAILY.read_bytes() + b"x")        # F1's committed dailies changed
    with pytest.raises(SystemExit, match="different f1_daily_sha256"):
        x1.cmd_stage_a()


def test_stage_a_refuses_to_rerun_over_its_outputs_when_the_log_was_restored(monkeypatch, tmp_path):
    log, calls, committed, pins = guard_env(monkeypatch, tmp_path)
    committed.append(pins)
    js = x1.result_path(x1.STAGE_A_JSON)
    js.write_text("{}")
    with pytest.raises(SystemExit, match="X1_stage_a.json.*runs once"):
        x1.cmd_stage_a()
    js.unlink()
    pd.DataFrame({"F1X_0": [0.0]}).to_parquet(x1.trial_file())
    with pytest.raises(SystemExit, match="F1X_trial_returns.parquet.*runs once"):
        x1.cmd_stage_a()
    x1.trial_file().unlink()
    registry.record([{"strategy": "F1X", "version": 1, "params": {"T": 5}, "period": "dev", "sr": 0.0, "T": 10,
                      "skew": 0.0, "kurt": 3.0, "trades": 1, "sharpe_ann": 0.0}])
    with pytest.raises(SystemExit, match="registry.*runs once"):
        x1.cmd_stage_a()
    assert not calls and kinds(log) == []


def test_stage_a_needs_the_screened_trades_committed_before_it_logs_a_run(monkeypatch, tmp_path):
    log, calls, committed, pins = guard_env(monkeypatch, tmp_path)
    committed.append(pins)
    seen = []

    def require_committed(paths):
        seen.append([Path(p).name for p in paths])
        raise SystemExit("commit these first")
    monkeypatch.setattr(guards, "require_committed", require_committed)
    with pytest.raises(SystemExit, match="commit these first"):
        x1.cmd_stage_a()
    assert seen == [[f"{c}_dev_trades.parquet" for c in x1.SCREEN]] and kinds(log) == [] and not calls


def test_a_version_bump_names_its_own_outputs():
    assert x1.versioned("stage_a_f1_legs.csv", 1) == "stage_a_f1_legs.csv"
    assert x1.versioned("stage_a_f1_legs.csv", 2) == "stage_a_f1_legs_v2.csv"
    assert x1.trial_file(1).name == "F1X_trial_returns.parquet"
    assert x1.trial_file(3).name == "F1Xv3_trial_returns.parquet"
    assert x1.trial_column(0, 1) == "F1X_0" and x1.trial_column(4, 2) == "F1X_v2_4"


def test_section_guard_is_the_registered_one(monkeypatch):
    seen = []
    monkeypatch.setattr(guards, "require_section", lambda *a: seen.append(a) or SEC)
    assert x1._section_sha() == guards.sha_str(SEC)
    assert seen == [(x1.PREREG, "## Execution study X1", x1.REQUIRED_PHRASES)]
    assert "+0.5%/yr" in x1.REQUIRED_PHRASES and "post, then cross" in x1.REQUIRED_PHRASES


# ---------------------------------------------------------------- downloads (jobs only; no network)

def fake_listing(days=None):
    days = days or [d.strftime("%Y-%m-%d") for d in pd.date_range("2023-05-01", "2024-01-03", freq="D")]
    lst = {}
    for sym in SYMBOLS:
        for ds in ticks.DATASETS:
            lst[ticks.DATASETS[ds].format(sym=sym)] = [(ticks.daily_key(ds, sym, d), 1000) for d in days] + \
                [(ticks.daily_key(ds, sym, "2023-06-01") + ".CHECKSUM", 10)]
    return lambda prefix: lst[prefix]


def test_b1_jobs_are_seeded_dev_days_of_both_datasets():
    jobs, rep = x1.download_jobs("B1", fake_listing())
    assert len(jobs) == len(SYMBOLS) * 15 * 2 and rep["bytes"] == len(jobs) * 1000 and not rep["unlisted"]
    assert jobs["day"].between("2023-05-16", "2023-12-30").all()
    assert sorted(set(jobs["dataset"])) == ["aggTrades", "bookTicker"]
    assert [list(x) for x in ticks.select_b1_days(ticks.dev_days(SYMBOLS, lister=fake_listing()), seed=1101)] \
        == rep["days"]
    for w in jobs["windows"]:
        assert len(w) == 1 and w[0][1] - w[0][0] == DAY


def write_legs_csv(path, rows):
    cols = ["symbol", "leg", "side", "post_ms", "qty", "saving_bp"]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


def test_b2_jobs_cover_the_legs_in_the_book_span_and_drop_the_holdout(monkeypatch, tmp_path):
    monkeypatch.setattr(x1, "X1_DIR", tmp_path)
    t = ms("2023-07-01 12:01")
    write_legs_csv(tmp_path / x1.LEGS_CSV, [
        ("BTCUSDT", "exit", 1, t, 1.0, 3.0), ("BTCUSDT", "entry", 1, t, 1.0, 3.0),         # a reversal: one order
        ("ETHUSDT", "entry", -1, ms("2023-05-15 08:01"), 1.0, 3.0),                       # before the book archive
        ("SOLUSDT", "entry", 1, ms("2023-12-31 22:01"), 1.0, 3.0)])                       # window reaches 2024
    jobs, rep = x1.download_jobs("B2", fake_listing())
    assert rep["legs"] == 3 and rep["orders"] == 2 and [d["symbol"] for d in rep["dropped"]] == ["SOLUSDT"]
    assert jobs[["dataset", "symbol", "day"]].values.tolist() == [["aggTrades", "BTCUSDT", "2023-07-01"],
                                                                   ["bookTicker", "BTCUSDT", "2023-07-01"]]
    assert jobs["windows"].iloc[0] == [(t - x1.B2_PRE_MS, t + x1.B2_POST_MS)]
    gone = fake_listing([d.strftime("%Y-%m-%d") for d in pd.date_range("2023-08-01", "2023-12-31", freq="D")])
    jobs, rep = x1.download_jobs("B2", gone)
    assert len(jobs) == 0 and [d["symbol"] for d in rep["missing"]] == ["BTCUSDT"]


def stage_a_committed(committed, pins, strict_pass):
    committed.append({"kind": "x1-stage-a", "version": 1, "decision": "x", "strict_pass": strict_pass,
                      **{k: v for k, v in pins.items() if k not in ("kind", "version")}})


def test_download_needs_stage_a_and_b2_a_strict_pass(monkeypatch, tmp_path):
    log, calls, committed, pins = guard_env(monkeypatch, tmp_path)
    committed.append(pins)
    built = []
    def fake_build(jobs, manifest, getter=None, workers=2):
        built.append(jobs)
        empty = pd.DataFrame(columns=ticks.MANIFEST_COLS)
        empty.to_csv(manifest, index=False)
        return empty
    monkeypatch.setattr(ticks, "build", fake_build)
    with pytest.raises(SystemExit, match="no committed X1 Stage A result"):
        x1.cmd_download("B1", lister=fake_listing())
    stage_a_committed(committed, pins, False)
    with pytest.raises(SystemExit, match="strict pass"):
        x1.cmd_download("B2", lister=fake_listing())
    jobs = x1.cmd_download("B1", lister=fake_listing(), dry_run=True)
    assert len(jobs) == 150 and not built and kinds(log) == []
    x1.cmd_download("B1", lister=fake_listing())
    assert len(built) == 1 and kinds(log) == ["x1-download-start", "x1-download"]
    with pytest.raises(SystemExit, match="unknown tick set"):
        x1.cmd_download("B3")


def test_a_download_guard_trip_is_a_logged_hard_error_that_closes_the_version(monkeypatch, tmp_path):
    log, calls, committed, pins = guard_env(monkeypatch, tmp_path)
    committed.append(pins)
    stage_a_committed(committed, pins, True)

    def late_build(jobs, manifest, getter=None, workers=2):
        man = pd.DataFrame([{c: None for c in ticks.MANIFEST_COLS} | {"last_ts": HOLD, "bytes": 1}],
                           columns=ticks.MANIFEST_COLS)
        man.to_csv(manifest, index=False)
        return man
    monkeypatch.setattr(ticks, "build", late_build)
    with pytest.raises(SystemExit, match="hard error, logged"):
        x1.cmd_download("B1", lister=fake_listing())
    assert kinds(log) == ["x1-download-start", "x1-download"]
    assert "a manifest row reaches the holdout" in json.loads(log.read_text().splitlines()[-1])["hard_error"]
    for cmd in (x1.stage_a_record, x1.cmd_stage_a, lambda: x1.cmd_download("B1", lister=fake_listing()),
                lambda: x1.cmd_download("B2", lister=fake_listing()), lambda: x1.cmd_stage_b("B1"),
                lambda: x1.cmd_stage_b("B2")):
        with pytest.raises(SystemExit, match="hard error: the version is closed"):
            cmd()
    assert kinds(log) == ["x1-download-start", "x1-download"] and not calls


def test_download_failures_are_resumable_but_a_holdout_error_from_the_archive_is_not(monkeypatch, tmp_path):
    log, calls, committed, pins = guard_env(monkeypatch, tmp_path)
    committed.append(pins)
    stage_a_committed(committed, pins, False)
    errors = [RuntimeError("3 of 150 jobs failed, first: ConnectTimeout"),
              PermissionError(13, "the manifest is open in another program"),
              ticks.HoldoutError("x.zip: 1 row(s) of this batch are stamped on or after HOLDOUT_START")]

    def build(jobs, manifest, getter=None, workers=2):
        raise errors.pop(0)
    monkeypatch.setattr(ticks, "build", build)
    with pytest.raises(SystemExit, match="again to resume"):
        x1.cmd_download("B1", lister=fake_listing())
    rec = json.loads(log.read_text().splitlines()[-1])
    assert rec["kind"] == "x1-download" and "ConnectTimeout" in rec["error"] and "hard_error" not in rec
    assert x1.stage_a_record()["decision"] == "x"                         # still open
    with pytest.raises(SystemExit, match="again to resume"):
        x1.cmd_download("B1", lister=fake_listing())                    # an OS error is no guard trip
    assert "hard_error" not in json.loads(log.read_text().splitlines()[-1])
    with pytest.raises(SystemExit, match="hard error, logged"):
        x1.cmd_download("B1", lister=fake_listing())
    assert "HoldoutError" in json.loads(log.read_text().splitlines()[-1])["hard_error"]
    with pytest.raises(SystemExit, match="the version is closed"):
        x1.stage_a_record()


def stage_b_env(monkeypatch, tmp_path):
    """A committed Stage A record pinning the SHA256 of its outputs, and a committed completed download of each
    tick set; Stage B's simulations are stubs. Returns (log, committed, seen require_committed calls, sims)."""
    log, calls, committed, pins = guard_env(monkeypatch, tmp_path)
    committed.append(pins)
    x1.X1_DIR.mkdir(parents=True, exist_ok=True)
    outs = x1.stage_a_outputs()
    for name in (x1.TICK_CSV, x1.Q3_CSV, x1.SCREEN_CSV):
        pd.DataFrame({"a": [1, 2]}).to_csv(outs[name], index=False)
    write_legs_csv(outs[x1.LEGS_CSV], [("BTCUSDT", "entry", 1, ms("2023-07-01 12:01"), 1.0, 3.0)])
    pd.DataFrame({"leg_id": [0]}).to_parquet(outs[x1.Q3_LEGS])
    pd.DataFrame({"F1X_0": [0.0]}).to_parquet(outs["trial_returns"])
    committed.append({"kind": "x1-stage-a", "version": 1, "decision": "x", "strict_pass": True,
                      "outputs_sha256": {n: x1._sha(p) for n, p in outs.items()}})
    for s in x1.TICK_SETS:
        pd.DataFrame(columns=ticks.MANIFEST_COLS).to_csv(x1.manifest_path(s), index=False)
        committed.append({"kind": "x1-download", "version": 1, "tick_set": s,
                          "manifest_sha256": guards.sha_text(x1.manifest_path(s))})
    seen = []
    monkeypatch.setattr(guards, "require_committed", lambda paths: seen.append(sorted(Path(p).name for p in paths)))
    sims = []

    def sim(*a, **k):
        sims.append(1)
        raise RuntimeError("simulated")
    monkeypatch.setattr(x1, "b1_simulate", sim)
    monkeypatch.setattr(x1, "b2_simulate", sim)
    return log, committed, seen, sims


def test_stage_b_reads_only_committed_stage_a_outputs_unchanged_since_stage_a(monkeypatch, tmp_path):
    log, committed, seen, sims = stage_b_env(monkeypatch, tmp_path)
    q3c = x1.x1_path(x1.Q3_CSV)
    good = q3c.read_bytes()
    q3c.write_text("a\n1\n9\n")                                           # e.g. a c_eff edited after Stage A
    with pytest.raises(SystemExit, match=r"differ.*stage_a_q3_costs.csv"):
        x1.cmd_stage_b("B1")
    q3c.write_bytes(good)
    legs = x1.x1_path(x1.LEGS_CSV)
    legs.write_text(legs.read_text().replace(",3.0", ",9.0"))          # e.g. the strict 1m savings (f) compares to
    with pytest.raises(SystemExit, match=r"differ.*stage_a_f1_legs.csv"):
        x1.cmd_stage_b("B2")
    with pytest.raises(SystemExit, match=r"differ.*stage_a_f1_legs.csv"):
        x1.cmd_download("B2", lister=fake_listing())
    assert not sims and kinds(log) == []
    man = x1.manifest_path("B1")
    man.write_text(man.read_text() + "\n")                              # not what the completed download wrote
    with pytest.raises(SystemExit, match="not the manifest of a completed"):
        x1.cmd_stage_b("B1")
    pd.DataFrame(columns=ticks.MANIFEST_COLS).to_csv(man, index=False)
    with pytest.raises(SystemExit, match="hard error, logged"):
        x1.cmd_stage_b("B1")                                            # every check passes: the stub runs
    assert sims == [1] and ["X1_q3_legs.parquet", "stage_a_q3_costs.csv"] in seen and ["tick_manifest_B1.csv"] in seen
    run = json.loads(log.read_text().splitlines()[0])
    assert run["kind"] == "x1-stage-b-run" and set(run["inputs_sha256"]) == {x1.Q3_LEGS, x1.Q3_CSV}
    # a Stage B hard error closes the version for the other set (and every later stage) as well
    for cmd in (lambda: x1.cmd_stage_b("B2"), lambda: x1.cmd_download("B2", lister=fake_listing())):
        with pytest.raises(SystemExit, match="hard error: the version is closed"):
            cmd()


def test_stage_b_runs_once_even_when_the_log_was_restored(monkeypatch, tmp_path):
    log, committed, seen, sims = stage_b_env(monkeypatch, tmp_path)
    p = x1.result_path(x1.STAGE_B_JSON)
    p.write_text(json.dumps({"B1": {"legs": 1}}))
    with pytest.raises(SystemExit, match="B1 in X1_stage_b.json.*runs once"):
        x1.cmd_stage_b("B1")
    p.unlink()
    x1.x1_path(x1.COST_CSV).write_text("x\n")
    with pytest.raises(SystemExit, match="x1_cost_v1.csv.*runs once"):
        x1.cmd_stage_b("B1")
    assert not sims and kinds(log) == []


# ---------------------------------------------------------------- Stage B: tick files, the merged order, (f)

def write_tick(path, dataset, data: dict, windows):
    schema = ticks.SCHEMAS[dataset].with_metadata({"x1_windows": json.dumps(windows)})
    n = len(next(iter(data.values())))
    cols = []
    for f in schema:
        v = data.get(f.name, np.zeros(n, np.int64) if pa.types.is_integer(f.type) else np.zeros(n))
        cols.append(pa.array(np.asarray(v), type=f.type))
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_arrays(cols, schema=schema), path)
    return path


def tick_day(tmp_path, sym, day, t_live, sells_at_bid, bid=100.0, ask=100.1, bid_qty=5.0):
    """Book rows every 10 s over [t_live - 10 min, t_live + 2 h) and seller prints at the bid after t_live."""
    a, b = t_live - 600_000, t_live + 7_200_000
    bts = np.arange(a, b, 10_000, dtype=np.int64)
    book = {"transaction_time": bts, "event_time": bts, "best_bid_price": np.full(len(bts), bid),
            "best_bid_qty": np.full(len(bts), bid_qty), "best_ask_price": np.full(len(bts), ask),
            "best_ask_qty": np.full(len(bts), 3.0)}
    tts = t_live + 1000 * np.arange(1, len(sells_at_bid) + 1, dtype=np.int64)
    tape = {"transact_time": tts, "price": np.full(len(tts), bid), "quantity": np.asarray(sells_at_bid, float),
            "is_buyer_maker": np.ones(len(tts), bool)}
    w = [[int(a), int(b)]]
    pb = write_tick(tmp_path / f"{sym}-bookTicker-{day}.parquet", "bookTicker", book, w)
    pt = write_tick(tmp_path / f"{sym}-aggTrades-{day}.parquet", "aggTrades", tape, w)
    return [{"dataset": "bookTicker", "symbol": sym, "day": day, "out_file": str(pb)},
            {"dataset": "aggTrades", "symbol": sym, "day": day, "out_file": str(pt)}]


def test_load_streams_sorts_stably_and_drops_bad_rows(tmp_path):
    w = [[0, 100]]
    pb = write_tick(tmp_path / "b.parquet", "bookTicker",
                    {"transaction_time": [5, 3, 3, 9], "best_bid_price": [1.0, 2.0, 3.0, 4.0],
                     "best_bid_qty": [1.0, 1.0, 0.0, 1.0], "best_ask_price": [1.1, 2.1, 3.1, 4.1],
                     "best_ask_qty": [1.0, 1.0, 1.0, 1.0]}, w)
    pt = write_tick(tmp_path / "t.parquet", "aggTrades", {"transact_time": [4, 4, 2], "price": [1.0, 2.0, 3.0],
                                                          "quantity": [1.0, 2.0, 3.0],
                                                          "is_buyer_maker": [True, False, True]}, w)
    book, tape, windows, dropped = x1.load_streams([pb], [pt])
    assert book.ts.tolist() == [3, 5, 9] and book.bid_px.tolist() == [2.0, 1.0, 4.0]      # the zero-size row is out
    assert tape.ts.tolist() == [2, 4, 4] and tape.px.tolist() == [3.0, 1.0, 2.0]          # file order at equal ms
    assert windows == w and dropped == {"book_dropped": 1, "tape_dropped": 0}
    late = write_tick(tmp_path / "late.parquet", "aggTrades", {"transact_time": [HOLD], "price": [1.0],
                                                               "quantity": [1.0], "is_buyer_maker": [True]}, w)
    with pytest.raises(x1.HardError, match="holdout"):
        x1.load_streams([pb], [late])


def test_merged_split_fills_the_exit_first():
    r = pd.Series({"filled_qty": 1.5, "post_px": 100.0, "taker_px": 100.1, "fallback_px": 100.2})
    ex, en = x1.merged_split(r, 1.0, 1.0, 1.0, 0.0005, 0.0002)
    assert ex["maker_share"] == 1.0 and en["maker_share"] == 0.5
    assert ex["saving_bp"] == pytest.approx((0.1 / 100.1 + 0.0003) * 1e4)
    assert en["avg_px"] == pytest.approx(100.1)
    assert en["saving_bp"] == pytest.approx((0.0 + 0.5 * 0.0003) * 1e4)
    r0 = r.copy()
    r0["filled_qty"] = 0.0
    none = x1.merged_split(r0, -1.0, 2.0, 1.0, 0.0005, 0.0002)
    assert none[0]["maker_share"] == 0.0 and none[0]["avg_px"] == 100.2


def test_merged_split_of_a_whole_fill_never_mixes_in_the_missing_fallback_price():
    r = pd.Series({"filled_qty": 0.7 + 0.1, "post_px": 100.0, "taker_px": 100.1, "fallback_px": np.nan})
    assert r["filled_qty"] - 0.7 < 0.1                                    # float dust: 0.09999999999999987
    ex, en = x1.merged_split(r, 1.0, 0.7, 0.1, 0.0005, 0.0002)
    assert ex["maker_share"] == 1.0 and en["maker_share"] == 1.0 and en["avg_px"] == 100.0
    assert np.isfinite(en["saving_bp"]) and en["saving_bp"] == ex["saving_bp"]
    rng = np.random.default_rng(7)
    fee_d = 0.0003
    for _ in range(400):
        qx, qn = (float(v) for v in np.round(rng.uniform(0.001, 3.0, 2), 3))
        total = qx + qn
        frac = [0.0, 1.0, float(rng.uniform())][rng.integers(0, 3)]
        full = frac == 1.0
        got = (qn + qx) if full else frac * total                         # summed in another order than the split
        side = float(rng.choice([-1.0, 1.0]))
        whole = pd.Series({"filled_qty": got, "post_px": 100.0, "taker_px": 100.0 + 0.1 * side,
                           "fallback_px": np.nan if full else 100.0 + 0.2 * side})
        ex, en = x1.merged_split(whole, side, qx, qn, 0.0005, 0.0002)
        assert np.isfinite([ex["saving_bp"], en["saving_bp"], ex["avg_px"], en["avg_px"]]).all()
        assert ex["maker_share"] * qx + en["maker_share"] * qn == pytest.approx(total if full else got, abs=1e-9)
        share = 1.0 if full else got / total
        avg = 100.0 if full else share * 100.0 + (1 - share) * whole["fallback_px"]
        s_whole = (side * (whole["taker_px"] - avg) / whole["taker_px"] + share * fee_d) * 1e4
        assert (qx * ex["saving_bp"] + qn * en["saving_bp"]) / total == pytest.approx(s_whole, abs=1e-9)


def test_b2_merged_reversal_fully_filled_keeps_both_legs_in_f(tmp_path):
    t = ms("2023-07-01 12:01")
    man = pd.DataFrame(tick_day(tmp_path, "BTCUSDT", "2023-07-01", t, sells_at_bid=[0.5, 5.0], bid_qty=0.5))
    legs = pd.DataFrame({"symbol": "BTCUSDT", "leg": ["exit", "entry"], "side": [1, 1], "post_ms": [t, t],
                         "qty": [0.7, 0.1], "saving_bp": [3.0, 3.0]})
    b2, _ = x1.b2_simulate(man, legs)
    assert (b2["conservative_maker_share"] == 1.0).all() and (b2["prorata_maker_share"] == 1.0).all()
    assert np.isfinite(b2["conservative_saving_bp"]).all() and b2["conservative_saving_bp"].nunique() == 1
    f = x1.check_f(b2)
    assert f["legs_used"] == 2 and f["tick_conservative_mean_bp"] == pytest.approx((0.1 / 100.1 + 0.0003) * 1e4)
    b2.loc[1, "conservative_saving_bp"] = np.nan
    with pytest.raises(x1.HardError, match="different legs"):
        x1.check_f(b2)


def write_rows(path, dataset, data: dict, windows, row_group_size):
    schema = ticks.SCHEMAS[dataset].with_metadata({"x1_windows": json.dumps(windows)})
    n = len(next(iter(data.values())))
    cols = [pa.array(np.asarray(data.get(f.name, np.zeros(n, np.int64) if pa.types.is_integer(f.type)
                                         else np.zeros(n))), type=f.type) for f in schema]
    pq.write_table(pa.Table.from_arrays(cols, schema=schema), path, row_group_size=row_group_size)
    return path


def random_day(tmp_path, sym, day, seed=3):
    """A whole UTC day (B1-style window): book rows every 10 s on a moving 0.1 grid, prints at either side."""
    rng = np.random.default_rng(seed)
    d0 = ms(day)
    bts = np.arange(d0, d0 + DAY, 10_000, dtype=np.int64)
    bid = 100 + 0.1 * np.cumsum(rng.integers(-1, 2, len(bts)))
    book = {"transaction_time": bts, "event_time": bts, "best_bid_price": bid, "best_ask_price": bid + 0.1,
            "best_bid_qty": rng.uniform(0.5, 5, len(bts)), "best_ask_qty": rng.uniform(0.5, 5, len(bts))}
    tts = np.sort(rng.integers(d0, d0 + DAY, 20_000)).astype(np.int64)
    k = np.clip(np.searchsorted(bts, tts, "right") - 1, 0, None)
    seller = rng.uniform(size=len(tts)) < 0.5
    tape = {"transact_time": tts, "price": np.where(seller, bid[k], bid[k] + 0.1),
            "quantity": rng.uniform(0.01, 2, len(tts)), "is_buyer_maker": seller}
    w = [[d0, d0 + DAY]]
    pb = write_rows(tmp_path / f"{sym}-bookTicker-{day}.parquet", "bookTicker", book, w, 1000)
    pt = write_rows(tmp_path / f"{sym}-aggTrades-{day}.parquet", "aggTrades", tape, w, 1000)
    return [{"dataset": "bookTicker", "symbol": sym, "day": day, "out_file": str(pb)},
            {"dataset": "aggTrades", "symbol": sym, "day": day, "out_file": str(pt)}]


def test_load_streams_reads_only_a_span_and_skips_the_row_groups_outside_it(tmp_path, monkeypatch):
    rows = random_day(tmp_path, "BTCUSDT", "2023-07-01")
    pb, pt = Path(rows[0]["out_file"]), Path(rows[1]["out_file"])
    full_b, full_t, full_w, _ = x1.load_streams([pb], [pt])
    a, e = ms("2023-07-01 06:00"), ms("2023-07-01 08:17")
    asked = []
    real = pq.ParquetFile.iter_batches

    def iter_batches(self, *args, **kw):
        asked.append(len(kw.get("row_groups") or []))
        return real(self, *args, **kw)
    monkeypatch.setattr(pq.ParquetFile, "iter_batches", iter_batches)
    b, t, w, dropped = x1.load_streams([pb], [pt], span=(a, e))
    keep_b = (full_b.ts >= a) & (full_b.ts < e)
    keep_t = (full_t.ts >= a) & (full_t.ts < e)
    np.testing.assert_array_equal(b.ts, full_b.ts[keep_b])
    np.testing.assert_array_equal(b.bid_qty, full_b.bid_qty[keep_b])
    np.testing.assert_array_equal(t.ts, full_t.ts[keep_t])
    np.testing.assert_array_equal(t.buyer_maker, full_t.buyer_maker[keep_t])
    assert w == [[a, e]] and full_w == [[ms("2023-07-01"), ms("2023-07-02")]] and dropped["book_dropped"] == 0
    n_groups = pq.ParquetFile(pb).metadata.num_row_groups
    assert n_groups == 9 and asked[0] <= 3                              # 2 h 17 min of a 24 h day
    near_end = x1.load_streams([pb], [pt], span=(ms("2023-07-01 23:00"), ms("2023-07-02 01:00")))[2]
    assert near_end == [[ms("2023-07-01 23:00"), ms("2023-07-02")]]       # cut to the file's window


def test_b1_by_spans_matches_the_whole_day(tmp_path):
    man = pd.DataFrame(random_day(tmp_path, "BTCUSDT", "2023-07-01"))
    d0 = ms("2023-07-01")
    m0 = d0 + MIN * np.array([2, 30, 31, 200, 700, 701, 1300, 1400, 1439])     # overlaps, the day's two ends
    legs = pd.DataFrame({"leg_id": np.arange(len(m0)), "class": "random", "symbol": "BTCUSDT", "m0_ms": m0,
                         "side": np.tile([1, -1], 5)[:len(m0)], "open_px": 100.0, "slip": 0.0001, "complete": True})
    for T in x1.T_GRID:
        for m in x1.MODELS:
            legs[f"saving_{m}_T{T}"] = 1.0
            legs[f"filled_{m}_T{T}"] = False
    b1, info = x1.b1_simulate(man, legs)
    assert info["days"] == 1 and info["spans"] == 4
    book, tape, windows, _ = x1.load_streams([Path(man["out_file"][0])], [Path(man["out_file"][1])])
    mks = tuple(k * 60_000 for k in x1.MARKOUTS)
    for T in x1.T_GRID:
        for q in x1.QUEUES:
            want = queue_sim.simulate_legs(book, tape, m0, legs["side"].to_numpy(), x1.Q3_NOTIONAL / 100.0,
                                           T * 60_000, queue=q, markouts_ms=mks, windows_ms=windows)
            got = b1[(b1["T"] == T) & (b1["queue"] == q)].reset_index(drop=True)
            assert got["leg_id"].tolist() == list(range(len(m0)))
            np.testing.assert_array_equal(got["status"], want["status"])
            np.testing.assert_array_equal(got["complete_tick"], want["complete"])
            np.testing.assert_array_equal(got["saving_bp"], want["saving_bp"])
            np.testing.assert_array_equal(got["maker_share"], want["maker_share"])
    assert not b1.loc[b1["m0_ms"] == m0[-1], "complete_tick"].any()      # its markouts pass the day's end


def test_b2_simulates_a_reversal_as_one_order_and_checks_f(tmp_path):
    t = ms("2023-07-01 12:01")
    man = pd.DataFrame(tick_day(tmp_path, "BTCUSDT", "2023-07-01", t, sells_at_bid=[4.0, 2.0]))   # 6 at P, 5 ahead
    legs = pd.DataFrame({"symbol": "BTCUSDT", "leg": ["exit", "entry"], "side": [1, 1], "post_ms": [t, t],
                         "qty": [1.0, 1.0], "saving_bp": [3.0, 3.0]})
    b2, info = x1.b2_simulate(man, legs)
    assert info == {"orders": 1, "no_files": 0} and b2["merged"].all()
    ex, en = b2.set_index("leg").loc["exit"], b2.set_index("leg").loc["entry"]
    assert ex["conservative_maker_share"] == 1.0 and en["conservative_maker_share"] == 0.0
    assert ex["conservative_saving_bp"] == pytest.approx((0.1 / 100.1 + 0.0003) * 1e4)
    assert en["conservative_saving_bp"] == pytest.approx(0.0)          # fell back at the ask: the taker price
    f = x1.check_f(b2)
    assert f["legs_used"] == 2 and f["strict_1m_mean_bp"] == 3.0
    assert f["tick_conservative_mean_bp"] == pytest.approx(ex["conservative_saving_bp"] / 2)
    assert f["f_pass"] is (f["tick_conservative_mean_bp"] >= 3.0 - x1.F_TOL_BP)
    alone = x1.b2_simulate(man, legs.iloc[[1]])[0].iloc[0]                # an entry alone gets the whole fill
    assert alone["conservative_maker_share"] == 1.0 and not alone["merged"]
    other = legs.assign(symbol="ETHUSDT")
    gone, info = x1.b2_simulate(man, other)
    assert info["no_files"] == 2 and (gone["status"] == "no_files").all() and x1.check_f(gone)["f_pass"] is False


def test_b1_cost_model_corrections(tmp_path, monkeypatch):
    t = ms("2023-07-01 12:01")
    man = pd.DataFrame(tick_day(tmp_path, "BTCUSDT", "2023-07-01", t, sells_at_bid=[30.0]))
    legs = pd.DataFrame({"leg_id": [0, 1, 2], "class": "random", "symbol": "BTCUSDT", "m0_ms": [t, t, t],
                         "side": [1, -1, 1], "open_px": 100.05, "slip": 0.0001, "complete": [True, True, False]})
    for T in x1.T_GRID:
        for m in x1.MODELS:
            legs[f"saving_{m}_T{T}"] = [3.0, -2.0, 0.0] if m == "strict" else [4.0, -1.0, 0.0]
            legs[f"filled_{m}_T{T}"] = [True, False, False]
    b1, info = x1.b1_simulate(man, legs)
    assert info["days"] == 1 and len(b1) == 2 * len(x1.T_GRID) * len(x1.QUEUES)
    g = b1[(b1["T"] == 15) & (b1["queue"] == "conservative")].set_index("leg_id")
    assert g.loc[0, "maker_share"] == 1.0 and g.loc[1, "maker_share"] == 0.0        # buyers never print here
    q3c = pd.DataFrame([{"class": "random", "symbol": s, "T": T, "model": m, "c_eff_bp": 6.0}
                        for s in [*SYMBOLS, "ALL"] for T in x1.T_GRID for m in x1.MODELS])
    cm = x1.cost_model(b1, q3c)
    r = cm[(cm["class"] == "random") & (cm["symbol"] == "BTCUSDT") & (cm["T"] == 15)].iloc[0]
    tick_mean = g["saving_bp"].astype(float).mean()
    assert r["n"] == 2 and r["tick_conservative_bp"] == pytest.approx(tick_mean)
    assert r["m1_strict_bp"] == pytest.approx(0.5) and r["corr_strict_bp"] == pytest.approx(tick_mean - 0.5)
    assert r["corr_central_bp"] == pytest.approx(tick_mean - 1.5)
    assert r["c_eff_v1_strict_bp"] == pytest.approx(6.0 - (tick_mean - 0.5))
    assert cm[(cm["symbol"] == "ALL") & (cm["class"] == "random") & (cm["T"] == 15)]["n"].iloc[0] == 2


# ---------------------------------------------------------------- Stage A end to end on a synthetic store

def store_env(monkeypatch, tmp_path, n_days=75):
    """Five synthetic markets from 2023-07-01 (a short dev window), committed results of the screened families,
    a sandboxed registry and the F1 reference daily built independently with engine.run."""
    log, calls, committed, pins = guard_env(monkeypatch, tmp_path)
    monkeypatch.setattr(x1, "stage_a", STAGE_A)                          # the real Stage A, not the stub
    monkeypatch.setattr(x1, "START", pd.Timestamp("2023-07-03", tz="UTC"))
    monkeypatch.setattr(x1, "Q3_RANDOM_PER_COIN", 200)
    mks = {s: synth_market(s, "2023-07-01", n_days, seed=i, tick=0.5) for i, s in enumerate(SYMBOLS)}
    monkeypatch.setattr(engine.Market, "load", lambda period, symbol, with_metrics=True: mks[symbol])
    parts = []
    for s in SYMBOLS:
        mk = mks[s]
        r = engine.run(mk, S1Frozen().compute(mk, **x1.F1_PARAMS), Costs(slippage=SLIPPAGE[s]), sleeve_risk(5),
                       start=x1.START, latency_min=1)
        parts.append((aligned_daily(r, mk), r.trades))
    ref = assemble(parts)[1].rename("ret").to_frame()
    ref.to_parquet(x1.F1_DEV_DAILY)
    pins = {"kind": "x1-start", "version": 1, **x1.start_pins(guards.sha_str(SEC))}
    committed.append(pins)
    rng = np.random.default_rng(0)
    for code in x1.SCREEN:
        n = 20
        t = pd.DataFrame({"qty": 1.0, "entry_px": 100.0, "exit_px": 100.0 + rng.normal(0, 1, n),
                          "gross": rng.normal(0.05, 0.5, n), "maker": 0,
                          "reason": rng.choice(["stop", "exit_signal", "time"], n),
                          "entry_time": pd.date_range("2019-06-01", periods=n, freq="60D", tz="UTC"),
                          "exit_time": pd.date_range("2019-06-02", periods=n, freq="60D", tz="UTC")})
        if code.startswith("F"):
            t["symbol"] = rng.choice(SYMBOLS, n)
        t.to_parquet(x1.RESULTS / f"{code}_dev_trades.parquet")
    monkeypatch.setattr(research_run, "RESULTS", x1.RESULTS)
    monkeypatch.setattr(registry, "PATH", tmp_path / "trials.csv")
    monkeypatch.setattr(registry, "NEFF_LOG", tmp_path / "neff.jsonl")
    return log, committed, pins, mks


def test_stage_a_end_to_end_on_a_synthetic_store(monkeypatch, tmp_path, capsys):
    log, committed, pins, mks = store_env(monkeypatch, tmp_path)
    r = x1.cmd_stage_a()
    assert kinds(log) == ["x1-stage-a-run", "x1-stage-a"]
    rec = json.loads(log.read_text().splitlines()[-1])
    assert rec["decision"] == r["decision"] and rec["strict_pass"] is r["strict_pass"]
    assert r["decision"] == x1.FAIL_DECISION                         # one year of data: (d) cannot pass
    assert not r["gates"]["strict"]["(d) >=3 of 4 years and >=4 of 5 coins positive"]
    assert set(r["runs"]) == set(x1.policies()) and r["runs"]["taker"]["trades"] > 10
    assert all(v["info"]["funding_in_window"] == 0 for v in r["runs"].values())
    lg = r["legs"][x1.PRIMARY]
    assert lg["legs"] > 10 and 0 < lg["fill_rate"] < 1 and lg["mean_saving_filled_bp"] == pytest.approx(3.0)
    assert r["legs"]["taker"]["mean_saving_bp"] == 0.0
    for k in ("strict_T15_nofb", "strict_T60_nofb", x1.PRIMARY):
        info, s = r["runs"][k]["info"], r["legs"][k]
        assert s["fill_rate"] == pytest.approx((info["entry_maker_fills"] + info["exit_maker_fills"])
                                               / info["orders_posted"])
        assert s["orders_without_leg"] == info["entries_missed"] + info["exits_cancelled"] \
            + info["exits_preempted_by_stop"]
    nofb = r["legs"]["strict_T15_nofb"]
    assert nofb["leg_fill_rate"] == 1.0 and nofb["orders_without_leg"] > 0 and nofb["fill_rate"] < 1.0
    assert set(r["outputs_sha256"]) == {x1.TICK_CSV, x1.LEGS_CSV, x1.Q3_CSV, x1.SCREEN_CSV, x1.Q3_LEGS,
                                        "trial_returns"}
    assert rec["outputs_sha256"] == r["outputs_sha256"] and rec["stage_a_json_sha256"]
    assert rec["outputs_sha256"][x1.LEGS_CSV] == guards.sha_text(x1.x1_path(x1.LEGS_CSV))
    assert set(rec["screen_inputs_sha256"]) == {f"{c}_dev_trades.parquet" for c in x1.SCREEN}
    tr = pd.read_parquet(x1.trial_file())
    assert list(tr.columns) == [f"F1X_{i}" for i in range(5)] and len(tr) == len(pd.read_parquet(x1.F1_DEV_DAILY))
    reg = registry.load()
    assert len(reg) == 5 and set(reg["strategy"]) == {"F1X"}
    assert json.loads(reg["params"].iloc[3]) == {"T": 15, "fallback": False, "model": "strict"}
    assert r["n_eff_family"] >= 1 and r["n_eff_project"] >= r["n_eff_family"] - 1e-9
    for f in (x1.TICK_CSV, x1.LEGS_CSV, x1.Q3_CSV, x1.SCREEN_CSV):
        assert x1.x1_path(f).exists(), f
    ticks_csv = pd.read_csv(x1.x1_path(x1.TICK_CSV))
    assert (ticks_csv["tick"] == 0.5).all() and (ticks_csv["source"] == "1m").all()
    legs_csv = pd.read_csv(x1.x1_path(x1.LEGS_CSV))
    assert len(legs_csv) == lg["legs"] and {"post_ms", "qty", "side", "saving_bp"} <= set(legs_csv.columns)
    q3c = pd.read_csv(x1.x1_path(x1.Q3_CSV))
    assert (q3c[(q3c["class"] == "random") & (q3c["symbol"] == "ALL")]["n"] > 900).all()
    q3l = pd.read_parquet(x1.result_path(x1.Q3_LEGS))
    assert q3l["m0_ms"].max() < HOLD and len(q3l) >= 1000
    js = json.loads(x1.result_path(x1.STAGE_A_JSON).read_text())
    assert js["decision"] == r["decision"] and js["code_sha256"] == pins["code_sha256"] and "n_eff_family" in js
    out = capsys.readouterr().out
    assert "DECISION: F1 NO-GO" in out and "strict_T15_nofb" in out
    x1.cmd_report()
    assert "DECISION" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="runs once"):
        x1.cmd_stage_a()


def test_stage_a_hard_error_when_the_taker_run_does_not_reproduce_f1(monkeypatch, tmp_path):
    log, committed, pins, mks = store_env(monkeypatch, tmp_path, n_days=45)
    ref = pd.read_parquet(x1.F1_DEV_DAILY)
    ref.iloc[20, 0] += 1e-9
    ref.to_parquet(x1.F1_DEV_DAILY)
    committed[0] = {"kind": "x1-start", "version": 1, **x1.start_pins(guards.sha_str(SEC))}
    with pytest.raises(SystemExit, match="does not reproduce F1_dev_daily"):
        x1.cmd_stage_a()
    rec = json.loads(log.read_text().splitlines()[-1])
    assert rec["kind"] == "x1-stage-a" and "HardError" in rec["hard_error"] and "decision" not in rec
    assert not x1.result_path(x1.STAGE_A_JSON).exists()


def test_stage_a_hard_error_when_a_maker_run_charges_the_wrong_fees(monkeypatch, tmp_path):
    log, committed, pins, mks = store_env(monkeypatch, tmp_path, n_days=45)
    real = ME.run

    def taxed(*a, **k):
        res = real(*a, **k)
        t = res.trades
        mf = t["maker"] == 1                                            # a maker entry billed at the taker fee
        t.loc[mf, "fees"] += 0.0003 * t.loc[mf, "qty"] * t.loc[mf, "entry_px"]
        return res
    monkeypatch.setattr(ME, "run", taxed)
    with pytest.raises(SystemExit, match="fees other than"):
        x1.cmd_stage_a()
    rec = json.loads(log.read_text().splitlines()[-1])
    assert rec["kind"] == "x1-stage-a" and "HardError" in rec["hard_error"]


def test_a_version_bump_keeps_the_earlier_trial_series_and_counts_them_in_the_family(monkeypatch, tmp_path):
    log, committed, pins, mks = store_env(monkeypatch, tmp_path)
    x1.cmd_stage_a()
    v1 = pd.read_parquet(x1.trial_file())
    v1_json = x1.result_path(x1.STAGE_A_JSON).read_text()
    monkeypatch.setattr(x1, "VERSION", 2)
    assert x1.result_path(x1.STAGE_A_JSON).name == "X1_stage_a_v2.json"
    committed.append(pins | {"version": 2})
    r = x1.cmd_stage_a()
    pd.testing.assert_frame_equal(pd.read_parquet(x1.trial_file(1)), v1)
    assert (x1.RESULTS / x1.STAGE_A_JSON).read_text() == v1_json
    assert list(pd.read_parquet(x1.trial_file()).columns) == [f"F1X_v2_{i}" for i in range(5)]
    assert r["family_columns"] == [f"F1X_{i}" for i in range(5)] + [f"F1X_v2_{i}" for i in range(5)]
    reg = registry.load()
    assert len(reg) == 10 and sorted(set(reg["version"])) == [1, 2]
    assert research_run.all_trial_returns().shape[1] == 10
    for f in (x1.TICK_CSV, x1.LEGS_CSV, x1.Q3_CSV, x1.SCREEN_CSV):
        assert x1.x1_path(f).name.endswith("_v2.csv") and x1.x1_path(f).exists()
        assert (x1.X1_DIR / f).exists()                                   # v1's file is still there


def test_check_runs_catches_funding_in_a_window_and_holdout_dates():
    idx = days()[:3]
    d = pd.Series([0.0, 0.01, -0.01], index=idx)
    ok = {"daily": d, "sleeves": {"BTCUSDT": d}, "trades": pd.DataFrame(columns=["entry_time", "exit_time"]),
          "info": {"funding_in_window": 0}}
    x1.check_runs({x1.TAKER: ok}, d)
    with pytest.raises(x1.HardError, match="funding"):
        x1.check_runs({x1.TAKER: ok, "strict_T60": ok | {"info": {"funding_in_window": 1}}}, d)
    late = pd.Series([0.0], index=pd.DatetimeIndex([HOLDOUT_START]))
    with pytest.raises(x1.HardError, match="holdout"):
        x1.check_runs({x1.TAKER: ok | {"sleeves": {"BTCUSDT": late}}}, d)
    with pytest.raises(x1.HardError, match="reproduce"):
        x1.check_runs({x1.TAKER: ok}, d.iloc[:2])
    assert x1.parity_problem(d, d + 1e-13) is None and "max" in x1.parity_problem(d, d + 1e-11)


# ---------------------------------------------------------------- registration plumbing

def test_code_files_exist_and_the_cli_knows_x1():
    for f in x1.CODE_FILES:
        assert (x1.ROOT / f).exists(), f
    assert "src/btcq/backtest/engine.py" in x1.CODE_FILES and len(set(x1.CODE_FILES)) == 20
    import inspect
    src = inspect.getsource(research_run.main)
    for c in ("x1-start", "x1-stage-a", "x1-download", "x1-stage-b", "x1-report", "--set"):
        assert c in src


def test_policies_and_trial_columns():
    p = x1.policies()
    assert len(p) == 1 + 9 + 2 and p["taker"].T == 0
    assert p["strict_T15_nofb"] == ME.MakerPolicy(15, "strict", False) and p[x1.PRIMARY] == ME.MakerPolicy(15)
    assert x1.TRIAL_KEYS == ["strict_T5", "strict_T15", "strict_T60", "strict_T15_nofb", "strict_T60_nofb"]
