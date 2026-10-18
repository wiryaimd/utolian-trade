"""Round 5: D1 Donchian ensemble, XS cross-sectional momentum, nulls, gates, guards (synthetic data)."""
import json

import numpy as np
import pandas as pd
import pytest

from btcq.backtest import stats
from btcq.config import SYMBOLS
from btcq.data import store
from btcq.research import round5 as R
from btcq.strategies import r5_d1 as D
from btcq.strategies import r5_xsmom as X
from conftest import make_m1

DAY = 86_400_000
T0 = int(pd.Timestamp("2022-01-01", tz="UTC").value // 10**6)
W = R.Window("t", "dev", pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2022-09-01", tz="UTC"))


# ---------------------------------------------------------------- D1 state machine

def reference_state(c, L, ready):
    """Plain-Python reference of the D1 rule."""
    n = len(c)
    st, ex, tr = [0] * n, [False] * n, [np.nan] * n
    pos, T = 0, None
    for t in range(L, n):
        U = max(c[t - L:t])
        win = c[t - L + 1:t + 1]
        M = (max(win) + min(win)) / 2
        if pos == 0:
            if c[t] > U and ready[t]:
                pos, T = 1, M
        else:
            T = max(T, M)
            if c[t] < T:
                pos, T, ex[t] = 0, None, True
        st[t] = pos
        if pos:
            tr[t] = T
    return np.array(st), np.array(ex), np.array(tr, float)


@pytest.mark.parametrize("seed,L", [(0, 3), (1, 5), (2, 20), (3, 5), (4, 20)])
def test_donchian_state_matches_reference(seed, L):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, 400)))
    ready = np.ones(400, bool)
    ready[:30] = False
    st, ex, tr, _ = D.donchian_state(c, L, ready)
    rs, rex, rtr = reference_state(list(c), L, ready)
    assert np.array_equal(st, rs) and np.array_equal(ex, rex)
    assert np.allclose(np.nan_to_num(tr, nan=-1), np.nan_to_num(rtr, nan=-1))
    assert st[:30].sum() == 0 or st[:max(L, 30)].max() == 0


def test_donchian_hand_checked_path():
    c = np.array([10, 11, 12, 12, 13, 12.4, 12.6, 11.0, 13.5])
    st, ex, tr, mid = D.donchian_state(c, 3, np.ones(len(c), bool))
    # t=3: 12 is not > max(11,12,12)? prior closes 10,11,12 -> 12 not > 12: no entry
    # t=4: 13 > 12 -> long, T = (13+12)/2 = 12.5
    assert list(st[:5]) == [0, 0, 0, 0, 1] and tr[4] == 12.5
    # t=5: MID(12,13,12.4)=12.5, close 12.4 < 12.5 -> exit
    assert ex[5] and st[5] == 0
    # t=6: 12.6 not > max(12,13,12.4)=13 -> flat; t=7 flat; t=8: 13.5 > 13 -> long again
    assert list(st[6:]) == [0, 0, 1]


def test_d1_causal_under_truncation():
    rng = np.random.default_rng(9)
    c = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.03, 500)))
    s = pd.Series(c, index=T0 + DAY * np.arange(500))
    full = D.sub_signals(s, 20)
    for cut in (120, 260, 499):
        part = D.sub_signals(s.iloc[:cut], 20)
        assert np.array_equal(part.state, full.state[:cut]) and np.array_equal(part.exit, full.exit[:cut])
        assert np.allclose(np.nan_to_num(part.size), np.nan_to_num(full.size[:cut]))


def test_vol_size_is_nan_until_ready_and_capped():
    c = 100 * np.exp(np.cumsum(np.full(200, 0.0001)))
    v = D.vol_size(c)
    assert np.isnan(v[:D.VOL_DAYS]).all() and np.all(v[D.VOL_DAYS + 1:] == D.LEV_CAP)


# ---------------------------------------------------------------- synthetic coins

def daily_path(days, drift, vol, seed):
    rng = np.random.default_rng(seed)
    n = days * 1440
    lr = rng.normal(drift / 1440, vol / np.sqrt(1440), n)
    c = 100 * np.exp(np.cumsum(lr))
    o = np.r_[100, c[:-1]]
    return list(zip(o, np.maximum(o, c) * 1.0001, np.minimum(o, c) * 0.9999, c))


@pytest.fixture(scope="module")
def coins():
    days = 243
    drifts = [0.004, 0.002, 0.0, -0.002, -0.004]
    out = {}
    for i, s in enumerate(SYMBOLS):
        m1 = make_m1(daily_path(days, drifts[i], 0.03, i), T0)
        out[s] = R.coin_from_m1(s, m1, None)
    return out


def test_engine_follows_the_state_machine(coins):
    closes = {s: c.perp_close for s, c in coins.items()}
    subs = R.d1_build(coins, closes)
    parts, floors = R.d1_run(subs, coins, W)
    assert floors == 0 and R.d1_mismatches(subs, parts, coins, W) == 0
    n_tr = 0
    for (s, L), (_, tr) in parts.items():
        if not len(tr):
            continue
        n_tr += len(tr)
        assert not (tr["reason"] == "stop").any()
        mk = coins[s].mk
        # fills happen at 00:01 UTC, one minute after a daily close
        assert (mk.ts[tr["entry_i"].to_numpy()] % DAY == 60_000).all()
        sub = subs[(s, L)]
        lvl = sub.trail[tr["entry_tf"].to_numpy()]
        assert np.allclose(tr["init_risk"], tr["qty"] * (tr["entry_px"] - lvl))
        signal_exits = tr[tr["reason"] == "exit_signal"]
        assert sub.sig.exit_long[(mk.ts[signal_exits["exit_i"].to_numpy()] - 60_000 - DAY) // DAY
                                 - (coins[s].bar_ts[0] // DAY)].all()
    assert n_tr > 20


def test_d1_slot_assembly(coins):
    closes = {s: c.perp_close for s, c in coins.items()}
    parts, _ = R.d1_run(R.d1_build(coins, closes), coins, W)
    for cfg, n in (("all9", 45), ("fast5", 25), ("slow5", 25)):
        _, daily = R.d1_assemble(parts, W, cfg)
        want = sum(d for (s, L), (d, _) in parts.items() if L in D.CONFIGS[cfg]) / n
        assert np.allclose(daily.to_numpy(), want.to_numpy())


def test_d1_shift_null_preserves_time_in_market():
    st = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0, 0], np.int8)
    sig = D.to_engine(D.SubSignal(np.arange(10) * DAY, st, np.zeros(10, bool), np.ones(10), np.ones(10)),
                      np.arange(10) * DAY)[0]
    ts = np.arange(10) * DAY
    sh = R._shift_sig(sig, st, ts, ts, 3)
    assert sh.entry.sum() == st.sum() and list(sh.entry) == list(np.roll(st, 3))
    assert list(sh.exit_long) == list(np.roll(st, 3) == 0)
    # a coin listed later: its bars sit inside the shared calendar, pre-listing days are flat
    cal = np.arange(-3, 10) * DAY
    sh2 = R._shift_sig(sig, st, ts, cal, 3)
    assert list(sh2.entry) == list(np.roll(np.r_[np.zeros(3, np.int8), st], 3)[3:])


# ---------------------------------------------------------------- XS

def px_frame(rows, cols=("A", "B", "C", "D", "E")):
    return pd.DataFrame(rows, columns=list(cols), index=T0 + DAY * np.arange(len(rows)))


def test_xs_ranks_and_sides_for_n_2_to_5():
    sc = px_frame([[0.3, 0.1, 0.2, -0.1, 0.0], [0.3, 0.3, np.nan, 0.1, 0.2], [0.2, np.nan, np.nan, 0.1, 0.0],
                   [0.1, np.nan, np.nan, np.nan, -0.1]])
    el = sc.notna()
    rk = X.ranks(sc, el)
    assert list(rk.iloc[0]) == [1, 3, 2, 5, 4]
    assert list(rk.iloc[1]) == [1, 2, 0, 4, 3]           # tie 0.3/0.3 broken by symbol (A before B)
    sd = X.sides(rk)
    assert list(sd.iloc[0]) == [1, 0, 1, -1, -1]          # n=5, k=2
    assert list(sd.iloc[1]) == [1, 1, 0, -1, -1]          # n=4, k=2
    assert list(sd.iloc[2]) == [1, 0, 0, 0, -1]           # n=3, k=1
    assert list(sd.iloc[3]) == [1, 0, 0, 0, -1]           # n=2, k=1


def test_xs_permutation_preserves_rank_multiset():
    rk = X.ranks(px_frame([[0.3, 0.1, 0.2, -0.1, np.nan]] * 20), px_frame([[0.3, 0.1, 0.2, -0.1, np.nan]] * 20).notna())
    pr = X.permute_ranks(rk, np.random.default_rng(0))
    for i in range(20):
        assert sorted(pr.iloc[i]) == sorted(rk.iloc[i]) and pr.iloc[i, 4] == 0


def test_xs_weights_sizes_and_stops():
    rng = np.random.default_rng(3)
    vols = [0.01, 0.02, 0.03, 0.04, 0.05]
    px = px_frame(100 * np.exp(np.cumsum(rng.normal(0, vols, (80, 5)), axis=0)))
    plan = X.leg_plan(px, {"mode": "weekly", "L": 14})
    i = 70
    side, wgt = plan["side"].iloc[i], plan["weight"].iloc[i]
    for s in (1, -1):
        assert wgt[side == s].sum() == pytest.approx(1.0)
    assert np.allclose(plan["size"].iloc[i], 2.5 * wgt)
    _, _, mad = X.features(px, 14)
    long = side[side == 1].index[0]
    assert plan["stop"].iloc[i][long] == pytest.approx(px.iloc[i][long] * (1 - 4 * mad.iloc[i][long]))


def test_xs_hysteresis_holds_for_min_days():
    want = px_frame([[1, -1, 0, 0, 0]] * 3 + [[-1, 1, 0, 0, 0]] * 10).astype(np.int8)
    out = X.hysteresis(want, np.ones(13, bool), 7)
    assert list(out["A"]) == [1] * 7 + [-1] * 6            # flips once the leg has been held 7 days


def test_xs_engine_legs(coins):
    px = R.xs_panel(coins)
    res, daily, parts, plan = R.xs_run(coins, px, X.CONFIGS[X.PRIMARY], W)
    t = res.trades
    assert len(t) > 10
    # entries only on Monday 00:01
    days = pd.to_datetime(t["entry_time"])
    assert (days.dt.dayofweek == 0).all() and (days.dt.hour == 0).all() and (days.dt.minute == 1).all()
    # R comes from the per-leg stop set on the decision bar
    for _, tr in t.head(20).iterrows():
        dec = int(tr["entry_time"].value // 10**6) - 60_000 - DAY
        stop = plan["stop"].loc[dec, tr["symbol"]]
        assert tr["init_risk"] == pytest.approx(tr["qty"] * abs(tr["entry_px"] - stop), rel=1e-9)
    # a continuing leg is not re-traded: no two consecutive trades of a coin share a direction without a gap
    for s, g in t.groupby("symbol"):
        g = g.sort_values("entry_time")
        same = (g["dir"].to_numpy()[1:] == g["dir"].to_numpy()[:-1]) & \
               (g["entry_i"].to_numpy()[1:] == g["exit_i"].to_numpy()[:-1]) & (g["reason"].to_numpy()[:-1] == "exit_signal")
        assert not same.any()


def test_xs_loco_runs_five_universes(coins):
    lo = R.xs_loco(coins, X.CONFIGS[X.PRIMARY], W)
    assert sorted(lo) == sorted(SYMBOLS)


def test_xs_rank_null_detects_planted_edge():
    """Strong persistent drifts: the true momentum ranking makes money and beats permuted ranks."""
    drifts = [0.02, 0.01, 0.0, -0.01, -0.02]
    tc = {s: R.coin_from_m1(s, make_m1(daily_path(243, drifts[i], 0.01, 10 + i), T0), None)
          for i, s in enumerate(SYMBOLS)}
    px = R.xs_panel(tc)
    cfg = X.CONFIGS[X.PRIMARY]
    res, gross, _, _ = R.xs_run(tc, px, cfg, W, zero=True)
    assert res.trades["R"].mean() > 0
    nl = R.xs_rank_null(tc, px, cfg, W, float(gross.mean()), n_draws=40)
    assert nl["p_value"] < 0.05


# ---------------------------------------------------------------- stats, gates, guards

def test_hac_ols_matches_double_loop():
    rng = np.random.default_rng(1)
    x = rng.normal(size=300)
    y = 0.001 + 0.4 * x + np.convolve(rng.normal(size=302), [1, 0.5, 0.25], "valid")[:300]
    h = stats.hac_ols(y, x, lag=5)
    X_ = np.column_stack([np.ones(300), x])
    b = np.linalg.lstsq(X_, y, rcond=None)[0]
    e = y - X_ @ b
    S = np.zeros((2, 2))
    for t in range(300):
        S += e[t] ** 2 * np.outer(X_[t], X_[t])
    for l in range(1, 6):
        w = 1 - l / 6
        for t in range(l, 300):
            S += w * e[t] * e[t - l] * (np.outer(X_[t], X_[t - l]) + np.outer(X_[t - l], X_[t]))
    Bi = np.linalg.inv(X_.T @ X_)
    V = Bi @ S @ Bi * 300 / 298
    assert h["alpha"] == pytest.approx(b[0]) and h["t_alpha"] == pytest.approx(b[0] / np.sqrt(V[0, 0]))
    assert stats.nw_lag(1461) == 7


def test_year_share_daily_and_power_bounds():
    idx = pd.date_range("2020-01-01", periods=730, freq="D", tz="UTC")
    d = pd.Series(np.r_[np.full(366, 0.002), np.full(364, 0.001)], index=idx)
    assert stats.year_share_daily(d) == pytest.approx(0.732 / 1.096, rel=1e-3)
    p = stats.power_table((0.0, 1.0), n_sims=4000)
    assert 0.0005 <= p[0]["p_replicated"] <= 0.012 and p[1]["p_replicated"] > p[0]["p_replicated"]


def test_gate_edges_and_headline():
    ev = {"weekly": {"t": 2.0}, "stress_2x_mean_daily": 1e-6, "year_share": 0.5, "coins_positive": 3,
          "null": {"p_value": 0.049}, "pbo": 0.5, "plateau": 1.0, "alpha_vs_ew": {"psr_alpha": 0.95}}
    assert all(R.gates_d1(ev).values())
    assert not R.gates_d1({**ev, "plateau": 0.5})["plateau>=70%"]
    assert not R.gates_d1({**ev, "alpha_vs_ew": {"psr_alpha": 0.949}})["alpha PSR vs EW hold>=0.95"]
    rng = np.random.default_rng(2)
    days = pd.date_range("2020-01-01", periods=2465, freq="D", tz="UTC")
    strong = pd.Series(rng.normal(0.003, 0.02, len(days)), index=days)
    h = R.headline_r5(True, True, strong, R.NEFF_FLOOR)
    assert h["REPLICATED"] and h["n"] == pytest.approx(16.8496, abs=1e-4)
    assert not R.headline_r5(True, False, strong, R.NEFF_FLOOR)["REPLICATED"]
    weak = pd.Series(rng.normal(0.0, 0.02, len(days)), index=days)
    assert R.headline_r5(True, True, weak, R.NEFF_FLOOR)["REPLICATED"] and not R.headline_r5(True, True, weak, R.NEFF_FLOOR)["STRICT"]


def test_trial_rows_are_unique():
    ev_d = {"cfg_daily": {c: pd.Series([0.001, 0.0, -0.001]) for c in D.CONFIGS}, "cfg_trades": {c: 3 for c in D.CONFIGS}}
    ev_x = {"cfg_daily": {i: pd.Series([0.001, 0.0, -0.001]) for i in range(6)}, "cfg_trades": {i: 3 for i in range(6)}}
    rows = R._trial_rows("D1", ev_d, lambda c: {"set": c}) + R._trial_rows("XS", ev_x, lambda i: X.CONFIGS[i])
    keys = {(r["strategy"], json.dumps(r["params"], sort_keys=True)) for r in rows}
    assert len(rows) == 9 and len(keys) == 9


def test_download_refuses_without_prereg(tmp_path, monkeypatch):
    p = tmp_path / "PREREGISTRATION.md"
    p.write_text("# nothing\n")
    monkeypatch.setattr(R, "PREREG", p)
    from btcq.research import round4
    monkeypatch.setattr(round4, "_require_committed", lambda paths: None)   # committed, but no Round 5 section
    monkeypatch.setattr(store, "build_spot_1d", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no download")))
    with pytest.raises(SystemExit):
        R.cmd_download_spot()


def test_spot_build_never_reaches_the_holdout():
    with pytest.raises(ValueError):
        store.build_spot_1d("ETHUSDT", "2024-01")


def test_preflight_and_resume_once(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "RESULTS", tmp_path)
    monkeypatch.setattr(R.store, "HOLDOUT_LOG", tmp_path / "holdout_log.jsonl")
    calls = []
    monkeypatch.setattr(R.store, "unlock_holdout", lambda reason: calls.append(reason))
    (tmp_path / "XS.json").write_text(json.dumps({"dev_pass": False}))
    (tmp_path / "D1.json").write_text(json.dumps({"dev_pass": False}))
    probs, resume = R.preflight_holdout()
    assert any("did not pass dev" in p for p in probs) and not resume
    with pytest.raises(SystemExit):
        R.cmd_validate("x")
    (tmp_path / "XS.json").write_text(json.dumps({"dev_pass": True}))
    (tmp_path / "holdout_log.jsonl").write_text('{"reason": "x [R5: XS,D1]"}\n')
    probs, resume = R.preflight_holdout()
    assert resume and not any("unlock" in p for p in probs)
    (tmp_path / "holdout_log.jsonl").write_text('{"reason": "x [R5: XS,D1]"}\n{"reason": "x [R5 resume XS,D1]"}\n')
    probs, _ = R.preflight_holdout()
    assert any("already resumed" in p for p in probs)
    assert calls == []


def test_stitch_daily_rule():
    spot = pd.Series([1.0, 2.0, 3.0, 5.0], index=np.array([0, 1, 2, 4]) * DAY)
    perp = pd.Series([10.0, 11.0, 12.0], index=np.array([5, 6, 7]) * DAY)
    s = store.stitch_daily(spot, perp)
    assert list(s.index // DAY) == list(range(0, 8))
    assert list(s) == [1.0, 2.0, 3.0, 3.0, 5.0, 10.0, 11.0, 12.0]   # day 3 forward-filled; perp from D0 = day 5
    with pytest.raises(ValueError):
        store.stitch_daily(pd.Series([1.0, 2.0], index=np.array([0, 4]) * DAY), perp)



def regime_path(days, drift, seed, vol=0.01):
    """1m path with the given daily drift sequence (shared by all coins) plus coin-specific noise."""
    rng = np.random.default_rng(seed)
    lr = rng.normal(np.repeat(drift, 1440) / 1440, vol / np.sqrt(1440))
    c = 100 * np.exp(np.cumsum(lr))
    o = np.r_[100, c[:-1]]
    return list(zip(o, np.maximum(o, c) * 1.0001, np.minimum(o, c) * 0.9999, c))


def test_d1_shift_null_detects_planted_timing_edge():
    """Trending regimes: trend timing earns, the same positions shifted in time (one offset for all coins) do not."""
    days = 600

    def drift(seed):                                     # irregular regimes, independent per coin
        lens = np.random.default_rng(seed).integers(15, 40, 60)
        return np.concatenate([np.full(n, 0.03 if j % 2 == 0 else -0.03) for j, n in enumerate(lens)])[:days]

    tc = {s: R.coin_from_m1(s, make_m1(regime_path(days, drift(70 + i), 20 + i), T0), None)
          for i, s in enumerate(SYMBOLS)}
    w = R.Window("t", "dev", pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2022-01-01", tz="UTC") + pd.Timedelta(days=days))
    closes = {s: c.perp_close for s, c in tc.items()}
    subs = R.d1_build(tc, closes)
    parts, _ = R.d1_run(subs, tc, w)
    actual = float(R.d1_assemble(parts, w)[1].mean())
    nl = R.d1_shift_null(subs, tc, w, actual, n_draws=30)
    assert actual > 0 and nl["p_value"] < 0.05


def test_d1_mismatches_catch_a_broken_position(coins):
    closes = {s: c.perp_close for s, c in coins.items()}
    subs = R.d1_build(coins, closes)
    parts, _ = R.d1_run(subs, coins, W)
    assert R.d1_mismatches(subs, parts, coins, W) == 0
    key = next(k for k, (_, tr) in parts.items() if (tr["reason"] == "exit_signal").sum() >= 1)
    d, tr = parts[key]
    tr = tr.copy()
    i = tr.index[tr["reason"] == "exit_signal"][0]
    tr.loc[i, "exit_i"] = tr.loc[i, "exit_i"] + 3 * 1440          # pretend the engine exited 3 days late
    assert R.d1_mismatches(subs, {**parts, key: (d, tr)}, coins, W) >= 3
    ex = R.d1_exposure(subs, parts, coins, W)
    assert ex["mean_ratio"] is not None and 0.2 < ex["mean_ratio"] < 5


def test_xs_daily_book_stays_k_long_k_short():
    rng = np.random.default_rng(4)
    px = px_frame(100 * np.exp(np.cumsum(rng.normal(0, 0.03, (200, 5)), axis=0)))
    for cfg in X.CONFIGS[4:]:
        side = X.leg_plan(px, cfg)["side"]
        score, sigma, _ = X.features(px, cfg["L"])
        full = X.eligible(px, score, sigma).all(axis=1)
        assert ((side[full] == 1).sum(axis=1) == 2).all() and ((side[full] == -1).sum(axis=1) == 2).all()


def test_xs_plan_causal_under_truncation():
    rng = np.random.default_rng(6)
    px = px_frame(100 * np.exp(np.cumsum(rng.normal(0, 0.03, (220, 5)), axis=0)))
    for cfg in (X.CONFIGS[X.PRIMARY], X.CONFIGS[4]):
        full = X.leg_plan(px, cfg)
        for cut in (90, 150, 219):
            part = X.leg_plan(px.iloc[:cut], cfg)
            for k in ("side", "size", "stop"):
                assert np.allclose(np.nan_to_num(part[k].to_numpy()), np.nan_to_num(full[k].iloc[:cut].to_numpy()))


def test_spot_red_flags():
    base = {"status": "ok", "longest_gap_days": 0, "median_overlap_basis": 0.001, "stitch_jump": 0.002, "bad_prices": 0}
    rows = [base, {**base, "longest_gap_days": 3}, {**base, "median_overlap_basis": 0.006},
            {**base, "stitch_jump": -0.02}, {**base, "stitch_jump": None}, {**base, "bad_prices": 1},
            {**base, "status": "no_spot_1d"}, {**base, "median_overlap_basis": None}]
    flags = list(R.spot_red_flags(pd.DataFrame(rows))["red_flag"])
    assert flags == [False, True, True, True, True, True, True, False]


def test_d1_info_run_needs_the_committed_deviation_runs_once_and_leaves_xs_alone(tmp_path, monkeypatch):
    from btcq.research import round5 as R
    (tmp_path / "D1.json").write_text('{"dev_pass": false}')
    log = tmp_path / "holdout_log.jsonl"
    monkeypatch.setattr(R, "RESULTS", tmp_path)
    monkeypatch.setattr(R.store, "HOLDOUT_LOG", log)
    monkeypatch.setattr(R, "SYMBOLS", [])
    monkeypatch.setattr(R, "_git", lambda *a: "")
    assert any("not committed" in p for p in R.preflight_d1_info())
    monkeypatch.setattr(R, "_git", lambda *a: "## Deviation 2: D1 on 2024-2026")
    assert R.preflight_d1_info() == []
    log.write_text('{"reason": "x [R5 D1-info: D1 only, information; XS stays locked]"}\n')
    assert any("already unlocked" in p for p in R.preflight_d1_info())
    assert "[R5:" not in R.D1_INFO_TAG and "[R5 resume" not in R.D1_INFO_TAG    # XS's one unlock stays unused
