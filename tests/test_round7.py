"""Round 7 (RT): the rotation rule, the stratified null, gates, the data panel and the pre-registration guards.
Synthetic data only: no real price is loaded here."""
import json

import numpy as np
import pandas as pd
import pytest

from btcq.backtest import rotation_sim as RS
from btcq.backtest import stats
from btcq.data import store
from btcq.research import guards, round7 as r7, round7_calib as rc
from btcq.strategies import r7_rotation as R


def mk_panel(close, member=None, blocked=None, fill=None):
    close = np.asarray(close, float)
    n, k = close.shape
    days = pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC").as_unit("ms")
    f = np.zeros((n, k), bool)
    fill = np.vstack([close[:1], close[:-1]]) if fill is None else np.asarray(fill, float)
    member = np.ones((n, k), bool) if member is None else np.asarray(member, bool)
    return RS.Panel(days=days, symbols=[f"C{j}USDT" for j in range(k)], close=close, traded=np.isfinite(close),
                    fill_px=fill, fund_pre=np.zeros((n, k)), fund_post=np.zeros((n, k)), fx_pre=f.copy(),
                    fx_post=f.copy(), fx_px=np.full((n, k), np.nan), fund_fx=np.zeros((n, k)),
                    blocked=f.copy() if blocked is None else np.asarray(blocked, bool),
                    days_since_gap=np.full((n, k), np.inf), member=member, rank=np.where(member, 12, 99),
                    slip=np.full((n, k), 0.0003),
                    is_meme=np.zeros(k, bool), funding_filled_h=np.zeros(k))


def path_from_scores(score_rows, H=6, K=3):
    """sel for hand-written candidate scores (NaN = not a candidate), decisions 0..n-1."""
    sc = np.asarray(score_rows, float)
    nf = np.zeros(sc.shape, bool)
    return R.select_path(sc, nf, np.zeros(sc.shape, bool), K, H, 0, sc.shape[0] - 1)


# ---------------------------------------------------------------- the rule

def test_grid_primary_and_neighbours():
    assert len(R.CONFIGS) == 6 and R.CONFIGS[R.PRIMARY] == {"L": 14, "H": 6} and R.K == 3
    assert [R.CONFIGS[i] for i in R.NEIGHBOURS] == [{"L": 7, "H": 6}, {"L": 28, "H": 6}, {"L": 14, "H": 3}]


def test_top_three_absolute_momentum_and_cash():
    sel = path_from_scores([[0.5, 0.4, 0.3, 0.2, -0.1], [-0.1, -0.2, -0.3, -0.4, -0.5], [0.1, -0.2, np.nan, -0.4, 0.2]])
    np.testing.assert_array_equal(sel[0], [1, 1, 1, 0, 0])
    np.testing.assert_array_equal(sel[1], [0, 0, 0, 0, 0])       # nothing positive: all cash
    np.testing.assert_array_equal(sel[2], [1, 0, 0, 0, 1])       # 2 eligible: one slot stays cash


def test_hysteresis_keeps_a_holding_while_rank_is_within_H():
    rows = [[0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.8, 0.7, 0.9, 0.95, 0.85, 0.6, 0.5],     # C0 ranks 7th (> H=6) -> dropped; C1, C2 kept
            [0.1, 0.8, 0.5, 0.9, 0.95, 0.85, 0.6, 0.45]]    # C2 ranks 6th -> still kept
    sel = path_from_scores(rows, H=6)
    np.testing.assert_array_equal(np.flatnonzero(sel[1]), [1, 2, 4])
    np.testing.assert_array_equal(np.flatnonzero(sel[2]), [1, 2, 4])
    sel3 = path_from_scores(rows, H=3)
    np.testing.assert_array_equal(np.flatnonzero(sel3[1]), [3, 4, 5])


def test_ties_are_broken_by_symbol_order():
    sel = path_from_scores([[0.2, 0.3, 0.3, 0.3, 0.3]])
    np.testing.assert_array_equal(sel[0], [0, 1, 1, 1, 0])


def test_a_stuck_holding_is_kept_and_fills_a_slot():
    sc = np.array([[0.5, 0.4, 0.3, 0.2], [np.nan, 0.4, 0.3, 0.9], [0.5, 0.4, 0.3, 0.2]])
    nf = np.zeros(sc.shape, bool)
    nf[1, 0] = True                                     # C0 has no fill row on the next day
    sel = R.select_path(sc, nf, np.zeros(sc.shape, bool), 3, 6, 0, 1)
    np.testing.assert_array_equal(sel[1], [R.STUCK, 1, 1, 0])
    W = R.to_weights(sel, 3, 0, 1)
    assert np.isnan(W[2, 0]) and np.allclose(W[2, 1:], [1 / 3, 1 / 3, 0])
    blocked = np.zeros(sc.shape, bool)
    blocked[1, 0] = True                                # in a gap: it was force-closed, the slot is free
    sel = R.select_path(sc, nf, blocked, 3, 6, 0, 1)
    np.testing.assert_array_equal(sel[1], [0, 1, 1, 1])


@pytest.mark.parametrize("t", [45, 80, 117])
def test_the_rule_is_causal(t):
    """Everything after decision day t may change, except the two facts the rule may know about day t + 1:
    whether a fill row exists and the fill month's membership. The decisions up to t must not move."""
    rng = np.random.default_rng(3)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.04, (120, 12)), axis=0))
    p = mk_panel(close)
    p.blocked[rng.random(p.shape) < 0.03] = True
    W, cb, sel = R.path(p, {"L": 14, "H": 6}, 40, 119)
    q = mk_panel(close.copy(), blocked=p.blocked.copy())
    after = slice(t + 1, None)
    q.close[after] *= np.exp(rng.normal(0, 0.3, q.close[after].shape))
    q.fill_px[t + 2:] *= np.exp(rng.normal(0, 0.3, q.fill_px[t + 2:].shape))
    q.fill_px[t + 1] *= 1.37                                  # its value is unknown at t; only its existence
    q.traded[after] = rng.random(q.traded[after].shape) < 0.5
    q.blocked[after] = rng.random(q.blocked[after].shape) < 0.5
    q.days_since_gap[after] = rng.uniform(0, 30, q.days_since_gap[after].shape)
    q.member[t + 2:] = rng.random(q.member[t + 2:].shape) < 0.5
    W2, cb2, sel2 = R.path(q, {"L": 14, "H": 6}, 40, 119)
    np.testing.assert_array_equal(sel[:t + 1], sel2[:t + 1])
    np.testing.assert_array_equal(W[:t + 2], W2[:t + 2])
    np.testing.assert_array_equal(cb[:t + 2], cb2[:t + 2])
    st, st2 = (R.strata(R.ewma_vol(x.close), R.candidates(x, 14)) for x in (p, q))
    np.testing.assert_array_equal(st[:t + 1], st2[:t + 1])


def test_candidates_need_fill_day_membership_trading_and_a_clean_lookback():
    close = np.tile(np.linspace(100, 130, 40)[:, None], (1, 4))
    member = np.ones((40, 4), bool)
    member[31:, 1] = False                                   # C1 leaves the universe on day 31 (a new month)
    p = mk_panel(close, member=member)
    p.traded[20, 2] = False
    p.days_since_gap[25, 3] = 13.5
    p.days_since_gap[26, 3] = 14.0                           # the gap ended at 00:00 of day D-L+1: C_{D-L} is stale
    c = R.candidates(p, 14)
    assert c[29, 1] and not c[30, 1]                         # the decision on day 30 fills on day 31
    assert not c[20, 2] and c[21, 2]
    assert not c[25, 3] and R.candidates(p, 7)[25, 3]
    assert not c[26, 3] and R.candidates(p, 13)[26, 3]
    assert not c[:14].any() and not c[-1].any()


def test_month_boundary_membership_uses_the_fill_day():
    gu = pd.DataFrame({"month": ["2021-01", "2021-02"], "symbol": ["AUSDT", "BUSDT"], "rank": [5, 15]})
    days = pd.date_range("2021-01-30", periods=4, freq="D", tz="UTC").as_unit("ms")
    member, rank, slip = r7.membership_grid(gu, ["AUSDT", "BUSDT"], days)
    np.testing.assert_array_equal(member, [[1, 0], [1, 0], [0, 1], [0, 1]])
    np.testing.assert_allclose(slip, [[0.0002, 0.0005], [0.0002, 0.0005], [0.0005, 0.0003], [0.0005, 0.0003]])
    close = np.full((4, 2), 100.0)
    close[:, 1] = [100, 101, 102, 103]
    p = mk_panel(close, member=member)
    c = R.candidates(p, 1)
    assert c[1, 1] and not c[1, 0]                           # Jan 31's decision fills on Feb 1


def test_strata_split_candidates_by_volatility():
    vol = np.array([[0.01, 0.05, 0.03, 0.02, 0.04, 0.06, np.nan, 0.07]])
    cand = np.array([[1, 1, 1, 1, 1, 1, 1, 0]], bool)
    st = R.strata(vol, cand)
    np.testing.assert_array_equal(st[0], [0, 2, 1, 0, 1, 2, 3, -1])   # by vol: C0 C3 | C2 C4 | C1 C5


def test_permutation_keeps_the_scores_within_each_group():
    rng = np.random.default_rng(1)
    sc = rng.normal(0, 1, (50, 9))
    sc[rng.random(sc.shape) < 0.2] = np.nan
    st = np.where(np.isfinite(sc), np.arange(9) % 3, -1)
    out = R.permute_scores(sc, st, 0, 49)
    for t in range(50):
        for g in range(3):
            m = st[t] == g
            np.testing.assert_allclose(np.sort(out[t, m]), np.sort(sc[t, m]))
        assert (np.isnan(out[t]) == np.isnan(sc[t])).all()
        assert (out[t] > 0).sum() == (sc[t] > 0).sum()


def test_null_identity_reproduces_the_actual_statistic():
    rng = np.random.default_rng(4)
    p, first, last = rc.synthetic_panel(rng, "factor", n_days=300)
    cfg = {"L": 14, "H": 6}
    W, cb, _ = R.path(p, cfg, first, last)
    actual = R.actual_stat(p, cfg, first, last, 0.01)
    zero = RS.simulate(p, W, cb, first, last, taker=0.0, slip_mult=0.0, penalty=0.01)
    assert actual == pytest.approx(zero.daily.mean(), rel=1e-12)
    cand = R.candidates(p, 14)
    unique = np.where(cand, np.arange(p.shape[1])[None, :], -1)       # one coin per group: nothing moves
    res = R.null_test(p, cfg, first, last, 20, 1, strata_=unique)
    assert res["actual"] == actual
    np.testing.assert_array_equal(res["dist"], actual)                # bit for bit: ties count as >=
    assert res["p_value"] == 1.0


def test_null_flags_a_planted_edge_and_not_noise():
    rng = np.random.default_rng(11)
    p, first, last = rc.synthetic_panel(rng, "factor", edge=0.05)
    assert rc.calibration_p(p, first, last, 199, 3) < 0.05
    ps = [rc.calibration_p(*rc.synthetic_panel(np.random.default_rng([9, i]), "iid", n_days=500), 99, i)
          for i in range(12)]
    assert 0.2 < np.mean(ps) < 0.8


def test_calibration_does_not_depend_on_chunking():
    for si in (2, 3, 4):
        assert rc._cal_task(si, 5, 8, 19) == rc._cal_task(si, 5, 6, 19) + rc._cal_task(si, 6, 8, 19)
    with pytest.raises(ValueError, match="unknown"):
        rc.synthetic_panel(np.random.default_rng(1), "beta")


# ---------------------------------------------------------------- evaluation and gates

def test_ew_benchmark_masks_non_members():
    close = np.array([[100, 100], [110, 50], [121, 100]], float)
    member = np.array([[1, 1], [1, 0], [1, 1]], bool)
    p = mk_panel(close, member=member)
    b = r7.ew_benchmark(p, 1, 2)
    np.testing.assert_allclose(b.to_numpy(), [0.10, (0.10 + 1.0) / 2])


def fake_ev(**kw):
    ev = {"summary": {"trades": 100, "sharpe": 1.0, "mean_daily": 0.001, "max_dd": 0.2},
          "stress_2x_mean_daily": 1e-6, "null": {"p_value": 0.049}, "dsr_family": {"dsr": 0.95},
          "year_share": 0.65, "plateau": {"1": True, "5": True, "2": True},
          "breadth": {"qualifying": 10, "positive": 6, "share": 0.6}}
    for k, v in kw.items():
        ev[k] = v
    return ev


def test_dev_gates_pass_exactly_at_the_thresholds():
    assert all(r7.gates_dev(fake_ev()).values())
    bad = [fake_ev(summary={"trades": 99}), fake_ev(stress_2x_mean_daily=0.0), fake_ev(null={"p_value": 0.05}),
           fake_ev(dsr_family={"dsr": 0.9499}), fake_ev(year_share=0.6501), fake_ev(plateau={"1": True, "5": False}),
           fake_ev(breadth={"qualifying": 9, "positive": 9, "share": 1.0}),
           fake_ev(breadth={"qualifying": 20, "positive": 11, "share": 0.55})]
    for ev in bad:
        g = r7.gates_dev(ev)
        assert sum(not v for v in g.values()) == 1, g


def test_holdout_gates():
    evh = {"summary": {"mean_daily": 1e-5, "sharpe": 0.5, "max_dd": 0.3}}
    assert all(r7.gates_holdout(evh, 1.0, 0.3).values())
    assert not r7.gates_holdout(evh, 1.01, 0.3)["holdout Sharpe>=50% of dev"]
    assert not r7.gates_holdout(evh, 1.0, 0.299)["holdout worst drawdown<=dev bootstrap p95"]


def test_family_dsr_has_no_deflation_below_two_effective_trials():
    rng = np.random.default_rng(2)
    d = pd.Series(rng.normal(0.001, 0.02, 1300), index=pd.date_range("2020-06-01", periods=1300, tz="UTC"))
    one = r7._dsr(d, 1.6)
    assert one["dsr"] == pytest.approx(one["psr0"]) and r7._dsr(d, 6.0)["dsr"] < one["dsr"]
    same = np.tile(d.to_numpy()[:, None], (1, 6)) + rng.normal(0, 1e-6, (1300, 6))
    assert stats.effective_trials(same) < 1.01


def test_headline_needs_dev_and_holdout():
    d = pd.Series(np.random.default_rng(1).normal(0.003, 0.02, 2300))
    assert r7.headline(True, True, d, 18.85)["PASSES_IN_PRACTICE"]
    assert not r7.headline(True, False, d, 18.85)["PASSES_IN_PRACTICE"]
    assert not r7.headline(False, True, d, 18.85)["STRICT_TIER"]


# ---------------------------------------------------------------- the panel from a (synthetic) store

def synth_store(monkeypatch, syms=("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT"), days=200, delist=None):
    """A fake gated universe and store for the 2023-07..2023-12 window (dev period only)."""
    rng = np.random.default_rng(8)
    t0 = int(pd.Timestamp("2023-06-15", tz="UTC").value // 10**6)
    data = {}
    for i, s in enumerate(syms):
        n = days * 1440
        ts = t0 + 60_000 * np.arange(n, dtype=np.int64)
        c = 10.0 * np.exp(np.cumsum(rng.normal(0.00001 * i, 0.0015, n)))
        vol = np.ones(n)
        if s == delist:
            vol[n // 2:] = 0.0
        m1 = pd.DataFrame({"ts": ts, "open": np.r_[10.0, c[:-1]], "high": c * 1.001, "low": c * 0.999, "close": c,
                           "volume": vol, "quote_volume": c, "taker_buy_volume": vol / 2, "is_perp": True})
        f = pd.DataFrame({"ts": t0 + np.arange(days * 3, dtype=np.int64) * 8 * 3_600_000,
                          "rate": rng.normal(0.0001, 0.0002, days * 3)})
        data[s] = (m1, f)
    cut = int(pd.Timestamp("2024-01-01", tz="UTC").value // 10**6)
    monkeypatch.setattr(store, "load_1m", lambda period, s: data[s][0][data[s][0]["ts"] < cut].reset_index(drop=True))
    monkeypatch.setattr(store, "load_funding", lambda period, s: data[s][1][data[s][1]["ts"] < cut].reset_index(drop=True))
    months = [m.strftime("%Y-%m") for m in pd.date_range("2023-06-01", "2023-12-01", freq="MS")]
    gu = pd.DataFrame([{"month": m, "symbol": s, "rank": 5 + j, "gated": True, "is_major": False}
                       for m in months for j, s in enumerate(syms)])
    monkeypatch.setattr(r7, "gated_universe", lambda: gu)
    monkeypatch.setattr(r7, "meme_flags", lambda s: np.zeros(len(s), bool))
    return r7.Window("dev", "dev", pd.Timestamp("2023-09-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC"))


def test_build_panel_and_evaluate_end_to_end(monkeypatch):
    w = synth_store(monkeypatch, delist="CCCUSDT")
    p, info = r7.build_panel(w)
    first, last = r7.window_bounds(p, w)
    assert p.days[first] == w.start and p.days[last] == w.end - pd.Timedelta(days=1)
    assert info.set_index("symbol").loc["CCCUSDT", "delisted"]
    j = p.symbols.index("CCCUSDT")
    assert p.fx_pre[:, j].sum() + p.fx_post[:, j].sum() == 1
    monkeypatch.setattr(r7, "NULL_SEED", 1)
    ev = r7.evaluate(p, w, n_draws=20)
    assert ev["daily"].index[0] == w.start and ev["daily"].index[-1] == w.end - pd.Timedelta(days=1)
    assert ev["summary"]["trades"] == len(ev["sim"].episodes) and set(ev["plateau"]) == {"1", "5", "2"}
    assert 0 < ev["null"]["p_value"] <= 1 and len(ev["null"]["dist"]) == 20
    assert ev["stress_2x"]["fees_total"] > ev["summary"]["fees_total"]
    frame = r7.panel_frame(p)
    assert len(frame) == p.shape[0] * p.shape[1] and frame["day"].max() < pd.Timestamp("2024-01-01", tz="UTC")
    g = r7.gates_dev(ev)
    assert len(g) == 7


def test_a_member_without_data_is_a_hard_error(monkeypatch):
    w = synth_store(monkeypatch)
    real = store.load_1m
    monkeypatch.setattr(store, "load_1m", lambda period, s: real(period, s).iloc[:0] if s == "BBBUSDT"
                        else real(period, s))
    with pytest.raises(r7.HardError, match="BBBUSDT"):
        r7.build_panel(w)


# ---------------------------------------------------------------- data handling and guards

def test_quality_counts_extreme_bars_only_before_the_cut():
    ts = 1_700_000_000_000 // 60_000 * 60_000 + 60_000 * np.arange(6, dtype=np.int64)
    c = np.array([10, 10, 20, 20, 40, 40], float)
    perp = pd.DataFrame({"ts": ts, "open": np.r_[10, c[:-1]], "high": c, "low": np.r_[10, c[:-1]], "close": c,
                         "volume": np.ones(6)})
    fund = pd.DataFrame({"ts": pd.Series([], dtype="int64"), "rate": pd.Series([], dtype=float)})
    assert store._quality(perp, None, fund)["extreme_bars"] == 2
    q = store._quality(perp, None, fund, returns_before_ms=int(ts[3]))
    assert q["extreme_bars"] == 1 and "extreme_bars_before" in q


def test_data_report_reads_timestamps_and_volume_only(monkeypatch, tmp_path):
    d = tmp_path / "XXXUSDT"
    d.mkdir()
    ts = int(pd.Timestamp("2024-01-01", tz="UTC").value // 10**6) + 60_000 * np.arange(31 * 1440, dtype=np.int64)
    ts = ts[(ts < ts[0] + 10 * 86_400_000) | (ts >= ts[0] + 15 * 86_400_000)]     # 5 days missing: coverage 0.84
    pd.DataFrame({"ts": ts, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}).to_parquet(
        d / "perp_1m.parquet")
    pd.DataFrame({"ts": ts[::480], "rate": 0.0001}).to_parquet(d / "funding.parquet")
    (d / "build_report.json").write_text(json.dumps({"status": "ok", "rows": len(ts)}))
    monkeypatch.setattr(store, "symbol_dir", lambda s: tmp_path / s)
    gu = pd.DataFrame({"month": ["2024-01"], "symbol": ["XXXUSDT"], "rank": [8], "gated": [True],
                       "is_major": [False]})
    monkeypatch.setattr(r7, "gated_universe", lambda: gu)
    seen = []
    real = pd.read_parquet

    def spy(path, *a, **kw):
        seen.append((str(path), kw.get("columns")))
        return real(path, *a, **kw)
    monkeypatch.setattr(pd, "read_parquet", spy)
    df = r7.data_report()
    assert seen and all(cols is not None for _, cols in seen)
    for path, cols in seen:
        allowed = {"ts", "volume"} if path.endswith("perp_1m.parquet") else {"ts"}
        assert set(cols) <= allowed, (path, cols)
    row = df.iloc[0]
    assert row["holdout_member_months"] == 1 and bool(row["red_flag"])
    assert row["holdout_min_month_coverage"] == pytest.approx(26 / 31, abs=1e-3)
    assert row["holdout_ends_early"] and not row["holdout_archive_short"]     # data ends 2024-01-31: a delisting


def test_universe_sha_is_pinned(monkeypatch):
    monkeypatch.setattr(r7, "UNIVERSE_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="SHA256"):
        r7.gated_universe()


def test_registered_universe_is_the_non_majors():
    gu = r7.gated_universe()
    sizes = gu[gu["month"].isin(r7.window_months(r7.DEV))].groupby("month").size()
    assert sizes.min() >= 15 and sizes.max() <= 16
    assert not set(gu["symbol"]) & {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT"}


def test_structured_unlock_records(monkeypatch, tmp_path):
    log = tmp_path / "holdout_log.jsonl"
    monkeypatch.setattr(store, "HOLDOUT_LOG", log)
    monkeypatch.setattr(store, "_HOLDOUT_UNLOCKED", False)
    monkeypatch.setattr(r7, "RESULTS", tmp_path)
    monkeypatch.setattr(guards, "clean_src", lambda: None)
    log.write_text(json.dumps({"utc": "x", "reason": "old [R6: M1 v1]"}) + "\n")
    store.unlock_holdout("Round 7 holdout [R7: RT v1]", round=7, kind="unlock", version=1)
    with pytest.raises(PermissionError, match="no Round 7 holdout verdict"):
        r7.verify_unlock("before the verdict")              # checks may never come before the verdict
    (tmp_path / "RT.json").write_text(json.dumps({"holdout_gates": {"x": False}}))
    r7.verify_unlock("resim")
    recs = r7.r7_unlock_records()
    assert [r["kind"] for r in recs] == ["unlock", "verify"] and recs[0]["version"] == 1
    assert "[R7 verify resim]" in recs[1]["reason"]


def fake_rt(**kw):
    return {"dev_pass": True, "gates": {"a": True}, "version": 1, "code_sha256": r7.code_sha(),
            "universe_sha256": r7.UNIVERSE_SHA256, "manifest_sha256": None, "f1_daily_sha256": None,
            "holdout_dd_limit": 0.4, "prereg_section_sha256": "sec", **r7.lib_versions(), **kw}


def preflight_env(monkeypatch, tmp_path, rt: dict, log_lines=(), verdict=True):
    res = tmp_path / "results"
    res.mkdir()
    (res / "RT.json").write_text(json.dumps(rt))
    monkeypatch.setattr(r7, "RESULTS", res)
    monkeypatch.setattr(r7, "MANIFEST", tmp_path / "missing_manifest.json")
    monkeypatch.setattr(r7, "F1_DEV_DAILY", tmp_path / "missing_f1.parquet")
    monkeypatch.setattr(r7, "verify_manifest", lambda: [])
    monkeypatch.setattr(guards, "require_committed", lambda paths: None)
    log = tmp_path / "holdout_log.jsonl"
    log.write_text("".join(json.dumps(x) + "\n" for x in log_lines))
    monkeypatch.setattr(store, "HOLDOUT_LOG", log)
    runs = tmp_path / "runs.jsonl"
    rec = {"kind": "r7-sweep", "version": 1, "verdict": True, "code_sha256": rt["code_sha256"],
           "holdout_dd_limit": rt["holdout_dd_limit"]}
    runs.write_text(json.dumps(rec) + "\n" if verdict else "")
    monkeypatch.setattr(r7, "RUN_LOG", runs)
    monkeypatch.setattr(guards, "committed_versions", lambda q: [q.read_text()] if q == runs else [])
    monkeypatch.setattr(guards, "committed_section", lambda *a: "")
    monkeypatch.setattr(guards, "sha_str", lambda x: "sec")


def test_preflight_allows_one_unlock_and_one_resume(monkeypatch, tmp_path):
    preflight_env(monkeypatch, tmp_path, fake_rt())
    assert r7.preflight_holdout() == ([], False)
    u = {"round": 7, "kind": "unlock", "version": 1, "reason": "x [R7: RT v1]"}
    preflight_env_2 = tmp_path / "b"
    preflight_env_2.mkdir()
    preflight_env(monkeypatch, preflight_env_2, fake_rt(), [u])
    assert r7.preflight_holdout() == ([], True)                      # crashed after the unlock: one resume
    c = tmp_path / "c"
    c.mkdir()
    preflight_env(monkeypatch, c, fake_rt(), [u, u | {"kind": "resume"}])
    probs, _ = r7.preflight_holdout()
    assert any("already resumed" in x for x in probs)
    d = tmp_path / "d"
    d.mkdir()
    preflight_env(monkeypatch, d, fake_rt(holdout={}), [u])
    assert any("already evaluated" in x for x in r7.preflight_holdout()[0])


def test_preflight_refuses_a_dev_fail_or_changed_code(monkeypatch, tmp_path):
    preflight_env(monkeypatch, tmp_path, fake_rt(dev_pass=False, code_sha256="x"))
    probs, _ = r7.preflight_holdout()
    assert any("did not pass dev" in x for x in probs) and any("code changed" in x for x in probs)


def test_preflight_needs_the_committed_verdict_and_consistent_gates(monkeypatch, tmp_path):
    preflight_env(monkeypatch, tmp_path, fake_rt(), verdict=False)
    assert any("no committed r7-sweep pass record" in x for x in r7.preflight_holdout()[0])
    b = tmp_path / "b"
    b.mkdir()
    preflight_env(monkeypatch, b, fake_rt(gates={"a": True, "b": False}))
    assert any("does not match its gates" in x for x in r7.preflight_holdout()[0])
    c = tmp_path / "c"
    c.mkdir()
    preflight_env(monkeypatch, c, fake_rt(numba_version="0.0"))
    assert any("versions changed" in x for x in r7.preflight_holdout()[0])


def test_a_validate_start_record_counts_as_an_unlock(monkeypatch, tmp_path):
    preflight_env(monkeypatch, tmp_path, fake_rt())
    with r7.RUN_LOG.open("a") as f:                         # the holdout log was restored, the run log wasn't
        f.write(json.dumps({"kind": "r7-validate-start", "version": 1, "resume": False}) + "\n")
    assert r7.preflight_holdout() == ([], True)
    with r7.RUN_LOG.open("a") as f:
        f.write(json.dumps({"kind": "r7-validate-start", "version": 1, "resume": True}) + "\n")
    assert any("already resumed" in x for x in r7.preflight_holdout()[0])


def sweep_env(monkeypatch, tmp_path):
    """A sandbox for cmd_sweep; returns (run log, panel-build calls, the committed start records, the pins)."""
    monkeypatch.setattr(r7, "_require_prereg", lambda: "")
    monkeypatch.setattr(r7, "_require_data_report", lambda: None)
    monkeypatch.setattr(guards, "clean_src", lambda: None)
    res = tmp_path / "results"
    res.mkdir(exist_ok=True)
    monkeypatch.setattr(r7, "RESULTS", res)
    log = tmp_path / "runs.jsonl"
    monkeypatch.setattr(r7, "RUN_LOG", log)
    monkeypatch.setattr(store, "HOLDOUT_LOG", tmp_path / "holdout_log.jsonl")
    for f in ("MANIFEST", "UNIVERSE_CSV", "F1_DEV_DAILY"):
        pth = tmp_path / f.lower()
        pth.write_text(f)
        monkeypatch.setattr(r7, f, pth)
    committed = []
    real = guards.log_records

    def log_records(path, kind, *, committed_only=False, **match):
        mine = [r for r in committed if r.get("kind") == kind and all(r.get(k) == v for k, v in match.items())]
        return mine if committed_only else real(path, kind, committed_only=False, **match) + mine
    monkeypatch.setattr(guards, "log_records", log_records)
    monkeypatch.setattr(guards, "committed_versions", lambda p: [])
    calls = []

    def build(w):
        calls.append(w)
        raise RuntimeError("ran")
    monkeypatch.setattr(r7, "build_panel", build)
    pins = {"kind": "r7-sweep-start", "version": 1, **r7.sweep_pins("")}
    return log, calls, committed, pins


def test_sweep_first_writes_a_start_record_and_stops(monkeypatch, tmp_path):
    log, calls, committed, pins = sweep_env(monkeypatch, tmp_path)
    with pytest.raises(SystemExit, match="start record"):
        r7.cmd_sweep()
    recs = [json.loads(x) for x in log.read_text().splitlines()]
    assert [r["kind"] for r in recs] == ["r7-sweep-start"]
    assert all(recs[0][k] == v for k, v in pins.items())
    with pytest.raises(SystemExit, match="start record"):            # not committed yet: still no run
        r7.cmd_sweep()
    assert len(log.read_text().splitlines()) == 1 and not calls


def test_sweep_runs_only_after_the_committed_start_and_with_the_same_pins(monkeypatch, tmp_path):
    log, calls, committed, pins = sweep_env(monkeypatch, tmp_path)
    committed.append(pins)
    with pytest.raises(SystemExit, match="hard error, logged"):     # any failure before the verdict is logged
        r7.cmd_sweep()
    assert len(calls) == 1
    kinds = [json.loads(x)["kind"] for x in log.read_text().splitlines()]
    assert kinds == ["r7-sweep-run", "r7-sweep"]
    with pytest.raises(SystemExit, match="hard error|already ran"):  # the same version never runs twice
        r7.cmd_sweep()
    log.write_text("")
    committed[0] = pins | {"code_sha256": "other"}
    with pytest.raises(SystemExit, match="different code_sha256"):
        r7.cmd_sweep()
    committed[0] = pins | {"manifest_sha256": "other"}
    with pytest.raises(SystemExit, match="different manifest_sha256"):
        r7.cmd_sweep()
    committed[0] = pins | {"numba_version": "0.0"}
    with pytest.raises(SystemExit, match="different numba_version"):
        r7.cmd_sweep()
    assert len(calls) == 1


def test_sweep_runs_once(monkeypatch, tmp_path):
    sweep_env(monkeypatch, tmp_path)
    (r7.RESULTS / "RT.json").write_text(json.dumps({"version": 1}))
    with pytest.raises(SystemExit, match="runs once"):
        r7.cmd_sweep()
    with pytest.raises(SystemExit, match="version bump"):
        r7.cmd_sweep(rerun=True)
    (r7.RESULTS / "RT.json").write_text(json.dumps({"version": 1, "holdout": {}}))
    with pytest.raises(SystemExit, match="closed"):
        r7.cmd_sweep(rerun=True)


def test_registration_section_guard(monkeypatch, tmp_path):
    text = "# x\n\n## Round 6\nold\n\n## Round 7 (RT)\nNull and Dev gates {N_EFF}\n\n## Later\n"
    assert guards.section(text, "## Round 7").startswith("## Round 7 (RT)")
    assert "Later" not in guards.section(text, "## Round 7")
    monkeypatch.setattr(guards, "require_committed", lambda paths: None)
    monkeypatch.setattr(guards, "git", lambda *a, **k: text)
    with pytest.raises(SystemExit, match="placeholders"):
        guards.require_section(r7.PREREG, "## Round 7", ("Null",))
    with pytest.raises(SystemExit, match="lacks"):
        guards.require_section(r7.PREREG, "## Round 7", ("Power table",))


def test_code_files_exist_and_the_cli_knows_round7():
    for f in r7.CODE_FILES:
        assert (r7.ROOT / f).exists(), f
    import inspect
    from btcq.research import run
    src = inspect.getsource(run.main)
    for c in ("r7-power", "r7-calibrate", "r7-download", "r7-sweep", "r7-validate"):
        assert c in src


def test_a_rerun_after_a_hard_error_needs_no_trial_returns(monkeypatch, tmp_path):
    log, calls, committed, pins = sweep_env(monkeypatch, tmp_path)
    log.write_text(json.dumps({"kind": "r7-sweep", "version": 1, "hard_error": "x"}) + "\n")
    assert r7.family_prior(2) is None                         # v1 stopped before a verdict: no family members
    log.write_text(json.dumps({"kind": "r7-sweep", "version": 1, "verdict": False}) + "\n")
    monkeypatch.setattr(guards, "require_committed", lambda paths: (_ for _ in ()).throw(SystemExit("commit")))
    with pytest.raises(SystemExit, match="commit"):           # v1 has a verdict: its trial returns must be committed
        r7.family_prior(2)


def test_preflight_refuses_after_a_recorded_holdout_failure(monkeypatch, tmp_path):
    preflight_env(monkeypatch, tmp_path, fake_rt())
    with r7.RUN_LOG.open("a") as f:                         # RT.json restored to dev after a recorded FAIL
        f.write(json.dumps({"kind": "r7-validate", "version": 1, "hard_error": "MemoryError"}) + "\n")
    assert any("recorded holdout failure" in x for x in r7.preflight_holdout()[0])


def test_sweep_refuses_a_version_that_already_ran(monkeypatch, tmp_path):
    log, calls, committed, pins = sweep_env(monkeypatch, tmp_path)
    committed.append(pins)
    log.write_text(json.dumps({"kind": "r7-sweep-run", "version": 1}) + "\n")     # e.g. killed mid-run
    with pytest.raises(SystemExit, match="already ran"):
        r7.cmd_sweep()
    assert not calls


def validate_env(monkeypatch, tmp_path, fail=None):
    res = tmp_path / "results"
    res.mkdir()
    monkeypatch.setattr(r7, "RESULTS", res)
    monkeypatch.setattr(r7, "R7_DIR", tmp_path)
    monkeypatch.setattr(r7, "RUN_LOG", tmp_path / "runs.jsonl")
    monkeypatch.setattr(r7, "preflight_holdout", lambda: ([], False))
    monkeypatch.setattr(guards, "clean_src", lambda: None)
    rt = {"version": 1, "dev_pass": True, "dev": {"sharpe": 1.0}, "holdout_dd_limit": 0.5, "code_sha256": "c",
          "n_eff_family": 1.2, "n_strict": 18.85}
    (res / "RT.json").write_text(json.dumps(rt))
    days = pd.date_range("2020-06-01", "2023-12-31", tz="UTC").as_unit("ms")
    pd.DataFrame({"ret": 0.001}, index=days).to_parquet(res / "RT_dev_daily.parquet")
    unlocks, order = [], []
    monkeypatch.setattr(store, "unlock_holdout", lambda reason, **kw: unlocks.append((reason, kw)))
    hd = pd.date_range("2024-01-01", "2026-09-30", tz="UTC").as_unit("ms")

    def build(w):
        order.append("load")
        if fail:
            raise fail
        return "panel", pd.DataFrame({"symbol": ["X"]})
    monkeypatch.setattr(r7, "build_panel", build)
    evh = {"summary": {"mean_daily": 0.001, "sharpe": 0.9, "max_dd": 0.2, "total_return": 1.0}, "coins": [],
           "daily": pd.Series(0.001, index=hd)}
    monkeypatch.setattr(r7, "evaluate", lambda p, w, with_null: evh)

    def write(tag, ev, p):
        order.append("write:" + ("verdict saved" if "holdout_gates" in json.loads((res / "RT.json").read_text())
                                 else "no verdict"))
    monkeypatch.setattr(r7, "_write", write)
    monkeypatch.setattr(r7, "without_best_coin", lambda *a: {})
    return res, unlocks, order


def test_validate_unlocks_once_and_saves_the_verdict_first(monkeypatch, tmp_path):
    res, unlocks, order = validate_env(monkeypatch, tmp_path)
    r = r7.cmd_validate("Round 7 holdout (pre-registered)")
    assert len(unlocks) == 1 and unlocks[0][0].endswith("[R7: RT v1]") and unlocks[0][1]["kind"] == "unlock"
    assert order == ["load", "write:verdict saved"]
    assert r["final_pass"] is True and all(r["holdout_gates"].values())
    kinds = [json.loads(x)["kind"] for x in (tmp_path / "runs.jsonl").read_text().splitlines()]
    assert kinds == ["r7-validate-start", "r7-validate"]


def test_any_error_after_the_unlock_is_a_recorded_fail(monkeypatch, tmp_path):
    res, unlocks, order = validate_env(monkeypatch, tmp_path, fail=KeyError("a coin"))
    with pytest.raises(SystemExit, match="recorded as a FAIL"):
        r7.cmd_validate("x")
    r = json.loads((res / "RT.json").read_text())
    assert r["final_pass"] is False and "KeyError" in r["holdout"]["integrity_error"] and len(unlocks) == 1


def test_a_member_that_never_traded_is_a_red_flag(monkeypatch, tmp_path):
    d = tmp_path / "ZZZUSDT"
    d.mkdir()
    ts = int(pd.Timestamp("2024-01-01", tz="UTC").value // 10**6) + 60_000 * np.arange(31 * 1440, dtype=np.int64)
    pd.DataFrame({"ts": ts, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 0.0}).to_parquet(
        d / "perp_1m.parquet")
    pd.DataFrame({"ts": ts[::480], "rate": 0.0001}).to_parquet(d / "funding.parquet")
    (d / "build_report.json").write_text(json.dumps({"status": "ok"}))
    monkeypatch.setattr(store, "symbol_dir", lambda s: tmp_path / s)
    gu = pd.DataFrame({"month": ["2024-01"], "symbol": ["ZZZUSDT"], "rank": [8], "gated": [True],
                       "is_major": [False]})
    monkeypatch.setattr(r7, "gated_universe", lambda: gu)
    df = r7.data_report()
    assert bool(df["red_flag"].iloc[0]) and not df["holdout_traded_in_member_months"].iloc[0]
    edited = df.assign(red_flag=False)
    assert len(r7.red_flags(edited)) == 1                     # recomputed: a hand-edited column can't hide it


def registered_calibration(**over):
    rows = [{"scenario": sc, "kind": kind, "planted_edge": edge, "reps": reps, "draws": rc.CAL_DRAWS,
             "ok": True if kind == "no-edge" else None, "calib_sha256": rc.calib_sha()}
            for sc, edge, reps, kind in rc.CAL_SCENARIOS]
    df = pd.DataFrame(rows)
    for k, v in over.items():
        df.loc[1, k] = v
    return df


def test_prereg_needs_the_registered_passing_calibration(monkeypatch, tmp_path):
    cal = tmp_path / "cal.csv"
    monkeypatch.setattr(r7, "CALIB_CSV", cal)
    monkeypatch.setattr(guards, "require_committed", lambda paths: None)
    monkeypatch.setattr(guards, "require_section", lambda *a: "section text")
    for over, msg in (({"ok": False}, "above its limit"), ({"reps": 999}, "registered scenarios"),
                      ({"calib_sha256": "x"}, "other calibration code")):
        registered_calibration(**over).to_csv(cal, index=False)
        with pytest.raises(SystemExit, match=msg):
            r7._require_prereg()
    registered_calibration().to_csv(cal, index=False)
    assert r7._require_prereg() == guards.sha_str("section text")


def test_manifest_pins_the_report_and_the_universe(monkeypatch, tmp_path):
    rep_csv, man = tmp_path / "report.csv", tmp_path / "manifest.json"
    rep_csv.write_text("symbol,red_flag\nAUSDT,False\n")
    monkeypatch.setattr(r7, "DATA_REPORT", rep_csv)
    monkeypatch.setattr(r7, "MANIFEST", man)
    r7.write_manifest(pd.DataFrame({"symbol": []}))
    assert r7.verify_manifest() == []
    rep_csv.write_text("symbol,red_flag\nAUSDT,False\nBUSDT,False\n")
    assert any("data report changed" in x for x in r7.verify_manifest())


def test_no_edge_generators_have_equal_expected_returns():
    """Pooled over many panels, the mean simple return of high- and low-volatility coins is the same (the
    calibration's no-edge panels must not carry a volatility or beta premium)."""
    for sc in ("factor", "garch", "cluster"):
        hi, lo = [], []
        for i in range(30):
            p, first, last = rc.synthetic_panel(np.random.default_rng([5, i]), sc)
            r = p.close[1:] / p.close[:-1] - 1
            v = np.nanstd(r, axis=0)
            order = np.argsort(v)
            lo.append(np.nanmean(r[:, order[:5]]))
            hi.append(np.nanmean(r[:, order[-5:]]))
        diff = np.mean(hi) - np.mean(lo)
        se = np.std(np.subtract(hi, lo), ddof=1) / np.sqrt(len(hi))
        assert abs(diff) < 4 * se + 2e-4, (sc, diff, se)

