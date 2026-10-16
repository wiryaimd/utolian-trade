"""Round 4 runner: membership masks, the exact position cap, slot accounting, gates, guards."""
import json

import numpy as np
import pandas as pd
import pytest

from btcq.backtest import engine, stats
from btcq.config import Costs
from btcq.research import portfolio, registry, round4
from btcq.research.round4 import SlimMarket, Sleeve, Spec, Stage
from conftest import T0, flat, make_m1, needs_data

SPEC0 = Spec(slots=2, latency_min=0, taker=0.0, maker=0.0, tier_slip=(0.0, 0.0), nonmember_slip=0.0,
             delist_penalty=0.0)


def sleeve(sym, closes, entry, exit_long=None, member=None, rank=None, stop=1.0):
    """15m-bar sleeve on a synthetic path: one 15-minute flat block per close value."""
    paths = [x for c in closes for x in flat(c, 15)]
    mk = engine.Market(make_m1(paths), None, synthetic_funding=0.0)
    ex = mk.exec_idx("15m", 0)
    n = len(ex)
    e = np.asarray(entry, np.int8)
    s = np.where(e != 0, stop, np.nan)
    nanv = np.full(n, np.nan)
    xl = np.zeros(n, np.bool_) if exit_long is None else np.asarray(exit_long, np.bool_)
    arrays = (e, s, nanv, nanv, xl, np.zeros(n, np.bool_), nanv, nanv)
    slim = SlimMarket(mk.ts, mk.o, mk.h, mk.l, mk.c, mk.fund, mk.day_marks, mk.days, ex, "15m", 0)
    member = np.ones(n, np.bool_) if member is None else np.asarray(member, np.bool_)
    rank = np.ones(n, np.int64) if rank is None else np.asarray(rank, np.int64)
    bar_time = mk.bars("15m")["ts"].to_numpy(np.int64) + 15 * 60_000
    return Sleeve(sym, slim, {"U1": arrays}, 0, member, rank, bar_time, np.zeros(len(mk.ts)),
                  np.zeros(len(mk.ts), np.bool_), False, False, False, 1)


def test_membership_blocks_entries_but_not_exits():
    n = 8
    e = np.zeros(n, np.int8)
    e[0] = e[5] = 1
    xl = np.zeros(n, np.bool_)
    xl[3] = True
    member = np.array([1, 1, 0, 0, 0, 0, 0, 0], bool)   # leaves after bar 1
    sl = sleeve("AUSDT", [100] * n, e, xl, member)
    t = round4.run_sleeve(sl, "U1", set(), SPEC0).trades
    assert list(t["reason"]) == ["exit_signal"]           # bar-5 entry blocked, bar-3 exit still taken
    assert t["entry_i"].iloc[0] == 15 and t["exit_i"].iloc[0] == 60


def greedy_reference(sleeves, slots):
    """Step-by-step shared account: at each bar, signal exits first, then entries by (rank, symbol)."""
    n = len(next(iter(sleeves.values())).sigs["U1"][0])
    open_ = {}
    trades = []
    for b in range(n - 1):                # the last bar's decision never executes
        for s, sl in sleeves.items():
            if s in open_ and sl.sigs["U1"][4][b]:
                trades.append((s, open_.pop(s), b))
        cands = sorted((sl.bar_rank[b], s) for s, sl in sleeves.items()
                       if s not in open_ and sl.sigs["U1"][0][b] != 0 and sl.member_bar[b])
        for _, s in cands[:max(slots - len(open_), 0)]:
            open_[s] = b
    trades += [(s, b0, None) for s, b0 in open_.items()]
    return sorted(trades, key=lambda x: (x[0], x[1]))


@pytest.mark.parametrize("seed", range(6))
def test_cap_pass_equals_step_by_step_shared_account(seed):
    rng = np.random.default_rng(seed)
    n, k = 60, 9
    sleeves = {}
    for i in range(k):
        e = (rng.random(n) < 0.25).astype(np.int8)
        xl = rng.random(n) < 0.3
        member = rng.random(n) < 0.85
        sleeves[f"S{i}USDT"] = sleeve(f"S{i}USDT", [100] * n, e, xl, member, rng.integers(1, 21, n))
    spec = Spec(**{**SPEC0.__dict__, "slots": 3})
    results, masks, skipped, it = round4.cap_pass(sleeves, "U1", spec)
    got = []
    for s, r in results.items():
        for _, t in r.trades.iterrows():
            got.append((s, int(t["entry_tf"]), None if t["reason"] == "end" else int(t["exit_i"]) // 15 - 1))
    assert sorted(got, key=lambda x: (x[0], x[1])) == greedy_reference(sleeves, 3)
    ev = round4._slot_events(results, sleeves)
    assert round4.first_violation(ev, 3) is None
    assert round4.concurrency(ev, 3, pd.Timestamp(T0, unit="ms", tz="UTC"),
                              pd.Timestamp(T0 + n * 15 * 60_000, unit="ms", tz="UTC"))["max"] <= 3
    if len(skipped):
        assert it >= 1


def test_assemble_slots_sums_and_divides():
    cal = pd.date_range("2022-01-01", periods=3, freq="D", tz="UTC").as_unit("ms")
    a = pd.Series([0.02, 0.0, -0.01], index=cal)
    b = pd.Series([0.04], index=cal[1:2])
    ta = pd.DataFrame({"entry_time": [cal[0]], "symbol": ["A"], "ret": [0.02], "R": [1.0]})
    tb = pd.DataFrame({"entry_time": [cal[1]], "symbol": ["B"], "ret": [0.04], "R": [2.0]})
    res, daily = portfolio.assemble_slots([(a, ta), (b, tb)], 20, cal)
    assert daily.tolist() == pytest.approx([0.001, 0.002, -0.0005])
    assert res.trades["ret"].tolist() == pytest.approx([0.001, 0.002])
    assert res.equity.iloc[-1] == pytest.approx(10_000 * 1.001 * 1.002 * 0.9995)


def test_weekly_t_and_beta_alpha():
    days = pd.date_range("2022-01-03", periods=21, freq="D", tz="UTC")   # Monday..Sunday x3
    d = pd.Series(0.0, index=days)
    d.iloc[[0, 7, 14]] = [0.02, -0.01, 0.03]
    w = stats.weekly_returns(d)
    assert w.tolist() == pytest.approx([0.02, -0.01, 0.03])
    wt = stats.weekly_t(d)
    x = np.array([0.02, -0.01, 0.03])
    assert wt["t"] == pytest.approx(x.mean() / x.std(ddof=1) * np.sqrt(3)) and wt["n"] == 3
    rng = np.random.default_rng(0)
    xb = rng.normal(0, 0.02, 500)
    e = rng.normal(0, 0.01, 500)
    e = e - e.mean() - np.cov(e, xb)[0, 1] / xb.var(ddof=1) * (xb - xb.mean())   # orthogonal to xb
    ba = stats.beta_alpha(0.5 * xb + e + 0.001, xb)
    assert ba["beta"] == pytest.approx(0.5, abs=1e-9) and ba["alpha_ann"] == pytest.approx(0.365, rel=1e-6)


def test_null_candidates_come_only_from_member_bars(monkeypatch):
    n = 40
    e = np.zeros(n, np.int8)
    e[2] = 1
    xl = np.zeros(n, np.bool_)
    xl[5] = True
    member = np.r_[np.ones(10, bool), np.zeros(n - 10, bool)]
    sl = sleeve("AUSDT", [100] * n, e, xl, member)
    trades = round4.run_sleeve(sl, "U1", set(), SPEC0).trades
    seen = {}

    def fake(o, cand, *a, **k):
        seen["cand"] = np.asarray(cand)
        return np.zeros(3)

    monkeypatch.setattr(round4.stats, "random_entry_null", fake)
    round4._null_for(sl, trades, SPEC0)
    assert set(seen["cand"]) == set(sl.mk._ex[:10])


def ev_stub(**over):
    ev = {"summary": {"trades": 120, "trades_per_year": 60, "pf": 1.2, "max_year_share": 0.5},
          "weekly": {"t": 2.0, "sharpe_ann": 1.0}, "stress_2x": {"avg_R": 0.01},
          "worst_minute": {"avg_R": 0.01}, "null": {"p_value": 0.049}, "consistency": {"share": 0.6}}
    for k, v in over.items():
        ev[k] = {**ev[k], **v} if isinstance(v, dict) else v
    return ev


def test_stage_a_gates_at_their_edges():
    s1 = {"pbo": 0.348, "plateau": 1.0}
    assert all(round4.gates_a(ev_stub(), s1).values())
    assert not round4.gates_a(ev_stub(weekly={"t": 1.999}), s1)["weekly t>=2"]
    assert not round4.gates_a(ev_stub(null={"p_value": 0.05}), s1)["null p<0.05"]
    assert not round4.gates_a(ev_stub(consistency={"share": 0.59}), s1)[">=60% symbols (10+ trades) R>0"]
    assert not round4.gates_a(ev_stub(summary={"trades_per_year": 51.9}), s1)["trades/yr>=52"]
    assert not round4.gates_a(ev_stub(worst_minute={"avg_R": 0.0}), s1)["worst-minute stops avg_R>0"]


def test_headline_needs_both_stages_and_psr():
    rng = np.random.default_rng(1)
    days = pd.date_range("2020-04-01", periods=2000, freq="D", tz="UTC")
    strong = pd.Series(rng.normal(0.004, 0.02, len(days)), index=days)
    h = round4.headline(strong, 14.85, True, True)
    assert h["REPLICATED"] and h["psr0"] > 0.95 and h["STRICT"] == (h["dsr"] >= 0.95)
    assert not round4.headline(strong, 14.85, False, True)["REPLICATED"]
    weak = pd.Series(rng.normal(0.0, 0.02, len(days)), index=days)
    assert not round4.headline(weak, 14.85, True, True)["REPLICATED"]


def test_preflight_blocks_unlock(tmp_path, monkeypatch):
    monkeypatch.setattr(round4, "RESULTS", tmp_path)
    monkeypatch.setattr(round4.store, "HOLDOUT_LOG", tmp_path / "holdout_log.jsonl")
    monkeypatch.setattr(round4.U, "ROUND4_DIR", tmp_path)
    calls = []
    monkeypatch.setattr(round4.store, "unlock_holdout", lambda reason: calls.append(reason))
    (tmp_path / "U1.json").write_text(json.dumps({"dev_pass": False}))
    with pytest.raises(SystemExit, match="did not pass Stage A"):
        round4.cmd_validate("x")
    (tmp_path / "U1.json").write_text(json.dumps({"dev_pass": True, "holdout": {}}))
    probs, resume = round4.preflight_b()
    assert any("already evaluated" in p for p in probs) and any("U1L.json missing" in p for p in probs)
    assert any("U1_dev_benchmark.parquet missing" in p for p in probs)
    assert any("data report missing" in p for p in probs) and not resume
    # a logged unlock that never produced results may be resumed once, and only once
    for c in ("U1", "U1L"):
        (tmp_path / f"{c}.json").write_text(json.dumps({"dev_pass": True}))
    (tmp_path / "holdout_log.jsonl").write_text('{"reason": "x [R4: U1,U1L]"}\n')
    probs, resume = round4.preflight_b()
    assert resume and not any("unlock" in p for p in probs)
    (tmp_path / "holdout_log.jsonl").write_text('{"reason": "x [R4: U1,U1L]"}\n{"reason": "x [R4 resume U1,U1L]"}\n')
    probs, resume = round4.preflight_b()
    assert any("already resumed" in p for p in probs)
    assert calls == []


def test_registry_replaces_rows_with_the_same_key(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "PATH", tmp_path / "trials.csv")
    row = lambda code: {"strategy": code, "version": 1, "params": {"n": 40, "universe": "x"}, "period": "dev",  # noqa: E731
                        "sr": 0.05, "T": 1000, "skew": 1.0, "kurt": 9.0, "trades": 500, "sharpe_ann": 0.9}
    registry.record([row("F1")])
    registry.record([row("U1"), row("U1L")])
    assert len(registry.load()) == 3
    registry.record([row("U1"), row("U1L")])
    assert len(registry.load()) == 3


def trend_path(days, seed=0):
    rng = np.random.default_rng(seed)
    n = days * 1440
    steps = rng.normal(0, 0.0007, n) + 0.00003 * np.sin(np.arange(n) / (1440 * 9))
    c = 100 * np.exp(np.cumsum(steps))
    o = np.r_[100, c[:-1]]
    return [(oo, max(oo, cc) * 1.0002, min(oo, cc) * 0.9998, cc) for oo, cc in zip(o, c)]


def test_make_sleeve_slicing_matches_a_full_run(monkeypatch):
    """The cut-down sleeve must produce exactly the trades of a full-market run with entries
    masked outside member months."""
    t0 = int(pd.Timestamp("2021-01-01", tz="UTC").value // 10**6)
    m1 = make_m1(trend_path(150), t0)
    mk = engine.Market(m1, None, synthetic_funding=Costs.synthetic_funding)
    st = Stage("T", "dev", pd.Timestamp("2021-02-01", tz="UTC"), pd.Timestamp("2021-06-01", tz="UTC"))
    months = {"2021-02": 3, "2021-04": 12}
    spec = Spec(clean_data=False)
    monkeypatch.setattr(round4, "load_market", lambda sym, stage, sp: (mk, {"last_ts": int(mk.ts[-1])}))
    sl = round4.make_sleeve("XUSDT", st, months, {"status": "TRADING"}, spec, codes=["U1"])
    got = round4.run_sleeve(sl, "U1", set(), spec).trades
    # reference: the whole market, entries outside the member months zeroed
    sig = round4.BY_CODE["U1"].compute(mk, **round4.PARAMS)
    close = mk.bars("4h")["ts"].to_numpy() + 4 * 3_600_000
    month = pd.to_datetime(close, unit="ms", utc=True).strftime("%Y-%m")
    ok = np.isin(month, list(months)) & (close >= st.start.value // 10**6) & (close < st.end.value // 10**6)
    sig.entry = np.where(ok, sig.entry, 0).astype(np.int8)
    slip = np.array([spec.slip("XUSDT", months.get(m)) for m in
                     pd.to_datetime(mk.ts, unit="ms", utc=True).strftime("%Y-%m")])
    ref = engine.run(mk, sig, Costs(taker=spec.taker, maker=spec.maker, slippage=0.0), spec.risk(),
                     latency_min=spec.latency_min, slippage=slip).trades
    assert len(ref) >= 2
    cols = ["dir", "entry_px", "exit_px", "qty", "fees", "funding", "net", "R", "reason", "entry_time", "exit_time"]
    pd.testing.assert_frame_equal(got[cols].reset_index(drop=True), ref[cols].reset_index(drop=True))
    assert not sl.delisted and not sl.force_exit.any()


def test_no_trade_gaps():
    m = 60_000
    ts = np.arange(0, 4000 * m, m, dtype=np.int64)
    vol = np.ones(len(ts))
    vol[1000:2600] = 0.0                      # 26.7h flat, zero-volume stretch
    idx, gaps = round4.no_trade_gaps(ts, vol, int(ts[-1]) + m)
    assert list(idx) == [999] and gaps == [(999 * m, 2600 * m)]
    idx, gaps = round4.no_trade_gaps(ts, vol, int(ts[-1]) + 30 * 3_600_000)   # data ends > 24h early
    assert list(idx) == [999, 3999] and gaps[-1][1] == int(ts[-1]) + 30 * 3_600_000


def test_make_sleeve_delisting_settlement_gap_and_cooldown(monkeypatch):
    t0 = int(pd.Timestamp("2021-01-01", tz="UTC").value // 10**6)
    m1 = make_m1(trend_path(100, 3), t0)
    g0 = int((pd.Timestamp("2021-03-01", tz="UTC").value // 10**6 - t0) // 60_000)
    m1.loc[g0:g0 + 3 * 1440, "volume"] = 0.0          # a 3-day settlement stretch inside the data
    mk = engine.Market(m1, None, synthetic_funding=0.0)
    st = Stage("T", "dev", pd.Timestamp("2021-02-01", tz="UTC"), pd.Timestamp("2021-06-01", tz="UTC"))
    monkeypatch.setattr(round4, "load_market", lambda sym, stage, sp: (mk, {"last_ts": int(mk.ts[-1])}))
    months = {"2021-02": 1, "2021-03": 1}
    sl = round4.make_sleeve("XUSDT", st, months, {"status": "TRADING"}, Spec(), codes=["U1"])
    # data ends on 2021-04-10, long before the stage end: a delisting whatever the snapshot says
    assert sl.delisted and sl.force_exit[-1] and sl.settlement_gaps == 1 and sl.force_exit.sum() == 2
    fx_ts = sl.mk.ts[np.flatnonzero(sl.force_exit)[0]]
    assert fx_ts == t0 + (g0 - 1) * 60_000                       # last traded minute before the stretch
    gap_end = t0 + (g0 + 3 * 1440 + 1) * 60_000
    blocked = (sl.bar_time >= fx_ts) & (sl.bar_time < gap_end + round4.COOLDOWN_MS)
    assert blocked.any() and not sl.member_bar[blocked].any()
    april = int(pd.Timestamp("2021-04-01", tz="UTC").value // 10**6)
    after = (sl.bar_time >= gap_end + round4.COOLDOWN_MS) & (sl.bar_time < april)
    assert after.any() and sl.member_bar[after].all()


@needs_data
def test_round4_machinery_reproduces_round3_f1():
    """Static universe of the 5 majors, 5 slots, 1% risk, round-3 slippage and seed: the new pipeline
    must reproduce F1's committed dev trades, trade statistics and null p-value exactly."""
    from btcq.config import REPORTS_DIR, SLIPPAGE, SYMBOLS
    f1 = json.loads((REPORTS_DIR / "results" / "F1.json").read_text())
    ref = pd.read_parquet(REPORTS_DIR / "results" / "F1_dev_trades.parquet")
    st = Stage("X", "dev", pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC"))
    spec = Spec(slots=5, account_risk=0.01, slip_by_symbol=dict(SLIPPAGE), clean_data=False, null_seed=11)
    months = {m: 1 for m in st.months}
    sleeves = {s: round4.make_sleeve(s, st, months, None, spec, codes=["U1"]) for s in SYMBOLS}
    uni = pd.DataFrame([{"month": m, "symbol": s, "rank": 1} for m in st.months for s in SYMBOLS])
    ev = round4.evaluate("U1", st, sleeves, uni, spec, with_stress=False)
    assert ev["iterations"] == 0
    cols = ["entry_time", "exit_time", "dir", "entry_px", "exit_px", "qty", "fees", "funding", "net", "R", "symbol"]
    key = ["symbol", "entry_time"]
    got = ev["trades"][cols].sort_values(key).reset_index(drop=True)
    want = ref[cols].sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(got, want, check_exact=False, rtol=1e-12, check_dtype=False)
    for k in ("trades", "avg_R", "pf", "t_stat", "max_year_share"):
        assert ev["summary"][k] == pytest.approx(f1["dev"][k], rel=1e-12)
    assert ev["null"]["p_value"] == pytest.approx(f1["null"]["p_value"])
