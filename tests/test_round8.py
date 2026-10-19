"""Round 8 (IBS): account, evaluation, gates, data report, power and the pre-registration guards. Synthetic data only:
no real price is loaded here."""
import json

import numpy as np
import pandas as pd
import pytest

from btcq.backtest import stats
from btcq.data import external_ohlc as X
from btcq.data import store
from btcq.research import guards, round8 as r8, round8_calib as C
from btcq.research.round5 import Window
from btcq.strategies import r8_ibs as R


def synth(seed=0, scenario="garch", kappa=0.0, n_days=2600):
    """Synthetic markets named like the real ones (for their slippage tiers) and a window over most of them."""
    mk, a, b = C.synthetic_markets(np.random.default_rng([31, seed]), scenario, kappa, n_days=n_days)
    mk = [R.MarketRows(s, m.ts, m.open, m.high, m.low, m.close) for s, m in zip(R.MARKETS, mk)]
    start = pd.Timestamp(a, unit="ms", tz="UTC")
    end = pd.Timestamp(b, unit="ms", tz="UTC")
    return mk, Window("dev", "dev", start, end)


@pytest.fixture(scope="module")
def ev():
    mk, w = synth(scenario="planted", kappa=0.10)
    return mk, w, r8.evaluate(mk, w, n_draws=99)


# ---------------------------------------------------------------- the account and the evaluation

def test_account_is_the_sum_of_three_sleeves_over_three(ev):
    mk, w, e = ev
    for name, rec in e["primaries"].items():
        sl = rec["sleeves"]
        tot = sum(sl.sdaily[s] for s in R.MARKETS) / 3
        np.testing.assert_allclose(rec["daily"].to_numpy(), tot.to_numpy(), rtol=0, atol=1e-15)
        assert rec["daily"].index[0] == w.start and rec["daily"].index[-1] == w.end - pd.Timedelta(days=1)
        assert rec["summary"]["trades"] == len(sl.trades()) > 100
        assert sum(rec["market_contrib"].values()) == pytest.approx(rec["daily"].sum(), rel=1e-12)
        # markets 1 and 2 earn nothing before they are ready
        first2 = pd.Timestamp(int(mk[2].ts[90]), unit="ms", tz="UTC")
        assert (sl.sdaily["EWYUSDT"][sl.sdaily["EWYUSDT"].index < first2] == 0).all()


def test_planted_edge_passes_the_null_and_costs_are_ordered(ev):
    _, _, e = ev
    for name, rec in e["primaries"].items():
        assert rec["null"]["p_value"] < 0.025
        assert rec["stress_2x_mean_daily"] < rec["summary"]["mean_daily"]
        assert rec["funding_mean_daily"] < rec["summary"]["mean_daily"]
        assert rec["stress_2x_plus_funding_mean_daily"] < rec["stress_2x_mean_daily"]
        assert rec["maker_first"]["mean_daily"] > rec["summary"]["mean_daily"]
        assert "not TradFi books" in rec["maker_first"]["label"]
        g = r8.gates_dev(rec)
        assert len(g) == 9
    assert 1.0 <= e["n_eff_family"] <= 12 and 0.0 <= e["pbo"] <= 1.0
    assert set(e["primaries"]) == {"IBS-L", "IBS-LS"}


def test_the_null_statistic_is_the_accounts_gross_sized_pnl(ev):
    """At zero cost with daily resizing, a sleeve's P&L on a held day is size x side x the next return: the null's
    actual statistic is the mean of exactly these terms."""
    mk, w, e = ev
    rec = e["primaries"]["IBS-L"]
    sp = R.strata_panel(mk, r8._ms(w.start), r8._ms(w.end))
    v = R.held_vectors(sp, mk, R.CONFIGS[R.PRIMARIES["IBS-L"]])
    terms = []
    for j, m in enumerate(mk):
        pos, size = R.positions(m.high, m.low, m.close, R.CONFIGS[R.PRIMARIES["IBS-L"]])
        nxt = np.r_[m.close[1:] / m.close[:-1] - 1, np.nan]
        inw = (m.ts >= r8._ms(w.start)) & (m.ts < r8._ms(w.end))
        ok = inw & (pos != 0) & np.isfinite(nxt)
        terms.append(pos[ok] * size[ok] * nxt[ok])
    allt = np.concatenate(terms)
    assert (v != 0).sum() == len(allt)
    assert rec["null"]["actual"] == pytest.approx(allt.mean(), rel=1e-10)


def test_no_dev_output_reaches_2016(monkeypatch):
    days = pd.bdate_range("2015-06-01", "2016-01-08", tz="UTC").as_unit("ms").asi8
    c = np.linspace(100, 110, len(days))
    df = pd.DataFrame({"ts": days, "open": c, "high": c * 1.01, "low": c * 0.99, "close": c})
    monkeypatch.setattr(X, "load_ohlc", lambda s, period: df)
    with pytest.raises(r8.HardError, match="window's end"):
        r8.load_markets(r8.DEV)
    monkeypatch.setattr(X, "load_ohlc", lambda s, period: df[df["ts"] < r8._ms(r8.HOLD.start)])
    mk = r8.load_markets(r8.DEV)
    assert all(int(m.ts.max()) < r8._ms(r8.HOLD.start) for m in mk)


def test_ew_benchmark_is_scaled_like_the_account():
    days = pd.bdate_range("2001-01-01", periods=6, tz="UTC").as_unit("ms").asi8
    c = np.array([100.0, 101, 102, 103, 104, 105])
    mk = [R.MarketRows("SPYUSDT", days, c, c, c, c), R.MarketRows("QQQUSDT", days[3:], c[3:], c[3:], c[3:], c[3:]),
          R.MarketRows("EWYUSDT", days[:0], c[:0], c[:0], c[:0], c[:0])]
    w = Window("dev", "dev", pd.Timestamp("2001-01-01", tz="UTC"), pd.Timestamp("2001-01-09", tz="UTC"))
    b = r8.ew_benchmark(mk, w)
    assert b.loc["2001-01-02"] == pytest.approx(0.01 / 3)                   # one market live: 1/3, not the mean
    assert b.loc["2001-01-05"] == pytest.approx(2 * (104 / 103 - 1) / 3)        # QQQ from its second row
    assert b.loc["2001-01-06"] == 0.0


# ---------------------------------------------------------------- gates

def fake_rec(**kw):
    rec = {"summary": {"trades": 100}, "stress_2x_mean_daily": 1e-9, "funding_mean_daily": 1e-9,
           "null": {"p_value": 0.0249}, "dsr_family": {"dsr": 0.95}, "alpha_vs_ew": {"psr_alpha": 0.95},
           "year_share": 0.65, "plateau": {"1": True, "2": True, "3": True}, "markets_positive": 2}
    for k, v in kw.items():
        rec[k] = v
    return rec


def test_dev_gates_pass_exactly_at_the_thresholds():
    assert all(r8.gates_dev(fake_rec()).values())
    for kw in ({"summary": {"trades": 99}}, {"stress_2x_mean_daily": 0.0}, {"funding_mean_daily": 0.0},
               {"null": {"p_value": 0.025}}, {"dsr_family": {"dsr": 0.9499}}, {"alpha_vs_ew": {"psr_alpha": 0.9499}},
               {"year_share": 0.6501}, {"plateau": {"1": True, "2": False, "3": True}}, {"markets_positive": 1}):
        g = r8.gates_dev(fake_rec(**kw))
        assert sum(not x for x in g.values()) == 1, kw


def test_candidate_choice():
    d = {"IBS-L": {"dev_pass": True, "alpha_psr": 0.97, "alpha_sharpe": 0.9},
         "IBS-LS": {"dev_pass": True, "alpha_psr": 0.99, "alpha_sharpe": 0.5}}
    assert r8.candidate(d) == "IBS-LS"
    d["IBS-LS"]["alpha_psr"] = 0.97
    assert r8.candidate(d) == "IBS-L"                                       # a PSR tie: the higher alpha Sharpe
    d["IBS-LS"]["alpha_sharpe"] = 0.9
    assert r8.candidate(d) == "IBS-L"                                       # an exact tie goes to IBS-L
    d["IBS-LS"]["alpha_sharpe"] = 0.91
    assert r8.candidate(d) == "IBS-LS"
    d["IBS-L"]["dev_pass"] = False
    assert r8.candidate(d) == "IBS-LS"
    d["IBS-LS"]["dev_pass"] = False
    assert r8.candidate(d) is None


def test_holdout_gates():
    dev = {"sharpe": 1.0, "alpha_sharpe": 0.8}
    rec = {"summary": {"mean_daily": 1e-5, "sharpe": 0.5, "max_dd": 0.2},
           "alpha_vs_ew": {"alpha_ann": 0.001, "sharpe_alpha_ann": 0.4}}
    assert all(r8.gates_holdout(rec, dev, 0.2).values())
    rec["alpha_vs_ew"]["sharpe_alpha_ann"] = 0.399
    rec["summary"]["max_dd"] = 0.21
    g = r8.gates_holdout(rec, dev, 0.2)
    assert not g["holdout alpha Sharpe>=50% of dev"] and not g["holdout worst drawdown<=dev bootstrap p95"]
    assert sum(g.values()) == 3


def test_holdout_drawdown_bootstrap_uses_the_full_book_source():
    days = pd.date_range("1993-07-01", "2015-12-31", tz="UTC").as_unit("ms")
    d = pd.Series(np.where(days < pd.Timestamp("2000-09-20", tz="UTC"), -0.01, 0.0005), index=days)
    src = r8.boot_source(d, "2000-09-20")
    assert src.index[0] == pd.Timestamp("2000-09-20", tz="UTC") and (src > 0).all()
    assert r8.holdout_dd_limit(d, "2000-09-20") < 1e-12                    # the losing early years are not drawn


# ---------------------------------------------------------------- data report

def frame(n=300, hold=50, **bad):
    days = np.r_[pd.bdate_range("2015-01-02", periods=n - hold, tz="UTC").as_unit("ms").asi8,
                 pd.bdate_range("2016-01-04", periods=hold, tz="UTC").as_unit("ms").asi8]
    c = np.linspace(100, 120, n)
    df = pd.DataFrame({"ts": days, "open": c, "high": c * 1.01, "low": c * 0.99, "close": c})
    for k, (i, v) in bad.items():
        df.loc[i, k] = v
    return df


def test_data_report_red_flags():
    ok = r8.market_report("SPYUSDT", frame())
    assert ok["rows_dev"] == 250 and ok["rows_holdout"] == 50 and ok["closes_match_round6"]
    assert not r8.red_flags(pd.DataFrame([ok]))["red_flag"].any()
    rep = r8.market_report("SPYUSDT", frame(high=(10, np.nan), low=(20, np.nan)))     # 2 of 300 > 0.5%
    assert r8.red_flags(pd.DataFrame([rep]))["red_flag_reasons"][0] == "null H or L >0.5% of rows"
    rep = r8.market_report("SPYUSDT", frame(low=(260, 130.0)))                          # a holdout violation
    assert rep["holdout_ohlc_violations"] == 1 and rep["dev_ohlc_violations"] == 0
    assert r8.red_flags(pd.DataFrame([rep]))["red_flag"].any()                          # 1 of 300 > 0.1%
    rep = r8.market_report("SPYUSDT", frame(high=(5, 200.0)))
    assert rep["dev_range_gt_30"] == 1 and "dev range" in r8.red_flags(pd.DataFrame([rep]))["red_flag_reasons"][0]
    rep = r8.market_report("SPYUSDT", frame(high=(5, 115.0)))
    assert rep["dev_range_15_30_dates"] and rep["dev_range_gt_30"] == 0
    df = frame()
    df.loc[:4, "high"] = df.loc[:4, "low"]
    rep = r8.market_report("SPYUSDT", df)
    assert rep["dev_high_eq_low"] == 5 and "H = L" in r8.red_flags(pd.DataFrame([rep]))["red_flag_reasons"][0]


def test_data_report_never_summarises_holdout_prices():
    a = r8.market_report("SPYUSDT", frame())
    df = frame()
    df.loc[260:, ["open", "high", "low", "close"]] *= 3.7                               # holdout prices change
    b = r8.market_report("SPYUSDT", df)
    assert a == b


# ---------------------------------------------------------------- power

def test_sufficient_statistics_match_brute_force():
    rng = np.random.default_rng(1)
    groups = [(1 / 3, 40), (2 / 3, 30), (1.0, 50)]
    w = np.concatenate([np.full(n, g) for g, n in groups])
    mu = 0.1
    m, sd = r8._uni(rng, 200_000, groups, mu)
    x = w * mu + np.sqrt(w) * rng.standard_normal((40_000, len(w)))
    assert m.mean() == pytest.approx(x.mean(1).mean(), abs=2e-3) and m.std() == pytest.approx(x.mean(1).std(), rel=0.02)
    assert sd.mean() == pytest.approx(x.std(1, ddof=1).mean(), rel=0.01)
    assert sd.std() == pytest.approx(x.std(1, ddof=1).std(), rel=0.03)
    Mr, Sr, Am, As = r8._bi(rng, 100_000, groups, 0.05, mu, 1.0, 0.5)
    b = w * 0.05 + 0.5 * np.sqrt(w) * rng.standard_normal((20_000, len(w)))
    a = w * mu + np.sqrt(w) * rng.standard_normal((20_000, len(w)))
    y = b + a
    bh = np.array([stats.beta_alpha(y[i], b[i])["beta"] for i in range(2000)])
    al = y[:2000] - bh[:, None] * b[:2000]
    assert Am.mean() == pytest.approx(al.mean(1).mean(), abs=5e-3)
    assert As.mean() == pytest.approx(al.std(1, ddof=1).mean(), rel=0.02)
    assert Sr.mean() == pytest.approx(y.std(1, ddof=1).mean(), rel=0.01)


def test_power_table_rows_and_known_values():
    rows = pd.DataFrame(r8.power_table(sharpes=(0.0, 0.5), n_sims=40_000, variants=("constant",)))
    z = rows[(rows["sharpe_ann"] == 0) & (rows["n_eff"] == 1) & (rows["rho"] == 1)].iloc[0]
    assert z["p_dev"] == pytest.approx(0.05, abs=0.005)
    assert z["sharpe_needed_ann"] == pytest.approx(1.6449 / np.sqrt(8218) * np.sqrt(365), rel=1e-3)
    h = rows[(rows["sharpe_ann"] == 0.5) & (rows["n_eff"] == 1)].set_index("rho")
    assert h.loc[1.0, "p_pass"] > h.loc[0.5, "p_pass"] > h.loc[0.0, "p_pass"]
    assert (h["p_dev"] == h["p_dev"].iloc[0]).all()
    assert h.loc[1.0, "p_dev"] == pytest.approx(0.77, abs=0.02)            # the plan's approximate table
    assert len(rows) == 2 * 4 * 3


def test_ramp_weights_follow_the_source_dates():
    w = r8.ramp_weights()
    assert len(w) == 8219 and set(np.round(np.unique(w), 6)) == {round(1 / 3, 6), round(2 / 3, 6), 1.0}
    rd = r8.ready_dates()
    assert rd["SPYUSDT"] < r8.DEV.start
    assert pd.Timestamp("1999-07-01", tz="UTC") < rd["QQQUSDT"] < pd.Timestamp("1999-08-01", tz="UTC")
    assert pd.Timestamp("2000-09-01", tz="UTC") < rd["EWYUSDT"] < pd.Timestamp("2000-10-01", tz="UTC")
    assert 17 < w.sum() / 365 < 19


def test_maker_first_savings_are_x1_cost_v1():
    cal = pd.read_csv(r8.ROOT / "research" / "x1" / "x1_cost_v1.csv")
    sel = cal[(cal["symbol"] == "ALL") & (cal["T"] == 15)].set_index("class")["c_eff_v1_strict_bp"]
    assert r8.MAKER_ENTRY_SAVING == pytest.approx((6.6 - sel["reversion"]) * 1e-4, rel=1e-12)
    assert r8.MAKER_EXIT_SAVING == pytest.approx((6.6 - sel["random"]) * 1e-4, rel=1e-12)


# ---------------------------------------------------------------- calibration guard

def cal_frame(**over):
    rows = []
    for sc, kappa, reps, kind in C.CAL_ROWS:
        for rule in R.PRIMARIES:
            rows.append({"scenario": sc, "kind": kind, "rule": rule, "planted_edge": kappa, "reps": reps,
                         "draws": C.CAL_DRAWS, "reject_05": 0.9 if kind == "planted edge" else 0.05,
                         "reject_025": 0.9 if kind == "planted edge" else 0.02,
                         "ok": True if kind == "no-edge" else None, "calib_sha256": C.calib_sha(),
                         **C.lib_versions_calib()})
    df = pd.DataFrame(rows)
    for k, v in over.items():
        df.loc[0, k] = v
    return df


def test_calibration_guard():
    assert r8.calibration_problems(cal_frame()) == []
    assert any("above its limit" in p for p in r8.calibration_problems(cal_frame(ok=False)))
    assert any("above its limit" in p for p in r8.calibration_problems(cal_frame(reject_05=0.0701)))   # recomputed
    assert any("above its limit" in p for p in r8.calibration_problems(cal_frame(reject_025=0.0351)))
    assert r8.calibration_problems(cal_frame(reject_05=0.07, reject_025=0.035)) == []                    # inclusive
    assert any("other calibration code" in p for p in r8.calibration_problems(cal_frame(calib_sha256="x")))
    assert any("registered scenarios" in p for p in r8.calibration_problems(cal_frame(reps=999)))
    assert any("library versions" in p for p in r8.calibration_problems(cal_frame(numpy_version="0.0")))
    weak = cal_frame()
    weak.loc[(weak["planted_edge"] == 0.10) & (weak["rule"] == "IBS-L"), "reject_025"] = 0.79
    assert any("not detected" in p for p in r8.calibration_problems(weak))
    assert r8.calibration_problems(pd.DataFrame({"a": [1]})) == ["the calibration file is not in the registered format"]


# ---------------------------------------------------------------- run-order guards

def sweep_env(monkeypatch, tmp_path):
    monkeypatch.setattr(r8, "_require_prereg", lambda: "")
    monkeypatch.setattr(r8, "_require_data_report", lambda: None)
    monkeypatch.setattr(guards, "clean_src", lambda: None)
    res = tmp_path / "results"
    res.mkdir(exist_ok=True)
    monkeypatch.setattr(r8, "R8_RESULTS", res)
    monkeypatch.setattr(r8, "VERDICT", res / "IBS.json")
    log = tmp_path / "runs.jsonl"
    monkeypatch.setattr(r8, "RUN_LOG", log)
    monkeypatch.setattr(store, "HOLDOUT_LOG", tmp_path / "holdout_log.jsonl")
    for name in ("MANIFEST", "R6_MANIFEST"):
        pth = tmp_path / name.lower()
        pth.write_text(name)
        monkeypatch.setattr(X, name, pth)
    for name in ("DATA_REPORT", "F1_DEV_DAILY"):
        pth = tmp_path / name.lower()
        pth.write_text(name)
        monkeypatch.setattr(r8, name, pth)
    committed = []
    real = guards.log_records

    def log_records(path, kind, *, committed_only=False, **match):
        mine = [r for r in committed if r.get("kind") == kind and all(r.get(k) == v for k, v in match.items())]
        return mine if committed_only else real(path, kind, committed_only=False, **match) + mine
    monkeypatch.setattr(guards, "log_records", log_records)
    monkeypatch.setattr(guards, "committed_versions", lambda p: [])
    calls = []

    def load(w):
        calls.append(w)
        raise RuntimeError("ran")
    monkeypatch.setattr(r8, "load_markets", load)
    pins = {"kind": "r8-sweep-start", "version": 1, **r8.sweep_pins("")}
    return log, calls, committed, pins


def test_sweep_first_writes_a_start_record_and_stops(monkeypatch, tmp_path):
    log, calls, committed, pins = sweep_env(monkeypatch, tmp_path)
    with pytest.raises(SystemExit, match="start record"):
        r8.cmd_sweep()
    recs = [json.loads(x) for x in log.read_text().splitlines()]
    assert [r["kind"] for r in recs] == ["r8-sweep-start"] and all(recs[0][k] == v for k, v in pins.items())
    with pytest.raises(SystemExit, match="start record"):
        r8.cmd_sweep()
    assert len(log.read_text().splitlines()) == 1 and not calls


def test_sweep_runs_once_after_the_committed_start_with_the_same_pins(monkeypatch, tmp_path):
    log, calls, committed, pins = sweep_env(monkeypatch, tmp_path)
    committed.append(pins)
    with pytest.raises(SystemExit, match="hard error, logged"):
        r8.cmd_sweep()
    assert len(calls) == 1
    assert [json.loads(x)["kind"] for x in log.read_text().splitlines()] == ["r8-sweep-run", "r8-sweep"]
    with pytest.raises(SystemExit, match="hard error|already ran"):
        r8.cmd_sweep()
    log.write_text("")
    for k in ("code_sha256", "manifest_sha256", "r6_manifest_sha256", "data_report_sha256", "numba_version"):
        committed[0] = pins | {k: "other"}
        with pytest.raises(SystemExit, match=f"different {k}"):
            r8.cmd_sweep()
    assert len(calls) == 1
    committed[0] = pins
    log.write_text(json.dumps({"kind": "r8-sweep-run", "version": 1}) + "\n")      # e.g. killed mid-run
    with pytest.raises(SystemExit, match="already ran"):
        r8.cmd_sweep()


def test_sweep_is_closed_after_any_verdict_or_holdout_touch(monkeypatch, tmp_path):
    log, calls, committed, pins = sweep_env(monkeypatch, tmp_path)
    r8.VERDICT.write_text(json.dumps({"version": 1}))
    with pytest.raises(SystemExit, match="runs once"):
        r8.cmd_sweep()
    monkeypatch.setattr(R.IBS, "version", 2)
    with pytest.raises(SystemExit, match="no reruns or variants"):                 # not even with a version bump
        r8.cmd_sweep()
    with pytest.raises(SystemExit, match="has no reruns"):
        r8.cmd_sweep(rerun=True)
    r8.VERDICT.unlink()
    log.write_text(json.dumps({"kind": "r8-sweep", "version": 1, "verdict": {"IBS-L": False}}) + "\n")
    with pytest.raises(SystemExit, match="no reruns or variants"):                 # deleting IBS.json is no way out
        r8.cmd_sweep()
    log.write_text(json.dumps({"kind": "r8-sweep", "version": 1, "hard_error": "x"}) + "\n")
    with pytest.raises(SystemExit, match="start record"):                          # a hard error allows a fix as v2
        r8.cmd_sweep()
    store.HOLDOUT_LOG.write_text(json.dumps({"round": 8, "kind": "verify", "reason": "peek [R8 verify x]"}) + "\n")
    with pytest.raises(SystemExit, match="closed"):                                # any load of the holdout closes it
        r8.cmd_sweep()
    assert not calls


def test_a_version_bump_needs_a_committed_hard_error_of_every_earlier_version(monkeypatch, tmp_path):
    log, calls, committed, pins = sweep_env(monkeypatch, tmp_path)
    committed.append(pins)                                       # v1 started (committed) ...
    log.write_text("")                                           # ... its uncommitted verdict was reverted away
    monkeypatch.setattr(R.IBS, "version", 2)
    with pytest.raises(SystemExit, match="started without a committed hard error"):
        r8.cmd_sweep()
    log.write_text(json.dumps({"kind": "r8-sweep", "version": 1, "hard_error": "x"}) + "\n")
    with pytest.raises(SystemExit, match="started without a committed hard error"):   # not committed yet
        r8.cmd_sweep()
    committed.append({"kind": "r8-sweep", "version": 1, "hard_error": "x"})
    with pytest.raises(SystemExit, match="start record"):                          # now v2 may start
        r8.cmd_sweep()
    assert not calls


def fake_verdict(**kw):
    g = {"a": True}
    dev = {"IBS-L": {"dev_pass": True, "gates": g, "sharpe": 0.9, "alpha_psr": 0.97, "alpha_sharpe": 0.7,
                     "holdout_dd_limit": 0.3},
           "IBS-LS": {"dev_pass": False, "gates": {"a": False}, "sharpe": 0.5, "alpha_psr": 0.9, "alpha_sharpe": 0.4,
                      "holdout_dd_limit": 0.35}}
    r = {"version": 1, "candidate": "IBS-L", "dev": dev, "code_sha256": r8.code_sha(), "manifest_sha256": None,
         "r6_manifest_sha256": None, "data_report_sha256": None, "f1_daily_sha256": None,
         "prereg_section_sha256": "sec", **r8.lib_versions()}
    r.update(kw)
    return r


def preflight_env(monkeypatch, tmp_path, verdict: dict, log_lines=(), sweep_rec=None):
    res = tmp_path / "results"
    res.mkdir()
    monkeypatch.setattr(r8, "R8_RESULTS", res)
    monkeypatch.setattr(r8, "VERDICT", res / "IBS.json")
    r8.VERDICT.write_text(json.dumps(verdict))
    monkeypatch.setattr(X, "MANIFEST", tmp_path / "missing_m8.csv")
    monkeypatch.setattr(X, "R6_MANIFEST", tmp_path / "missing_m6.csv")
    monkeypatch.setattr(r8, "DATA_REPORT", tmp_path / "missing_report.csv")
    monkeypatch.setattr(r8, "F1_DEV_DAILY", tmp_path / "missing_f1.parquet")
    monkeypatch.setattr(r8, "_require_data_report", lambda: None)
    monkeypatch.setattr(guards, "require_committed", lambda paths: None)
    log = tmp_path / "holdout_log.jsonl"
    log.write_text("".join(json.dumps(x) + "\n" for x in log_lines))
    monkeypatch.setattr(store, "HOLDOUT_LOG", log)
    runs = tmp_path / "runs.jsonl"
    rec = {"kind": "r8-sweep", "version": 1, "verdict": {"IBS-L": True, "IBS-LS": False},
           "candidate": verdict.get("candidate"), "yardsticks": r8.yardsticks(verdict["dev"]),
           **{k: verdict.get(k) for k in r8.PIN_KEYS}}
    if sweep_rec is not None:
        rec = sweep_rec(rec)
    runs.write_text(json.dumps(rec) + "\n" if rec else "")
    monkeypatch.setattr(r8, "RUN_LOG", runs)
    monkeypatch.setattr(guards, "committed_versions", lambda q: [q.read_text()] if q == runs else [])
    monkeypatch.setattr(guards, "committed_section", lambda *a: "")
    monkeypatch.setattr(guards, "sha_str", lambda x: "sec")


def test_preflight_passes_once_and_refuses_after_any_touch(monkeypatch, tmp_path):
    preflight_env(monkeypatch, tmp_path, fake_verdict())
    assert r8.preflight_holdout() == []
    assert r8._holdout_touched(json.loads(r8.VERDICT.read_text())) == []
    for sub, line in (("b", {"round": 8, "kind": "unlock", "version": 1, "reason": "x [R8: IBS v1]"}),
                      ("c", {"reason": "x [R8 resume IBS v1]"}), ("d", {"round": 8, "kind": "verify", "reason": "y"})):
        d = tmp_path / sub
        d.mkdir()
        preflight_env(monkeypatch, d, fake_verdict(), [line])
        assert r8._holdout_touched(json.loads(r8.VERDICT.read_text()))
    e = tmp_path / "e"
    e.mkdir()
    preflight_env(monkeypatch, e, fake_verdict(holdout={}))
    assert any("already evaluated" in x for x in r8.preflight_holdout())
    f = tmp_path / "f"
    f.mkdir()
    preflight_env(monkeypatch, f, fake_verdict(), [{"round": 6, "kind": "unlock", "reason": "x [R6: M1 v1]"}])
    assert any("Round 6 record" in x for x in r8.preflight_holdout())


def test_preflight_refuses_no_pass_changed_code_or_a_wrong_candidate(monkeypatch, tmp_path):
    v = fake_verdict(candidate=None, code_sha256="x")
    v["dev"]["IBS-L"]["dev_pass"] = False
    v["dev"]["IBS-L"]["gates"] = {"a": False}
    preflight_env(monkeypatch, tmp_path, v)
    probs = r8.preflight_holdout()
    assert any("no Round 8 primary passed" in x for x in probs) and any("code changed" in x for x in probs)
    b = tmp_path / "b"
    b.mkdir()
    preflight_env(monkeypatch, b, fake_verdict(candidate="IBS-LS"))
    assert any("not the registered choice" in x for x in r8.preflight_holdout())
    c = tmp_path / "c"
    c.mkdir()
    preflight_env(monkeypatch, c, fake_verdict(), sweep_rec=lambda rec: {})
    assert any("no committed r8-sweep record" in x for x in r8.preflight_holdout())
    d = tmp_path / "d"
    d.mkdir()
    v = fake_verdict()
    v["dev"]["IBS-L"]["gates"] = {"a": True, "b": False}
    preflight_env(monkeypatch, d, v)
    assert any("does not match its gates" in x for x in r8.preflight_holdout())


def test_preflight_compares_every_yardstick_and_pin_with_the_committed_record(monkeypatch, tmp_path):
    def lowered(rec):
        rec["yardsticks"]["IBS-L"]["sharpe"] = 1.5                # IBS.json's dev Sharpe was edited down after the run
        return rec
    preflight_env(monkeypatch, tmp_path, fake_verdict(), sweep_rec=lowered)
    assert any("no committed r8-sweep record" in x for x in r8.preflight_holdout())
    b = tmp_path / "b"
    b.mkdir()
    preflight_env(monkeypatch, b, fake_verdict(), sweep_rec=lambda rec: rec | {"manifest_sha256": "other"})
    assert any("no committed r8-sweep record" in x for x in r8.preflight_holdout())


def test_an_unlock_without_a_result_is_a_fail(monkeypatch, tmp_path):
    preflight_env(monkeypatch, tmp_path, fake_verdict())
    monkeypatch.setattr(guards, "clean_src", lambda: None)
    with r8.RUN_LOG.open("a") as f:                              # killed after the start record
        f.write(json.dumps({"kind": "r8-validate-start", "version": 1}) + "\n")
    with pytest.raises(SystemExit, match="recorded as a FAIL"):
        r8.cmd_validate("Round 8 holdout")
    r = json.loads(r8.VERDICT.read_text())
    assert r["final_pass"] is False and "left no result" in r["holdout"]["integrity_error"]
    with pytest.raises(SystemExit, match="stays locked"):
        r8.cmd_validate("Round 8 holdout")
    assert not store.HOLDOUT_LOG.read_text()                     # no unlock was written


def validate_env(monkeypatch, tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    monkeypatch.setattr(r8, "R8_RESULTS", res)
    monkeypatch.setattr(r8, "VERDICT", res / "IBS.json")
    monkeypatch.setattr(r8, "RUN_LOG", tmp_path / "runs.jsonl")
    monkeypatch.setattr(r8, "preflight_holdout", lambda: [])
    monkeypatch.setattr(guards, "clean_src", lambda: None)
    monkeypatch.setattr(store, "HOLDOUT_LOG", tmp_path / "holdout_log.jsonl")
    monkeypatch.setattr(X, "_R8_UNLOCKED", False)
    r8.VERDICT.write_text(json.dumps(fake_verdict()))


def test_validate_records_a_fail_on_any_error_after_the_unlock(monkeypatch, tmp_path):
    validate_env(monkeypatch, tmp_path)
    monkeypatch.setattr(r8, "load_markets", lambda w: (_ for _ in ()).throw(MemoryError("boom")))
    with pytest.raises(SystemExit, match="recorded as a FAIL"):
        r8.cmd_validate("Round 8 holdout")
    r = json.loads(r8.VERDICT.read_text())
    assert r["final_pass"] is False and "MemoryError" in r["holdout"]["integrity_error"]
    hl = [json.loads(x) for x in store.HOLDOUT_LOG.read_text().splitlines()]
    assert len(hl) == 1 and hl[0]["round"] == 8 and hl[0]["reason"].endswith("[R8: IBS v1]")
    kinds = [json.loads(x)["kind"] for x in r8.RUN_LOG.read_text().splitlines()]
    assert kinds == ["r8-validate-start", "r8-validate"]


def test_validate_end_to_end_on_synthetic_data(monkeypatch, tmp_path):
    """The candidate is gated, a dev-failed primary is not evaluated, the verdict is saved before the result files,
    and a failing save after the result is still a recorded FAIL."""
    validate_env(monkeypatch, tmp_path)
    mk, w = synth(seed=3, n_days=5000)
    monkeypatch.setattr(r8, "HOLD", w)
    monkeypatch.setattr(r8, "load_markets", lambda win: mk)
    days = pd.date_range("1993-07-01", "2015-12-31", tz="UTC").as_unit("ms")
    pd.DataFrame({"ret": 0.0002}, index=days).to_parquet(r8.R8_RESULTS / "IBS-L_dev_daily.parquet")
    order = []
    real_write = r8._write
    monkeypatch.setattr(r8, "_write", lambda tag, ev: order.append(
        "holdout_gates" in json.loads(r8.VERDICT.read_text())) or real_write(tag, ev))
    r = r8.cmd_validate("Round 8 holdout")
    assert order == [True] and set(r["holdout_gates"]) == {"IBS-L"} and set(r["holdout"]["primaries"]) == {"IBS-L"}
    assert r["headline"]["candidate"] == "IBS-L" and isinstance(r["final_pass"], bool)
    assert "config_totals" not in r["holdout"] and not (r8.R8_RESULTS / "IBS_holdout_config_daily.parquet").exists()
    assert len(r["holdout_gates"]["IBS-L"]) == 5


def test_validate_save_error_is_a_fail(monkeypatch, tmp_path):
    validate_env(monkeypatch, tmp_path)
    mk, w = synth(seed=3, n_days=3000)
    monkeypatch.setattr(r8, "HOLD", w)
    monkeypatch.setattr(r8, "load_markets", lambda win: mk)
    days = pd.date_range("1993-07-01", "2015-12-31", tz="UTC").as_unit("ms")
    pd.DataFrame({"ret": 0.0002}, index=days).to_parquet(r8.R8_RESULTS / "IBS-L_dev_daily.parquet")
    real = r8._save_json
    calls = []

    def flaky(p, obj):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("disk full")
        real(p, obj)
    monkeypatch.setattr(r8, "_save_json", flaky)
    with pytest.raises(SystemExit, match="recorded as a FAIL"):
        r8.cmd_validate("Round 8 holdout")
    assert json.loads(r8.VERDICT.read_text())["final_pass"] is False


def test_registration_section_guard(monkeypatch):
    text = ("# x\n\n## Round 7 (RT)\nold\n\n## Round 8 (IBS) - registered\nIBS-L IBS-LS {N_EFF}\n\n"
            "## Round 8 dev outcome\nlater text\n")
    sec = guards.section(text, r8.SECTION)
    assert sec.startswith("## Round 8 (IBS)") and "later text" not in sec        # an outcome section is outside
    monkeypatch.setattr(guards, "require_committed", lambda paths: None)
    monkeypatch.setattr(guards, "git", lambda *a, **k: text)
    with pytest.raises(SystemExit, match="lacks"):
        guards.require_section(r8.PREREG, r8.SECTION, r8.REQUIRED_PHRASES)


def test_code_files_exist_and_the_cli_knows_round8():
    for f in r8.CODE_FILES + C.CALIB_FILES:
        assert (r8.ROOT / f).exists(), f
    import inspect
    from btcq.research import run
    src = inspect.getsource(run.main)
    for c in ("r8-power", "r8-calibrate", "r8-data", "r8-sweep", "r8-validate"):
        assert c in src


def test_round8_never_unlocks_the_crypto_store(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "HOLDOUT_LOG", tmp_path / "holdout_log.jsonl")
    monkeypatch.setattr(store, "_HOLDOUT_UNLOCKED", False)
    monkeypatch.setattr(X, "_R8_UNLOCKED", False)
    X.unlock_r8("x", version=1)
    assert X._R8_UNLOCKED and not store._HOLDOUT_UNLOCKED
    import inspect
    assert "store.unlock_holdout" not in inspect.getsource(r8) and "unlock_r6" not in inspect.getsource(r8)
