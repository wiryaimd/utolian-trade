"""Round 3: 5-market portfolio families, frequency/consistency gates, F7 ensemble, holdout.

    python -m btcq.research.run r3-sweep
    python -m btcq.research.run r3-validate --finalists --unlock-holdout
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from btcq.backtest import engine, metrics, stats
from btcq.config import HOLDOUT_START, SLIPPAGE, SYMBOLS, Costs, Risk
from btcq.data import store
from btcq.research import registry
from btcq.research.portfolio import assemble, combine_null, sleeve_risk
from btcq.research.run import (GATES, N_BOOT, N_PERM, RESULTS, _save_json, aligned_daily,
                               trade_equity_mdd)
from btcq.strategies import BY_CODE, ROUND3

LATENCY_MIN = 1  # auto-execute: market orders fill one minute after the bar close
CODES = [s.code for s in ROUND3]
ENSEMBLE = "F7"
ALTS = [s for s in SYMBOLS if s != "BTCUSDT"]
MIN_FREQ = 52
MIN_FREQ_COMPONENT = 30


def _costs(sym: str, mult: float = 1.0) -> Costs:
    return Costs(slippage=SLIPPAGE[sym]).scaled(mult)


def _confirm_years(R: np.ndarray, per_year: float) -> float:
    """Years of forward trading for the mean-R t-stat to reach 2 at the observed edge."""
    R = np.asarray(R, dtype=np.float64)
    if R.size < 2 or R.mean() <= 0 or per_year <= 0:
        return float("inf")
    return float((2 * R.std(ddof=1) / R.mean()) ** 2 / per_year)


def _buy_hold(closes: dict[str, pd.Series]) -> dict:
    r = pd.concat([c.pct_change() for c in closes.values()], axis=1).mean(axis=1).dropna()
    eq = (1 + r).cumprod()
    years = len(eq) / 365
    return {"total_return": eq.iloc[-1] - 1, "cagr": eq.iloc[-1] ** (1 / years) - 1,
            "sharpe": r.mean() / r.std() * np.sqrt(365),
            "max_dd": metrics.max_drawdown(np.r_[1.0, eq.to_numpy()])}


def _daily_close(mk: engine.Market, start) -> pd.Series:
    c = pd.Series(mk.c, index=pd.to_datetime(mk.ts, unit="ms", utc=True))
    return c[c.index >= start].resample("D").last().dropna()


def _run(strat, mk, sym, params, start, mult=1.0) -> engine.Result:
    return engine.run(mk, strat.compute(mk, **params), _costs(sym, mult), sleeve_risk(len(SYMBOLS)),
                      start=start, latency_min=LATENCY_MIN)


def _symbol_null(mk: engine.Market, trades: pd.DataFrame, tf: str, sym: str, start) -> np.ndarray:
    if len(trades) == 0:
        return np.zeros(N_PERM)
    ex = mk.exec_idx(tf, LATENCY_MIN)
    start_i = mk.index_at(start)
    cand = ex[(ex >= start_i) & (ex < len(mk.ts) - 1)]
    c = _costs(sym)
    return stats.random_entry_null(mk.o, cand, (trades["exit_i"] - trades["entry_i"]).to_numpy(),
                                   trades["dir"].to_numpy(), 2 * (c.taker + c.slippage), N_PERM)


# ---------------------------------------------------------------- dev sweep

def sweep_family(strat) -> dict:
    t0 = time.time()
    configs = strat.configs()
    start = strat.data_start
    by_sym: dict[str, list] = {}
    for sym in SYMBOLS:
        mk = engine.Market.load("dev", sym)
        by_sym[sym] = [(aligned_daily(r, mk), r.trades.assign(symbol=sym))
                       for r in (_run(strat, mk, sym, p, start) for p in configs)]
        del mk

    rows, cols, trial_rows = [], [], []
    for i, p in enumerate(configs):
        res, daily = assemble([by_sym[s][i] for s in SYMBOLS])
        summ = metrics.summarize(res)
        sym_R = {s: float(by_sym[s][i][1]["R"].mean()) if len(by_sym[s][i][1]) else 0.0 for s in SYMBOLS}
        sr, T, sk, ku = stats.sharpe_moments(daily)
        rows.append({"cfg": i, **p, **summ, "sr": sr, "T": T, "skew": sk, "kurt": ku,
                     "symbols_positive": sum(v > 0 for v in sym_R.values())})
        cols.append(daily.rename(f"{strat.code}_{i}"))
        trial_rows.append({"strategy": strat.code, "version": strat.version, "params": p,
                           "period": "dev", "sr": sr, "T": T, "skew": sk, "kurt": ku,
                           "trades": summ["trades"], "sharpe_ann": summ["sharpe"]})
    registry.record(trial_rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    trial = pd.concat(cols, axis=1)
    trial.fillna(0.0).to_parquet(RESULTS / f"{strat.code}_trial_returns.parquet")

    df = pd.DataFrame(rows)
    if len(configs) > 1:
        pbo, logits = stats.pbo_cscv(trial.fillna(0.0).to_numpy())
        nbrs = stats.grid_neighbors(strat.grid, configs)
        sh = df["sharpe"].to_numpy()
        df["nbhd_sharpe"] = [np.mean(sh[[i] + nbrs[i]]) for i in range(len(df))]
        inherited = []
    else:
        # One frozen config (F1): PBO/plateau come from the family it was selected in (S1).
        s1 = json.loads((RESULTS / "S1.json").read_text())
        pbo, logits, nbrs = s1["pbo"], np.array([]), [[]]
        df["nbhd_sharpe"] = df["sharpe"]
        inherited = ["PBO", "plateau"]
    eligible = df[(df["trades"] >= GATES["min_trades"]) & (df["trades_per_year"] >= MIN_FREQ)]
    pick = int((eligible if len(eligible) else df)["nbhd_sharpe"].idxmax())
    params = configs[pick]
    nb = nbrs[pick]
    plateau = (float(np.mean(df.loc[nb, "total_return"].to_numpy() > 0)) if nb
               else (json.loads((RESULTS / "S1.json").read_text())["plateau"] if inherited else 0.0))

    # Second pass on the chosen config: cost stress, per-symbol nulls, benchmark.
    chosen = {s: by_sym[s][pick] for s in SYMBOLS}
    stress_parts, nulls, counts, closes, sym_rows = [], [], [], {}, []
    for sym in SYMBOLS:
        mk = engine.Market.load("dev", sym)
        st = _run(strat, mk, sym, params, start, mult=2.0)
        stress_parts.append((aligned_daily(st, mk), st.trades))
        tr = chosen[sym][1]
        nulls.append(_symbol_null(mk, tr, strat.tf, sym, start))
        counts.append(len(tr))
        closes[sym] = _daily_close(mk, start)
        sym_rows.append({"symbol": sym, "trades": len(tr),
                         "avg_R": float(tr["R"].mean()) if len(tr) else 0.0,
                         "win_rate": float((tr["R"] > 0).mean()) if len(tr) else 0.0,
                         "long_avg_R": float(tr.loc[tr["dir"] == 1, "R"].mean()) if (tr["dir"] == 1).any() else 0.0,
                         "short_avg_R": float(tr.loc[tr["dir"] == -1, "R"].mean()) if (tr["dir"] == -1).any() else 0.0,
                         "sharpe": float(chosen[sym][0].mean() / chosen[sym][0].std() * np.sqrt(365))
                         if chosen[sym][0].std() > 0 else 0.0})
        del mk
    res, daily = assemble(list(chosen.values()))
    summ = metrics.summarize(res)
    stress = metrics.summarize(assemble(stress_parts)[0])
    null = combine_null(nulls, counts)
    t = res.trades
    actual = float(((t["gross"] - t["fees"]) / (t["qty"] * t["entry_px"])).mean()) if len(t) else 0.0
    p_val = (1 + np.sum(null >= actual)) / (1 + len(null))
    boot_tot, boot_mdd = stats.block_bootstrap(t["ret"].to_numpy(), N_BOOT) if len(t) > 1 \
        else (np.zeros(1), np.zeros(1))
    sr, T, sk, ku = stats.sharpe_moments(daily)

    out = {
        "code": strat.code, "name": strat.name, "tf": strat.tf, "version": strat.version, "round": 3,
        "hypothesis": strat.hypothesis, "rules": strat.rules, "grid": strat.grid,
        "n_configs": len(configs), "chosen_cfg": pick, "params": params, "symbols": SYMBOLS,
        "holdout_scope": strat.holdout_scope, "latency_min": LATENCY_MIN,
        "start": res.start.isoformat(), "end": daily.index[-1].isoformat(),
        "dev": summ, "stress_2x": {k: stress.get(k) for k in ("avg_R", "pf", "total_return", "sharpe")},
        "sr_daily": sr, "T": T, "skew": sk, "kurt": ku, "pbo": pbo, "plateau": plateau,
        "n_neighbors": len(nb), "inherited_gates": inherited,
        "null": {"actual_mean_ret": actual, "null_mean": float(null.mean()),
                 "null_p95": float(np.quantile(null, 0.95)), "p_value": float(p_val)},
        "bootstrap": {"ret_p5": np.quantile(boot_tot, 0.05), "ret_p50": np.quantile(boot_tot, 0.5),
                      "mdd_p50": np.quantile(boot_mdd, 0.5), "mdd_p95": np.quantile(boot_mdd, 0.95)},
        "benchmark": _buy_hold(closes), "symbol_table": sym_rows,
        "symbols_positive": sum(r["avg_R"] > 0 for r in sym_rows),
        "years": metrics.year_table(res).reset_index(names="year").to_dict("records"),
        "confirm_years": _confirm_years(t["R"].to_numpy(), summ.get("trades_per_year", 0)),
        "weeks_with_entry": _weeks_with_entry(t, res.start, daily.index[-1]),
        "runtime_s": time.time() - t0,
    }
    if strat.holdout_scope == "alts":
        scope_res, _ = assemble([chosen[s] for s in ALTS])
        out["dev_scope"] = metrics.summarize(scope_res)
        scope_trades = scope_res.trades
    else:
        scope_trades = t
    df.to_csv(RESULTS / f"{strat.code}_configs.csv", index=False)
    t.to_parquet(RESULTS / f"{strat.code}_dev_trades.parquet")
    scope_trades.to_parquet(RESULTS / f"{strat.code}_dev_scope_trades.parquet")
    res.equity.rename("equity").to_frame().to_parquet(RESULTS / f"{strat.code}_dev_equity.parquet")
    daily.rename("ret").to_frame().to_parquet(RESULTS / f"{strat.code}_dev_daily.parquet")
    assemble(stress_parts)[0].trades.to_parquet(RESULTS / f"{strat.code}_stress_trades.parquet")
    np.savez_compressed(RESULTS / f"{strat.code}_dev_dists.npz", null=null, boot_tot=boot_tot,
                        boot_mdd=boot_mdd, pbo_logits=np.asarray(logits), null_counts=np.asarray(counts))
    _save_json(RESULTS / f"{strat.code}.json", out)
    return out


def _weeks_with_entry(trades: pd.DataFrame, start, end) -> float:
    if not len(trades):
        return 0.0
    weeks = trades["entry_time"].dt.tz_localize(None).dt.to_period("W").nunique()
    return float(weeks / max((pd.Timestamp(end) - pd.Timestamp(start)).days / 7, 1))


# ---------------------------------------------------------------- gates

def augment_gates(r: dict) -> dict:
    """Round-3 gates on top of run.dev_gates: frequency and cross-symbol consistency."""
    g = r["gates"]
    per_year = r["dev"].get("trades_per_year", 0)
    g["trades/yr>=52"] = per_year >= MIN_FREQ
    g[">=3/5 symbols R>0"] = r.get("symbols_positive", 0) >= 3
    r["dev_pass"] = all(g.values())
    r["dev_pass_ex_freq"] = all(v for k, v in g.items() if k != "trades/yr>=52") \
        and per_year >= MIN_FREQ_COMPONENT
    return r


# ---------------------------------------------------------------- F7 ensemble

def build_ensemble() -> dict | None:
    """Equal-weight ensemble of families passing every dev gate except frequency (pre-registered)."""
    fam = {c: json.loads((RESULTS / f"{c}.json").read_text()) for c in CODES
           if (RESULTS / f"{c}.json").exists()}
    comps = [c for c, r in fam.items() if r.get("dev_pass_ex_freq")]
    path = RESULTS / f"{ENSEMBLE}.json"
    if len(comps) < 2:
        _save_json(path, {"code": ENSEMBLE, "formed": False, "components": comps,
                          "reason": "needs at least two families passing every dev gate except frequency"})
        return None
    dailies = [pd.read_parquet(RESULTS / f"{c}_dev_daily.parquet")["ret"] for c in comps]
    daily = pd.concat(dailies, axis=1).dropna().mean(axis=1)
    k = len(comps)
    tr = pd.concat([pd.read_parquet(RESULTS / f"{c}_dev_trades.parquet").assign(component=c) for c in comps],
                   ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    tr = tr[tr["entry_time"] >= daily.index[0]]
    tr["ret"] = tr["ret"] / k
    eq0 = Risk().initial_equity
    res = engine.Result(tr, eq0 * (1 + daily).cumprod(), eq0, daily.index[0])
    summ = metrics.summarize(res)
    stress_tr = pd.concat([pd.read_parquet(RESULTS / f"{c}_stress_trades.parquet") for c in comps])
    sr, T, sk, ku = stats.sharpe_moments(daily)
    registry.record([{"strategy": ENSEMBLE, "version": 1, "params": {"components": comps}, "period": "dev",
                      "sr": sr, "T": T, "skew": sk, "kurt": ku, "trades": summ["trades"],
                      "sharpe_ann": summ["sharpe"]}])
    daily.rename(f"{ENSEMBLE}_0").to_frame().to_parquet(RESULTS / f"{ENSEMBLE}_trial_returns.parquet")
    d = [np.load(RESULTS / f"{c}_dev_dists.npz") for c in comps]
    null = combine_null([x["null"] for x in d], [int(x["null_counts"].sum()) for x in d])
    actual = float(((tr["gross"] - tr["fees"]) / (tr["qty"] * tr["entry_px"])).mean())
    boot_tot, boot_mdd = stats.block_bootstrap(tr["ret"].to_numpy(), N_BOOT)
    sym_R = tr.groupby("symbol")["R"].mean()
    out = {
        "code": ENSEMBLE, "name": "Ensemble of " + " + ".join(comps), "tf": "mixed", "version": 1, "round": 3,
        "formed": True, "components": comps,
        "hypothesis": "Several uncorrelated weekly-ish edges combined reach 2+ trades a week.",
        "rules": "Equal-weight portfolio of " + ", ".join(comps) + " at their chosen dev configs.",
        "grid": {}, "n_configs": 1, "chosen_cfg": 0, "params": {c: fam[c]["params"] for c in comps},
        "symbols": SYMBOLS, "holdout_scope": "mixed", "latency_min": LATENCY_MIN,
        "start": res.start.isoformat(), "end": daily.index[-1].isoformat(), "dev": summ,
        "stress_2x": {"avg_R": float(stress_tr["R"].mean())},
        "sr_daily": sr, "T": T, "skew": sk, "kurt": ku,
        "pbo": max(fam[c]["pbo"] for c in comps), "plateau": min(fam[c]["plateau"] for c in comps),
        "n_neighbors": 0, "inherited_gates": ["PBO", "plateau"],
        "null": {"actual_mean_ret": actual, "null_mean": float(null.mean()),
                 "null_p95": float(np.quantile(null, 0.95)),
                 "p_value": float((1 + np.sum(null >= actual)) / (1 + len(null)))},
        "bootstrap": {"ret_p5": np.quantile(boot_tot, 0.05), "ret_p50": np.quantile(boot_tot, 0.5),
                      "mdd_p50": np.quantile(boot_mdd, 0.5), "mdd_p95": np.quantile(boot_mdd, 0.95)},
        "benchmark": fam[comps[0]]["benchmark"],
        "symbols_positive": int((sym_R > 0).sum()),
        "symbol_table": [{"symbol": s, "trades": int((tr["symbol"] == s).sum()), "avg_R": float(v)}
                         for s, v in sym_R.items()],
        "years": metrics.year_table(res).reset_index(names="year").to_dict("records"),
        "confirm_years": _confirm_years(tr["R"].to_numpy(), summ.get("trades_per_year", 0)),
        "weeks_with_entry": _weeks_with_entry(tr, res.start, daily.index[-1]),
    }
    pd.DataFrame([{"cfg": 0, **summ, "sr": sr, "nbhd_sharpe": summ["sharpe"]}]) \
        .to_csv(RESULTS / f"{ENSEMBLE}_configs.csv", index=False)
    tr.to_parquet(RESULTS / f"{ENSEMBLE}_dev_trades.parquet")
    tr.to_parquet(RESULTS / f"{ENSEMBLE}_dev_scope_trades.parquet")
    res.equity.rename("equity").to_frame().to_parquet(RESULTS / f"{ENSEMBLE}_dev_equity.parquet")
    daily.rename("ret").to_frame().to_parquet(RESULTS / f"{ENSEMBLE}_dev_daily.parquet")
    np.savez_compressed(RESULTS / f"{ENSEMBLE}_dev_dists.npz", null=null, boot_tot=boot_tot,
                        boot_mdd=boot_mdd, pbo_logits=np.array([]))
    _save_json(path, out)
    return out


# ---------------------------------------------------------------- holdout

def _holdout_parts(code: str) -> dict[str, tuple[pd.Series, pd.DataFrame]]:
    """Run (once) and cache a family's chosen config on the holdout, per symbol."""
    cache = RESULTS / f"{code}_holdout_parts.parquet"
    if cache.exists():
        allt = pd.read_parquet(cache)
        days = pd.read_parquet(RESULTS / f"{code}_holdout_sym_daily.parquet")
        return {s: (days[s].dropna(), allt[allt["symbol"] == s]) for s in SYMBOLS}
    r = json.loads((RESULTS / f"{code}.json").read_text())
    strat = BY_CODE[code]
    parts = {}
    for sym in SYMBOLS:
        mk = engine.Market.load("full", sym)
        res = _run(strat, mk, sym, r["params"], HOLDOUT_START)
        parts[sym] = (aligned_daily(res, mk), res.trades.assign(symbol=sym))
        del mk
    pd.concat([p[1] for p in parts.values()], ignore_index=True).to_parquet(cache)
    pd.concat({s: p[0] for s, p in parts.items()}, axis=1).to_parquet(
        RESULTS / f"{code}_holdout_sym_daily.parquet")
    return parts


def _holdout_eval(r: dict, res: engine.Result, dev_ref: dict, dev_trades: pd.DataFrame) -> dict:
    s = metrics.summarize(res)
    n_h = max(len(res.trades), 2)
    _, mdd_dist = stats.block_bootstrap(dev_trades["ret"].to_numpy(), N_BOOT, n_out=n_h)
    mdd = trade_equity_mdd(res.trades["ret"].to_numpy()) if len(res.trades) else 0.0
    hg = {
        "holdout trades>=30": s["trades"] >= GATES["min_holdout_trades"],
        "holdout trades/yr>=52": s.get("trades_per_year", 0) >= MIN_FREQ,
        "holdout avg_R>0": s.get("avg_R", 0) > 0,
        "holdout Sharpe>=50% dev": s.get("sharpe", 0) >= GATES["holdout_sharpe_frac"] * dev_ref["sharpe"],
        "holdout MDD<=boot p95": mdd <= np.quantile(mdd_dist, 0.95),
    }
    return s | {"trade_mdd": mdd, "boot_mdd_p95": float(np.quantile(mdd_dist, 0.95)),
                "years": metrics.year_table(res).reset_index(names="year").to_dict("records")}, hg


def validate(codes: list[str], reason: str) -> None:
    done = [c for c in codes if "holdout" in json.loads((RESULTS / f"{c}.json").read_text())]
    if done:
        raise SystemExit(f"holdout already evaluated for {done}; it is used once per strategy")
    store.unlock_holdout(f"{reason} [{','.join(codes)}]")
    for code in codes:
        r = json.loads((RESULTS / f"{code}.json").read_text())
        if code == ENSEMBLE:
            comps = r["components"]
            parts_by = {c: _holdout_parts(c) for c in comps}
            scoped = []
            for c in comps:
                scope = ALTS if BY_CODE[c].holdout_scope == "alts" else SYMBOLS
                res_c, daily_c = assemble([parts_by[c][s] for s in scope])
                scoped.append((daily_c, res_c.trades))
            res, _ = assemble(scoped)
            res_all = res
            dev_ref, status = r["dev"], "partly alts-only" if any(
                BY_CODE[c].holdout_scope == "alts" for c in comps) else "clean"
        else:
            parts = _holdout_parts(code)
            scope = ALTS if r["holdout_scope"] == "alts" else SYMBOLS
            res, _ = assemble([parts[s] for s in scope])
            res_all, _ = assemble([parts[s] for s in SYMBOLS])
            dev_ref = r.get("dev_scope", r["dev"]) if r["holdout_scope"] == "alts" else r["dev"]
            status = "clean (4 alts; BTC info only)" if r["holdout_scope"] == "alts" else "clean"
        dev_trades = pd.read_parquet(RESULTS / f"{code}_dev_scope_trades.parquet")
        h, hg = _holdout_eval(r, res, dev_ref, dev_trades)
        r["holdout"] = h
        r["holdout_all"] = metrics.summarize(res_all)
        r["holdout_gates"] = hg
        r["holdout_status"] = status
        r["final_pass"] = bool(r.get("dev_pass")) and all(hg.values())
        res.trades.to_parquet(RESULTS / f"{code}_holdout_trades.parquet")
        res.equity.rename("equity").to_frame().to_parquet(RESULTS / f"{code}_holdout_equity.parquet")
        _save_json(RESULTS / f"{code}.json", r)
        print(f"{code} holdout [{status}]: trades={h['trades']} ({h.get('trades_per_year', 0):.0f}/yr) "
              f"avgR={h.get('avg_R', 0):.3f} sharpe={h.get('sharpe', 0):.2f} mdd={h['trade_mdd']:.1%} -> "
              f"{'PASS' if r['final_pass'] else 'FAIL'} {hg}")


def cmd_sweep(codes: list[str]) -> None:
    from btcq.research.run import cmd_gates
    for code in codes:
        r = sweep_family(BY_CODE[code])
        d = r["dev"]
        print(f"{code} swept {r['n_configs']} configs in {r['runtime_s']:.0f}s -> {r['params']} "
              f"trades={d['trades']} ({d.get('trades_per_year', 0):.0f}/yr) avgR={d.get('avg_R', 0):.3f} "
              f"sharpe={d.get('sharpe', 0):.2f} symbols+={r['symbols_positive']}/5", flush=True)
    cmd_gates()
    if build_ensemble() is not None:
        print("F7 ensemble formed; re-gating with it in the registry")
        cmd_gates()
    else:
        print("F7 not formed:", json.loads((RESULTS / f"{ENSEMBLE}.json").read_text())["reason"])
