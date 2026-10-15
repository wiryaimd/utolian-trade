"""Research CLI.

    python -m btcq.research.run download
    python -m btcq.research.run sweep --strategy all
    python -m btcq.research.run validate --finalists --unlock-holdout
    python -m btcq.research.run report
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from btcq.backtest import engine, metrics, stats
from btcq.config import HOLDOUT_START, REPORTS_DIR, Costs, Risk
from btcq.data import store
from btcq.research import registry
from btcq.strategies import ALL, BY_CODE

RESULTS = REPORTS_DIR / "results"

GATES = {"min_trades": 100, "min_pf": 1.2, "min_t": 2.0, "min_dsr": 0.95, "max_pbo": 0.5,
         "min_plateau": 0.7, "max_null_p": 0.05, "max_year_share": 0.5,
         "min_holdout_trades": 30, "holdout_sharpe_frac": 0.5}
N_PERM = 1000
N_BOOT = 10_000


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(obj), indent=1))


def aligned_daily(res: engine.Result, market: engine.Market) -> pd.Series:
    """Daily returns over every day from the run's start (flat days count as 0)."""
    days = market.days[market.days >= res.start.floor("D")]
    eq = res.equity.reindex(days).ffill().fillna(res.initial_equity)
    prev = eq.shift(1)
    prev.iloc[0] = res.initial_equity
    return eq / prev - 1.0


def trade_equity_mdd(rets: np.ndarray) -> float:
    return metrics.max_drawdown(np.r_[1.0, np.cumprod(1.0 + np.asarray(rets))])


def null_test(market: engine.Market, res: engine.Result, tf: str, costs: Costs) -> dict:
    t = res.trades
    actual = ((t["gross"] - t["fees"]) / (t["qty"] * t["entry_px"])).mean()
    ex = market.exec_idx(tf)
    start_i = market.index_at(res.start)
    cand = ex[(ex >= start_i) & (ex < len(market.ts) - 1)]
    dist = stats.random_entry_null(market.o, cand, (t["exit_i"] - t["entry_i"]).to_numpy(),
                                   t["dir"].to_numpy(), 2 * (costs.taker + costs.slippage), N_PERM)
    p = (1 + np.sum(dist >= actual)) / (1 + len(dist))
    return {"actual_mean_ret": actual, "null_mean": dist.mean(), "null_p95": np.quantile(dist, 0.95),
            "p_value": p, "dist": dist}


def benchmark(market: engine.Market, start: pd.Timestamp) -> dict:
    close = market.m1.set_index(pd.to_datetime(market.m1["ts"], unit="ms", utc=True))["close"]
    d = close[close.index >= start].resample("D").last().dropna()
    r = d.pct_change().dropna()
    years = len(d) / 365
    return {"total_return": d.iloc[-1] / d.iloc[0] - 1,
            "cagr": (d.iloc[-1] / d.iloc[0]) ** (1 / years) - 1,
            "sharpe": r.mean() / r.std() * np.sqrt(365),
            "max_dd": metrics.max_drawdown(d.to_numpy())}


# ---------------------------------------------------------------- sweep (dev)

def sweep_strategy(strat, market: engine.Market, costs: Costs = Costs()) -> dict:
    t0 = time.time()
    configs = strat.configs()
    rows, cols, trial_rows = [], [], []
    for i, p in enumerate(configs):
        res = engine.run(market, strat.compute(market, **p), costs, start=strat.data_start)
        s = metrics.summarize(res)
        dr = aligned_daily(res, market)
        sr, T, sk, ku = stats.sharpe_moments(dr)
        rows.append({"cfg": i, **p, **s, "sr": sr, "T": T, "skew": sk, "kurt": ku})
        cols.append(dr.to_numpy())
        trial_rows.append({"strategy": strat.code, "version": strat.version, "params": p,
                           "period": "dev", "sr": sr, "T": T, "skew": sk, "kurt": ku,
                           "trades": s["trades"], "sharpe_ann": s["sharpe"]})
    registry.record(trial_rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(np.column_stack(cols), index=market.days[-len(cols[0]):],
                 columns=[f"{strat.code}_{i}" for i in range(len(cols))]) \
        .to_parquet(RESULTS / f"{strat.code}_trial_returns.parquet")
    df = pd.DataFrame(rows)
    pbo, logits = stats.pbo_cscv(np.column_stack(cols))
    nbrs = stats.grid_neighbors(strat.grid, configs)
    sh = df["sharpe"].to_numpy()
    df["nbhd_sharpe"] = [np.mean(sh[[i] + nbrs[i]]) for i in range(len(df))]
    eligible = df[df["trades"] >= GATES["min_trades"]]
    pick = int((eligible if len(eligible) else df)["nbhd_sharpe"].idxmax())
    params = configs[pick]
    nb = nbrs[pick]
    plateau = float(np.mean(df.loc[nb, "total_return"].to_numpy() > 0)) if nb else 0.0

    res = engine.run(market, strat.compute(market, **params), costs, start=strat.data_start)
    summ = metrics.summarize(res)
    stress = metrics.summarize(engine.run(market, strat.compute(market, **params), costs.scaled(2),
                                          start=strat.data_start))
    nt = null_test(market, res, strat.tf, costs)
    boot_tot, boot_mdd = (stats.block_bootstrap(res.trades["ret"].to_numpy(), N_BOOT)
                          if len(res.trades) > 1 else (np.zeros(1), np.zeros(1)))
    out = {
        "code": strat.code, "name": strat.name, "tf": strat.tf, "version": strat.version,
        "hypothesis": strat.hypothesis, "rules": strat.rules, "grid": strat.grid,
        "n_configs": len(configs), "chosen_cfg": pick, "params": params,
        "start": res.start.isoformat(), "end": market.days[-1].isoformat(),
        "dev": summ, "stress_2x": {k: stress.get(k) for k in ("avg_R", "pf", "total_return", "sharpe")},
        "sr_daily": rows[pick]["sr"], "T": rows[pick]["T"], "skew": rows[pick]["skew"],
        "kurt": rows[pick]["kurt"], "pbo": pbo, "plateau": plateau, "n_neighbors": len(nb),
        "null": {k: v for k, v in nt.items() if k != "dist"},
        "bootstrap": {"ret_p5": np.quantile(boot_tot, 0.05), "ret_p50": np.quantile(boot_tot, 0.5),
                      "mdd_p50": np.quantile(boot_mdd, 0.5), "mdd_p95": np.quantile(boot_mdd, 0.95)},
        "benchmark": benchmark(market, res.start),
        "years": metrics.year_table(res).reset_index(names="year").to_dict("records"),
        "runtime_s": time.time() - t0,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / f"{strat.code}_configs.csv", index=False)
    res.trades.to_parquet(RESULTS / f"{strat.code}_dev_trades.parquet")
    res.equity.rename("equity").to_frame().to_parquet(RESULTS / f"{strat.code}_dev_equity.parquet")
    np.savez_compressed(RESULTS / f"{strat.code}_dev_dists.npz", null=nt["dist"], boot_tot=boot_tot,
                        boot_mdd=boot_mdd, pbo_logits=logits)
    _save_json(RESULTS / f"{strat.code}.json", out)
    return out


def dev_gates(r: dict, n_trials: int, var_sr: float, n_eff: float) -> dict:
    d = r["dev"]
    sr, T, sk, ku = r["sr_daily"], r["T"], r["skew"], r["kurt"]
    v0 = stats.null_sr_variance(T)
    # Amendment 1 (see research/PREREGISTRATION.md): zero-skill sampling variance and the
    # effective number of independent trials. The pre-registered and all-trials-independent
    # variants are kept alongside for transparency.
    r["dsr"] = stats.dsr(sr, T, sk, ku, v0, max(int(round(n_eff)), 2))
    r["dsr_n_all"] = stats.dsr(sr, T, sk, ku, v0, n_trials)
    r["dsr_prereg"] = stats.dsr(sr, T, sk, ku, var_sr, n_trials)
    r["psr0"] = stats.psr(sr, T, sk, ku)
    r["exp_max_sharpe_ann"] = stats.expected_max_sr(v0, max(int(round(n_eff)), 2)) * np.sqrt(365)
    r["n_trials"], r["n_eff"], r["var_sr"] = n_trials, n_eff, var_sr
    g = {
        "trades>=100": d["trades"] >= GATES["min_trades"],
        "PF>=1.2": d.get("pf", 0) >= GATES["min_pf"],
        "t-stat>=2": d.get("t_stat", 0) >= GATES["min_t"],
        "2x-cost avg_R>0": (r["stress_2x"].get("avg_R") or 0) > 0,
        "DSR>=0.95": r["dsr"] >= GATES["min_dsr"],
        "PBO<=0.5": r["pbo"] is not None and r["pbo"] <= GATES["max_pbo"],
        "plateau>=70%": r["plateau"] >= GATES["min_plateau"],
        "null p<0.05": r["null"]["p_value"] < GATES["max_null_p"],
        "year share<=50%": (d.get("max_year_share") or np.inf) <= GATES["max_year_share"],
    }
    r["gates"] = g
    r["dev_pass"] = all(g.values())
    return r


def cmd_sweep(codes: list[str]) -> None:
    market = engine.Market.load("dev")
    for code in codes:
        if code == "C12":
            from btcq.research import combo
            r = combo.run_dev(market)
        else:
            r = sweep_strategy(BY_CODE[code], market)
        print(f"{code} swept {r['n_configs']} configs in {r['runtime_s']:.0f}s -> chosen {r['params']} "
              f"trades={r['dev']['trades']} avgR={r['dev'].get('avg_R', 0):.3f} "
              f"sharpe={r['dev'].get('sharpe', 0):.2f}")
    cmd_gates()


def all_trial_returns() -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in sorted(RESULTS.glob("*_trial_returns.parquet"))]
    return pd.concat(frames, axis=1).fillna(0.0)


def cmd_gates() -> None:
    n, var = registry.multiple_testing_inputs("dev")
    trial_rets = all_trial_returns()
    pr_now = stats.effective_trials(trial_rets.to_numpy())
    n_eff = registry.ratcheted_effective_trials(pr_now, trial_rets.shape[1])
    if trial_rets.shape[1] != n:
        print(f"WARNING: registry has {n} dev trials but {trial_rets.shape[1]} return series on disk")
    from btcq.research import round3
    for code in [s.code for s in ALL] + ["C12"] + round3.CODES + [round3.ENSEMBLE]:
        p = RESULTS / f"{code}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        if r.get("formed") is False:
            continue
        r = dev_gates(r, n, var, n_eff)
        if r.get("round") == 3:
            r = round3.augment_gates(r)
        if "holdout" in r and r.get("round") != 3:
            r.setdefault("holdout_status", holdout_status(code))
        if r.get("holdout_gates") is not None and r.get("holdout_status") != "contaminated":
            r["final_pass"] = bool(r["dev_pass"]) and all(r["holdout_gates"].values())
        _save_json(p, r)
        failed = [k for k, v in r["gates"].items() if not v]
        print(f"{code} DSR={r['dsr']:.3f} PBO={r['pbo']:.2f} null_p={r['null']['p_value']:.3f} "
              f"-> {'PASS' if r['dev_pass'] else 'FAIL: ' + ', '.join(failed)}")
    print(f"registry: {n} dev trials, effective N used={n_eff:.1f} (current participation ratio "
          f"{pr_now:.1f}), cross-sectional var(SR)={var:.2e}")


# ---------------------------------------------------------------- holdout

def holdout_status(code: str) -> str:
    """'clean', or 'contaminated' when the idea was shaped by an already-seen holdout (pre-registered)."""
    if code == "C12" or BY_CODE[code].holdout_contaminated:
        return "contaminated"
    return "clean"


def cmd_validate(codes: list[str], reason: str) -> None:
    done = [c for c in codes if "holdout" in json.loads((RESULTS / f"{c}.json").read_text())]
    if done:
        raise SystemExit(f"holdout already evaluated for {done}; it is used once per strategy")
    store.unlock_holdout(f"{reason} [{','.join(codes)}]")
    market = engine.Market.load("full")
    for code in codes:
        p = RESULTS / f"{code}.json"
        r = json.loads(p.read_text())
        extra = {}
        if code == "C12":
            from btcq.research import combo
            res, extra = combo.run_holdout(market)
        else:
            res = engine.run(market, BY_CODE[code].compute(market, **r["params"]), start=HOLDOUT_START)
        s = metrics.summarize(res)
        dev_rets = pd.read_parquet(RESULTS / f"{code}_dev_trades.parquet")["ret"].to_numpy()
        n_h = max(len(res.trades), 2)
        _, mdd_dist = stats.block_bootstrap(dev_rets, N_BOOT, n_out=n_h)
        mdd_trades = trade_equity_mdd(res.trades["ret"].to_numpy())
        hg = {
            "holdout trades>=30": s["trades"] >= GATES["min_holdout_trades"],
            "holdout avg_R>0": s.get("avg_R", 0) > 0,
            "holdout Sharpe>=50% dev": s.get("sharpe", 0) >= GATES["holdout_sharpe_frac"] * r["dev"]["sharpe"],
            "holdout MDD<=boot p95": mdd_trades <= np.quantile(mdd_dist, 0.95),
        }
        r["holdout"] = s | extra | {"trade_mdd": mdd_trades, "boot_mdd_p95": np.quantile(mdd_dist, 0.95),
                                    "benchmark": benchmark(market, res.start),
                                    "years": metrics.year_table(res).reset_index(names="year").to_dict("records")}
        r["holdout_gates"] = hg
        r["holdout_status"] = holdout_status(code)
        # A contaminated holdout is informational only: it can neither pass nor fail.
        r["final_pass"] = (None if r["holdout_status"] == "contaminated"
                           else bool(r.get("dev_pass")) and all(hg.values()))
        res.trades.to_parquet(RESULTS / f"{code}_holdout_trades.parquet")
        res.equity.rename("equity").to_frame().to_parquet(RESULTS / f"{code}_holdout_equity.parquet")
        _save_json(p, r)
        print(f"{code} holdout: trades={s['trades']} avgR={s.get('avg_R', 0):.3f} "
              f"sharpe={s.get('sharpe', 0):.2f} mdd={mdd_trades:.1%} -> "
              f"{ {True: 'PASS', False: 'FAIL', None: 'INFO ONLY (contaminated)'}[r['final_pass']] } {hg}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["download", "sweep", "gates", "validate", "report", "r3-sweep",
                                    "r3-validate"])
    ap.add_argument("--strategy", default="all")
    ap.add_argument("--finalists", action="store_true")
    ap.add_argument("--unlock-holdout", action="store_true")
    ap.add_argument("--reason", default="final holdout evaluation of dev-gate finalists")
    a = ap.parse_args()
    codes = [s.code for s in ALL] if a.strategy == "all" else a.strategy.split(",")
    if a.cmd == "download":
        print(store.build())
    elif a.cmd == "sweep":
        cmd_sweep(codes)
    elif a.cmd == "gates":
        cmd_gates()
    elif a.cmd == "validate":
        if not a.unlock_holdout:
            raise SystemExit("validate needs --unlock-holdout")
        if a.finalists:
            codes = [c for c in codes if json.loads((RESULTS / f"{c}.json").read_text()).get("dev_pass")]
            if not codes:
                raise SystemExit("no strategy passed the dev gates; holdout stays locked")
        cmd_validate(codes, a.reason)
    elif a.cmd == "r3-sweep":
        from btcq.research import round3
        round3.cmd_sweep(round3.CODES if a.strategy == "all" else a.strategy.split(","))
    elif a.cmd == "r3-validate":
        from btcq.research import round3
        if not a.unlock_holdout:
            raise SystemExit("r3-validate needs --unlock-holdout")
        codes = round3.CODES + [round3.ENSEMBLE] if a.strategy == "all" else a.strategy.split(",")
        rs = {c: json.loads((RESULTS / f"{c}.json").read_text()) for c in codes
              if (RESULTS / f"{c}.json").exists()}
        codes = [c for c, r in rs.items() if r.get("formed", True) and (r.get("dev_pass") or not a.finalists)]
        if not codes:
            raise SystemExit("no round-3 family passed the dev gates; holdout stays locked")
        round3.validate(codes, a.reason)
    elif a.cmd == "report":
        from btcq.research import report
        print(report.build())


if __name__ == "__main__":
    main()
