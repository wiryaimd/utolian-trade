"""Round 2 portfolio C12: S1 and S2 at their chosen dev configs, each risking 1% on one account."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from btcq.backtest import engine, metrics, stats
from btcq.config import HOLDOUT_START, Costs, Risk
from btcq.research import registry
from btcq.strategies import BY_CODE

CODE = "C12"
COMPONENTS = ("S1", "S2")
VERSION = 1


def _component_results():
    from btcq.research.run import RESULTS
    return {c: json.loads((RESULTS / f"{c}.json").read_text()) for c in COMPONENTS}


def _sleeves(market, comps, costs: Costs, start=None):
    return {c: engine.run(market, BY_CODE[c].compute(market, **comps[c]["params"]), costs,
                          start=start) for c in COMPONENTS}


def combine(market, sleeves: dict) -> tuple[engine.Result, pd.Series]:
    """Shared-account approximation: the combined daily return is the sum of the sleeves'."""
    from btcq.research.run import aligned_daily
    rets = pd.concat([aligned_daily(r, market).rename(c) for c, r in sleeves.items()], axis=1).fillna(0.0)
    combo = rets.sum(axis=1)
    eq0 = Risk().initial_equity
    equity = eq0 * (1.0 + combo).cumprod()
    trades = pd.concat([r.trades.assign(sleeve=c) for c, r in sleeves.items()], ignore_index=True)
    trades = trades.sort_values("entry_i").reset_index(drop=True)
    start = min(r.start for r in sleeves.values())
    return engine.Result(trades, equity, eq0, start), combo


def max_concurrent_notional(trades: pd.DataFrame) -> float:
    """Peak combined notional / equity across overlapping sleeve positions."""
    frac = (trades["qty"] * trades["entry_px"] / trades["eq_before"]).to_numpy()
    ev = sorted([(e, 1, f) for e, f in zip(trades["entry_i"], frac)] +
                [(x, 0, -f) for x, f in zip(trades["exit_i"], frac)])
    cur = peak = 0.0
    for _, _, f in ev:
        cur += f
        peak = max(peak, cur)
    return float(peak)


def run_dev(market, costs: Costs = Costs()) -> dict:
    from btcq.research.run import (N_BOOT, RESULTS, _save_json, benchmark, null_test)
    t0 = time.time()
    comps = _component_results()
    res, daily = combine(market, _sleeves(market, comps, costs))
    summ = metrics.summarize(res)
    sr, T, sk, ku = stats.sharpe_moments(daily)
    params = {c: comps[c]["params"] for c in COMPONENTS}
    registry.record([{"strategy": CODE, "version": VERSION, "params": params, "period": "dev", "sr": sr,
                      "T": T, "skew": sk, "kurt": ku, "trades": summ["trades"],
                      "sharpe_ann": summ["sharpe"]}])
    daily.rename(f"{CODE}_0").to_frame().to_parquet(RESULTS / f"{CODE}_trial_returns.parquet")
    stress_res, _ = combine(market, _sleeves(market, comps, costs.scaled(2)))
    stress = metrics.summarize(stress_res)
    nt = null_test(market, res, "4h", costs)
    boot_tot, boot_mdd = stats.block_bootstrap(res.trades["ret"].to_numpy(), N_BOOT)
    out = {
        "code": CODE, "name": "S1 + S2 combo", "tf": "4h", "version": VERSION,
        "hypothesis": ("Round 2: two 4h trend/breakout sleeves with different entry logic should "
                       "diversify each other's losing streaks."),
        "rules": ("S1 (n=40, k=3.5, channel) and S2 (bb_n=20, q=0.3, stop=atr, exit=trail) run "
                  "side by side, each risking 1% of the shared account per trade."),
        "grid": {}, "n_configs": 1, "chosen_cfg": 0, "params": params,
        "start": res.start.isoformat(), "end": market.days[-1].isoformat(),
        "dev": summ, "stress_2x": {k: stress.get(k) for k in ("avg_R", "pf", "total_return", "sharpe")},
        "sr_daily": sr, "T": T, "skew": sk, "kurt": ku,
        # Not defined for one fixed portfolio: inherit the worse component (pre-registered).
        "pbo": max(comps[c]["pbo"] for c in COMPONENTS),
        "plateau": min(comps[c]["plateau"] for c in COMPONENTS),
        "n_neighbors": 0, "inherited_gates": ["PBO", "plateau"],
        "null": {k: v for k, v in nt.items() if k != "dist"},
        "bootstrap": {"ret_p5": np.quantile(boot_tot, 0.05), "ret_p50": np.quantile(boot_tot, 0.5),
                      "mdd_p50": np.quantile(boot_mdd, 0.5), "mdd_p95": np.quantile(boot_mdd, 0.95)},
        "benchmark": benchmark(market, res.start),
        "years": metrics.year_table(res).reset_index(names="year").to_dict("records"),
        "max_notional": max_concurrent_notional(res.trades),
        "sleeve_corr": _sleeve_corr(market, comps, costs),
        "runtime_s": time.time() - t0,
    }
    pd.DataFrame([{"cfg": 0, **summ, "sr": sr, "nbhd_sharpe": summ["sharpe"]}]) \
        .to_csv(RESULTS / f"{CODE}_configs.csv", index=False)
    res.trades.to_parquet(RESULTS / f"{CODE}_dev_trades.parquet")
    res.equity.rename("equity").to_frame().to_parquet(RESULTS / f"{CODE}_dev_equity.parquet")
    np.savez_compressed(RESULTS / f"{CODE}_dev_dists.npz", null=nt["dist"], boot_tot=boot_tot,
                        boot_mdd=boot_mdd, pbo_logits=np.array([]))
    _save_json(RESULTS / f"{CODE}.json", out)
    return out


def _sleeve_corr(market, comps, costs) -> float:
    from btcq.research.run import aligned_daily
    s = _sleeves(market, comps, costs)
    a, b = (aligned_daily(s[c], market) for c in COMPONENTS)
    return float(a.corr(b))


def run_holdout(market) -> tuple[engine.Result, dict]:
    comps = _component_results()
    res, _ = combine(market, _sleeves(market, comps, Costs(), start=HOLDOUT_START))
    return res, {"max_notional": max_concurrent_notional(res.trades)}
