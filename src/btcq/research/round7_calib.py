"""Round 7 null calibration: the registered null's rejection rate on synthetic panels without an edge (and its
power on planted edges), run before any Round 7 data is touched.

    python -m btcq.research.run r7-calibrate
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from btcq.backtest import rotation_sim as RS
from btcq.config import ROOT
from btcq.data import universe as U
from btcq.research import guards
from btcq.strategies import r7_rotation as R

CALIB_CSV = ROOT / "research" / "round7" / "null_calibration.csv"
CALIB_FILES = ["src/btcq/backtest/rotation_sim.py", "src/btcq/strategies/r7_rotation.py",
               "src/btcq/research/round7_calib.py"]
PENALTY = 0.01                  # Round 4's delisting penalty (round4.SPEC.delist_penalty)
MAX_NULL_P = 0.05
CAL_REPS = 1000
CAL_DRAWS = 499
CAL_SEED = 77
CAL_MAX_FPR = 0.07
SCENARIO_NAMES = ("iid", "factor", "garch", "churn", "cluster", "beta_tilt")


def synthetic_panel(rng: np.random.Generator, scenario: str, n_days: int = 1400, n_coins: int = 16,
                    edge: float = 0.0) -> tuple[RS.Panel, int, int]:
    """A daily panel for the null calibration. Prices are built from simple returns (close = cumprod(1 + R)).

    No-edge scenarios: on every day every coin has the same conditional expected simple return (the common
    drift), so no rule can beat a random choice of coins:
      iid      equal volatility, no factor;
      factor   volatility 2-8%/day, a market factor (bull, bear and flat regimes in the common drift) whose
               shocks scale with beta proportional to volatility, t(3) shocks;
      garch    factor plus volatility clustering (GARCH(1,1) on each coin's own shocks);
      churn    factor with 26 coins: 15 members a month, late listings, a delisting with a forced exit and a
               -30% exit price, random positive funding;
      cluster  beta independent of volatility, three sector factors, and common jump days (2%) with random
               loadings.
    beta_tilt (information, not a no-edge panel): the drift itself scales with beta (high-beta coins earn more in
    the bull regime), the structural tilt a momentum rule can pick up without skill.
    edge > 0 plants cross-sectional momentum: each coin's next return gains edge x vol x its standardised
    14-day idiosyncratic return."""
    if scenario not in SCENARIO_NAMES:
        raise ValueError(f"unknown calibration scenario {scenario!r}")
    k = 26 if scenario == "churn" else n_coins
    n = n_days
    t_ = np.arange(n)
    regime = np.where(t_ < 0.4 * n, 0.004, np.where(t_ < 0.7 * n, -0.003, 0.0005))
    if scenario == "iid":
        vol, beta = np.full(k, 0.04), np.zeros(k)
    elif scenario == "cluster":
        vol, beta = rng.uniform(0.02, 0.08, k), rng.uniform(0.5, 2.0, k)
    else:
        vol = rng.uniform(0.02, 0.08, k)
        beta = vol / 0.04
    z = rng.standard_normal((n, k)) if scenario == "iid" else rng.standard_t(3, size=(n, k)) / np.sqrt(3.0)
    sig = np.ones((n, k))
    if scenario == "garch":
        h = np.ones(k)
        for t in range(n):
            sig[t] = np.sqrt(h)
            h = 0.05 + 0.10 * (z[t] ** 2) * h + 0.85 * h
        sig = sig / np.sqrt((sig ** 2).mean())
    scale = vol if scenario == "iid" else 0.7 * vol
    idio = scale * sig * z
    shock = 0.03 * rng.standard_normal(n)
    common = np.zeros((n, k))
    if scenario == "cluster":
        sector = rng.integers(0, 3, k)
        f = 0.02 * rng.standard_normal((n, 3))
        jumps = (rng.random(n) < 0.02) * 0.08 * rng.standard_normal(n)
        common = f[:, sector] + jumps[:, None] * rng.uniform(0.5, 1.5, k)
    drift = (beta[None, :] * regime[:, None]) if scenario == "beta_tilt" else np.repeat(regime[:, None], k, axis=1)
    if scenario == "iid":
        drift = np.zeros((n, k))
    r = drift + beta * shock[:, None] + common + idio
    if edge:
        past = pd.DataFrame(idio).rolling(14).sum().shift(1).to_numpy()
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            zs = (past - np.nanmean(past, axis=1, keepdims=True)) / (np.nanstd(past, axis=1, keepdims=True) + 1e-12)
        r = r + edge * np.nan_to_num(zs) * vol
    r = np.clip(r, -0.9, None)
    close = 100.0 * np.cumprod(1.0 + r, axis=0)
    fill = np.vstack([close[:1], close[:-1]]) * np.exp(0.001 * rng.standard_normal((n, k)))
    days = pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC").as_unit("ms")
    member = np.ones((n, k), np.bool_)
    blocked = np.zeros((n, k), np.bool_)
    fx_pre = np.zeros((n, k), np.bool_)
    fx_px = np.full((n, k), np.nan)
    fund_pre = np.zeros((n, k))
    fund_post = np.zeros((n, k))
    if scenario == "churn":
        start = rng.integers(0, 400, k)
        start[:12] = 0
        for j in range(k):
            close[:start[j], j] = np.nan
            fill[:start[j] + 1, j] = np.nan
        month = days.month.to_numpy() + 12 * days.year.to_numpy()
        for mo in np.unique(month):
            rows = month == mo
            live = np.flatnonzero(np.isfinite(close[rows][0]))
            pick = rng.choice(live, size=min(15, len(live)), replace=False)
            m = np.zeros(k, np.bool_)
            m[pick] = True
            member[rows] = m
        j, a = k - 1, int(0.55 * n)                     # one coin stops trading for good
        fx_pre[a, j] = True
        fx_px[a, j] = close[a - 1, j] * 0.7
        close[a:, j] = np.nan
        fill[a:, j] = np.nan
        blocked[a - 1:, j] = True
        rate = 0.0001 + 0.0002 * rng.standard_normal((n, k))
        fund_pre = rate * np.nan_to_num(fill)
        fund_post = 2 * rate * np.nan_to_num(fill)
    p = RS.Panel(days=days, symbols=[f"C{j:02d}USDT" for j in range(k)], close=close, traded=np.isfinite(close),
                 fill_px=fill, fund_pre=fund_pre, fund_post=fund_post, fx_pre=fx_pre,
                 fx_post=np.zeros((n, k), np.bool_), fx_px=fx_px, fund_fx=np.zeros((n, k)), blocked=blocked,
                 days_since_gap=np.full((n, k), np.inf), member=member, rank=np.where(member, 12, 99),
                 slip=np.full((n, k), 0.0003), is_meme=np.zeros(k, np.bool_), funding_filled_h=np.zeros(k))
    return p, 70, n - 1


def calibration_p(p: RS.Panel, first: int, last: int, draws: int, seed: int, cfg=None) -> float:
    return R.null_test(p, cfg or R.CONFIGS[R.PRIMARY], first, last, draws, seed, PENALTY)["p_value"]


def calib_sha() -> str:
    """SHA256 of the code that decides the calibration: the simulator, the rule and null, this module."""
    return guards.code_sha(CALIB_FILES)


# (scenario, planted edge, replications, kind); the 7% limit applies to every "no-edge" row
CAL_SCENARIOS = [("iid", 0.0, CAL_REPS, "no-edge"), ("factor", 0.0, CAL_REPS, "no-edge"),
                 ("garch", 0.0, CAL_REPS, "no-edge"), ("churn", 0.0, CAL_REPS, "no-edge"),
                 ("cluster", 0.0, CAL_REPS, "no-edge"), ("beta_tilt", 0.0, 400, "tilt (information)"),
                 ("factor", 0.015, 200, "planted edge"), ("factor", 0.03, 200, "planted edge")]


def _cal_task(si: int, i0: int, i1: int, draws: int) -> list[float]:
    sc, edge, _, _ = CAL_SCENARIOS[si]
    out = []
    for i in range(i0, i1):
        rng = np.random.default_rng([CAL_SEED, si, i])        # one stream per replicate: chunking can't matter
        p, first, last = synthetic_panel(rng, sc, edge=edge)
        out.append(calibration_p(p, first, last, draws, CAL_SEED + 7919 * si + i))
    return out


def cmd_calibrate(draws: int = CAL_DRAWS, workers: int | None = None, chunk: int = 50) -> pd.DataFrame:
    """Rejection rate (p < 0.05) of the registered null on no-edge panels, and detection of planted edges."""
    import os
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    sha0 = calib_sha()
    tasks = [(si, i0, min(i0 + chunk, reps), draws) for si, (_, _, reps, _) in enumerate(CAL_SCENARIOS)
             for i0 in range(0, reps, chunk)]
    with ProcessPoolExecutor(max_workers=workers or max(1, (os.cpu_count() or 2) - 2)) as ex:
        res = list(ex.map(_cal_task, *zip(*tasks)))
    ps = {si: [] for si in range(len(CAL_SCENARIOS))}
    for (si, *_), r in zip(tasks, res):
        ps[si] += r
    rows = []
    for si, (sc, edge, reps, kind) in enumerate(CAL_SCENARIOS):
        x = np.asarray(ps[si])
        rej = float((x < MAX_NULL_P).mean())
        rows.append({"scenario": sc, "kind": kind, "planted_edge": edge, "reps": len(x), "draws": draws,
                     "reject_rate": rej, "se": float(np.sqrt(rej * (1 - rej) / len(x))), "mean_p": float(x.mean()),
                     "limit": CAL_MAX_FPR if kind == "no-edge" else None,
                     "ok": bool(rej <= CAL_MAX_FPR) if kind == "no-edge" else None, "calib_sha256": sha0})
    df = pd.DataFrame(rows)
    if calib_sha() != sha0:
        raise SystemExit("the calibration code changed while it ran: nothing written")
    CALIB_CSV.parent.mkdir(parents=True, exist_ok=True)
    U.write_csv(df, CALIB_CSV)
    print(df.to_string(index=False))
    print(f"calibration runtime {time.time() - t0:.0f}s, calibration code {sha0[:12]}")
    return df
