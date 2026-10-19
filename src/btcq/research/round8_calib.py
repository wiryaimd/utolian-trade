"""Round 8 null calibration: the registered date-level null's rejection rate on synthetic OHLC panels without an edge
(and its power on planted edges), run before any Round 8 data is touched.

    python -m btcq.research.run r8-calibrate

Generator. Each market's daily simple return is R = mu_t + sigma_t z (z Gaussian or a unit-variance t(4)), so the
expected simple return is mu_t on every day by construction (Round 7's lesson: a fixed log drift creates a fake
edge). The day is one Brownian bridge in log price from the previous close to log(1 + R): the first step, the
overnight gap, carries 25% of the variance, then 78 equal intraday steps run from the open to the close, so the high
and the low are the path's extremes between the open and the close. Panels follow the dev ramp: market 0 alone for
about 6 years, then market 1 and, about 1.2 years later, market 2, about 5,600 window days on a Monday-Friday
calendar.

No-edge scenarios (the expected return never depends on IBS given the date-level strata):
  iid      constant volatility, drift 3 bp/day, Gaussian shocks;
  garch    GJR-GARCH with the leverage effect, t(4) shocks;
  varprem  garch, plus an expected return mu0 + 3 sigma_t^2: low-IBS days predict higher volatility and so a higher
           return; the volatility strata must absorb it;
  regime   a persistent market-wide bull/bear drift of +/-8 bp/day (switching about once a year); the trend strata
           must absorb it;
  factor   3 correlated markets (rho about 0.85 / 0.7) with a common GARCH factor and shared intraday paths, so IBS
           signals coincide across markets;
  weekend  a 3-day variance Friday -> Monday (the extra variance overnight) and a Monday drift of -5 bp/day.
Information rows: a strong bull drift of 8 bp/day, and planted edges on the combined garch + factor + regime base:
+kappa sigma_t after IBS <= 0.2 and -kappa sigma_t after IBS >= 0.8, kappa in {0.05, 0.10}.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from numba import njit

from btcq.config import ROOT
from btcq.research import guards
from btcq.strategies import r8_ibs as R

CALIB_CSV = ROOT / "research" / "round8" / "null_calibration.csv"
CALIB_FILES = ["src/btcq/strategies/r8_ibs.py", "src/btcq/strategies/r6_macro.py", "src/btcq/features/ta.py",
               "src/btcq/research/round8_calib.py"]
CAL_REPS = 1000
CAL_DRAWS = 499
CAL_SEED = 88
CAL_MAX_P05 = 0.07          # every no-edge row, both rules: rejection at p < 0.05 at most 7%
CAL_MAX_P025 = 0.035        # and at p < 0.025 (the gate's own threshold) at most 3.5%
CAL_MIN_DETECT = 0.80       # the planted kappa = 0.10 edge must be rejected at p < 0.025 in >= 80% of panels, both rules
SCENARIOS = ("iid", "garch", "varprem", "regime", "factor", "weekend", "bull", "planted")
# (scenario, planted kappa, replications, kind); the limits apply to every "no-edge" row
CAL_ROWS = [("iid", 0.0, CAL_REPS, "no-edge"), ("garch", 0.0, CAL_REPS, "no-edge"),
            ("varprem", 0.0, CAL_REPS, "no-edge"), ("regime", 0.0, CAL_REPS, "no-edge"),
            ("factor", 0.0, CAL_REPS, "no-edge"), ("weekend", 0.0, CAL_REPS, "no-edge"),
            ("bull", 0.0, 400, "drift (information)"), ("planted", 0.05, 200, "planted edge"),
            ("planted", 0.10, 200, "planted edge")]

N_DAYS = 5700               # Monday-Friday rows of market 0
WIN_START = 105             # the window starts at row 105 (dev: 1993-01-29 -> 1993-07-01)
STARTS = (0, 1526, 1828)    # first rows of markets 1 and 2: ready (90 returns) about 6.0 and 7.2 years into the window
VOLS = np.array([0.012, 0.017, 0.021])
LOADINGS = np.array([0.95, 0.895, 0.737])       # one-factor loadings: rho 0.85 (0-1), 0.70 (0-2), 0.66 (1-2)
STEPS = 78
GAP_SHARE = 0.25
MU0 = 3e-4
BULL_MU = 8e-4
REGIME_MU = 8e-4
REGIME_SWITCH = 1 / 252
VARPREM_LAMBDA = 3.0
WEEKEND_VAR = 3.0
MONDAY_MU = -5e-4
GJR = (0.02, 0.03, 0.10, 0.90)                  # omega, alpha, gamma (leverage), beta: unit unconditional variance


def _t4(rng, shape):
    return rng.standard_t(4, size=shape) / np.sqrt(2.0)


def _gjr(z: np.ndarray) -> np.ndarray:
    """Conditional variance path h_t (unit unconditional mean) driven by the shocks z (h_t uses z up to t-1)."""
    om, a, g, b = GJR
    n = z.shape[0]
    h = np.ones(z.shape)
    for t in range(1, n):
        h[t] = om + (a + g * (z[t - 1] < 0)) * h[t - 1] * z[t - 1] ** 2 + b * h[t - 1]
    return h


@njit(cache=True)
def _paths(B, tau, sig, mu, z, kappa, theta):
    """Day by day: R = mu + edge + sig z (edge = +kappa sig after IBS <= theta, -kappa sig after IBS >= 1 - theta on
    the previous day), clipped at -50%; the log path is sig x bridge + tau x log(1 + R) from the previous close."""
    n, M, K = B.shape
    O = np.empty((n, M))
    H = np.empty((n, M))
    L = np.empty((n, M))
    C = np.empty((n, M))
    ib = np.full(M, np.nan)
    prev = np.full(M, 100.0)
    for t in range(n):
        for m in range(M):
            e = 0.0
            if kappa != 0.0 and not np.isnan(ib[m]):
                if ib[m] <= theta:
                    e = kappa * sig[t, m]
                elif ib[m] >= 1.0 - theta:
                    e = -kappa * sig[t, m]
            r = mu[t, m] + e + sig[t, m] * z[t, m]
            if r < -0.5:
                r = -0.5
            ell = np.log1p(r)
            hi = -1e300
            lo = 1e300
            op = 0.0
            for k in range(K):
                x = sig[t, m] * B[t, m, k] + tau[t, k] * ell
                if k == 0:
                    op = x
                if x > hi:
                    hi = x
                if x < lo:
                    lo = x
            O[t, m] = prev[m] * np.exp(op)
            H[t, m] = prev[m] * np.exp(hi)
            L[t, m] = prev[m] * np.exp(lo)
            C[t, m] = prev[m] * np.exp(ell)
            ib[m] = (C[t, m] - L[t, m]) / (H[t, m] - L[t, m]) if H[t, m] > L[t, m] else np.nan
            prev[m] = C[t, m]
    return O, H, L, C


def synthetic_markets(rng: np.random.Generator, scenario: str, kappa: float = 0.0,
                      n_days: int = N_DAYS) -> tuple[list[R.MarketRows], int, int]:
    """(3 markets, window start ms, window end ms) for one calibration panel."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown calibration scenario {scenario!r}")
    n, M = n_days, 3
    days = pd.bdate_range("1993-01-29", periods=n, tz="UTC").as_unit("ms")
    ts = days.asi8.astype(np.int64)
    after_gap = np.r_[False, np.diff(ts) >= 2 * R.DAY_MS]
    gaussian = scenario in ("iid", "regime", "weekend", "bull")
    garch = scenario in ("garch", "varprem", "factor", "planted")
    factor = scenario in ("factor", "planted")
    # daily shocks and volatility
    if factor:
        f = rng.standard_normal(n) if gaussian else _t4(rng, n)
        e = rng.standard_normal((n, M)) if gaussian else _t4(rng, (n, M))
        z = LOADINGS * f[:, None] + np.sqrt(1 - LOADINGS ** 2) * e
        h = np.repeat(_gjr(f)[:, None], M, axis=1)
    else:
        z = rng.standard_normal((n, M)) if gaussian else _t4(rng, (n, M))
        h = _gjr(z) if garch else np.ones((n, M))
    var_mult = np.ones(n)
    gap_share = np.full(n, GAP_SHARE)
    if scenario == "weekend":
        var_mult = np.where(after_gap, WEEKEND_VAR, 1.0)
        gap_share = np.where(after_gap, (GAP_SHARE + WEEKEND_VAR - 1) / WEEKEND_VAR, GAP_SHARE)
    sig = VOLS * np.sqrt(h * var_mult[:, None])
    # expected simple returns
    mu = np.full((n, M), BULL_MU if scenario == "bull" else MU0)
    if scenario == "varprem":
        mu = MU0 + VARPREM_LAMBDA * sig ** 2
    if scenario in ("regime", "planted"):
        flips = rng.random(n) < REGIME_SWITCH
        state = (np.cumsum(flips) + int(rng.integers(0, 2))) % 2
        mu = np.repeat(np.where(state == 0, REGIME_MU, -REGIME_MU)[:, None], M, axis=1)
    if scenario == "weekend":
        mu = np.where(after_gap[:, None], MONDAY_MU, MU0) * np.ones((1, M))
    # bridges: one standard Brownian motion per day and market on the variance-time grid (the gap, then 78 steps)
    var_steps = np.concatenate([gap_share[:, None], np.repeat(((1 - gap_share) / STEPS)[:, None], STEPS, axis=1)],
                               axis=1)                                       # (n, 79)
    tau = np.cumsum(var_steps, axis=1)
    tau[:, -1] = 1.0
    if factor:
        wf = rng.standard_normal((n, 1, STEPS + 1))
        we = rng.standard_normal((n, M, STEPS + 1))
        inc = (LOADINGS[None, :, None] * wf + np.sqrt(1 - LOADINGS ** 2)[None, :, None] * we)
    else:
        inc = rng.standard_normal((n, M, STEPS + 1))
    W = np.cumsum(inc * np.sqrt(var_steps)[:, None, :], axis=2)
    B = W - tau[:, None, :] * W[:, :, -1:]
    O, H, L, C = _paths(np.ascontiguousarray(B), np.ascontiguousarray(tau), np.ascontiguousarray(sig),
                        np.ascontiguousarray(mu), np.ascontiguousarray(z), float(kappa), 0.20)
    mk = [R.MarketRows(f"M{j}", ts[a:].copy(), O[a:, j].copy(), H[a:, j].copy(), L[a:, j].copy(), C[a:, j].copy())
          for j, a in enumerate(STARTS)]                  # a market's rows before its start are dropped
    return mk, int(ts[WIN_START]), int(ts[-1] + R.DAY_MS)


def calibration_p(mkts: list[R.MarketRows], start_ms: int, end_ms: int, draws: int, seed: int) -> tuple[float, ...]:
    """The null's p-value for each primary (IBS-L, IBS-LS) on one panel."""
    sp = R.strata_panel(mkts, start_ms, end_ms)
    out = []
    for k, ci in enumerate(R.PRIMARIES.values()):
        v = R.held_vectors(sp, mkts, R.CONFIGS[ci])
        out.append(R.null_test(sp, v, draws, seed + 1_000_003 * k)["p_value"])
    return tuple(out)


def lib_versions_calib() -> dict:
    """The random streams (numpy) and the strata (pandas) depend on these: recorded with the calibration."""
    import platform

    import numba
    return {"numpy_version": np.__version__, "pandas_version": pd.__version__, "numba_version": numba.__version__,
            "python_version": platform.python_version()}


def calib_sha() -> str:
    """SHA256 of the code that decides the calibration: the rule and null, the sizing it reuses, this module."""
    return guards.code_sha(CALIB_FILES)


def _cal_task(ri: int, i0: int, i1: int, draws: int) -> list[tuple[float, ...]]:
    sc, kappa, _, _ = CAL_ROWS[ri]
    out = []
    for i in range(i0, i1):
        rng = np.random.default_rng([CAL_SEED, ri, i])          # one stream per replicate: chunking can't matter
        mk, a, b = synthetic_markets(rng, sc, kappa)
        out.append(calibration_p(mk, a, b, draws, CAL_SEED * 100_003 + 7919 * ri + i))
    return out


def run_rows(draws: int = CAL_DRAWS, workers: int | None = None, chunk: int = 25, rows=None,
             reps: int | None = None) -> dict[int, np.ndarray]:
    """{row index: (reps, 2) p-values}, over a process pool; rows/reps restrict the run (tests)."""
    import os
    from concurrent.futures import ProcessPoolExecutor
    rows = list(range(len(CAL_ROWS))) if rows is None else list(rows)
    tasks = [(ri, i0, min(i0 + chunk, reps or CAL_ROWS[ri][2]), draws) for ri in rows
             for i0 in range(0, reps or CAL_ROWS[ri][2], chunk)]
    if workers == 1:
        res = [_cal_task(*t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers or max(1, (os.cpu_count() or 2) - 2)) as ex:
            res = list(ex.map(_cal_task, *zip(*tasks)))
    ps = {ri: [] for ri in rows}
    for (ri, *_), r in zip(tasks, res):
        ps[ri] += r
    return {ri: np.asarray(v, np.float64) for ri, v in ps.items()}


def cmd_calibrate(draws: int = CAL_DRAWS, workers: int | None = None) -> pd.DataFrame:
    """Rejection rates of the registered null (p < 0.05 and p < 0.025) on no-edge panels, for both primaries, and
    detection of planted edges. Writes research/round8/null_calibration.csv."""
    t0 = time.time()
    sha0 = calib_sha()
    ps = run_rows(draws, workers)
    rows = []
    for ri, (sc, kappa, reps, kind) in enumerate(CAL_ROWS):
        for k, rule in enumerate(R.PRIMARIES):
            x = ps[ri][:, k]
            r05, r025 = float((x < 0.05).mean()), float((x < 0.025).mean())
            ok = bool(r05 <= CAL_MAX_P05 and r025 <= CAL_MAX_P025) if kind == "no-edge" else None
            rows.append({"scenario": sc, "kind": kind, "rule": rule, "planted_edge": kappa, "reps": int(len(x)),
                         "draws": draws, "reject_05": r05, "se_05": float(np.sqrt(r05 * (1 - r05) / len(x))),
                         "reject_025": r025, "se_025": float(np.sqrt(r025 * (1 - r025) / len(x))),
                         "mean_p": float(x.mean()), "limit_05": CAL_MAX_P05 if kind == "no-edge" else None,
                         "limit_025": CAL_MAX_P025 if kind == "no-edge" else None, "ok": ok, "calib_sha256": sha0,
                         **lib_versions_calib()})
    df = pd.DataFrame(rows)
    if calib_sha() != sha0:
        raise SystemExit("the calibration code changed while it ran: nothing written")
    CALIB_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CALIB_CSV, index=False, lineterminator="\n")
    with pd.option_context("display.width", 220):
        print(df.drop(columns=["calib_sha256"]).to_string(index=False))
    print(f"calibration runtime {time.time() - t0:.0f}s, calibration code {sha0[:12]}")
    bad = df[(df["kind"] == "no-edge") & ~df["ok"].astype(bool)]
    if len(bad):
        print("NO-EDGE ROWS ABOVE THE LIMIT: " + ", ".join(f"{a}/{b}" for a, b in zip(bad["scenario"], bad["rule"])))
    det = df[(df["kind"] == "planted edge") & (df["planted_edge"] == max(e for _, e, _, _ in CAL_ROWS))]
    if not (det["reject_025"] >= CAL_MIN_DETECT).all():
        print(f"THE PLANTED EDGE (kappa 0.10) IS DETECTED IN FEWER THAN {CAL_MIN_DETECT:.0%} OF PANELS")
    return df
