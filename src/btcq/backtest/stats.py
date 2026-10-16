"""Overfitting-aware statistics: PSR/DSR, block bootstrap, random-entry null, CSCV-PBO."""
from __future__ import annotations

from itertools import combinations, product

import numpy as np
from numba import njit
from scipy import stats as sps

EULER_GAMMA = 0.5772156649015329


def sharpe_moments(rets) -> tuple[float, int, float, float]:
    """Per-period Sharpe, sample length, skewness, (non-excess) kurtosis."""
    r = np.asarray(rets, dtype=np.float64)
    sd = r.std(ddof=1)
    if r.size < 3 or sd == 0:
        return 0.0, r.size, 0.0, 3.0
    return float(r.mean() / sd), r.size, float(sps.skew(r)), float(sps.kurtosis(r, fisher=False))


def psr(sr: float, T: int, skew: float, kurt: float, sr_star: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio (Bailey & Lopez de Prado 2012). SRs are per-period."""
    den = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    if den <= 0 or T < 2:
        return 0.0
    return float(sps.norm.cdf((sr - sr_star) * np.sqrt(T - 1) / np.sqrt(den)))


def expected_max_sr(var_sr: float, n_trials: int) -> float:
    """Expected maximum Sharpe among n_trials skill-less strategies (False Strategy Theorem)."""
    if n_trials < 2 or var_sr <= 0:
        return 0.0
    z1 = sps.norm.ppf(1 - 1.0 / n_trials)
    z2 = sps.norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(var_sr) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def dsr(sr: float, T: int, skew: float, kurt: float, var_sr: float, n_trials: int) -> float:
    """Deflated Sharpe Ratio: PSR against the expected max Sharpe of n_trials."""
    return psr(sr, T, skew, kurt, expected_max_sr(var_sr, n_trials))


def null_sr_variance(T: int) -> float:
    """Sampling variance of a per-period Sharpe estimate when the true Sharpe is zero."""
    return 1.0 / max(T - 1, 1)


def effective_trials(returns: np.ndarray) -> float:
    """Effective number of independent trials: participation ratio of the eigenvalues of the
    trials' return correlation matrix, (sum l)^2 / sum l^2. Equals N for uncorrelated trials
    and 1 for identical ones."""
    R = np.asarray(returns, dtype=np.float64)
    R = R[:, R.std(axis=0) > 0]
    if R.shape[1] < 2:
        return float(R.shape[1])
    ev = np.clip(np.linalg.eigvalsh(np.corrcoef(R.T)), 0, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def block_bootstrap(rets, n_sims: int = 10_000, block: int = 10, n_out: int | None = None,
                    seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Circular block bootstrap of per-trade equity returns -> (total return, max drawdown)."""
    x = np.asarray(rets, dtype=np.float64)
    n = x.size
    n_out = n_out or n
    rng = np.random.default_rng(seed)
    nb = -(-n_out // block)
    totals, mdds = [], []
    for chunk in np.array_split(np.arange(n_sims), max(1, n_sims * n_out // 2_000_000)):
        starts = rng.integers(0, n, size=(chunk.size, nb))
        idx = ((starts[:, :, None] + np.arange(block)) % n).reshape(chunk.size, -1)[:, :n_out]
        paths = np.cumprod(1.0 + x[idx], axis=1)
        paths = np.concatenate([np.ones((chunk.size, 1)), paths], axis=1)
        peak = np.maximum.accumulate(paths, axis=1)
        totals.append(paths[:, -1] - 1.0)
        mdds.append(np.max(1.0 - paths / peak, axis=1))
    return np.concatenate(totals), np.concatenate(mdds)


@njit(cache=True)
def _null_means(o, cand, durs, dirs, cost, n_perm, seed):
    np.random.seed(seed)
    n = durs.shape[0]
    out = np.empty(n_perm)
    last = o.shape[0] - 1
    for p in range(n_perm):
        pd_ = np.random.permutation(durs)
        pdir = np.random.permutation(dirs)
        s = 0.0
        for k in range(n):
            e = cand[np.random.randint(cand.shape[0])]
            x = min(e + pd_[k], last)
            s += pdir[k] * (o[x] / o[e] - 1.0) - cost
        out[p] = s / n
    return out


def random_entry_null(o, candidates, durations, directions, cost: float, n_perm: int = 1000,
                      seed: int = 11) -> np.ndarray:
    """Mean per-trade notional return of random entries with the strategy's own holding
    times and long/short mix (both shuffled), entered at random strategy-bar opens."""
    return _null_means(np.asarray(o, np.float64), np.asarray(candidates, np.int64),
                       np.asarray(durations, np.int64), np.asarray(directions, np.float64),
                       float(cost), int(n_perm), int(seed))


def pbo_cscv(M: np.ndarray, S: int = 16) -> tuple[float, np.ndarray]:
    """Probability of Backtest Overfitting via Combinatorially Symmetric Cross-Validation.

    M: T x N matrix of per-period returns, one column per configuration.
    Returns (PBO, logits). PBO = share of splits where the in-sample winner ranks below
    the out-of-sample median.
    """
    M = np.asarray(M, dtype=np.float64)
    T, N = M.shape
    if N < 2:
        return float("nan"), np.array([])
    L = T // S
    blocks = M[T - L * S:].reshape(S, L, N)
    s1, s2 = blocks.sum(axis=1), (blocks ** 2).sum(axis=1)
    combos = list(combinations(range(S), S // 2))
    C = np.zeros((len(combos), S))
    for k, cmb in enumerate(combos):
        C[k, list(cmb)] = 1.0
    n_half = L * S / 2

    def _sr(a1, a2):
        mu = a1 / n_half
        var = a2 / n_half - mu * mu
        with np.errstate(divide="ignore", invalid="ignore"):
            sr = mu / np.sqrt(np.maximum(var, 0))
        return np.where(np.isfinite(sr), sr, 0.0)

    is1, is2 = C @ s1, C @ s2
    sr_is = _sr(is1, is2)
    sr_oos = _sr(s1.sum(0) - is1, s2.sum(0) - is2)
    best = np.argmax(sr_is, axis=1)
    val = sr_oos[np.arange(len(combos)), best]
    below = (sr_oos < val[:, None]).sum(axis=1)
    ties = (sr_oos == val[:, None]).sum(axis=1)
    rank = below + (ties + 1) / 2.0  # average rank, 1..N
    omega = rank / (N + 1)
    logits = np.log(omega / (1 - omega))
    return float(np.mean(logits <= 0)), logits


def grid_configs(grid: dict[str, list]) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in product(*(grid[k] for k in keys))]


def grid_neighbors(grid: dict[str, list], configs: list[dict]) -> list[list[int]]:
    """Configs differing from each config by one step in exactly one parameter."""
    keys = list(grid)
    pos = [tuple(grid[k].index(c[k]) for k in keys) for c in configs]
    lookup = {p: i for i, p in enumerate(pos)}
    out = []
    for p in pos:
        nb = []
        for d in range(len(keys)):
            for step in (-1, 1):
                q = list(p)
                q[d] += step
                j = lookup.get(tuple(q))
                if j is not None:
                    nb.append(j)
        out.append(nb)
    return out


def weekly_returns(daily) -> "pd.Series":
    """Compound daily returns into Monday-Sunday (UTC) weeks; partial edge weeks are kept."""
    import pandas as pd
    d = pd.Series(daily, dtype=np.float64)
    return (1.0 + d).groupby(pd.Grouper(freq="W-SUN")).prod() - 1.0


def weekly_t(daily) -> dict:
    """t-stat of the mean weekly return. Clustered entries across markets make trade-level
    t-stats too optimistic; portfolio returns per week are much closer to independent."""
    w = weekly_returns(daily).to_numpy()
    n = w.size
    sd = w.std(ddof=1) if n > 1 else 0.0
    if n < 2 or sd == 0:
        return {"t": 0.0, "n": int(n), "mean": float(w.mean()) if n else 0.0, "sd": float(sd), "sharpe_ann": 0.0}
    return {"t": float(w.mean() / sd * np.sqrt(n)), "n": int(n), "mean": float(w.mean()), "sd": float(sd),
            "sharpe_ann": float(w.mean() / sd * np.sqrt(52))}


def beta_alpha(y, x) -> dict:
    """OLS beta of daily returns y on a benchmark x (aligned days), and the PSR / annual Sharpe of
    the residual return y - beta*x."""
    import pandas as pd
    j = pd.concat([pd.Series(y, dtype=np.float64), pd.Series(x, dtype=np.float64)], axis=1).dropna()
    yy, xx = j.iloc[:, 0].to_numpy(), j.iloc[:, 1].to_numpy()
    vx = xx.var(ddof=1) if xx.size > 1 else 0.0
    beta = float(np.cov(yy, xx, ddof=1)[0, 1] / vx) if vx > 0 else 0.0
    a = yy - beta * xx
    sr, T, sk, ku = sharpe_moments(a)
    return {"beta": beta, "psr_alpha": psr(sr, T, sk, ku), "sharpe_alpha_ann": float(sr * np.sqrt(365)),
            "alpha_ann": float(a.mean() * 365), "days": int(T)}
