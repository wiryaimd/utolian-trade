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


def nw_lag(T: int) -> int:
    """Newey-West (1994) automatic lag: floor(4 (T/100)^(2/9))."""
    return int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))


def hac_ols(y, x, lag: int | None = None) -> dict:
    """OLS of y on [1, x] with Newey-West (Bartlett) standard errors and a T/(T-2) correction."""
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    T = y.size
    lag = nw_lag(T) if lag is None else int(lag)
    X = np.column_stack([np.ones(T), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    u = (y - X @ b)[:, None] * X
    S = u.T @ u
    for l in range(1, lag + 1):
        G = u[l:].T @ u[:-l]
        S += (1.0 - l / (lag + 1.0)) * (G + G.T)
    V = XtX_inv @ S @ XtX_inv * T / max(T - 2, 1)
    se = np.sqrt(np.diag(V))
    return {"alpha": float(b[0]), "beta": float(b[1]), "se_alpha": float(se[0]),
            "t_alpha": float(b[0] / se[0]) if se[0] > 0 else 0.0,
            "t_beta": float(b[1] / se[1]) if se[1] > 0 else 0.0, "lag": lag, "T": T}


def year_share_daily(daily) -> float:
    """Largest calendar year's share of the summed daily returns (inf if the total is <= 0)."""
    import pandas as pd
    d = pd.Series(daily, dtype=np.float64)
    tot = d.sum()
    if tot <= 0:
        return float("inf")
    return float(d.groupby(d.index.year).sum().max() / tot)


def power_table(sharpes=(0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0), n_sims: int = 40_000, dev_days: int = 1461,
                hold_days: int = 1004, n_dsr: float = 16.8496, seed: int = 5, chunk: int = 4000) -> list[dict]:
    """Round 5 protocol power under iid Gaussian daily returns: P(dev weekly t >= 2), P(+ holdout mean > 0
    and holdout Sharpe >= 50% of dev) = REPLICATED, P(+ DSR >= 0.95 on dev+holdout at n_dsr) = STRICT."""
    rng = np.random.default_rng(seed)
    T = dev_days + hold_days
    v0 = null_sr_variance(T)
    sr_star = expected_max_sr(v0, n_dsr)
    weeks = dev_days // 7
    out = []
    for s_ann in sharpes:
        mu = s_ann / np.sqrt(365) * 0.01
        k = {"dev": 0, "rep": 0, "strict": 0}
        for start in range(0, n_sims, chunk):
            m = min(chunk, n_sims - start)
            r = rng.normal(mu, 0.01, size=(m, T))
            w = r[:, : weeks * 7].reshape(m, weeks, 7).sum(axis=2)
            t_w = w.mean(1) / w.std(1, ddof=1) * np.sqrt(weeks)
            dev_sr = r[:, :dev_days].mean(1) / r[:, :dev_days].std(1, ddof=1)
            h = r[:, dev_days:]
            h_sr = h.mean(1) / h.std(1, ddof=1)
            dev = t_w >= 2.0
            rep = dev & (h.mean(1) > 0) & (h_sr >= 0.5 * dev_sr)
            sr_all = r.mean(1) / r.std(1, ddof=1)
            z = (sr_all - sr_star) * np.sqrt(T - 1) / np.sqrt(1 + 0.5 * sr_all ** 2)
            from scipy.stats import norm
            strict = rep & (norm.cdf(z) >= 0.95)
            k["dev"] += int(dev.sum()); k["rep"] += int(rep.sum()); k["strict"] += int(strict.sum())
        out.append({"sharpe_ann": s_ann, "p_dev_pass": k["dev"] / n_sims, "p_replicated": k["rep"] / n_sims,
                    "p_strict": k["strict"] / n_sims})
    return out


def power_table_td(sharpes=(0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0), n_sims: int = 20_000, dev_days: int = 7560,
                   hold_days: int = 2709, n_dsr: float = 17.849570528171945, ramp=None, hurdles=(0.25, 0.45, 0.6, 0.75),
                   seed: int = 6, chunk: int = 1000) -> list[dict]:
    """Round 6 protocol power on trading days (252 a year, 5-day weeks) under iid Gaussian returns.

    sharpes are full-book annual Sharpes. ramp: optional weight per dev day, the share of the book that is
    live (the mean scales with w and the s.d. with sqrt(w), as for independent markets); the holdout is the
    full book. Dev pass = weekly t >= 2; REPLICATED adds holdout mean > 0 and holdout Sharpe >= 50% of dev;
    STRICT adds DSR >= 0.95 at n_dsr over dev + holdout. A funding cost of h full-book annual Sharpe units
    (scaled by w): p_funding_h = P(dev mean - cost > 0), the funding gate alone; p_dev_hurdle_h = P(dev pass and
    that gate). min_sharpe_dev ~ 2 sqrt(252 / sum(w)) is the realized full-book Sharpe a dev pass needs, so a
    hurdle below it never binds."""
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    w = np.ones(dev_days) if ramp is None else np.asarray(ramp, np.float64)[:dev_days]
    if w.shape[0] != dev_days:
        raise ValueError("ramp needs one weight per dev day")
    wt = np.r_[w, np.ones(hold_days)]
    T = dev_days + hold_days
    sr_star = expected_max_sr(null_sr_variance(T), n_dsr)
    weeks = dev_days // 5
    out = []
    for s_ann in sharpes:
        mu = s_ann / np.sqrt(252) * 0.01
        k = {"dev": 0, "rep": 0, "strict": 0} | {h: 0 for h in hurdles} | {("f", h): 0 for h in hurdles}
        for start in range(0, n_sims, chunk):
            m = min(chunk, n_sims - start)
            r = wt * mu + np.sqrt(wt) * rng.normal(0.0, 0.01, size=(m, T))
            d = r[:, :dev_days]
            wk = d[:, : weeks * 5].reshape(m, weeks, 5).sum(axis=2)
            t_w = wk.mean(1) / wk.std(1, ddof=1) * np.sqrt(weeks)
            dev = t_w >= 2.0
            dev_sr = d.mean(1) / d.std(1, ddof=1)
            h = r[:, dev_days:]
            h_sr = h.mean(1) / h.std(1, ddof=1)
            rep = dev & (h.mean(1) > 0) & (h_sr >= 0.5 * dev_sr)
            sr_all = r.mean(1) / r.std(1, ddof=1)
            z = (sr_all - sr_star) * np.sqrt(T - 1) / np.sqrt(1 + 0.5 * sr_all ** 2)
            strict = rep & (norm.cdf(z) >= 0.95)
            k["dev"] += int(dev.sum()); k["rep"] += int(rep.sum()); k["strict"] += int(strict.sum())
            for hh in hurdles:
                cost = hh / np.sqrt(252) * 0.01 * w.sum()
                fund = d.sum(1) - cost > 0
                k[hh] += int((dev & fund).sum())
                k[("f", hh)] += int(fund.sum())
        out.append({"variant": "constant" if ramp is None else "ramp", "sharpe_ann": s_ann,
                    "p_dev_pass": k["dev"] / n_sims, "p_replicated": k["rep"] / n_sims,
                    "p_strict": k["strict"] / n_sims,
                    **{f"p_funding_{hh:g}": k[("f", hh)] / n_sims for hh in hurdles},
                    **{f"p_dev_hurdle_{hh:g}": k[hh] / n_sims for hh in hurdles},
                    "full_book_years_dev": float(w.sum() / 252),
                    "min_sharpe_dev": float(2 * np.sqrt(252 / w.sum()))})
    return out


def power_table_practical(sharpes=(0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0), n_sims: int = 40_000, dev_days: int = 1309,
                          hold_days: int = 1004, n_effs=(1.0, 2.0, 3.0, 6.0), seed: int = 7,
                          chunk: int = 4000) -> list[dict]:
    """Practical-rulebook power under iid Gaussian daily returns (365-day years). For each family N_eff:
    p_dev = P(dev mean > 0 and dev DSR >= 0.95 at N_eff, null variance 1/(T-1)); p_pass = P(dev and holdout
    mean > 0 and holdout Sharpe >= 50% of dev). The random-timing null, the 2x-cost, year-share, plateau,
    coin-breadth and drawdown gates are not modelled (each can only lower these numbers)."""
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    T = dev_days + hold_days
    v0 = null_sr_variance(dev_days)
    out = []
    for s_ann in sharpes:
        mu = s_ann / np.sqrt(365) * 0.01
        k = {n: [0, 0] for n in n_effs}
        for start in range(0, n_sims, chunk):
            m = min(chunk, n_sims - start)
            r = rng.normal(mu, 0.01, size=(m, T))
            d, h = r[:, :dev_days], r[:, dev_days:]
            sr_d = d.mean(1) / d.std(1, ddof=1)
            sr_h = h.mean(1) / h.std(1, ddof=1)
            hold = (h.mean(1) > 0) & (sr_h >= 0.5 * sr_d)
            for n in n_effs:
                star = expected_max_sr(v0, n)
                z = (sr_d - star) * np.sqrt(dev_days - 1) / np.sqrt(1 + 0.5 * sr_d ** 2)
                dev = (d.mean(1) > 0) & (norm.cdf(z) >= 0.95)
                k[n][0] += int(dev.sum())
                k[n][1] += int((dev & hold).sum())
        for n in n_effs:
            out.append({"sharpe_ann": s_ann, "n_eff": n, "sharpe_needed_ann":
                        float((expected_max_sr(v0, n) + norm.ppf(0.95) / np.sqrt(dev_days - 1)) * np.sqrt(365)),
                        "p_dev": k[n][0] / n_sims, "p_pass": k[n][1] / n_sims})
    return out


def block_bootstrap_means(x, n_sims: int = 10_000, block: int = 30, seed: int = 11) -> np.ndarray:
    """Circular moving-block bootstrap of the mean of x: each draw is len(x) values built from blocks of `block`
    consecutive values (wrapping past the end) at uniform starts, from numpy's Generator(PCG64(seed)). The chunking
    depends only on len(x) and block, so the draws are the same for a given (x, n_sims, block, seed)."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0 or block < 1 or n_sims < 1:
        raise ValueError("block_bootstrap_means needs a non-empty x, block >= 1 and n_sims >= 1")
    rng = np.random.Generator(np.random.PCG64(seed))
    nb = -(-n // block)
    step = max(1, 2_000_000 // (nb * block))
    out = np.empty(n_sims)
    for s in range(0, n_sims, step):
        m = min(step, n_sims - s)
        starts = rng.integers(0, n, size=(m, nb))
        idx = ((starts[:, :, None] + np.arange(block)) % n).reshape(m, -1)[:, :n]
        out[s:s + m] = x[idx].mean(axis=1)
    return out
