"""Round 7 RT: daily rotation into the strongest coins of a liquid non-major universe, long only.

At each UTC day's close, score every candidate by its L-day log return, ln(C_D / C_{D-L}), C = the day's last
1m close. A candidate is a member of the fill day's universe month that traded on day D, is outside a no-trade
gap and its 7-day cooldown, has a fill row on day D+1, both closes, and no finished gap inside the lookback.
Eligible = score > 0 (absolute momentum). A held coin is kept while it is eligible and ranks <= H among the
candidates; free slots go to the best-ranked eligible coins (ties by symbol). K = 3 slots of 1/3 of the equity
each; an empty slot is cash. Every held coin is resized to its 1/3 at each fill.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from btcq.backtest.rotation_sim import Panel, SimError, _sim, mean_daily_fast, mean_return
from btcq.strategies.base import Strategy

RULES = __doc__

K = 3
LOOKBACKS = (7, 14, 28)
HOLDS = (3, 6)
CONFIGS = [{"L": L, "H": H} for L in LOOKBACKS for H in HOLDS]
PRIMARY = CONFIGS.index({"L": 14, "H": 6})
NEIGHBOURS = [CONFIGS.index(c) for c in ({"L": 7, "H": 6}, {"L": 28, "H": 6}, {"L": 14, "H": 3})]
N_STRATA = 3
VOL_ALPHA = 0.06     # EWMA weight of the newest squared return (lambda = 0.94, RiskMetrics)
VOL_MIN = 30         # daily log returns needed before a coin has a volatility
STUCK = 2      # sel code: held, no fill row, not in a gap: the quantity is kept


class Rotation(Strategy):
    code = "RT"
    name = "Daily top-3 rotation, liquid non-major perps (long only)"
    tf = "1d"
    version = 1
    hypothesis = ("Among the most liquid non-major USDT-M perps, the coins with the strongest recent return keep "
                  "outperforming the others over the next days (cross-sectional momentum), and holding cash when "
                  "fewer than 3 have a positive return avoids the worst falls.")
    rules = RULES
    grid = {"L": list(LOOKBACKS), "H": list(HOLDS)}


def scores(close: np.ndarray, L: int) -> np.ndarray:
    """ln(C_t / C_{t-L}) per decision day t (NaN when either close is missing)."""
    s = np.full(close.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        s[L:] = np.log(close[L:] / close[:-L])
    return s


def candidates(p: Panel, L: int) -> np.ndarray:
    """(days, coins) bool for decision day t (it fills on t + 1); the last day has no fill day."""
    sc = scores(p.close, L)
    c = np.zeros(p.shape, np.bool_)
    c[:-1] = (p.member[1:] & p.traded[:-1] & ~p.blocked[:-1] & np.isfinite(p.fill_px[1:]) & np.isfinite(sc[:-1])
              & (p.days_since_gap[:-1] > L))   # strict: the gap ended at or before C_{D-L}'s minute (23:59 of D-L)
    return c


def candidate_scores(p: Panel, L: int) -> np.ndarray:
    sc = scores(p.close, L)
    return np.where(candidates(p, L), sc, np.nan)


def no_fill_next(p: Panel) -> np.ndarray:
    nf = np.ones(p.shape, np.bool_)
    nf[:-1] = ~np.isfinite(p.fill_px[1:])
    return nf


@njit(cache=True)
def select_path(score_c, no_fill_next, blocked, K, H, start, end):
    """sel[t, j] for decision days start..end: 1 = hold at 1/K after the next fill, 2 = held but stuck (no fill
    row, not in a gap: the quantity is kept), 0 = not held. Holdings before `start` are empty."""
    n, k = score_c.shape
    sel = np.zeros((n, k), np.int8)
    order = np.empty(k, np.int64)
    rank = np.zeros(k, np.int64)
    for t in range(start, end + 1):
        m = 0
        for j in range(k):
            if not np.isnan(score_c[t, j]):
                order[m] = j
                m += 1
        for a in range(1, m):                       # insertion sort: score descending, then column (symbol)
            x = order[a]
            b = a - 1
            while b >= 0 and (score_c[t, order[b]] < score_c[t, x]
                              or (score_c[t, order[b]] == score_c[t, x] and order[b] > x)):
                order[b + 1] = order[b]
                b -= 1
            order[b + 1] = x
        for j in range(k):
            rank[j] = 0
        for r in range(m):
            rank[order[r]] = r + 1
        used = 0
        if t > start:
            for j in range(k):
                if sel[t - 1, j] > 0:
                    if rank[j] > 0 and score_c[t, j] > 0.0 and rank[j] <= H:
                        sel[t, j] = 1
                        used += 1
                    elif rank[j] == 0 and no_fill_next[t, j] and not blocked[t, j]:
                        sel[t, j] = 2
                        used += 1
        for r in range(m):
            if used >= K:
                break
            j = order[r]
            if score_c[t, j] <= 0.0:
                break
            if sel[t, j] == 0:
                sel[t, j] = 1
                used += 1
    return sel


@njit(cache=True)
def to_weights(sel, K, start, end):
    """Fill-day weights: W[t + 1] from the decision at day t's close; NaN = keep the quantity (stuck)."""
    n, k = sel.shape
    W = np.zeros((n, k))
    for t in range(start, min(end, n - 2) + 1):
        for j in range(k):
            if sel[t, j] == 1:
                W[t + 1, j] = 1.0 / K
            elif sel[t, j] == 2:
                W[t + 1, j] = np.nan
    return W


def can_buy(cand: np.ndarray) -> np.ndarray:
    """Fill-day mask: a coin may be bought or increased on day t + 1 only if it was a candidate at day t."""
    cb = np.zeros(cand.shape, np.bool_)
    cb[1:] = cand[:-1]
    return cb


def ewma_vol(close: np.ndarray) -> np.ndarray:
    """Per day and coin: sqrt of the EWMA (alpha 0.06) of squared daily log close-to-close returns up to that
    day's close, NaN until 30 returns exist. Recent volatility, so the null's groups follow volatility bursts."""
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.diff(np.log(close), axis=0, prepend=np.nan)
    return np.sqrt(pd.DataFrame(lr ** 2).ewm(alpha=VOL_ALPHA, min_periods=VOL_MIN).mean().to_numpy())


def strata(vol: np.ndarray, cand: np.ndarray, n_strata: int = N_STRATA) -> np.ndarray:
    """Per decision day, candidates split into volatility groups of near-equal size (np.array_split of the
    candidates sorted by their EWMA volatility, ties by symbol); candidates without a volatility form one more
    group (id n_strata); non-candidates are -1."""
    n, k = cand.shape
    out = np.full((n, k), -1, np.int64)
    for t in range(n):
        idx = np.flatnonzero(cand[t])
        if not len(idx):
            continue
        v = vol[t, idx]
        ok = np.isfinite(v)
        good = idx[ok][np.lexsort((idx[ok], v[ok]))]
        for g, part in enumerate(np.array_split(good, n_strata)):
            out[t, part] = g
        out[t, idx[~ok]] = n_strata
    return out


@njit(cache=True)
def permute_scores(score_c, strata_, start, end):
    """Null: on each decision day, the candidates' scores are shuffled within their volatility group."""
    out = score_c.copy()
    n, k = score_c.shape
    idx = np.empty(k, np.int64)
    vals = np.empty(k)
    for t in range(start, end + 1):
        top = -1
        for j in range(k):
            if strata_[t, j] > top:
                top = strata_[t, j]
        for g in range(top + 1):
            m = 0
            for j in range(k):
                if strata_[t, j] == g:
                    idx[m] = j
                    vals[m] = score_c[t, j]
                    m += 1
            if m > 1:
                perm = np.random.permutation(m)
                for r in range(m):
                    out[t, idx[r]] = vals[perm[r]]
    return out


def path(p: Panel, cfg: dict, first: int, last: int, score_override: np.ndarray | None = None):
    """Weights and the buy mask for fill days first..last of one config."""
    cand = candidates(p, cfg["L"])
    sc = np.where(cand, scores(p.close, cfg["L"]), np.nan) if score_override is None else score_override
    sel = select_path(sc, no_fill_next(p), p.blocked, K, cfg["H"], first - 1, last - 1)
    return to_weights(sel, K, first - 1, last - 1), can_buy(cand), sel


@njit(cache=True)
def _null_kernel(score_c, strata_, nfn, blocked, K, H, start, end, close, fill_px, fund_pre, fund_post, fx_pre,
                 fx_post, fx_px, fund_fx, slip, cb, first, last, pen, eq0, n_draws, seed):
    np.random.seed(seed)
    out = np.empty(n_draws)
    for d in range(n_draws):
        sc = permute_scores(score_c, strata_, start, end)
        sel = select_path(sc, nfn, blocked, K, H, start, end)
        W = to_weights(sel, K, start, end)
        eq, f1, f2, f3, f4, ep, err, et, ej = _sim(close, fill_px, fund_pre, fund_post, fx_pre, fx_post, fx_px,
                                                    fund_fx, slip, W, cb, first, last, 0.0, 0.0, pen, eq0, True, K,
                                                    False)
        if err != 0:
            out[d] = np.nan
            continue
        out[d] = mean_return(eq, first, last, eq0)
    return out


def actual_stat(p: Panel, cfg: dict, first: int, last: int, penalty: float = 0.01, eq0: float = 10_000.0) -> float:
    """The null's statistic on the actual path: the mean daily return with fees and slippage at 0."""
    W, cb, _ = path(p, cfg, first, last)
    return mean_daily_fast(p, W, cb, first, last, taker=0.0, slip_mult=0.0, penalty=penalty, eq0=eq0, K=K)


def null_test(p: Panel, cfg: dict, first: int, last: int, n_draws: int, seed: int,
              penalty: float = 0.01, eq0: float = 10_000.0, strata_: np.ndarray | None = None) -> dict:
    """Stratified score-permutation null. Statistic: the mean daily return with fees and slippage at zero
    (funding and the delisting penalty kept), for the actual path and every permuted one, through the same
    simulator and summation (so the identity shuffle reproduces it bit for bit).
    p = (1 + #null >= actual) / (1 + n_draws)."""
    actual = actual_stat(p, cfg, first, last, penalty, eq0)
    cand = candidates(p, cfg["L"])
    sc = np.where(cand, scores(p.close, cfg["L"]), np.nan)
    st = strata(ewma_vol(p.close), cand) if strata_ is None else strata_
    dist = _null_kernel(sc, st, no_fill_next(p), p.blocked, K, cfg["H"], first - 1, last - 1, p.close, p.fill_px,
                        p.fund_pre, p.fund_post, p.fx_pre, p.fx_post, p.fx_px, p.fund_fx, p.slip, can_buy(cand),
                        first, last, float(penalty), float(eq0), int(n_draws), int(seed))
    if np.isnan(dist).any():
        raise SimError("a hard check failed inside a null draw")
    return {"actual": float(actual), "null_mean": float(dist.mean()), "null_p95": float(np.quantile(dist, 0.95)),
            "p_value": float((1 + np.sum(dist >= actual)) / (1 + len(dist))), "n_draws": int(n_draws),
            "seed": int(seed), "dist": dist}
