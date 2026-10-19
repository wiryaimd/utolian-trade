"""Round 8 IBS: daily internal-bar-strength mean reversion (Pagonidis, NAAIM 2014) on the SPY, QQQ and EWY proxies of
Binance TradFi perps.

At each trade date t of a market (its own rows), IBS_t = (C_t - L_t) / (H_t - L_t), where the close sits in the day's
range; IBS is undefined (no signal) when H <= L or H, L or C is missing.
- IBS-L: long at the close of t if IBS_t <= theta, held to the next trade date's close; renewed while the signal
  repeats, otherwise exited at that close.
- IBS-LS: IBS-L plus a short at the close of t if IBS_t >= 1 - theta, held and renewed the same way; a long <-> short
  switch is a flip at the same close.
Hold h = 2 (grid only): the position follows the latest signal in the last 2 closes, flat if there is none.
Sizing (Round 6's): min(0.25 / sigma, 2.0) x the sleeve's equity at entry or flip, sigma = std (ddof 0) of the last 90
daily log returns x sqrt(252); no position until sigma exists; never resized while the signal repeats.
Grid: direction {L, LS} x theta {0.15, 0.20, 0.25} x hold {1, 2} = 12 configs, used only for N_eff, the plateau and
PBO. Primaries (fixed): (L, 0.20, 1) and (LS, 0.20, 1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit

from btcq.strategies.base import Strategy
from btcq.strategies.r6_macro import ANN, LEV_CAP, TAKER, VOL_DAYS, VOL_TARGET, vol_size_td  # noqa: F401

RULES = __doc__

MARKETS = ["SPYUSDT", "QQQUSDT", "EWYUSDT"]
SLIP = {"SPYUSDT": 3e-4, "QQQUSDT": 2e-4, "EWYUSDT": 2e-4}       # Round 6's 2026-10-16 tiers
DIRECTIONS = ("L", "LS")
THETAS = (0.15, 0.20, 0.25)
HOLDS = (1, 2)
CONFIGS = [{"dir": d, "theta": th, "hold": h} for d in DIRECTIONS for th in THETAS for h in HOLDS]


def cfg_index(direction: str, theta: float, hold: int) -> int:
    return CONFIGS.index({"dir": direction, "theta": theta, "hold": hold})


PRIMARIES = {"IBS-L": cfg_index("L", 0.20, 1), "IBS-LS": cfg_index("LS", 0.20, 1)}
NEIGHBOURS = {name: [cfg_index(CONFIGS[i]["dir"], 0.15, 1), cfg_index(CONFIGS[i]["dir"], 0.25, 1),
                     cfg_index(CONFIGS[i]["dir"], 0.20, 2)] for name, i in PRIMARIES.items()}


class IBS(Strategy):
    """Metadata for the Round 8 registry and report; the rule is run by btcq.research.round8."""
    code = "IBS"
    name = "IBS daily mean reversion, SPY/QQQ/EWY proxies (IBS-L and IBS-LS co-primaries)"
    tf = "1d"
    version = 1
    holdout_contaminated = False
    hypothesis = ("A close near the day's low predicts a higher next close-to-close return in equity ETFs, and a "
                  "close near the high a lower one (Pagonidis 2014); held one day from the same close, the effect "
                  "survives Binance perp costs, a random-timing null and an alpha test against holding the proxies.")
    rules = RULES
    grid = {"dir": list(DIRECTIONS), "theta": list(THETAS), "hold": list(HOLDS)}

    def compute(self, data, **params):
        raise NotImplementedError("IBS is a portfolio rule; see btcq.research.round8")


def ibs(high, low, close) -> np.ndarray:
    """(C - L) / (H - L); NaN when H <= L or any of H, L, C is missing (no signal). Not clipped: a close outside
    the range is a data error, counted by the data report."""
    h, lo, c = (np.asarray(x, np.float64) for x in (high, low, close))
    ok = np.isfinite(h) & np.isfinite(lo) & np.isfinite(c) & (h > lo)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ok, (c - lo) / np.where(ok, h - lo, 1.0), np.nan)


def signals(ib: np.ndarray, theta: float, direction: str) -> np.ndarray:
    """+1 when IBS <= theta; -1 when IBS >= 1 - theta (LS only); 0 otherwise or when IBS is undefined."""
    ib = np.asarray(ib, np.float64)
    fin = np.isfinite(ib)
    s = np.where(fin & (ib <= theta), 1, 0)
    if direction == "LS":
        s = np.where(fin & (ib >= 1.0 - theta), -1, s)
    elif direction != "L":
        raise ValueError(direction)
    return s.astype(np.int8)


def hold_positions(sig: np.ndarray, hold: int) -> np.ndarray:
    """The position decided at each close: the latest non-zero signal among the last `hold` closes (this one
    included), flat if there is none."""
    sig = np.asarray(sig, np.int8)
    if hold < 1:
        raise ValueError(hold)
    pos = sig.copy()
    for k in range(1, hold):
        prev = np.r_[np.zeros(k, np.int8), sig[:-k]]
        pos = np.where(pos != 0, pos, prev).astype(np.int8)
    return pos


def positions(high, low, close, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """(position at each close, size at each close) for one market's own rows. The position is 0 until sigma
    exists; size = min(0.25 / sigma, 2.0) (NaN before sigma exists)."""
    size = vol_size_td(np.asarray(close, np.float64))
    pos = hold_positions(signals(ibs(high, low, close), cfg["theta"], cfg["dir"]), int(cfg["hold"]))
    return np.where(np.isfinite(size), pos, 0).astype(np.int8), size


def episodes(pos: np.ndarray) -> int:
    """Number of holding episodes (maximal runs of one non-zero side) in a position path."""
    pos = np.asarray(pos, np.int8)
    prev = np.r_[np.int8(0), pos[:-1]]
    return int(((pos != 0) & (pos != prev)).sum())


# ---------------------------------------------------------------- the null: date-level relocation of held days
#
# The unit is a decision date d of the union trade-date grid with its position vector over the markets (each entry
# the market's side, or 0). Each draw moves every held date, whole vector at once, to a date d' drawn uniformly with
# replacement from the eligible dates of d's stratum, and re-sizes there. Strata are date-level: the era (the set of
# ready markets), the EWMA volatility tercile (lambda 0.94) of the slot-scaled equal-weight index within the era, the
# index above or below its 200-day SMA, and whether the next trade date is at least 2 calendar days away (a weekend
# or holiday follows). Dates before the EWMA and the SMA exist form their own volatility/trend cell (still split by
# era and gap). Statistic: the mean over held market-days of side x min(0.25 / sigma, 2) x the next close-to-close
# simple return, at zero cost.

EWMA_LAMBDA = 0.94
EWMA_MIN = 30            # index returns needed before the EWMA volatility exists
SMA_DAYS = 200           # grid rows of the index SMA (this date included)
DAY_MS = 86_400_000
WARMUP = 3               # vol_bin code of dates before the EWMA and the SMA exist


@dataclass
class MarketRows:
    """One market's own trade dates (ms, 00:00 UTC) and prices."""
    sym: str
    ts: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray


@dataclass
class StrataPanel:
    dates: np.ndarray        # union trade-date grid (ms) of the loaded rows
    Y: np.ndarray            # (n, M) size x next close-to-close simple return (NaN: no row, sigma or next row)
    ready: np.ndarray        # (n, M) the market's sigma exists at its last own row up to this date
    era: np.ndarray          # (n,) bitmask of ready markets
    vol_bin: np.ndarray      # (n,) 0/1/2 EWMA-volatility tercile within the era, WARMUP before EWMA/SMA exist
    trend: np.ndarray        # (n,) 1 if the index is above its SMA (0 below or in the warm-up)
    gap: np.ndarray          # (n,) the next grid date is >= 2 calendar days later
    in_window: np.ndarray    # (n,) start <= date < end, and not the window's last grid date
    eligible: np.ndarray     # (n,) in the window, some market ready, every ready market has a finite Y
    stratum: np.ndarray      # (n,) stratum id (-1 outside the window)
    ewma_vol: np.ndarray     # (n,) the index's EWMA volatility (NaN before EWMA_MIN returns)


def _next_simple_return(close: np.ndarray) -> np.ndarray:
    c = np.asarray(close, np.float64)
    out = np.full(len(c), np.nan)
    out[:-1] = c[1:] / c[:-1] - 1.0
    return out


def strata_panel(mkts: list[MarketRows], start_ms: int, end_ms: int) -> StrataPanel:
    """The date grid, the sized next returns and the strata of a window [start_ms, end_ms). Everything except
    the volatility terciles (a split of the window's own dates) uses data up to each date only."""
    M = len(mkts)
    dates = np.unique(np.concatenate([np.asarray(m.ts, np.int64) for m in mkts]))
    n = len(dates)
    Y = np.full((n, M), np.nan)
    ready = np.zeros((n, M), np.bool_)
    prev_ret = np.zeros((n, M))
    for j, m in enumerate(mkts):
        ts = np.asarray(m.ts, np.int64)
        ix = np.searchsorted(dates, ts)
        c = np.asarray(m.close, np.float64)
        size = vol_size_td(c)
        nxt = _next_simple_return(c)
        nxt[np.r_[ts[1:] >= end_ms, True]] = np.nan          # a next close at or after the window's end is outside
        Y[ix, j] = size * nxt
        rdy = np.zeros(n, np.bool_)
        rdy[ix] = np.isfinite(size)
        ready[:, j] = np.maximum.accumulate(rdy)
        r = np.r_[np.nan, c[1:] / c[:-1] - 1.0]
        prev_ret[ix, j] = np.nan_to_num(r, nan=0.0)
    era = (ready * (1 << np.arange(M))).sum(axis=1).astype(np.int64)
    b = prev_ret.sum(axis=1) / M                       # the slot-scaled equal-weight index's return on each date
    b[0] = np.nan
    index = np.cumprod(1.0 + np.nan_to_num(b, nan=0.0))
    ewv = pd.Series(b * b).ewm(alpha=1.0 - EWMA_LAMBDA, adjust=False, ignore_na=True,
                               min_periods=EWMA_MIN).mean().to_numpy()
    ewv = np.sqrt(ewv)
    sma = pd.Series(index).rolling(SMA_DAYS, min_periods=SMA_DAYS).mean().to_numpy()
    warm = ~(np.isfinite(ewv) & np.isfinite(sma))
    trend = np.where(warm, 0, (index > np.where(warm, 0.0, sma)).astype(np.int64))
    gap = np.zeros(n, np.int64)
    gap[:-1] = (np.diff(dates) >= 2 * DAY_MS).astype(np.int64)
    inw = (dates >= start_ms) & (dates < end_ms)
    w_idx = np.flatnonzero(inw)
    if len(w_idx):
        inw[w_idx[-1]] = False                         # the window's last date has no next return inside it
    vol_bin = np.full(n, WARMUP, np.int64)
    for e in np.unique(era[inw]):
        idx = np.flatnonzero(inw & (era == e) & ~warm)
        if not len(idx):
            continue
        order = idx[np.lexsort((dates[idx], ewv[idx]))]
        for g, part in enumerate(np.array_split(order, 3)):
            vol_bin[part] = g
    fin = np.isfinite(Y)
    eligible = inw & (era > 0) & np.all(fin | ~ready, axis=1)
    stratum = np.where(inw, era * 100 + vol_bin * 10 + trend * 2 + gap, -1)
    return StrataPanel(dates, Y, ready, era, vol_bin, trend, gap, inw, eligible, stratum, ewv)


def held_vectors(sp: StrataPanel, mkts: list[MarketRows], cfg: dict) -> np.ndarray:
    """(n, M) int8: each market's position decided at each grid date, inside the window and only where the
    position has a next return (a position without one earns nothing)."""
    v = np.zeros(sp.Y.shape, np.int8)
    for j, m in enumerate(mkts):
        pos, _ = positions(m.high, m.low, m.close, cfg)
        v[np.searchsorted(sp.dates, np.asarray(m.ts, np.int64)), j] = pos
    return np.where(sp.in_window[:, None] & np.isfinite(sp.Y), v, 0).astype(np.int8)


@njit(cache=True)
def _stat_rows(vh, Y, J):
    """For each row of J (destination grid rows, one per held date), the mean over held market-days of
    side x Y at the destination. One fixed summation order, so the identity row reproduces the actual exactly."""
    k, nh = J.shape
    M = vh.shape[1]
    cnt = 0
    for h in range(nh):
        for m in range(M):
            if vh[h, m] != 0:
                cnt += 1
    out = np.empty(k)
    for r in range(k):
        s = 0.0
        for h in range(nh):
            j = J[r, h]
            for m in range(M):
                if vh[h, m] != 0:
                    s += vh[h, m] * Y[j, m]
        out[r] = s / cnt if cnt > 0 else 0.0
    return out


def relocation_draws(sp: StrataPanel, held: np.ndarray, n_draws: int, seed: int) -> tuple[np.ndarray, int]:
    """(J, kept): J[r, h] = the destination grid row of held date held[h] in draw r, uniform with replacement over
    the eligible dates of its stratum (numpy's PCG64 stream from seed). A held date whose stratum has no eligible
    date stays in place (counted in kept)."""
    st = sp.stratum[held]
    groups, inv = np.unique(st, return_inverse=True)
    pools = [np.flatnonzero(sp.eligible & (sp.stratum == g)) for g in groups]
    lens = np.array([len(p) for p in pools], np.int64)
    offs = np.r_[0, np.cumsum(lens)[:-1]].astype(np.int64)
    flat = np.concatenate(pools + [np.zeros(0, np.int64)]).astype(np.int64)
    rng = np.random.Generator(np.random.PCG64(seed))
    u = rng.random((n_draws, len(held)))
    L = lens[inv]
    has = L > 0
    pick = np.minimum((u * L).astype(np.int64), np.maximum(L - 1, 0))
    J = np.broadcast_to(np.asarray(held, np.int64), u.shape).copy()
    if len(flat):
        J = np.where(has[None, :], flat[np.where(has, offs[inv], 0)[None, :] + np.where(has[None, :], pick, 0)], J)
    return np.ascontiguousarray(J, np.int64), int((~has).sum())


def null_test(sp: StrataPanel, v: np.ndarray, n_draws: int, seed: int) -> dict:
    """Date-level relocation null for one rule. p = (1 + #{null >= actual}) / (1 + n_draws)."""
    held = np.flatnonzero((v != 0).any(axis=1))
    if not len(held):
        raise ValueError("no held date in the window")
    vh = np.ascontiguousarray(v[held], np.int8)
    Y = np.ascontiguousarray(sp.Y)
    actual = float(_stat_rows(vh, Y, held[None, :].astype(np.int64))[0])
    J, kept = relocation_draws(sp, held, n_draws, seed)
    dist = _stat_rows(vh, Y, J)
    if not np.isfinite(dist).all() or not np.isfinite(actual):
        raise ValueError("a relocated held date has no sized return")
    return {"actual": actual, "null_mean": float(dist.mean()), "null_sd": float(dist.std(ddof=1)),
            "null_p95": float(np.quantile(dist, 0.95)),
            "p_value": float((1 + np.sum(dist >= actual)) / (1 + n_draws)), "n_draws": int(n_draws),
            "seed": int(seed), "held_dates": int(len(held)), "held_market_days": int((vh != 0).sum()),
            "kept_in_place": kept, "strata_used": int(len(np.unique(sp.stratum[held]))), "dist": dist}
