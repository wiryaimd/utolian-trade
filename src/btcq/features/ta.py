"""Causal technical indicators. Value at index t uses only data at indices <= t."""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def shift(x, k: int = 1) -> np.ndarray:
    x = _arr(x)
    out = np.full_like(x, np.nan)
    if k > 0:
        out[k:] = x[:-k]
    elif k < 0:
        raise ValueError("negative shift would look ahead")
    else:
        out[:] = x
    return out


def sma(x, n: int) -> np.ndarray:
    return pd.Series(_arr(x)).rolling(n, min_periods=n).mean().to_numpy()


def rolling_std(x, n: int) -> np.ndarray:
    return pd.Series(_arr(x)).rolling(n, min_periods=n).std(ddof=0).to_numpy()


def rolling_max(x, n: int) -> np.ndarray:
    return pd.Series(_arr(x)).rolling(n, min_periods=n).max().to_numpy()


def rolling_min(x, n: int) -> np.ndarray:
    return pd.Series(_arr(x)).rolling(n, min_periods=n).min().to_numpy()


def rolling_median(x, n: int) -> np.ndarray:
    return pd.Series(_arr(x)).rolling(n, min_periods=n).median().to_numpy()


@njit(cache=True)
def _ema(x, alpha, n):
    out = np.full(x.shape[0], np.nan)
    s = np.nan
    count = 0
    for i in range(x.shape[0]):
        v = x[i]
        if np.isnan(v):
            out[i] = np.nan if count < n else s
            continue
        s = v if count == 0 else alpha * v + (1 - alpha) * s
        count += 1
        if count >= n:
            out[i] = s
    return out


def ema(x, n: int) -> np.ndarray:
    return _ema(_arr(x), 2.0 / (n + 1), n)


def rma(x, n: int) -> np.ndarray:
    """Wilder's moving average."""
    return _ema(_arr(x), 1.0 / n, n)


def true_range(h, l, c) -> np.ndarray:
    h, l, c = _arr(h), _arr(l), _arr(c)
    pc = shift(c)
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    tr[0] = h[0] - l[0]
    return tr


def atr(h, l, c, n: int = 14) -> np.ndarray:
    return rma(true_range(h, l, c), n)


def rsi(c, n: int = 14) -> np.ndarray:
    d = np.diff(_arr(c), prepend=np.nan)
    up = rma(np.where(d > 0, d, 0.0)[1:], n)
    dn = rma(np.where(d < 0, -d, 0.0)[1:], n)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 100 - 100 / (1 + up / dn)
    r = np.where(dn == 0, 100.0, r)
    return np.concatenate([[np.nan], r])


def adx(h, l, c, n: int = 14) -> np.ndarray:
    h, l, c = _arr(h), _arr(l), _arr(c)
    up = np.diff(h, prepend=np.nan)
    dn = -np.diff(l, prepend=np.nan)
    pdm = np.where((up > dn) & (up > 0), up, 0.0)[1:]
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)[1:]
    tr = rma(true_range(h, l, c)[1:], n)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = 100 * rma(pdm, n) / tr
        mdi = 100 * rma(mdm, n) / tr
        dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
    return np.concatenate([[np.nan], rma(dx, n)])


def bollinger(c, n: int = 20, k: float = 2.0):
    mid = sma(c, n)
    sd = rolling_std(c, n)
    return mid, mid + k * sd, mid - k * sd, (2 * k * sd) / mid


@njit(cache=True)
def _pct_rank(x, n):
    out = np.full(x.shape[0], np.nan)
    for i in range(n - 1, x.shape[0]):
        v = x[i]
        if np.isnan(v):
            continue
        le = 0
        tot = 0
        for j in range(i - n + 1, i + 1):
            if not np.isnan(x[j]):
                tot += 1
                if x[j] <= v:
                    le += 1
        if tot >= n // 2:
            out[i] = le / tot
    return out


def pct_rank(x, n: int) -> np.ndarray:
    """Fraction of the trailing n values (incl. current) that are <= current."""
    return _pct_rank(_arr(x), n)


def rolling_vwap(h, l, c, v, n: int):
    """Rolling volume-weighted mean of typical price and volume-weighted std around it."""
    tp = (_arr(h) + _arr(l) + _arr(c)) / 3
    v = _arr(v)
    sv = pd.Series(v).rolling(n, min_periods=n).sum().to_numpy()
    s1 = pd.Series(v * tp).rolling(n, min_periods=n).sum().to_numpy()
    s2 = pd.Series(v * tp * tp).rolling(n, min_periods=n).sum().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = s1 / sv
        var = np.maximum(s2 / sv - vwap * vwap, 0.0)
    return vwap, np.sqrt(var)


def cross_above(x, level) -> np.ndarray:
    x = _arr(x)
    level = np.broadcast_to(_arr(level), x.shape)
    return (x > level) & (shift(x) <= shift(level))


def cross_below(x, level) -> np.ndarray:
    x = _arr(x)
    level = np.broadcast_to(_arr(level), x.shape)
    return (x < level) & (shift(x) >= shift(level))


def recent(flag, n: int) -> np.ndarray:
    """True if flag was true in any of the last n bars (incl. current)."""
    f = pd.Series(np.asarray(flag, dtype=np.float64)).rolling(n, min_periods=1).max().to_numpy()
    return f > 0


def rolling_sum(x, n: int) -> np.ndarray:
    return pd.Series(_arr(x)).rolling(n, min_periods=n).sum().to_numpy()


def zscore(x, n: int) -> np.ndarray:
    """(x - trailing mean) / trailing std over n observations (incl. current)."""
    x = _arr(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (x - sma(x, n)) / rolling_std(x, n)
    return np.where(np.isfinite(z), z, np.nan)
