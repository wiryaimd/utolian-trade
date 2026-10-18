"""Round 7: one long-only account over many coins, rebalanced once a day, on daily arrays cut from the 1m path.

Every array is (days, coins) on one UTC calendar. A decision made at day d's close fills on the next day, the
"fill day" t = d + 1, at fill_px[t]: the open of the first minute at or after 00:01. On each fill day, for every
coin, in the minute order engine.run uses:
  1. funding on the position held before the fill, through the fill minute (engine.run settles funding before
     it executes decisions in the same minute);
  2. a forced exit (delisting rule) at a minute before the fill, at that minute's close less slippage and the
     penalty;
  3. the fill: buys at X (1 + slip), sells at X (1 - slip), taker fee on the traded notional;
  4. funding on the new position after the fill minute, through the day's last minute;
  5. a forced exit at or after the fill minute;
  6. the mark at the day's last close (the last known close when the coin has no row that day).
After the last day's mark every open position is closed at its last close less slippage, with the fee, and the
last mark becomes the cash (as engine.run does at the end of its data).

Weights W[t] are decided at day t-1's close: a finite weight is a target share of the equity at that close
(target qty = w * E[t-1] / X); NaN keeps the current quantity (the coin has no fill row and can't trade).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numba import njit

DAY_MS = 86_400_000
MINUTE_MS = 60_000

ERRORS = {1: "fill without a price", 2: "entry or increase on a coin that is not a candidate",
          3: "more holdings than slots", 4: "target weights sum above 1", 5: "equity at or below zero",
          6: "exit or resize asked on a coin without a fill price", 7: "equity identity broken"}
EP_COLS = ["coin", "entry_day", "exit_day", "reason", "entry_px", "exit_px", "qty0", "eq_before", "cash_flow",
           "fees", "funding", "net", "trades", "notional"]
EXIT_REASONS = ["rule", "delist", "end"]


class SimError(RuntimeError):
    """A hard check failed inside the simulation (a bug or a data problem, never a result)."""


@dataclass
class Panel:
    days: pd.DatetimeIndex       # UTC calendar days (ms unit)
    symbols: list[str]           # column order = tie-break order (sorted)
    close: np.ndarray            # last 1m close of the day (NaN: no row that day)
    traded: np.ndarray           # bool: a minute with volume > 0 that day
    fill_px: np.ndarray          # open of the first minute at/after 00:01 that day (NaN: none that day)
    fund_pre: np.ndarray         # sum of open x rate per unit qty over the day's minutes up to the fill minute
    fund_post: np.ndarray        # the same after the fill minute through the day's last minute
    fx_pre: np.ndarray           # bool: forced exit before the fill minute (any minute if no fill row)
    fx_post: np.ndarray          # bool: forced exit at or after the fill minute
    fx_px: np.ndarray            # close of the forced-exit minute
    fund_fx: np.ndarray          # funding per unit qty of the segment that ends at the forced-exit minute
    blocked: np.ndarray          # bool: the decision at this day's close falls in a no-trade gap or its cooldown
    days_since_gap: np.ndarray   # days from the end of the last finished gap to this day's close (inf: none)
    member: np.ndarray           # bool: universe member in this day's month
    rank: np.ndarray             # rank in this day's month (99: not a member)
    slip: np.ndarray             # slippage of fills on this day
    is_meme: np.ndarray = field(default_factory=lambda: np.zeros(0, np.bool_))
    funding_filled_h: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def shape(self) -> tuple[int, int]:
        return self.close.shape

    def index_of(self, day) -> int:
        return int(self.days.get_loc(pd.Timestamp(day)))

    def drop(self, symbol: str) -> "Panel":
        """The same panel without one coin (information runs)."""
        keep = [i for i, s in enumerate(self.symbols) if s != symbol]
        kw = {}
        for f in self.__dataclass_fields__:
            v = getattr(self, f)
            if f in ("days",):
                kw[f] = v
            elif f == "symbols":
                kw[f] = [self.symbols[i] for i in keep]
            elif isinstance(v, np.ndarray) and v.ndim == 2:
                kw[f] = v[:, keep]
            elif isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == len(self.symbols):
                kw[f] = v[keep]
            else:
                kw[f] = v
        return Panel(**kw)


# ---------------------------------------------------------------- minute path -> daily columns

def daily_columns(ts, o, c, vol, fund, cal_ms, fx_idx, gaps, cooldown_ms: int,
                  latency_ms: int = MINUTE_MS) -> dict[str, np.ndarray]:
    """One coin's daily columns on the calendar cal_ms (day starts, ms).

    ts/o/c/vol: the 1m rows; fund: engine.Market's per-minute funding rate (so funding lands on the same minute
    as in engine.run); fx_idx: the minute index of each forced exit; gaps: [(a, b), ...] no-trade stretches,
    a = last traded minute before, b = first traded minute after (or the data end)."""
    ts = np.asarray(ts, np.int64)
    o, c = np.asarray(o, np.float64), np.asarray(c, np.float64)
    cal_ms = np.asarray(cal_ms, np.int64)
    n = len(cal_ms)
    first = np.searchsorted(ts, cal_ms)
    end = np.searchsorted(ts, cal_ms + DAY_MS)
    has = end > first
    last = np.maximum(end - 1, 0)
    close = np.where(has, c[last] if len(c) else np.nan, np.nan)
    fi = np.searchsorted(ts, cal_ms + latency_ms)
    has_fill = fi < end
    fill_px = np.where(has_fill, o[np.minimum(fi, len(o) - 1)] if len(o) else np.nan, np.nan)
    F = np.r_[0.0, np.cumsum(o * np.asarray(fund, np.float64))]
    pre_end = np.where(has_fill, fi + 1, end)
    fund_pre = F[pre_end] - F[first]
    fund_post = np.where(has_fill, F[end] - F[np.minimum(fi + 1, len(F) - 1)], 0.0)
    V = np.r_[0, np.cumsum(np.asarray(vol) > 0)]
    traded = (V[end] - V[first]) > 0
    fx_pre = np.zeros(n, np.bool_)
    fx_post = np.zeros(n, np.bool_)
    fx_px = np.full(n, np.nan)
    fund_fx = np.zeros(n)
    for j in np.asarray(fx_idx, np.int64):
        t = int(np.searchsorted(cal_ms, ts[j] // DAY_MS * DAY_MS))
        if t >= n or cal_ms[t] != ts[j] // DAY_MS * DAY_MS:
            continue
        if fx_pre[t] or fx_post[t]:
            raise SimError("two forced exits on one day")
        if has_fill[t] and j >= fi[t]:
            fx_post[t] = True
            fund_fx[t] = F[j + 1] - F[fi[t] + 1]
        else:
            fx_pre[t] = True
            fund_fx[t] = F[j + 1] - F[first[t]]
        fx_px[t] = c[j]
    t_dec = cal_ms + DAY_MS
    blocked = np.zeros(n, np.bool_)
    since = np.full(n, np.inf)
    for a, b in gaps:
        blocked |= (t_dec >= a) & (t_dec < b + cooldown_ms)
        done = t_dec >= b
        since = np.where(done, np.minimum(since, (t_dec - b) / DAY_MS), since)
    return {"close": close, "traded": traded, "fill_px": fill_px, "fund_pre": fund_pre, "fund_post": fund_post,
            "fx_pre": fx_pre, "fx_post": fx_post, "fx_px": fx_px, "fund_fx": fund_fx, "blocked": blocked,
            "days_since_gap": since}


def stack(cols: dict[str, dict], days: pd.DatetimeIndex, member: np.ndarray, rank: np.ndarray, slip: np.ndarray,
          is_meme=None, funding_filled_h=None) -> Panel:
    """Panel from per-coin daily columns; coins are sorted by symbol (the tie-break order)."""
    syms = sorted(cols)
    if list(cols) != syms:
        raise ValueError("pass the coins sorted by symbol (member/rank/slip columns follow that order)")
    st = {k: np.column_stack([cols[s][k] for s in syms]) for k in cols[syms[0]]}
    k = len(syms)
    return Panel(days=days, symbols=syms, member=np.asarray(member, np.bool_), rank=np.asarray(rank, np.int64),
                 slip=np.asarray(slip, np.float64),
                 is_meme=np.zeros(k, np.bool_) if is_meme is None else np.asarray(is_meme, np.bool_),
                 funding_filled_h=np.zeros(k) if funding_filled_h is None else np.asarray(funding_filled_h, float),
                 **st)


# ---------------------------------------------------------------- the account

@njit(cache=True)
def _sim(close, fill_px, fund_pre, fund_post, fx_pre, fx_post, fx_px, fund_fx, slip, W, can_buy,
         first, last, fee, slip_mult, pen, eq0, resize, K, record):
    n, k = close.shape
    eq = np.full(n, np.nan)
    fees_d = np.zeros(n)
    fund_d = np.zeros(n)
    notional_d = np.zeros(n)
    held_d = np.zeros(n, np.int64)
    ep = np.zeros((n * k + k + 1 if record else 1, 14))   # at most one entry per coin and day
    ne = 0
    err, err_t, err_j = 0, -1, -1
    q = np.zeros(k)
    lastc = np.full(k, np.nan)
    lastc_t = np.full(k, -1, np.int64)
    e_t = np.zeros(k, np.int64)
    e_px = np.zeros(k)
    e_q0 = np.zeros(k)
    e_eq = np.zeros(k)
    e_cf = np.zeros(k)
    e_fee = np.zeros(k)
    e_fund = np.zeros(k)
    e_n = np.zeros(k, np.int64)
    e_not = np.zeros(k)
    cash = eq0
    prev = eq0
    for t in range(first, last + 1):
        # 1-2) before the fill: funding on the old position, a forced exit before the fill minute
        for j in range(k):
            if q[j] > 0.0:
                if fx_pre[t, j]:
                    f = q[j] * fund_fx[t, j]
                    cash -= f
                    fund_d[t] += f
                    e_fund[j] += f
                    px = fx_px[t, j] * (1.0 - (slip[t, j] * slip_mult + pen))
                    fe = q[j] * px * fee
                    cash += q[j] * px - fe
                    fees_d[t] += fe
                    notional_d[t] += q[j] * px
                    e_fee[j] += fe
                    e_cf[j] += q[j] * px
                    e_n[j] += 1
                    e_not[j] += q[j] * px
                    if record:
                        ep[ne, 0] = j; ep[ne, 1] = e_t[j]; ep[ne, 2] = t; ep[ne, 3] = 1
                        ep[ne, 4] = e_px[j]; ep[ne, 5] = px; ep[ne, 6] = e_q0[j]; ep[ne, 7] = e_eq[j]
                        ep[ne, 8] = e_cf[j]; ep[ne, 9] = e_fee[j]; ep[ne, 10] = e_fund[j]
                        ep[ne, 11] = e_cf[j] - e_fee[j] - e_fund[j]; ep[ne, 12] = e_n[j]; ep[ne, 13] = e_not[j]
                    ne += 1
                    q[j] = 0.0
                else:
                    f = q[j] * fund_pre[t, j]
                    cash -= f
                    fund_d[t] += f
                    e_fund[j] += f
        # 3) the fill
        sw = 0.0
        for j in range(k):
            if not np.isnan(W[t, j]):
                sw += W[t, j]
        if sw > 1.0 + 1e-9 and err == 0:
            err, err_t, err_j = 4, t, -1
        for j in range(k):
            w = W[t, j]
            if np.isnan(w):
                continue
            if w == 0.0 and q[j] == 0.0:
                continue
            X = fill_px[t, j]
            if np.isnan(X):
                if err == 0:
                    err, err_t, err_j = (1 if q[j] == 0.0 else 6), t, j
                continue
            if resize or q[j] == 0.0 or w == 0.0:
                target = w * prev / X
            else:
                target = q[j]
            d = target - q[j]
            if d == 0.0:
                continue
            if d > 0.0 and not can_buy[t, j] and err == 0:
                err, err_t, err_j = 2, t, j
            s = slip[t, j] * slip_mult
            px = X * (1.0 + s) if d > 0.0 else X * (1.0 - s)
            fe = abs(d) * px * fee
            cash -= d * px + fe
            fees_d[t] += fe
            notional_d[t] += abs(d) * px
            if q[j] == 0.0:
                e_t[j] = t
                e_px[j] = px
                e_q0[j] = target
                e_eq[j] = prev
                e_cf[j] = 0.0
                e_fee[j] = 0.0
                e_fund[j] = 0.0
                e_n[j] = 0
                e_not[j] = 0.0
            e_cf[j] -= d * px
            e_fee[j] += fe
            e_n[j] += 1
            e_not[j] += abs(d) * px
            q[j] = target
            if target == 0.0:
                if record:
                    ep[ne, 0] = j; ep[ne, 1] = e_t[j]; ep[ne, 2] = t; ep[ne, 3] = 0
                    ep[ne, 4] = e_px[j]; ep[ne, 5] = px; ep[ne, 6] = e_q0[j]; ep[ne, 7] = e_eq[j]
                    ep[ne, 8] = e_cf[j]; ep[ne, 9] = e_fee[j]; ep[ne, 10] = e_fund[j]
                    ep[ne, 11] = e_cf[j] - e_fee[j] - e_fund[j]; ep[ne, 12] = e_n[j]; ep[ne, 13] = e_not[j]
                ne += 1
        # 4-5) after the fill: funding on the new position, a forced exit at or after the fill minute
        for j in range(k):
            if q[j] > 0.0:
                if fx_post[t, j]:
                    f = q[j] * fund_fx[t, j]
                    cash -= f
                    fund_d[t] += f
                    e_fund[j] += f
                    px = fx_px[t, j] * (1.0 - (slip[t, j] * slip_mult + pen))
                    fe = q[j] * px * fee
                    cash += q[j] * px - fe
                    fees_d[t] += fe
                    notional_d[t] += q[j] * px
                    e_fee[j] += fe
                    e_cf[j] += q[j] * px
                    e_n[j] += 1
                    e_not[j] += q[j] * px
                    if record:
                        ep[ne, 0] = j; ep[ne, 1] = e_t[j]; ep[ne, 2] = t; ep[ne, 3] = 1
                        ep[ne, 4] = e_px[j]; ep[ne, 5] = px; ep[ne, 6] = e_q0[j]; ep[ne, 7] = e_eq[j]
                        ep[ne, 8] = e_cf[j]; ep[ne, 9] = e_fee[j]; ep[ne, 10] = e_fund[j]
                        ep[ne, 11] = e_cf[j] - e_fee[j] - e_fund[j]; ep[ne, 12] = e_n[j]; ep[ne, 13] = e_not[j]
                    ne += 1
                    q[j] = 0.0
                else:
                    f = q[j] * fund_post[t, j]
                    cash -= f
                    fund_d[t] += f
                    e_fund[j] += f
        # 6) the mark
        nh = 0
        mv = 0.0
        for j in range(k):
            if not np.isnan(close[t, j]):
                lastc[j] = close[t, j]
                lastc_t[j] = t
            if q[j] > 0.0:
                nh += 1
                mv += q[j] * lastc[j]
        held_d[t] = nh
        if nh > K and err == 0:
            err, err_t, err_j = 3, t, -1
        eq[t] = cash + mv
        if not (eq[t] > 0.0) and err == 0:
            err, err_t, err_j = 5, t, -1
        prev = eq[t]
    # the window closes at the last close
    for j in range(k):
        if q[j] > 0.0:
            px = lastc[j] * (1.0 - slip[lastc_t[j], j] * slip_mult)
            fe = q[j] * px * fee
            cash += q[j] * px - fe
            fees_d[last] += fe
            notional_d[last] += q[j] * px
            e_cf[j] += q[j] * px
            e_fee[j] += fe
            e_n[j] += 1
            e_not[j] += q[j] * px
            if record:
                ep[ne, 0] = j; ep[ne, 1] = e_t[j]; ep[ne, 2] = last; ep[ne, 3] = 2
                ep[ne, 4] = e_px[j]; ep[ne, 5] = px; ep[ne, 6] = e_q0[j]; ep[ne, 7] = e_eq[j]
                ep[ne, 8] = e_cf[j]; ep[ne, 9] = e_fee[j]; ep[ne, 10] = e_fund[j]
                ep[ne, 11] = e_cf[j] - e_fee[j] - e_fund[j]; ep[ne, 12] = e_n[j]; ep[ne, 13] = e_not[j]
            ne += 1
            q[j] = 0.0
    if last >= first:
        eq[last] = cash
    if record and err == 0:
        tot = 0.0
        for r in range(ne):
            tot += ep[r, 11]
        if abs(tot - (cash - eq0)) > 1e-7 * eq0:
            err, err_t, err_j = 7, last, -1
    return eq, fees_d, fund_d, notional_d, held_d, ep[:ne] if record else ep[:0], err, err_t, err_j


@njit(cache=True)
def mean_return(eq, first, last, eq0):
    """Mean daily return of the window; the one summation used for the null's statistic, actual and permuted."""
    s = 0.0
    prev = eq0
    for t in range(first, last + 1):
        s += eq[t] / prev - 1.0
        prev = eq[t]
    return s / (last - first + 1)


@dataclass
class SimResult:
    equity: pd.Series            # end-of-day equity over the window (the last day after the closing trades)
    daily: pd.Series             # daily returns over the window (the first day against the initial equity)
    episodes: pd.DataFrame       # holding episodes: entry to full exit (resizes inside)
    fees: pd.Series
    funding: pd.Series
    turnover: pd.Series          # traded notional / the decision equity
    held: pd.Series              # coins held after each day's fill
    initial_equity: float


def simulate(panel: Panel, W: np.ndarray, can_buy: np.ndarray, first: int, last: int, *, taker: float = 0.0005,
             slip_mult: float = 1.0, fee_mult: float = 1.0, penalty: float = 0.01, eq0: float = 10_000.0,
             resize: bool = True, K: int = 3) -> SimResult:
    """Run the account over fill days first..last (panel indices, inclusive). Raises SimError on a hard check."""
    n, k = panel.shape
    if W.shape != (n, k) or can_buy.shape != (n, k):
        raise ValueError("W and can_buy must be (days, coins)")
    if not (0 < first <= last < n):
        raise ValueError("the window needs a decision day before it")
    eq, fees, fund, notional, held, ep, err, et, ej = _sim(
        panel.close, panel.fill_px, panel.fund_pre, panel.fund_post, panel.fx_pre, panel.fx_post, panel.fx_px,
        panel.fund_fx, panel.slip, np.ascontiguousarray(W, np.float64), np.ascontiguousarray(can_buy, np.bool_),
        int(first), int(last), float(taker * fee_mult), float(slip_mult), float(penalty), float(eq0), bool(resize),
        int(K), True)
    if err:
        where = f"{panel.days[et].date()}" + (f" {panel.symbols[ej]}" if ej >= 0 else "")
        raise SimError(f"{ERRORS[err]} ({where})")
    days = panel.days[first:last + 1]
    e = pd.Series(eq[first:last + 1], index=days)
    prev = np.r_[eq0, eq[first:last]]
    daily = pd.Series(eq[first:last + 1] / prev - 1.0, index=days)
    epi = pd.DataFrame(ep, columns=EP_COLS)
    for col in ("coin", "entry_day", "exit_day", "reason", "trades"):
        epi[col] = epi[col].astype(np.int64)
    epi.insert(0, "symbol", [panel.symbols[j] for j in epi["coin"]])
    epi["entry_date"] = panel.days[epi["entry_day"].to_numpy()]
    epi["exit_date"] = panel.days[epi["exit_day"].to_numpy()]
    epi["reason"] = [EXIT_REASONS[r] for r in epi["reason"]]
    epi["ret"] = epi["net"] / epi["eq_before"]
    epi["days_held"] = epi["exit_day"] - epi["entry_day"] + (epi["reason"] != "rule").astype(int)
    dec_eq = pd.Series(prev, index=days)
    return SimResult(e, daily, epi, pd.Series(fees[first:last + 1], index=days),
                     pd.Series(fund[first:last + 1], index=days),
                     pd.Series(notional[first:last + 1], index=days) / dec_eq,
                     pd.Series(held[first:last + 1], index=days), float(eq0))


def mean_daily_fast(panel: Panel, W: np.ndarray, can_buy: np.ndarray, first: int, last: int, *, taker: float,
                    slip_mult: float, penalty: float, eq0: float = 10_000.0, resize: bool = True, K: int = 3) -> float:
    """Mean daily return only (no episode records); used by the null and the calibration."""
    eq, *_, err, et, ej = _sim(panel.close, panel.fill_px, panel.fund_pre, panel.fund_post, panel.fx_pre,
                               panel.fx_post, panel.fx_px, panel.fund_fx, panel.slip, W, can_buy, first, last,
                               taker, slip_mult, penalty, eq0, resize, K, False)
    if err:
        raise SimError(f"{ERRORS[err]} ({panel.days[et].date()})")
    return float(mean_return(eq, first, last, eq0))
