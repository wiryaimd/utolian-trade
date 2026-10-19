"""Single-market daily simulator (Round 8): signed positions decided at a trade date's close and filled at the same
close (lag 0), or one row later (lag 1, at the next close or the next open), sized at entry or flip only.

`engine.run` fills only on the next bar, so it can't book a fill at the decision's own close. With lag 1 and fills at
the close, this simulator reproduces `engine.run` on close-only rows (open = high = low = close, sentinel stops,
level signals as `r6_macro.engine_signals`): trades, fees and daily marks (a unit test).

Conventions (engine.run's):
- rows are one market's trade dates; the window is rows first..last; a decision at row k fills at row k + lag if that
  row is inside the window (so with lag 1 the decision at row first - 1 fills at row first);
- a fill at row i with target tau and position p: if p != 0 and tau != p the position is closed at
  fill_px[i] x (1 - slip x p), paying fee_out on the exit notional; then, if flat and tau != 0, a new position is
  opened at fill_px[i] x (1 + slip x tau) with quantity min(size, lev_cap) x cash / price, paying fee_in on the entry
  notional. A position that repeats is renewed for free and never resized;
- the mark at each row's close is cash + qty x (close - entry price) x side;
- any position still open after the window's last row is closed at that row's close (slip and fee_out), and the last
  mark is the cash after it. With lag 0 the decision at the last row is not opened (it would be held past the
  window): an open position is closed there instead. With lag 1 the engine's convention is kept (a decision at
  row last - 1 fills at row last and is closed at once);
- liquidation (Round 6's rule): at the first mark where equity <= 0 or |notional| >= liq_lev x equity, the position
  is closed at max(equity, 0) at that mark (only its entry fee was paid) and the sleeve stays flat; an exit (or the
  exit leg of a flip) that leaves the cash <= 0 is booked the same way, at 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit

REASONS = ["exit", "reverse", "end", "liquidation"]
TRADE_COLS = ["entry_i", "exit_i", "dir", "entry_px", "exit_px", "qty", "gross", "fees", "slip_cost", "net",
              "eq_before", "reason"]


@dataclass
class DailyResult:
    trades: pd.DataFrame        # one row per holding episode
    equity: np.ndarray          # mark at each row's close (NaN outside first..last)
    lev: np.ndarray             # signed notional / equity at each mark (0 when flat or outside the window)
    fees: np.ndarray            # fees paid at each row (currency)
    slip_cost: np.ndarray       # slippage paid at each row (currency, against the fill reference price)
    turnover: np.ndarray        # notional traded at each row
    liq_row: int                # row of the liquidation mark, -1 if none
    eq0: float
    first: int
    last: int


@njit(cache=True)
def _sim(close, fill_px, pos, size, first, last, lag, fee_in, fee_out, slip, eq0, lev_cap, liq_lev, liquidate):
    n = close.shape[0]
    tr = np.zeros((n + 2, 12))
    nt = 0
    eq = np.full(n, np.nan)
    lev = np.zeros(n)
    fees = np.zeros(n)
    slc = np.zeros(n)
    turn = np.zeros(n)
    cash = eq0
    p = 0
    q = 0.0
    epx = 0.0
    e_i = 0
    e_fee = 0.0
    e_slip = 0.0
    e_eq = 0.0
    dead = False
    liq_row = -1
    for i in range(first, last + 1):
        k = i - lag
        tau = 0
        size_k = np.nan
        if k >= 0:
            tau = pos[k]
            size_k = size[k]
        if lag == 0 and i == last:
            tau = 0
        if dead:
            tau = 0
        # exit (or the exit leg of a flip)
        if p != 0 and tau != p:
            ref = fill_px[i]
            px = ref * (1.0 - slip * p)
            gross = q * (px - epx) * p
            fee = q * px * fee_out
            sc = q * ref * slip
            cash += gross - fee
            fees[i] += fee
            slc[i] += sc
            turn[i] += q * px
            tr[nt, 0] = e_i; tr[nt, 1] = i; tr[nt, 2] = p; tr[nt, 3] = epx; tr[nt, 4] = px; tr[nt, 5] = q
            tr[nt, 6] = gross; tr[nt, 7] = e_fee + fee; tr[nt, 8] = e_slip + sc
            tr[nt, 9] = gross - e_fee - fee; tr[nt, 10] = e_eq
            tr[nt, 11] = 1 if tau == -p else (2 if (lag == 0 and i == last) else 0)
            p = 0
            if liquidate and cash <= 0.0:          # the loss is capped at the sleeve's capital (Round 6's rule)
                tr[nt, 9] = -e_eq
                tr[nt, 11] = 3
                cash = 0.0
                dead = True
                tau = 0
                liq_row = i
            nt += 1
        # entry (or the entry leg of a flip)
        if p == 0 and tau != 0 and not np.isnan(size_k):
            ref = fill_px[i]
            px = ref * (1.0 + slip * tau)
            qq = min(size_k * cash / px, lev_cap * cash / px)
            if qq > 0:
                p = tau
                q = qq
                epx = px
                e_i = i
                e_fee = qq * px * fee_in
                e_slip = qq * ref * slip
                e_eq = cash
                cash -= e_fee
                fees[i] += e_fee
                slc[i] += e_slip
                turn[i] += qq * px
        # mark at the close
        m = cash + (q * (close[i] - epx) * p if p != 0 else 0.0)
        eq[i] = m
        if p != 0:
            notional = q * close[i]
            if liquidate and (m <= 0.0 or notional >= liq_lev * m):
                booked = max(m, 0.0)
                tr[nt, 0] = e_i; tr[nt, 1] = i; tr[nt, 2] = p; tr[nt, 3] = epx; tr[nt, 4] = close[i]
                tr[nt, 5] = q; tr[nt, 6] = q * (close[i] - epx) * p; tr[nt, 7] = e_fee; tr[nt, 8] = e_slip
                tr[nt, 9] = booked - e_eq; tr[nt, 10] = e_eq; tr[nt, 11] = 3
                nt += 1
                cash = booked
                p = 0
                eq[i] = booked
                dead = True
                liq_row = i
            elif m > 0:
                lev[i] = notional * p / m
    if p != 0:
        i = last
        ref = close[i]
        px = ref * (1.0 - slip * p)
        gross = q * (px - epx) * p
        fee = q * px * fee_out
        sc = q * ref * slip
        cash += gross - fee
        fees[i] += fee
        slc[i] += sc
        turn[i] += q * px
        tr[nt, 0] = e_i; tr[nt, 1] = i; tr[nt, 2] = p; tr[nt, 3] = epx; tr[nt, 4] = px; tr[nt, 5] = q
        tr[nt, 6] = gross; tr[nt, 7] = e_fee + fee; tr[nt, 8] = e_slip + sc
        tr[nt, 9] = gross - e_fee - fee; tr[nt, 10] = e_eq; tr[nt, 11] = 2
        if liquidate and cash <= 0.0:
            tr[nt, 9] = -e_eq
            tr[nt, 11] = 3
            cash = 0.0
            liq_row = i
        nt += 1
        eq[i] = cash
        lev[i] = 0.0
    return tr[:nt], eq, lev, fees, slc, turn, liq_row


def simulate(close, fill_px, pos, size, first: int, last: int, *, lag: int = 0, fee: float = 5e-4,
             slip: float = 0.0, eq0: float = 10_000.0, fee_in: float | None = None, fee_out: float | None = None,
             lev_cap: float = 2.0, liq_lev: float = 20.0, liquidate: bool = True) -> DailyResult:
    """One market's sleeve over rows first..last (inclusive). pos: int8 target per decision row; size: notional /
    equity at entry for each decision row (NaN = can't enter). fee_in / fee_out default to fee."""
    close = np.ascontiguousarray(close, np.float64)
    fill_px = np.ascontiguousarray(fill_px, np.float64)
    pos = np.ascontiguousarray(pos, np.int8)
    size = np.ascontiguousarray(size, np.float64)
    n = close.shape[0]
    if not (fill_px.shape == pos.shape == size.shape == (n,)):
        raise ValueError("close, fill_px, pos and size need one value per row")
    if lag not in (0, 1):
        raise ValueError("lag must be 0 or 1")
    if not (0 <= first <= last < n):
        raise ValueError("the window must be rows 0 <= first <= last < n")
    if not np.isfinite(close[first:last + 1]).all() or not np.isfinite(fill_px[first:last + 1]).all():
        raise ValueError("a window row has no price")
    fi = fee if fee_in is None else fee_in
    fo = fee if fee_out is None else fee_out
    tr, eq, lev, fees, slc, turn, liq = _sim(close, fill_px, pos, size, int(first), int(last), int(lag), float(fi),
                                             float(fo), float(slip), float(eq0), float(lev_cap), float(liq_lev),
                                             bool(liquidate))
    t = pd.DataFrame(tr, columns=TRADE_COLS)
    for c in ("entry_i", "exit_i", "dir", "reason"):
        t[c] = t[c].astype(np.int64)
    t["reason"] = [REASONS[r] for r in t["reason"]]
    t["ret"] = t["net"] / t["eq_before"] if len(t) else pd.Series(dtype=float)
    return DailyResult(t, eq, lev, fees, slc, turn, int(liq), float(eq0), int(first), int(last))


def calendar_daily(ts: np.ndarray, equity: np.ndarray, calendar: pd.DatetimeIndex, eq0: float) -> pd.Series:
    """A sleeve's daily returns on a UTC calendar: the marks of its trade dates (ts in ms, 00:00 UTC) forward-filled
    over the days without a row, eq0 before the first mark; 0 once a liquidated sleeve has no equity."""
    ok = np.isfinite(equity)
    e = pd.Series(equity[ok], index=pd.to_datetime(np.asarray(ts, np.int64)[ok], unit="ms", utc=True).as_unit("ms"))
    e = e.reindex(calendar).ffill().fillna(eq0)
    prev = e.shift(1)
    prev.iloc[0] = eq0
    return (e / prev - 1.0).where(prev > 0, 0.0)


def funding_drag(ts: np.ndarray, lev: np.ndarray, calendar: pd.DatetimeIndex, rate_ann: float) -> pd.Series:
    """The funding stress (Round 6's drag) on the calendar: on each trade date, |notional / equity| at the
    previous trade date's mark x rate x calendar days since it / 365 (Mondays carry 3), long or short. Subtracted
    from the sleeve's daily return by the caller (floored at -100%); it never compounds into sizes."""
    ts = np.asarray(ts, np.int64)
    lev = np.asarray(lev, np.float64)
    out = pd.Series(0.0, index=calendar)
    if len(ts) < 2:
        return out
    days = np.diff(ts) / 86_400_000.0
    charge = np.abs(lev[:-1]) * rate_ann * days / 365.0
    s = pd.Series(charge, index=pd.to_datetime(ts[1:], unit="ms", utc=True).as_unit("ms"))
    s = s[s.index.isin(calendar)]
    out.loc[s.index] = s.to_numpy()
    return out
