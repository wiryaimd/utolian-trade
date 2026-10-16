"""Bar-signal backtester that executes on the 1-minute path.

Signals are produced per strategy bar (15m/1h/4h) at that bar's close. The engine
walks every 1m bar: actions decided at a strategy bar's close execute at the open
of the first minute of the next strategy bar; stops/targets are checked on each
1m high/low (stop assumed first if both are touched in the same minute).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numba import njit

from btcq.config import MINUTE_MS, SYMBOL, TF_MS, Costs, Risk
from btcq.data import store
from btcq.data.bardata import BarData

# 6 = "manual" matches the live book; 7 = "delist" is a forced exit (round 4)
REASONS = ["stop", "target", "exit_signal", "time", "reverse", "end", "manual", "delist"]
TRADE_COLS = ["entry_i", "exit_i", "dir", "entry_px", "exit_px", "qty", "init_risk", "gross",
              "fees", "funding", "net", "eq_before", "reason", "entry_tf", "maker"]


class Market(BarData):
    """1m execution path + funding for one period, with cached strategy-timeframe bars."""

    @classmethod
    def load(cls, period: str = "dev", symbol: str = SYMBOL, with_metrics: bool = True) -> "Market":
        m = cls(store.load_1m(period, symbol), store.load_funding(period, symbol),
                store.load_metrics(period, symbol) if with_metrics else None,
                premium=store.load_premium(period, symbol))
        m.period = period
        m.symbol = symbol
        return m

    def __init__(self, m1: pd.DataFrame, funding: pd.DataFrame | None = None,
                 metrics: pd.DataFrame | None = None,
                 synthetic_funding: float = Costs.synthetic_funding,
                 premium: pd.DataFrame | None = None):
        super().__init__(m1.reset_index(drop=True), funding, metrics, premium)
        self.period = "custom"
        self.symbol = SYMBOL
        self.ts = self.m1["ts"].to_numpy(np.int64)
        self.o = self.m1["open"].to_numpy(np.float64)
        self.h = self.m1["high"].to_numpy(np.float64)
        self.l = self.m1["low"].to_numpy(np.float64)
        self.c = self.m1["close"].to_numpy(np.float64)
        self.fund = self._funding_per_minute(synthetic_funding)
        day = self.ts // TF_MS["1d"]
        self.day_marks = np.flatnonzero(np.r_[day[1:] != day[:-1], True]).astype(np.int64)
        self.days = pd.to_datetime(day[self.day_marks] * TF_MS["1d"], unit="ms", utc=True)

    def _funding_per_minute(self, synthetic: float) -> np.ndarray:
        fund = np.zeros(len(self.ts))
        f = self.funding_df
        first_real = int(f["ts"].iloc[0]) if len(f) else int(self.ts[-1]) + 1
        # Synthetic 8h funding for the period before real history exists.
        step = 8 * 3_600_000
        syn = np.arange((self.ts[0] // step + 1) * step, min(first_real, self.ts[-1] + 1), step)
        for times, rates in ((syn, np.full(len(syn), synthetic)),
                             (f["ts"].to_numpy(np.int64), f["rate"].to_numpy(np.float64))):
            times = (times + MINUTE_MS // 2) // MINUTE_MS * MINUTE_MS
            idx = np.searchsorted(self.ts, times)
            ok = idx < len(self.ts)
            np.add.at(fund, idx[ok], rates[ok])
        return fund

    def exec_idx(self, tf: str, latency_min: int = 0) -> np.ndarray:
        """1m index where decisions made at each TF bar's close are executed, after an
        optional order-placement latency (resting stops/targets are not delayed)."""
        b = self.bars(tf)
        t = b["ts"].to_numpy(np.int64) + TF_MS[tf] + latency_min * MINUTE_MS
        return np.searchsorted(self.ts, t).astype(np.int64)

    def index_at(self, ts: pd.Timestamp) -> int:
        return int(np.searchsorted(self.ts, ts.value // 10**6))


@dataclass
class Signals:
    """Per-TF-bar decisions, all computed using information up to that bar's close."""
    tf: str
    entry: np.ndarray            # int8: +1 long, -1 short, 0 none
    stop: np.ndarray             # absolute initial stop for an entry at this bar
    target: np.ndarray           # absolute target (NaN = none)
    trail: np.ndarray | None = None       # trailing distance from extreme since entry (NaN = none)
    exit_long: np.ndarray | None = None   # close long at next open
    exit_short: np.ndarray | None = None
    limit: np.ndarray | None = None       # limit entry price valid for next bar (NaN = market)
    max_bars: int = 0            # time stop in TF bars (0 = none)
    size: np.ndarray | None = None        # notional as a multiple of equity (NaN = risk-at-stop sizing)

    def arrays(self, n: int):
        nanv = np.full(n, np.nan)
        falsev = np.zeros(n, dtype=np.bool_)
        return (np.asarray(self.entry, np.int8), np.asarray(self.stop, np.float64),
                np.asarray(self.target, np.float64),
                nanv if self.trail is None else np.asarray(self.trail, np.float64),
                falsev if self.exit_long is None else np.asarray(self.exit_long, np.bool_),
                falsev if self.exit_short is None else np.asarray(self.exit_short, np.bool_),
                nanv if self.limit is None else np.asarray(self.limit, np.float64),
                nanv if self.size is None else np.asarray(self.size, np.float64))


@dataclass
class Result:
    trades: pd.DataFrame
    equity: pd.Series            # daily mark-to-market equity
    initial_equity: float
    start: pd.Timestamp
    info: dict = field(default_factory=dict)


@njit(cache=True)
def _qty(size_mult, risk, max_lev, cash, px, dist):
    cap = max_lev * cash / px
    if np.isnan(size_mult):
        return min(risk * cash / dist, cap)
    return min(size_mult * cash / px, cap)


@njit(cache=True)
def _simulate(o, h, l, c, fund, exec_idx, entry, stop, target, trail, exit_long, exit_short,
              limit, size, max_bars, start_i, end_i, risk, max_lev, fee_t, fee_m, slip, eq0, marks,
              stop_mode, force_exit, force_pen):
    # slip: per-minute slippage, applied at the fill minute. stop_mode 1 = stops fill at the
    # minute's worst price (low for longs, high for shorts). force_exit[i] closes an open
    # position at c[i] with an extra adverse force_pen (e.g. delisting).
    n_tf = exec_idx.shape[0]
    tr = np.zeros((n_tf + 2, 15))
    nt = 0
    eq_marks = np.full(marks.shape[0], np.nan)
    mi = 0
    while mi < marks.shape[0] and marks[mi] < start_i:
        mi += 1

    cash = eq0
    pos = 0
    qty = 0.0
    epx = 0.0
    sl = 0.0
    tp = np.nan
    ext = 0.0
    held = 0
    e_i = 0
    e_tf = 0
    e_fee = 0.0
    e_fund = 0.0
    e_eq = 0.0
    e_risk = 0.0
    e_maker = 0
    # pending limit order
    p_on = False
    p_dir = 0
    p_px = 0.0
    p_sl = 0.0
    p_tp = np.nan
    p_exp = 0
    p_tf = 0
    p_size = np.nan

    t = 0
    while t < n_tf and exec_idx[t] < start_i:
        t += 1

    for i in range(start_i, end_i):
        # 1) funding settles on positions open before this minute
        if pos != 0 and fund[i] != 0.0:
            f = qty * o[i] * fund[i] * pos
            cash -= f
            e_fund += f

        # 2) decisions from strategy bars that closed at this minute's open
        while t < n_tf and exec_idx[t] == i:
            if pos != 0:
                held += 1
                reason = -1
                if (pos == 1 and exit_long[t]) or (pos == -1 and exit_short[t]):
                    reason = 2
                elif max_bars > 0 and held >= max_bars:
                    reason = 3
                elif entry[t] == -pos:
                    reason = 4
                if reason >= 0:
                    px = o[i] * (1.0 - slip[i] * pos)
                    gross = qty * (px - epx) * pos
                    fee = qty * px * fee_t
                    cash += gross - fee
                    tr[nt, 0] = e_i; tr[nt, 1] = i; tr[nt, 2] = pos; tr[nt, 3] = epx
                    tr[nt, 4] = px; tr[nt, 5] = qty; tr[nt, 6] = e_risk; tr[nt, 7] = gross
                    tr[nt, 8] = e_fee + fee; tr[nt, 9] = e_fund
                    tr[nt, 10] = gross - e_fee - fee - e_fund; tr[nt, 11] = e_eq
                    tr[nt, 12] = reason; tr[nt, 13] = e_tf; tr[nt, 14] = e_maker
                    nt += 1
                    pos = 0
                elif not np.isnan(trail[t]):
                    if pos == 1:
                        cand = ext - trail[t]
                        if cand > sl:
                            sl = cand
                    else:
                        cand = ext + trail[t]
                        if cand < sl:
                            sl = cand
            if pos == 0 and entry[t] != 0:
                d = entry[t]
                if not np.isnan(limit[t]):
                    p_on = True
                    p_dir = d
                    p_px = limit[t]
                    p_sl = stop[t]
                    p_tp = target[t]
                    p_exp = exec_idx[t + 1] if t + 1 < n_tf else end_i
                    p_tf = t
                    p_size = size[t]
                else:
                    px = o[i] * (1.0 + slip[i] * d)
                    s = stop[t]
                    if not np.isnan(s) and (s - px) * d < 0:
                        dist = abs(px - s)
                        q = _qty(size[t], risk, max_lev, cash, px, dist)
                        if q > 0:
                            pos = d
                            qty = q
                            epx = px
                            sl = s
                            tp = target[t]
                            ext = o[i]
                            held = 0
                            e_i = i
                            e_tf = t
                            e_fee = q * px * fee_t
                            e_fund = 0.0
                            e_eq = cash
                            e_risk = q * dist
                            e_maker = 0
                            cash -= e_fee
            t += 1

        # 3) pending limit entry: fills only if price trades through the level
        just_filled = False
        if p_on and pos == 0:
            if i >= p_exp:
                p_on = False
            elif (p_dir == 1 and l[i] < p_px) or (p_dir == -1 and h[i] > p_px):
                p_on = False
                if (p_sl - p_px) * p_dir < 0:
                    dist = abs(p_px - p_sl)
                    q = _qty(p_size, risk, max_lev, cash, p_px, dist)
                    pos = p_dir
                    qty = q
                    epx = p_px
                    sl = p_sl
                    tp = p_tp
                    ext = p_px
                    held = 0
                    e_i = i
                    e_tf = p_tf
                    e_fee = q * p_px * fee_m
                    e_fund = 0.0
                    e_eq = cash
                    e_risk = q * dist
                    e_maker = 1
                    cash -= e_fee
                    just_filled = True
        elif p_on and pos != 0:
            p_on = False

        # 4) stop / target on this minute's range (stop wins ties)
        if pos != 0:
            px = np.nan
            reason = -1
            if pos == 1:
                if l[i] <= sl:
                    if stop_mode == 1:
                        px = l[i]
                    else:
                        px = min(o[i], sl) if not just_filled else sl
                    reason = 0
                elif (not just_filled) and (not np.isnan(tp)) and h[i] >= tp:
                    px = max(o[i], tp)
                    reason = 1
            else:
                if h[i] >= sl:
                    if stop_mode == 1:
                        px = h[i]
                    else:
                        px = max(o[i], sl) if not just_filled else sl
                    reason = 0
                elif (not just_filled) and (not np.isnan(tp)) and l[i] <= tp:
                    px = min(o[i], tp)
                    reason = 1
            if reason >= 0:
                px = px * (1.0 - slip[i] * pos)
                gross = qty * (px - epx) * pos
                fee = qty * px * fee_t
                cash += gross - fee
                tr[nt, 0] = e_i; tr[nt, 1] = i; tr[nt, 2] = pos; tr[nt, 3] = epx
                tr[nt, 4] = px; tr[nt, 5] = qty; tr[nt, 6] = e_risk; tr[nt, 7] = gross
                tr[nt, 8] = e_fee + fee; tr[nt, 9] = e_fund
                tr[nt, 10] = gross - e_fee - fee - e_fund; tr[nt, 11] = e_eq
                tr[nt, 12] = reason; tr[nt, 13] = e_tf; tr[nt, 14] = e_maker
                nt += 1
                pos = 0
            elif pos == 1:
                if h[i] > ext:
                    ext = h[i]
            else:
                if l[i] < ext:
                    ext = l[i]

        # 4b) forced exit at this minute's close (a stop hit in the same minute wins)
        if pos != 0 and force_exit[i]:
            px = c[i] * (1.0 - (slip[i] + force_pen) * pos)
            gross = qty * (px - epx) * pos
            fee = qty * px * fee_t
            cash += gross - fee
            tr[nt, 0] = e_i; tr[nt, 1] = i; tr[nt, 2] = pos; tr[nt, 3] = epx
            tr[nt, 4] = px; tr[nt, 5] = qty; tr[nt, 6] = e_risk; tr[nt, 7] = gross
            tr[nt, 8] = e_fee + fee; tr[nt, 9] = e_fund
            tr[nt, 10] = gross - e_fee - fee - e_fund; tr[nt, 11] = e_eq
            tr[nt, 12] = 7; tr[nt, 13] = e_tf; tr[nt, 14] = e_maker
            nt += 1
            pos = 0

        # 5) daily mark-to-market
        while mi < marks.shape[0] and marks[mi] == i:
            eq_marks[mi] = cash + (qty * (c[i] - epx) * pos if pos != 0 else 0.0)
            mi += 1

        if cash <= 0:
            break

    if pos != 0:
        i = end_i - 1
        px = c[i] * (1.0 - slip[i] * pos)
        gross = qty * (px - epx) * pos
        fee = qty * px * fee_t
        cash += gross - fee
        tr[nt, 0] = e_i; tr[nt, 1] = i; tr[nt, 2] = pos; tr[nt, 3] = epx
        tr[nt, 4] = px; tr[nt, 5] = qty; tr[nt, 6] = e_risk; tr[nt, 7] = gross
        tr[nt, 8] = e_fee + fee; tr[nt, 9] = e_fund
        tr[nt, 10] = gross - e_fee - fee - e_fund; tr[nt, 11] = e_eq
        tr[nt, 12] = 5; tr[nt, 13] = e_tf; tr[nt, 14] = e_maker
        nt += 1
        if marks.shape[0] > 0 and marks[marks.shape[0] - 1] == i:
            eq_marks[marks.shape[0] - 1] = cash
    return tr[:nt], eq_marks


def run(market: Market, sig: Signals, costs: Costs = Costs(), risk: Risk = Risk(),
        start: pd.Timestamp | None = None, latency_min: int = 0, *,
        slippage: np.ndarray | None = None, stop_mode: int = 0,
        force_exit: np.ndarray | None = None, force_penalty: float = 0.0) -> Result:
    """Simulate a strategy's signals. Entries are only allowed from `start` onward.

    slippage: optional per-minute slippage array (replaces costs.slippage when given).
    stop_mode: 0 = stops fill at the stop (or the open on a gap); 1 = at the minute's worst price.
    force_exit: optional bool per minute; an open position is closed at that minute's close with
    an extra adverse force_penalty (reason "delist")."""
    ex = market.exec_idx(sig.tf, latency_min)
    n_tf = len(ex)
    entry, stop, target, trail, xl, xs, limit, size = sig.arrays(n_tf)
    start_i = market.index_at(start) if start is not None else 0
    end_i = len(market.ts)
    n = len(market.ts)
    slip = np.full(n, costs.slippage) if slippage is None else np.ascontiguousarray(slippage, np.float64)
    fx = np.zeros(n, np.bool_) if force_exit is None else np.ascontiguousarray(force_exit, np.bool_)
    if slip.shape != (n,) or fx.shape != (n,):
        raise ValueError("slippage and force_exit must have one value per minute")
    trades, eq = _simulate(market.o, market.h, market.l, market.c, market.fund, ex, entry, stop,
                           target, trail, xl, xs, limit, size, int(sig.max_bars), start_i, end_i,
                           risk.risk_per_trade, risk.max_leverage, costs.taker, costs.maker,
                           slip, risk.initial_equity, market.day_marks,
                           int(stop_mode), fx, float(force_penalty))
    tdf = pd.DataFrame(trades, columns=TRADE_COLS)
    for col in ("entry_i", "exit_i", "dir", "reason", "entry_tf", "maker"):
        tdf[col] = tdf[col].astype(np.int64)
    tdf["entry_time"] = pd.to_datetime(market.ts[tdf["entry_i"]], unit="ms", utc=True)
    tdf["exit_time"] = pd.to_datetime(market.ts[tdf["exit_i"]], unit="ms", utc=True)
    tdf["R"] = tdf["net"] / tdf["init_risk"]
    tdf["ret"] = tdf["net"] / tdf["eq_before"]
    tdf["reason"] = [REASONS[r] for r in tdf["reason"]]
    equity = pd.Series(eq, index=market.days).dropna()
    first = pd.Timestamp(market.ts[start_i], unit="ms", tz="UTC")
    return Result(tdf, equity, risk.initial_equity, first)
