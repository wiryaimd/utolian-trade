"""Maker-first ("post, then cross") execution on the 1-minute path (study X1).

A copy of engine._simulate in which decision exits (signal, time, reverse) and entries rest as post-only
limit orders for T minutes and then cross as takers. Stops, targets, forced and end exits stay taker; the
stop stays live while an exit rests and wins a same-minute tie. With T = 0 nothing is posted and the run is
bit-identical to engine.run.

At a decision executed at minute m0 (the engine's taker minute), for side +1 buy / -1 sell:
  ref = o[m0] * (1 + side * slip[m0])     the registered taker price (what engine.run pays)
  P   = o[m0] - side * tick[m0]           the post price, one tick inside the open
  exp = first row at/after m0 + T minutes  the order crosses as a taker at o[exp] (or is cancelled)
Fill models, at a minute m0 <= j < exp:
  strict      buy l[j] < P (sell h[j] > P), filled at ref: only the fee changes (the decision basis)
  central     same condition, filled at o[m0]: the registered slippage is saved too
  optimistic  buy l[j] <= P (sell h[j] >= P), filled at P
Prices within PX_TOL * P of P count as equal to it: P carries float noise from o - tick, which would otherwise
turn exact one-tick touches into trade-throughs (or hide them from the optimistic model).
Maker fills pay costs.maker; fallbacks pay costs.taker plus the slippage of the fallback minute. An entry's
quantity is fixed when it is posted (from cash, ref and the signal's stop) and kept for its fill or fallback.
A maker entry starts its trailing extreme at its market price (o[m0] for strict, whose ref only adds the
registered slippage, and the fill price otherwise); one that would fill at or beyond its own stop opens nothing
and is recorded as missed, as the engine's limit fills. An entry never opens on cash <= 0. With the fallback,
an entry still resting when the window ends or a forced exit fires crosses at that minute's close and is
closed with it (without the fallback it is cancelled).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit

from btcq.backtest.engine import REASONS, TRADE_COLS, Market, Result, Signals, _qty
from btcq.config import MINUTE_MS, Costs, Risk

MODELS = {"strict": 0, "central": 1, "optimistic": 2}
PX_TOL = 1e-9                     # relative tolerance of the fill condition at the post price
EXTRA_COLS = ["exit_maker", "entry_post_i", "exit_post_i", "entry_ref_px", "exit_ref_px"]
MF_COLS = TRADE_COLS + EXTRA_COLS
INFO_KEYS = ["orders_posted", "entry_maker_fills", "exit_maker_fills", "entry_fallbacks", "exit_fallbacks",
             "entries_missed", "exits_cancelled", "exits_preempted_by_stop", "forced_fallbacks",
             "funding_in_window"]
_INT_COLS = ("entry_i", "exit_i", "dir", "reason", "entry_tf", "maker", "exit_maker", "entry_post_i",
             "exit_post_i")


@dataclass(frozen=True)
class MakerPolicy:
    T: int = 0                # minutes the post-only order rests; 0 = pure taker, bit-identical to engine.run
    model: str = "strict"     # fill/price model (MODELS)
    fallback: bool = True     # False = post-and-cancel (information only: it changes which trades are taken)


@njit(cache=True)
def _close(tr, nt, i, px, fee_rate, pos, qty, epx, e_i, e_risk, e_fee, e_fund, e_eq, reason, e_tf, e_maker,
           x_maker, e_post, x_post, e_ref, x_ref):
    """Write trade row nt for a close at px and return the cash change (engine operation order)."""
    gross = qty * (px - epx) * pos
    fee = qty * px * fee_rate
    tr[nt, 0] = e_i; tr[nt, 1] = i; tr[nt, 2] = pos; tr[nt, 3] = epx
    tr[nt, 4] = px; tr[nt, 5] = qty; tr[nt, 6] = e_risk; tr[nt, 7] = gross
    tr[nt, 8] = e_fee + fee; tr[nt, 9] = e_fund
    tr[nt, 10] = gross - e_fee - fee - e_fund; tr[nt, 11] = e_eq
    tr[nt, 12] = reason; tr[nt, 13] = e_tf; tr[nt, 14] = e_maker
    tr[nt, 15] = x_maker; tr[nt, 16] = e_post; tr[nt, 17] = x_post; tr[nt, 18] = e_ref; tr[nt, 19] = x_ref
    return gross - fee


@njit(cache=True)
def _cross_and_close(tr, nt, i, c_i, slip_i, pen, fee_t, cash, d, q, sl, e_tf, e_post, e_ref, reason):
    """A resting entry crosses at c_i (taker) and is closed in the same minute (the window end, reason 5, or a
    forced exit, reason 7 with its penalty): a zero-gross trade before costs. Returns the new cash."""
    px = c_i * (1.0 + slip_i * d)
    e_fee = q * px * fee_t
    e_eq = cash
    cash -= e_fee
    xp = c_i * (1.0 - (slip_i + pen) * d)
    return cash + _close(tr, nt, i, xp, fee_t, d, q, px, i, q * abs(px - sl), e_fee, 0.0, e_eq, reason, e_tf,
                         0, 0, e_post, -1, e_ref, np.nan)


@njit(cache=True)
def _stop_hit(pos, o_i, h_i, l_i, sl, tp, just_filled, stop_mode):
    """The engine's stop/target rule for one minute: (price before slippage, reason) or (nan, -1)."""
    px = np.nan
    reason = -1
    if pos == 1:
        if l_i <= sl:
            if stop_mode == 1:
                px = l_i
            else:
                px = min(o_i, sl) if not just_filled else sl
            reason = 0
        elif (not just_filled) and (not np.isnan(tp)) and h_i >= tp:
            px = max(o_i, tp)
            reason = 1
    else:
        if h_i >= sl:
            if stop_mode == 1:
                px = h_i
            else:
                px = max(o_i, sl) if not just_filled else sl
            reason = 0
        elif (not just_filled) and (not np.isnan(tp)) and l_i <= tp:
            px = min(o_i, tp)
            reason = 1
    return px, reason


@njit(cache=True)
def _ext(pos, ext, h_i, l_i):
    if pos == 1:
        return h_i if h_i > ext else ext
    return l_i if l_i < ext else ext


@njit(cache=True)
def _touch(side, p, h_i, l_i, model):
    """Fill condition of a resting order at price p: trade-through, or a touch for the optimistic model."""
    eps = PX_TOL * p
    if side == 1:
        return l_i <= p + eps if model == 2 else l_i < p - eps
    return h_i >= p - eps if model == 2 else h_i > p + eps


@njit(cache=True)
def _model_px(model, ref, o_m0, p):
    if model == 0:
        return ref
    if model == 1:
        return o_m0
    return p


@njit(cache=True)
def _check_tick(tk):
    if not (tk > 0.0 and tk < np.inf):
        raise ValueError("tick must be positive and finite at every minute where an order is posted")


@njit(cache=True)
def _simulate_mf(o, h, l, c, ts, fund, exec_idx, entry, stop, target, trail, exit_long, exit_short, size,
                 max_bars, start_i, end_i, risk, max_lev, fee_t, fee_m, slip, tick, eq0, marks, stop_mode,
                 force_exit, force_pen, t_ms, model, fallback):
    # Steps follow engine._simulate (without its limit path); t_ms == 0 never posts, so every other step
    # below is skipped and the run reduces to the engine's operation for operation.
    n_tf = exec_idx.shape[0]
    tr = np.zeros((n_tf + 2, 20))
    nt = 0
    eq_marks = np.full(marks.shape[0], np.nan)
    mi = 0
    while mi < marks.shape[0] and marks[mi] < start_i:
        mi += 1
    cnt = np.zeros(10, np.int64)          # INFO_KEYS order
    missed = np.zeros(n_tf + 1, np.int64)
    nm = 0

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
    e_post = 0
    e_ref = 0.0
    # resting exit order X: closes the whole position, keeps its reason
    x_on = False
    x_reason = 0
    x_m0 = 0
    x_exp = 0
    x_ref = 0.0
    x_p = 0.0
    x_px = 0.0
    # resting entry order N: quantity, stop and target fixed at post
    n_on = False
    n_d = 0
    n_q = 0.0
    n_sl = 0.0
    n_tp = np.nan
    n_t = 0
    n_m0 = 0
    n_exp = 0
    n_ref = 0.0
    n_p = 0.0
    n_px = 0.0
    n_ext = 0.0

    t = 0
    while t < n_tf and exec_idx[t] < start_i:
        t += 1

    for i in range(start_i, end_i):
        # 1) funding settles on positions open before this minute
        if pos != 0 and fund[i] != 0.0:
            f = qty * o[i] * fund[i] * pos
            cash -= f
            e_fund += f
        if (x_on or n_on) and fund[i] != 0.0:
            cnt[9] += 1

        # 2) expiries, then 3) decisions; a decision first cuts short an order that still rests
        while True:
            dec = t < n_tf and exec_idx[t] == i
            if x_on and (i >= x_exp or dec):
                if i < x_exp:
                    cnt[8] += 1
                x_on = False
                if fallback:
                    px = o[i] * (1.0 - slip[i] * pos)
                    cash += _close(tr, nt, i, px, fee_t, pos, qty, epx, e_i, e_risk, e_fee, e_fund, e_eq,
                                   x_reason, e_tf, e_maker, 0, e_post, x_m0, e_ref, x_ref)
                    nt += 1
                    pos = 0
                    cnt[4] += 1
                else:
                    cnt[6] += 1
            if n_on and (i >= n_exp or dec):
                if i < n_exp:
                    cnt[8] += 1
                n_on = False
                if fallback and pos == 0 and cash > 0.0:
                    px = o[i] * (1.0 + slip[i] * n_d)
                    pos = n_d; qty = n_q; epx = px; sl = n_sl; tp = n_tp; ext = o[i]; held = 0
                    e_i = i; e_tf = n_t; e_fee = n_q * px * fee_t; e_fund = 0.0; e_eq = cash
                    e_risk = n_q * abs(px - n_sl); e_maker = 0; e_post = n_m0; e_ref = n_ref
                    cash -= e_fee
                    cnt[3] += 1
                else:
                    missed[nm] = n_t
                    nm += 1
                    cnt[5] += 1
            if not dec:
                break

            x_now = False
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
                    if t_ms == 0:
                        px = o[i] * (1.0 - slip[i] * pos)
                        cash += _close(tr, nt, i, px, fee_t, pos, qty, epx, e_i, e_risk, e_fee, e_fund, e_eq,
                                       reason, e_tf, e_maker, 0, e_post, i, e_ref, px)
                        nt += 1
                        pos = 0
                    else:
                        _check_tick(tick[i])
                        x_on = True
                        x_now = True
                        x_reason = reason
                        x_m0 = i
                        x_exp = np.searchsorted(ts, ts[i] + t_ms)
                        x_ref = o[i] * (1.0 - slip[i] * pos)
                        x_p = o[i] + pos * tick[i]
                        x_px = _model_px(model, x_ref, o[i], x_p)
                        cnt[0] += 1
                elif not np.isnan(trail[t]):
                    if pos == 1:
                        cand = ext - trail[t]
                        if cand > sl:
                            sl = cand
                    else:
                        cand = ext + trail[t]
                        if cand < sl:
                            sl = cand
            if (pos == 0 or x_now) and entry[t] != 0:
                d = entry[t]
                if t_ms == 0:
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
                            e_post = i
                            e_ref = px
                            cash -= e_fee
                else:
                    ref = o[i] * (1.0 + slip[i] * d)
                    s = stop[t]
                    if not np.isnan(s) and (s - ref) * d < 0:
                        # sized on the cash a taker exit at this minute would leave (merged reversal)
                        cp = cash
                        if x_now:
                            cp = cash + (qty * (x_ref - epx) * pos - qty * x_ref * fee_t)
                        q = _qty(size[t], risk, max_lev, cp, ref, abs(ref - s))
                        if q > 0:
                            _check_tick(tick[i])
                            n_on = True
                            n_d = d
                            n_q = q
                            n_sl = s
                            n_tp = target[t]
                            n_t = t
                            n_m0 = i
                            n_exp = np.searchsorted(ts, ts[i] + t_ms)
                            n_ref = ref
                            n_p = o[i] - d * tick[i]
                            n_px = _model_px(model, ref, o[i], n_p)
                            n_ext = o[i] if model == 0 else n_px
                            cnt[0] += 1
            t += 1

        # 4) while an exit rests the stop/target stays live and is checked first (the engine's step 4, so a
        #    minute without an exit still extends the trailing extreme)
        checked = False
        if pos != 0 and x_on:
            checked = True
            px, reason = _stop_hit(pos, o[i], h[i], l[i], sl, tp, False, stop_mode)
            if reason >= 0:
                x_on = False
                cnt[7] += 1
                px = px * (1.0 - slip[i] * pos)
                cash += _close(tr, nt, i, px, fee_t, pos, qty, epx, e_i, e_risk, e_fee, e_fund, e_eq,
                               reason, e_tf, e_maker, 0, e_post, -1, e_ref, np.nan)
                nt += 1
                pos = 0
            else:
                ext = _ext(pos, ext, h[i], l[i])

        # 5) the resting exit fills
        if x_on and _touch(-pos, x_p, h[i], l[i], model):
            x_on = False
            cash += _close(tr, nt, i, x_px, fee_m, pos, qty, epx, e_i, e_risk, e_fee, e_fund, e_eq,
                           x_reason, e_tf, e_maker, 1, e_post, x_m0, e_ref, x_ref)
            nt += 1
            pos = 0
            cnt[2] += 1

        # 6) the resting entry fills once flat; at or beyond its own stop it opens nothing (engine limit rule)
        just_filled = False
        if n_on and pos == 0 and cash > 0.0 and _touch(n_d, n_p, h[i], l[i], model):
            n_on = False
            if (n_sl - n_px) * n_d < 0:
                pos = n_d; qty = n_q; epx = n_px; sl = n_sl; tp = n_tp; ext = n_ext; held = 0
                e_i = i; e_tf = n_t; e_fee = n_q * n_px * fee_m; e_fund = 0.0; e_eq = cash
                e_risk = n_q * abs(n_px - n_sl); e_maker = 1; e_post = n_m0; e_ref = n_ref
                cash -= e_fee
                cnt[1] += 1
                just_filled = True
                checked = False
            else:
                missed[nm] = n_t
                nm += 1
                cnt[5] += 1

        # 7) stop / target on this minute's range (stop wins ties; engine step 4)
        if pos != 0 and not checked:
            px, reason = _stop_hit(pos, o[i], h[i], l[i], sl, tp, just_filled, stop_mode)
            if reason >= 0:
                px = px * (1.0 - slip[i] * pos)
                cash += _close(tr, nt, i, px, fee_t, pos, qty, epx, e_i, e_risk, e_fee, e_fund, e_eq,
                               reason, e_tf, e_maker, 0, e_post, -1, e_ref, np.nan)
                nt += 1
                pos = 0
            else:
                ext = _ext(pos, ext, h[i], l[i])

        # 8) forced exit at this minute's close (a stop hit in the same minute wins); an entry still resting
        #    crosses at the close and is closed with it, as at the window end
        if force_exit[i]:
            if pos != 0:
                if x_on:
                    x_on = False
                    cnt[7] += 1
                px = c[i] * (1.0 - (slip[i] + force_pen) * pos)
                cash += _close(tr, nt, i, px, fee_t, pos, qty, epx, e_i, e_risk, e_fee, e_fund, e_eq,
                               7, e_tf, e_maker, 0, e_post, -1, e_ref, np.nan)
                nt += 1
                pos = 0
            if n_on:
                n_on = False
                if fallback and cash > 0.0:
                    cash = _cross_and_close(tr, nt, i, c[i], slip[i], force_pen, fee_t, cash, n_d, n_q, n_sl,
                                            n_t, n_m0, n_ref, 7)
                    nt += 1
                    cnt[3] += 1
                else:
                    missed[nm] = n_t
                    nm += 1
                    cnt[5] += 1

        # 9) daily mark-to-market
        while mi < marks.shape[0] and marks[mi] == i:
            eq_marks[mi] = cash + (qty * (c[i] - epx) * pos if pos != 0 else 0.0)
            mi += 1

        if cash <= 0:
            break

    i = end_i - 1
    closed = False
    if pos != 0:
        if x_on:
            x_on = False
            cnt[6] += 1
        px = c[i] * (1.0 - slip[i] * pos)
        cash += _close(tr, nt, i, px, fee_t, pos, qty, epx, e_i, e_risk, e_fee, e_fund, e_eq,
                       5, e_tf, e_maker, 0, e_post, -1, e_ref, np.nan)
        nt += 1
        pos = 0
        closed = True
    if n_on:
        n_on = False
        if fallback and cash > 0.0:
            cash = _cross_and_close(tr, nt, i, c[i], slip[i], 0.0, fee_t, cash, n_d, n_q, n_sl, n_t, n_m0,
                                    n_ref, 5)
            nt += 1
            cnt[3] += 1
            closed = True
        else:
            missed[nm] = n_t
            nm += 1
            cnt[5] += 1
    if closed and marks.shape[0] > 0 and marks[marks.shape[0] - 1] == i:
        eq_marks[marks.shape[0] - 1] = cash
    return tr[:nt], eq_marks, cnt, missed[:nm]


def _model_code(model) -> int:
    if isinstance(model, str):
        if model not in MODELS:
            raise ValueError(f"unknown fill model {model!r}; expected one of {sorted(MODELS)}")
        return MODELS[model]
    if model not in MODELS.values():
        raise ValueError(f"unknown fill model code {model!r}")
    return int(model)


def _minutes(T) -> int:
    if int(T) != T or T < 0:
        raise ValueError(f"T must be a whole number of minutes >= 0, got {T!r}")
    return int(T)


def run(market: Market, sig: Signals, costs: Costs = Costs(), risk: Risk = Risk(),
        start: pd.Timestamp | None = None, latency_min: int = 0, *, policy: MakerPolicy = MakerPolicy(),
        tick: np.ndarray | None = None, slippage: np.ndarray | None = None, stop_mode: int = 0,
        force_exit: np.ndarray | None = None, force_penalty: float = 0.0) -> Result:
    """engine.run with maker-first execution of decision exits and entries (see the module docstring).

    tick: per-minute tick size, required (positive and finite where an order is posted) when policy.T > 0.
    Trades carry MF_COLS plus engine.run's derived columns; info counts the orders (INFO_KEYS) and lists the
    decision indices whose entry was cancelled (missed_entry_tf). Every posted order ends in exactly one of
    entry/exit maker fill, entry/exit fallback, entries_missed (fallback=False, a maker fill at or beyond its
    stop, or cash <= 0), exits_cancelled (fallback=False, or open at the window end) and exits_preempted_by_stop
    (a stop, target or forced exit closed the position first); forced_fallbacks counts the orders a later
    decision cut short (fell back, or cancelled without fallback)."""
    if sig.limit is not None:
        raise ValueError("limit entries are not supported by the maker-first engine")
    model = _model_code(policy.model)
    T = _minutes(policy.T)
    ex = market.exec_idx(sig.tf, latency_min)
    n_tf = len(ex)
    entry, stop, target, trail, xl, xs, _, size = sig.arrays(n_tf)
    start_i = market.index_at(start) if start is not None else 0
    end_i = len(market.ts)
    n = len(market.ts)
    slip = np.full(n, costs.slippage) if slippage is None else np.ascontiguousarray(slippage, np.float64)
    fx = np.zeros(n, np.bool_) if force_exit is None else np.ascontiguousarray(force_exit, np.bool_)
    if tick is None:
        if T > 0:
            raise ValueError("tick is required when policy.T > 0")
        tk = np.full(n, np.nan)
    else:
        tk = np.ascontiguousarray(tick, np.float64)
    if slip.shape != (n,) or fx.shape != (n,) or tk.shape != (n,):
        raise ValueError("slippage, force_exit and tick must have one value per minute")
    trades, eq, cnt, missed = _simulate_mf(
        market.o, market.h, market.l, market.c, market.ts, market.fund, ex, entry, stop, target, trail, xl,
        xs, size, int(sig.max_bars), start_i, end_i, risk.risk_per_trade, risk.max_leverage, costs.taker,
        costs.maker, slip, tk, risk.initial_equity, market.day_marks, int(stop_mode), fx,
        float(force_penalty), np.int64(T * MINUTE_MS), model, bool(policy.fallback))
    tdf = pd.DataFrame(trades, columns=MF_COLS)
    for col in _INT_COLS:
        tdf[col] = tdf[col].astype(np.int64)
    tdf["entry_time"] = pd.to_datetime(market.ts[tdf["entry_i"]], unit="ms", utc=True)
    tdf["exit_time"] = pd.to_datetime(market.ts[tdf["exit_i"]], unit="ms", utc=True)
    tdf["R"] = tdf["net"] / tdf["init_risk"]
    tdf["ret"] = tdf["net"] / tdf["eq_before"]
    tdf["reason"] = [REASONS[r] for r in tdf["reason"]]
    equity = pd.Series(eq, index=market.days).dropna()
    first = pd.Timestamp(market.ts[start_i], unit="ms", tz="UTC")
    info = {k: int(v) for k, v in zip(INFO_KEYS, cnt)}
    info["missed_entry_tf"] = missed.astype(np.int64)
    return Result(tdf, equity, risk.initial_equity, first, info)


def tick_array(market: Market, ticks_by_month: dict[str, float]) -> np.ndarray:
    """Per-minute tick size from {"YYYY-MM": tick}; NaN in months the table does not cover."""
    month = np.asarray(market.ts, np.int64).astype("datetime64[ms]").astype("datetime64[M]")
    uniq, inv = np.unique(month, return_inverse=True)
    vals = np.array([float(ticks_by_month.get(str(u), np.nan)) for u in uniq], np.float64)
    return vals[inv.reshape(-1)]


# ---------------------------------------------------------------- single legs (no account)

@njit(cache=True)
def _legs(o, h, l, c, ts, m0, side, tick, slip, t_ms, model, fallback, fee_t, fee_m, marks):
    nl = m0.shape[0]
    n = o.shape[0]
    nk = marks.shape[0]
    kmax = 0
    for k in range(nk):
        if marks[k] > kmax:
            kmax = marks[k]
    span = t_ms + kmax * MINUTE_MS            # m0 .. m0 + T + max(markouts) minutes
    span_rows = span // MINUTE_MS
    filled = np.zeros(nl, np.bool_)
    fell = np.zeros(nl, np.bool_)
    complete = np.zeros(nl, np.bool_)
    fill_i = np.full(nl, -1, np.int64)
    exp_i = np.zeros(nl, np.int64)
    wait = np.full(nl, np.nan)
    ref = np.zeros(nl)
    px = np.full(nl, np.nan)
    saving = np.full(nl, np.nan)
    mo = np.full((nl, nk), np.nan)
    for g in range(nl):
        m = m0[g]
        s = side[g]
        r = o[m] * (1.0 + s * slip[m])
        p = o[m] - s * tick[m]
        e = np.searchsorted(ts, ts[m] + t_ms)
        ref[g] = r
        exp_i[g] = e
        # every minute of the window is in the data (sorted 1m rows: no gap iff the row count matches)
        w = m + span_rows
        complete[g] = w < n and ts[w] - ts[m] == span
        j = m
        while j < e:
            if _touch(s, p, h[j], l[j], model):
                break
            j += 1
        if j < e:
            fp = _model_px(model, r, o[m], p)
            filled[g] = True
            fill_i[g] = j
            wait[g] = (ts[j] - ts[m]) / MINUTE_MS
            px[g] = fp
            saving[g] = (s * (r - fp) / r + (fee_t - fee_m)) * 1e4
            for k in range(nk):
                # the close of the minute k minutes after the fill minute; NaN if that minute is missing
                tk = ts[j] + marks[k] * MINUTE_MS
                r_k = np.searchsorted(ts, tk)
                if r_k < n and ts[r_k] == tk:
                    mo[g, k] = s * (c[r_k] - fp) / fp * 1e4
        elif e < n:
            wait[g] = (ts[e] - ts[m]) / MINUTE_MS
            if fallback:
                fp = o[e] * (1.0 + s * slip[e])
                fell[g] = True
                px[g] = fp
                saving[g] = (s * (r - fp) / r + (fee_t - fee_t)) * 1e4
    return filled, fell, complete, fill_i, exp_i, wait, ref, px, saving, mo


def _per_minute(x, n: int, name: str) -> np.ndarray:
    a = np.asarray(x, np.float64)
    a = np.full(n, float(a)) if a.ndim == 0 else np.ascontiguousarray(a)
    if a.shape != (n,):
        raise ValueError(f"{name} must be a scalar or have one value per minute")
    return a


def simulate_legs(o, h, l, c, ts, m0, side, tick, slip, T, model, fallback, markouts=(1, 5, 15, 60),
                  fee_t: float = 0.0005, fee_m: float = 0.0002) -> dict:
    """Maker-first execution of single legs (no account): the same post price, fill condition, fill price
    and fallback as run(). Leg arrays m0 (row of the taker reference minute) and side (+1 buy / -1 sell).

    Per leg: filled / maker (a maker fill), fell_back (crossed as a taker at exp), fill_i (maker fill row or
    -1), exp_i, wait_min (to the fill, else to exp), ref_px, px (fill or fallback price; NaN if cancelled),
    saving_bp = (side*(ref_px - px)/ref_px + fee_t - fee_used) * 1e4, markout_bp_<k> = side*(c[r] - px)/px
    * 1e4 for maker fills with r the minute k minutes after the fill minute (NaN if that minute is missing),
    and complete (every minute from m0 to m0 + T + max(markouts) is in the data, so expiry and markouts are
    measured without gaps). Times are minutes on the ts clock; ts are 1m rows."""
    o, h, l, c = (np.ascontiguousarray(a, np.float64) for a in (o, h, l, c))
    ts = np.ascontiguousarray(ts, np.int64)
    n = len(ts)
    if not (o.shape == h.shape == l.shape == c.shape == (n,)):
        raise ValueError("o, h, l, c and ts must have the same length")
    tick = _per_minute(tick, n, "tick")
    slip = _per_minute(slip, n, "slip")
    m0 = np.ascontiguousarray(m0, np.int64)
    side = np.ascontiguousarray(side, np.int64)
    if m0.ndim != 1 or side.shape != m0.shape:
        raise ValueError("m0 and side must be 1-D arrays of the same length")
    if len(m0) and (m0.min() < 0 or m0.max() >= n):
        raise ValueError("m0 out of range")
    if not np.isin(side, (-1, 1)).all():
        raise ValueError("side must be +1 or -1")
    tk = tick[m0]
    if not (np.isfinite(tk) & (tk > 0)).all():
        raise ValueError("tick must be positive and finite at every leg's m0")
    marks = np.asarray(markouts, np.int64).reshape(-1)
    if (marks < 0).any():
        raise ValueError("markouts must be >= 0 minutes")
    filled, fell, complete, fill_i, exp_i, wait, ref, px, saving, mo = _legs(
        o, h, l, c, ts, m0, side, tick, slip, np.int64(_minutes(T) * MINUTE_MS), _model_code(model),
        bool(fallback), float(fee_t), float(fee_m), marks)
    out = {"filled": filled, "fill_i": fill_i, "exp_i": exp_i, "wait_min": wait, "ref_px": ref, "px": px,
           "maker": filled.copy(), "fell_back": fell, "saving_bp": saving}
    for k, mk in enumerate(marks):
        out[f"markout_bp_{int(mk)}"] = mo[:, k].copy()
    out["complete"] = complete
    return out
