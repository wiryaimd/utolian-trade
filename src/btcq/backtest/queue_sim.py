"""X1 tick queue model: one post-only order per leg on archived bookTicker + aggTrades.

A buy joins the best bid of the last book row at or before t_live: post price P = that bid, taker benchmark = that
ask, queue ahead = that bid's size. It rests until t_exp = t_live + T. Events strictly between the two are merged by
timestamp (trades before book rows at equal timestamps, file order within a stream):
  - seller-initiated trades: below P the remainder fills at P (trade-through); at P the volume first eats the queue
    ahead and the excess fills us (partial fills tracked); above P nothing. Buyer-initiated trades do nothing.
  - book rows: a bid below P empties the queue ahead (our order is the best bid, unseen in the archive); a bid at P
    caps the queue ahead at the shown size, and "prorata" also moves us up by our share of the size drop the trades
    since the last observation do not explain ("conservative": every cancellation is behind us); a bid above P
    leaves the level unseen.
The remainder at t_exp crosses at the ask of the last book row at or before t_exp. Sells mirror everything (ask for
bid, buyer-initiated for seller-initiated, inequalities flipped). Prices are compared with a tolerance 1e-9 * P, and
an at-P print within 1e-9 * max(q_ahead0, print) of the queue ahead empties it exactly (float dust is not a fill).
Book rows whose opposite best reaches P (an ask at or below a buy's P) are not fills; such legs are flagged
book_crossed so they can be reported.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit

QUEUE = {"conservative": 0, "prorata": 1}
EPS_REL = 1e-9
QTY_REL = 1e-9     # queue-ahead dust, relative to max(q_ahead0, print): far below any lot step, far above float error
DUST = 1e-12       # a remainder below this share of the order counts as filled (float sums of partial fills)
LEG_COLS = ["status", "post_px", "taker_px", "q_ahead0", "filled_qty", "maker_share", "fill_ts", "first_fill_ts",
            "avg_maker_px", "fallback_px", "avg_px", "saving_bp", "trade_through", "level_vanished", "book_crossed"]


def _int_ms(x, name: str) -> np.ndarray:
    a = np.asarray(x)
    if a.dtype.kind in "iu":
        return np.ascontiguousarray(a, np.int64)
    if a.dtype.kind == "f" and np.all(np.isfinite(a)) and np.all(a == np.round(a)):
        return np.ascontiguousarray(a, np.int64)
    raise ValueError(f"{name} must be integer milliseconds")


def _check_sorted(ts: np.ndarray, name: str):
    if len(ts) > 1 and np.any(ts[1:] < ts[:-1]):
        raise ValueError(f"{name} timestamps are not sorted")


def _check_positive(a: np.ndarray, name: str):
    if not np.all(np.isfinite(a)) or np.any(a <= 0.0):
        raise ValueError(f"{name} must be finite and > 0")


@dataclass
class Book:
    """Best bid/ask rows (bookTicker), sorted by ts (int64 ms); equal timestamps keep file order."""
    ts: np.ndarray
    bid_px: np.ndarray
    bid_qty: np.ndarray
    ask_px: np.ndarray
    ask_qty: np.ndarray

    def __post_init__(self):
        self.ts = _int_ms(self.ts, "book ts")
        for f in ("bid_px", "bid_qty", "ask_px", "ask_qty"):
            setattr(self, f, np.ascontiguousarray(getattr(self, f), np.float64))

    def validate(self):
        if any(getattr(self, f).shape != self.ts.shape for f in ("bid_px", "bid_qty", "ask_px", "ask_qty")) \
                or self.ts.ndim != 1:
            raise ValueError("book columns must be 1-d and of equal length")
        _check_sorted(self.ts, "book")
        for f in ("bid_px", "bid_qty", "ask_px", "ask_qty"):
            _check_positive(getattr(self, f), f"book {f}")


@dataclass
class Tape:
    """Trades (aggTrades), sorted by ts (int64 ms); buyer_maker True = seller-initiated."""
    ts: np.ndarray
    px: np.ndarray
    qty: np.ndarray
    buyer_maker: np.ndarray

    def __post_init__(self):
        self.ts = _int_ms(self.ts, "tape ts")
        self.px = np.ascontiguousarray(self.px, np.float64)
        self.qty = np.ascontiguousarray(self.qty, np.float64)
        bm = np.asarray(self.buyer_maker)
        if bm.dtype != np.bool_:
            if bm.dtype.kind not in "iu" or not np.all((bm == 0) | (bm == 1)):
                raise ValueError("tape buyer_maker must be boolean")
        self.buyer_maker = np.ascontiguousarray(bm, np.bool_)

    def validate(self):
        if any(getattr(self, f).shape != self.ts.shape for f in ("px", "qty", "buyer_maker")) or self.ts.ndim != 1:
            raise ValueError("tape columns must be 1-d and of equal length")
        _check_sorted(self.ts, "tape")
        _check_positive(self.px, "tape px")
        _check_positive(self.qty, "tape qty")


@njit(cache=True)
def _legs(bts, bid, bidq, ask, askq, tts, tpx, tqty, tbm, side, qty, b0, t_lo, t_hi, b_hi, prorata):
    n = side.shape[0]
    post = np.full(n, np.nan)
    taker = np.full(n, np.nan)
    q0 = np.full(n, np.nan)
    got = np.full(n, np.nan)
    fill_ts = np.zeros(n, np.int64)
    first_ts = np.zeros(n, np.int64)
    done = np.zeros(n, np.bool_)
    started = np.zeros(n, np.bool_)
    through = np.zeros(n, np.bool_)
    vanished = np.zeros(n, np.bool_)
    crossed = np.zeros(n, np.bool_)
    for k in range(n):
        b = b0[k]
        if b < 0:
            continue
        buy = side[k] > 0
        s = 1.0 if buy else -1.0
        if buy:
            P, tk, qa = bid[b], ask[b], bidq[b]
        else:
            P, tk, qa = ask[b], bid[b], askq[b]
        post[k] = P
        taker[k] = tk
        q0[k] = qa
        eps = EPS_REL * P
        obs = qa
        traded = 0.0
        observed = True
        q = qty[k]
        rem = q
        i = t_lo[k]
        j = b + 1
        while rem > 0.0 and (i < t_hi[k] or j < b_hi[k]):
            if i < t_hi[k] and (j >= b_hi[k] or tts[i] <= bts[j]):
                # s * (P - px) > 0: the trade printed beyond P on the far side (below P for a buy)
                if tbm[i] == buy:
                    x = s * (P - tpx[i])
                    before = rem
                    if x > eps:
                        rem = 0.0
                        through[k] = True
                    elif x >= -eps:
                        tq = tqty[i]
                        d = tq - qa
                        tol = QTY_REL * max(q0[k], tq)
                        if qa <= 0.0 or d > tol:
                            rem -= min(d, rem)
                            qa = 0.0
                        elif d >= -tol:         # the print matches the queue ahead up to float dust
                            qa = 0.0
                        else:
                            qa -= tq
                        traded += tq
                    if rem < before:
                        if not started[k]:
                            started[k] = True
                            first_ts[k] = tts[i]
                        if rem <= DUST * q:
                            rem = 0.0
                            done[k] = True
                            fill_ts[k] = tts[i]
                i += 1
            else:
                lv = bid[j] if buy else ask[j]
                lq = bidq[j] if buy else askq[j]
                opp = ask[j] if buy else bid[j]
                if s * (P - opp) >= -eps:
                    crossed[k] = True
                x = s * (P - lv)
                if x > eps:
                    qa = 0.0
                    observed = False
                    vanished[k] = True
                elif x >= -eps:
                    if prorata and observed:
                        e = obs - traded
                        if lq < e and e > 0.0:
                            qa -= (e - lq) * qa / e
                    qa = min(qa, lq)
                    obs = lq
                    traded = 0.0
                    observed = True
                else:
                    observed = False
                j += 1
        got[k] = q - rem
    return post, taker, q0, got, fill_ts, first_ts, done, started, through, vanished, crossed


def _coverage(windows_ms) -> tuple[np.ndarray, np.ndarray]:
    """Sorted, merged [start, end) intervals (overlapping or touching windows join)."""
    w = np.asarray(windows_ms)
    if w.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    w = _int_ms(w, "windows_ms")
    if w.ndim != 2 or w.shape[1] != 2:
        raise ValueError("windows_ms must be (start_ms, end_ms) pairs")
    if np.any(w[:, 1] <= w[:, 0]):
        raise ValueError("every window must end after it starts")
    w = w[np.argsort(w[:, 0], kind="stable")]
    lo, hi = [int(w[0, 0])], [int(w[0, 1])]
    for a, e in w[1:].tolist():
        if a <= hi[-1]:
            hi[-1] = max(hi[-1], e)
        else:
            lo.append(a)
            hi.append(e)
    return np.array(lo, np.int64), np.array(hi, np.int64)


def simulate_legs(book: Book, tape: Tape, t_live_ms, side, qty, T_ms, queue: str = "conservative",
                  fee_t: float = 0.0005, fee_m: float = 0.0002,
                  markouts_ms=(60_000, 300_000, 900_000, 3_600_000), windows_ms=None) -> pd.DataFrame:
    """One row per leg, in input order. side +1 buy / -1 sell; qty in base units; t_live_ms, side, qty and T_ms
    broadcast against each other. Timestamps in the output are float ms (NaN = none).

    windows_ms: the [start, end) intervals in which both streams are complete, e.g. the extraction windows of the
    files concatenated into book and tape (touching windows join). Each leg is judged inside the window holding
    t_live: without a book row in that window at or before t_live it is no_book, and its markouts and `complete`
    end at that window's end. None: the streams are one gap-free stretch up to their last rows. A Book stitched
    from separate windows or days must pass windows_ms, or legs in and across the holes read stale rows.

    saving_bp = (side * (taker_px - avg_px) / taker_px + maker_share * (fee_t - fee_m)) * 1e4.
    markout_bp_<k> = side * (mid(t + k) - P) / P * 1e4 with t = fill_ts (first_fill_ts if only partly
    maker-filled), mid from the last book row at or before t + k; NaN with no maker fill or past the data.
    wait_ms: t_live to fill_ts, or T_ms when not fully maker-filled. book_age_ms: t_live minus the joined
    snapshot's ts. complete: [t_live, t_exp + max(markouts_ms)] lies inside the data, as in
    maker_engine.simulate_legs (inside t_live's window; without windows the book reaches t_exp + max(markouts_ms)
    and the tape t_exp). An incomplete leg may rest on rows past the data or across a hole: exclude it.
    book_crossed: a book row while the order rested, or the fallback row, showed the opposite best at or through
    P (an ask <= P for a buy). With our order in the book that ask would have traded with it; the model fills only
    on prints, so such a leg may miss a maker fill or fall back at or through its own post price."""
    if queue not in QUEUE:
        raise ValueError(f"unknown queue model {queue!r}; expected one of {sorted(QUEUE)}")
    if not (np.isfinite(fee_t) and np.isfinite(fee_m)):
        raise ValueError("fees must be finite")
    book.validate()
    tape.validate()
    t_live = _int_ms(np.atleast_1d(t_live_ms), "t_live_ms")
    T = _int_ms(np.atleast_1d(T_ms), "T_ms")
    side_a = np.asarray(np.atleast_1d(side), np.float64)
    qty_a = np.asarray(np.atleast_1d(qty), np.float64)
    t_live, T, side_a, qty_a = (np.array(a) for a in np.broadcast_arrays(t_live, T, side_a, qty_a))
    if t_live.ndim != 1:
        raise ValueError("legs must be 1-d")
    if not np.all((side_a == 1.0) | (side_a == -1.0)):
        raise ValueError("side must be +1 or -1")
    _check_positive(qty_a, "leg qty")
    if np.any(T < 0):
        raise ValueError("T_ms must be >= 0")
    mks = _int_ms(np.atleast_1d(markouts_ms), "markouts_ms")
    if mks.ndim != 1 or np.any(mks <= 0) or len(set(mks.tolist())) != len(mks):
        raise ValueError("markouts_ms must be distinct positive integers")
    side_i = side_a.astype(np.int64)
    t_exp = t_live + T
    n = len(t_live)
    none = np.iinfo(np.int64).min
    horizon = t_exp + (mks.max() if len(mks) else 0)       # complete needs data up to here

    b0 = np.searchsorted(book.ts, t_live, "right") - 1
    if windows_ms is None:
        end = (book.ts[-1] + 1) if len(book.ts) else none      # exclusive: markouts must read before it
        covered = (horizon < end) & (t_exp <= (tape.ts[-1] if len(tape.ts) else none))
    else:
        lo, hi = _coverage(windows_ms)
        w = np.searchsorted(lo, t_live, "right")                # 0 = before every window
        lo, hi = np.r_[none, lo], np.r_[none, hi]
        inside = t_live < hi[w]
        joined = (b0 >= 0) & ((book.ts[np.maximum(b0, 0)] if len(book.ts) else none) >= lo[w])
        b0 = np.where(inside & joined, b0, -1)
        end = np.where(inside, hi[w], none)
        covered = inside & (horizon < end)
    post, taker, q0, got, fill_ts, first_ts, done, part, through, vanished, crossed = _legs(
        book.ts, book.bid_px, book.bid_qty, book.ask_px, book.ask_qty,
        tape.ts, tape.px, tape.qty, tape.buyer_maker, side_i, qty_a, b0.astype(np.int64),
        np.searchsorted(tape.ts, t_live, "right").astype(np.int64),
        np.searchsorted(tape.ts, t_exp, "left").astype(np.int64),
        np.searchsorted(book.ts, t_exp, "left").astype(np.int64), QUEUE[queue] == 1)
    ok = b0 >= 0
    fb = np.full(n, np.nan)
    age = np.full(n, np.nan)
    if ok.any():
        bx = np.searchsorted(book.ts, t_exp[ok], "right") - 1
        fb[ok] = np.where(side_i[ok] > 0, book.ask_px[bx], book.bid_px[bx])
        age[ok] = t_live[ok] - book.ts[b0[ok]]
    fb = np.where(ok & ~done, fb, np.nan)
    share = got / qty_a
    avg = np.where(done, post, np.where(part, share * post + (1.0 - share) * fb, fb))
    saving = (side_a * (taker - avg) / taker + share * (fee_t - fee_m)) * 1e4
    crossed |= side_a * (post - fb) >= -EPS_REL * post      # the fallback row (fb is NaN without one)

    out = {"status": np.where(ok, "ok", "no_book"), "post_px": post, "taker_px": taker, "q_ahead0": q0,
           "filled_qty": got, "maker_share": share,
           "fill_ts": np.where(done, fill_ts, np.nan), "first_fill_ts": np.where(part, first_ts, np.nan),
           "avg_maker_px": np.where(part, post, np.nan), "fallback_px": fb, "avg_px": avg, "saving_bp": saving,
           "trade_through": through, "level_vanished": vanished, "book_crossed": crossed}
    ref = np.where(done, fill_ts, first_ts)
    for k in mks:
        m = np.full(n, np.nan)
        has = part & (ref + k < end)
        if has.any():
            idx = np.searchsorted(book.ts, ref[has] + k, "right") - 1
            mid = (book.bid_px[idx] + book.ask_px[idx]) / 2.0
            m[has] = side_a[has] * (mid - post[has]) / post[has] * 1e4
        out[f"markout_bp_{int(k)}"] = m
    out["wait_ms"] = np.where(ok, np.where(done, fill_ts - t_live, T), np.nan)
    out["book_age_ms"] = age
    out["complete"] = ok & covered
    return pd.DataFrame(out)
