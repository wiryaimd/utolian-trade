"""Round 6: M1, a long/short daily Donchian-ensemble trend sleeve on 11 macro markets (Binance TradFi perps),
tested on long external daily histories (Yahoo ETFs/futures, EIA spot prices).

    python -m btcq.research.run r6-power
    python -m btcq.research.run r6-download [--resume]
    python -m btcq.research.run r6-sweep [--rerun]
    python -m btcq.research.run r6-validate --unlock-holdout     # only if M1 passed dev

See research/PREREGISTRATION.md, Round 6. Round 6 keeps its own trial registry and N_eff history
(research/round6/), its trial returns in reports/results/round6/, and its own holdout lock (2016-01-01,
btcq.data.external). It never calls run.cmd_gates and never unlocks the crypto 2024+ data.

Engine mapping: one engine row per trade date, stamped 23:59 UTC, open = high = low = close; a decision at
a trade date's close fills at the next trading day's close (latency 0); base funding is exactly 0.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from btcq.backtest import engine, metrics, stats
from btcq.config import ROOT, Costs, Risk
from btcq.data import external as E
from btcq.data import store
from btcq.data import universe as U
from btcq.research import registry
from btcq.research.portfolio import assemble_slots
from btcq.research.round4 import SlimMarket
from btcq.research.round5 import Window, _clean_src, _coin_contrib, _dsr, _git, _plateau, f1_tests, headline_r5
from btcq.research.run import RESULTS, _save_json
from btcq.strategies import r6_macro as M

CODE = "M1"
R6_DIR = E.R6_DIR
R6_RESULTS = RESULTS / "round6"          # a subfolder: run.all_trial_returns (non-recursive glob) never sees it
TRIALS = R6_DIR / "trials.csv"
NEFF_LOG = R6_DIR / "neff_history.jsonl"
RUN_LOG = R6_DIR / "runs.jsonl"
DATA_REPORT = R6_DIR / "data_report.csv"
POWER_CSV = R6_DIR / "power_table.csv"
PREREG = ROOT / "research" / "PREREGISTRATION.md"
NEFF_FLOOR = 14.849570528171945 + 3      # global max N_eff ever recorded + one per post-round-4 family
N_DRAWS = 1000
NULL_SEED = 66
BOOT_BLOCK = 30
BOOT_SEED = 7
BOOT_N = 10_000                          # paths of the holdout MDD bootstrap (fixed here, not taken from run.py)
MIN_MARKETS = 7
SLOTS = len(E.MARKETS) * len(M.LOOKBACKS)
DAY = pd.Timedelta(days=1)
DAY_MS = E.DAY_MS
LAST_DATE_MIN = "2026-09-29"
KNOWN_CLOSURES = {"1994-04-27", "2001-09-11", "2001-09-12", "2001-09-13", "2001-09-14", "2004-06-11", "2007-01-02",
                  "2012-10-29", "2012-10-30", "2018-12-05", "2025-01-09"}     # NYSE closures outside the holiday rules
CL_NEG = ("CLUSDT", "2020-04-21")        # the row that carries the dropped 2020-04-20 interval
LIQ_LEV = 20.0                           # liquidation: equity <= 0 or open notional >= 20x equity at a daily mark
MIN_YEAR_REF_DAYS = 120                  # the yearly missing share is judged only on years with this many ref days
# Code whose behaviour the dev verdict depends on: validate refuses if any of it changed since the sweep.
CODE_FILES = ["src/btcq/data/external.py", "src/btcq/strategies/r6_macro.py", "src/btcq/strategies/r5_d1.py",
              "src/btcq/strategies/base.py", "src/btcq/research/round6.py", "src/btcq/research/round5.py",
              "src/btcq/research/round4.py", "src/btcq/research/portfolio.py", "src/btcq/research/registry.py",
              "src/btcq/research/run.py", "src/btcq/backtest/engine.py", "src/btcq/backtest/stats.py",
              "src/btcq/backtest/metrics.py", "src/btcq/config.py", "src/btcq/data/bardata.py",
              "src/btcq/data/store.py", "src/btcq/features/ta.py"]
PEER_ETFS = ["XAUUSDT", "XAGUSDT", "XPTUSDT", "XPDUSDT", "QQQUSDT", "EWYUSDT"]   # US-listed ETFs: SPY's calendar

DEV = Window("dev", "dev", pd.Timestamp("1986-01-01", tz="UTC"), E.R6_HOLDOUT_START)
HOLD = Window("holdout", "full", E.R6_HOLDOUT_START, E.R6_END)
RISK = Risk(risk_per_trade=0.01, max_leverage=M.LEV_CAP, initial_equity=Risk().initial_equity)


class HardError(RuntimeError):
    """A pre-registered hard error: the run stops, and a fix needs a strategy version bump."""


# ---------------------------------------------------------------- data

@dataclass
class Mkt:
    sym: str
    mk: SlimMarket
    bar_ts: np.ndarray           # trade dates (ms, 00:00 UTC) = the engine's 1d bar labels
    closes: pd.Series            # index = trade-date ms


def market_from_closes(sym: str, closes: pd.Series) -> Mkt:
    mk = engine.Market(E.rows_from_closes(closes), None, None, synthetic_funding=0.0)
    if mk.fund.any():
        raise AssertionError(f"{sym}: base funding must be exactly 0")
    if len(mk.day_marks) != len(mk.ts):
        raise AssertionError(f"{sym}: more than one engine row on a UTC day")
    bt = mk.bars("1d")["ts"].to_numpy(np.int64)
    if not np.array_equal(bt, closes.index.to_numpy(np.int64)):
        raise AssertionError(f"{sym}: engine bars are not the trade dates")
    ex = mk.exec_idx("1d", 0)
    slim = SlimMarket(mk.ts.copy(), mk.o.copy(), mk.h.copy(), mk.l.copy(), mk.c.copy(), mk.fund.copy(),
                      mk.day_marks.copy(), mk.days, ex.astype(np.int64), "1d", 0)
    return Mkt(sym, slim, bt, closes)


def closes_for(sym: str, period: str, adjusted: bool = False) -> pd.Series:
    """Split-adjusted price closes; adjusted=True (information only) uses Yahoo's dividend-adjusted closes
    for the Yahoo series (EIA spot and futures have none)."""
    adj = adjusted and E.SOURCES[sym].source == "yahoo"
    c = E.load_closes(sym, period, adjusted=adj)
    if c.isna().any() or (c <= 0).any():
        raise ValueError(f"{sym}: missing or non-positive closes")
    return c


def load_markets(w: Window, adjusted: bool = False) -> dict[str, Mkt]:
    return {s: market_from_closes(s, closes_for(s, w.period, adjusted)) for s in E.MARKETS}


def costs(sym: str, mult: float = 1.0) -> Costs:
    return Costs(taker=M.TAKER, maker=Costs().maker, slippage=M.SLIP[sym], synthetic_funding=0.0).scaled(mult)


def _cal_ms(w: Window) -> np.ndarray:
    return np.asarray(w.calendar.as_unit("ms").asi8, np.int64)


# ---------------------------------------------------------------- sub-sleeves

@dataclass
class Sub:
    sym: str
    L: int
    sig: engine.Signals
    trail: np.ndarray
    state: np.ndarray
    size: np.ndarray
    ready: int                   # first bar that can hold a position (L+1 closes and sigma ready)


def build(mkts: dict[str, Mkt]) -> dict[tuple, Sub]:
    out = {}
    for s, m in mkts.items():
        for L in M.LOOKBACKS:
            sub = M.sub_signals(m.closes, L)
            sig, trail, state = M.to_engine(sub, m.bar_ts)
            fin = np.flatnonzero(np.isfinite(sub.size))
            ready = max(int(L), int(fin[0])) if len(fin) else len(sub.size)
            out[(s, L)] = Sub(s, int(L), sig, trail, state, sub.size, ready)
    return out


def _stop_row(res: engine.Result, mk: SlimMarket, start: pd.Timestamp) -> int | None:
    """Row of the last daily mark before the engine stopped a sleeve (cash <= 0 breaks its loop); None if it
    marked every day."""
    start_i = mk.index_at(start)
    rows = mk.days.get_indexer(res.equity.index)
    expected = np.arange(start_i, len(mk.ts))
    gaps = np.setdiff1d(expected, rows)
    if not len(gaps):
        return None
    return int(gaps[0]) - 1


def liquidate(res: engine.Result, mk: SlimMarket, lev: float = LIQ_LEV) -> tuple[pd.Series, int | None]:
    """Exchange-style liquidation of one sub-sleeve (isolated margin). Positions are sized at entry only, so
    after a large adverse move the leverage grows without bound and daily returns explode near zero equity.
    At the first daily mark where equity <= 0 or open notional >= lev x equity, the sub-sleeve is closed at
    max(equity, 0) and stays flat to the end. Returns (equity per day, position of that mark or None)."""
    eq = res.equity
    if not len(eq):
        return eq, None
    rows = mk.days.get_indexer(eq.index)
    q = np.zeros(len(mk.ts))
    t = res.trades
    for e, x, qty in zip(t["entry_i"].to_numpy(), t["exit_i"].to_numpy(), t["qty"].to_numpy()):
        q[e:x] += qty                          # held at the marks of rows entry .. exit-1
    e = eq.to_numpy()
    notional = q[rows] * mk.c[rows]
    bad = (e <= 0) | (notional >= lev * e)
    if not bad.any():
        return eq, None
    j = int(np.argmax(bad))
    e2 = e.copy()
    e2[j:] = max(float(e[j]), 0.0)
    return pd.Series(e2, index=eq.index), j


def _liq_row(res: engine.Result, mk: SlimMarket, start: pd.Timestamp) -> int:
    """Row of the liquidation mark, or of the last mark before an engine stop; len(ts) if neither."""
    eq, j = liquidate(res, mk)
    rows = [len(mk.ts)]
    if j is not None:
        rows.append(int(mk.days.get_indexer(eq.index[j:j + 1])[0]))
    stop = _stop_row(res, mk, start)
    if stop is not None:
        rows.append(stop)
    return min(rows)


def sub_daily(eq: pd.Series, calendar: pd.DatetimeIndex, initial: float) -> pd.Series:
    """Daily returns on the calendar (round4.sleeve_daily), and 0 once a liquidated sub-sleeve has no equity."""
    e = eq.copy()
    e.index = e.index.as_unit("ms")
    e = e.reindex(calendar).ffill().fillna(initial)
    prev = e.shift(1)
    prev.iloc[0] = initial
    return (e / prev - 1.0).where(prev > 0, 0.0)


def _close_at(tr: pd.DataFrame, R: int, mk: SlimMarket, booked: float, taker: float) -> pd.DataFrame:
    """Book a liquidation in the trade list: trades entered after mark R are dropped (the engine's later
    trades never reach the account) and the last trade still open at R, or exiting at R, is closed there at the
    booked equity (so its loss is capped at its capital, like the sub-sleeve's)."""
    tr = tr[tr["entry_i"] <= R].reset_index(drop=True).copy()
    open_ = (tr["exit_i"] >= R).to_numpy()
    if open_.any():
        i = int(np.flatnonzero(open_)[-1])
        net = booked - float(tr.at[i, "eq_before"])
        if int(tr.at[i, "exit_i"]) != R:           # still open at R: closed at the mark, only the entry fee was paid
            tr.loc[i, ["exit_i", "exit_px"]] = [R, float(mk.c[R])]
            tr.loc[i, "fees"] = float(tr.at[i, "qty"]) * float(tr.at[i, "entry_px"]) * taker
        tr.loc[i, "net"] = net
        tr.loc[i, "gross"] = net + float(tr.at[i, "fees"]) + float(tr.at[i, "funding"])
        tr.loc[i, "exit_time"] = pd.Timestamp(int(mk.ts[R]), unit="ms", tz="UTC")
        tr.loc[i, "reason"] = "liquidation"
        tr.loc[i, "ret"] = net / float(tr.at[i, "eq_before"])
        tr.loc[i, "R"] = net / float(tr.at[i, "init_risk"]) if tr.at[i, "init_risk"] else np.nan
    return tr


def run_all(subs: dict, mkts: dict[str, Mkt], w: Window, mult: float = 1.0, sig_override: dict | None = None,
            with_r: bool = True):
    """Run every sub-sleeve; returns ({(sym, L): (daily on w.calendar, trades)}, info). info["eq"] holds each
    sub-sleeve's daily equity marks (after liquidation), info["liq_row"] the liquidation (or engine stop) row
    and info["ruins"] the liquidated sub-sleeves. A liquidated sub-sleeve's trade list ends at that mark."""
    parts = {}
    info = {"floors": 0, "ruins": [], "liquidations": 0, "eq": {}, "liq_row": {}}
    for key, sb in subs.items():
        m = mkts[sb.sym]
        sig = sig_override[key] if sig_override is not None else sb.sig
        res = engine.run(m.mk, sig, costs(sb.sym, mult), RISK, start=w.start, latency_min=0)
        eq, _ = liquidate(res, m.mk)
        R = _liq_row(res, m.mk, w.start)
        tr = res.trades
        if R < len(m.mk.ts):
            booked = float(eq.iloc[int(np.searchsorted(m.mk.days.get_indexer(eq.index), R))])
            eq = eq.copy()
            eq[m.mk.days.get_indexer(eq.index) > R] = booked                 # flat after the mark
            info["liq_row"][key] = R
            info["ruins"].append(f"{key[0]}/{key[1]}")
            info["liquidations"] += 1
            tr = _close_at(tr, R, m.mk, booked, costs(sb.sym, mult).taker)
        info["eq"][key] = eq
        if with_r:
            tr, nf = M.apply_r(tr, sb.trail)
            info["floors"] += nf
        parts[key] = (sub_daily(eq, w.calendar, RISK.initial_equity), tr.assign(symbol=sb.sym, lookback=sb.L))
    return parts, info


def assemble(parts: dict, w: Window, cfg: str = M.PRIMARY):
    """The account: slots = 11 x len(lookbacks); each sub-sleeve is 1/slots; sum / slots on the UTC calendar."""
    Ls = M.CONFIGS[cfg]
    sel = [v for (s, L), v in parts.items() if L in Ls]
    return assemble_slots(sel, len(E.MARKETS) * len(Ls), w.calendar)


def mismatches(subs: dict, parts: dict, mkts: dict[str, Mkt], w: Window, liq_rows: dict | None = None) -> dict:
    """Bars where the engine's signed position differs from the state machine, over the decisions executed
    inside the window (the last one excluded), plus stop exits (the sentinels must never fire). A liquidated
    sub-sleeve is checked only on decisions executed before its liquidation mark: from it on the sleeve is
    closed, and the engine may have stopped trading it."""
    bad = stops = 0
    for key, (_, tr) in parts.items():
        sb, mk = subs[key], mkts[key[0]].mk
        ex = mk._ex
        live = (ex >= mk.index_at(w.start)) & (ex < len(mk.ts))
        if liq_rows and key in liq_rows:
            live &= ex < liq_rows[key]
        pos = np.zeros(len(ex), np.int8)
        for b0, xi, d, why in zip(tr["entry_tf"], tr["exit_i"], tr["dir"], tr["reason"]):
            b1 = int(np.searchsorted(ex, int(xi))) if why != "end" else len(ex)
            pos[int(b0):b1] = d
            stops += int(why == "stop")
        idx = np.flatnonzero(live)
        if not (liq_rows and key in liq_rows):
            idx = idx[:-1]                          # the window's last decision meets the end-of-window close
        bad += int((pos[idx] != sb.state[idx]).sum())
    return {"state_mismatches": bad, "stop_exits": stops}


def hard_errors(mm: dict) -> list[str]:
    """Integrity errors (code or engine failures): state mismatches and stop exits, checked up to a
    liquidation. Liquidations are outcomes: booked and reported, never an error."""
    errs = []
    if mm["state_mismatches"]:
        errs.append(f"{mm['state_mismatches']} engine/state mismatches")
    if mm["stop_exits"]:
        errs.append(f"{mm['stop_exits']} stop exits")
    return errs


def exposure_path(m: Mkt, tr: pd.DataFrame, eq: pd.Series, cal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(lev, pos, on): a sub-sleeve's signed notional / equity at each daily mark (0 when flat, after a
    liquidation, or before the window), forward-filled over the calendar's non-trading days; pos/on map the
    market's bars to calendar slots."""
    mk = m.mk
    q = np.zeros(len(mk.ts))
    for e, x, dr, qty in zip(tr["entry_i"].to_numpy(), tr["exit_i"].to_numpy(), tr["dir"].to_numpy(),
                             tr["qty"].to_numpy()):
        q[e:x] += dr * qty                     # held at the marks of rows entry .. exit-1
    lev_row = np.zeros(len(mk.ts))
    if len(eq):
        rows = mk.days.get_indexer(eq.index)
        ev = eq.to_numpy()
        lev_row[rows] = np.where(ev > 0, q[rows] * mk.c[rows] / np.where(ev > 0, ev, 1.0), 0.0)
    pos = np.searchsorted(cal, m.bar_ts)
    on = (pos < len(cal)) & (cal[np.minimum(pos, len(cal) - 1)] == m.bar_ts)
    lev = np.full(len(cal), np.nan)
    lev[pos[on]] = lev_row[on]
    return pd.Series(lev).ffill().fillna(0.0).to_numpy(), pos, on


def funding_drag(subs: dict, mkts: dict[str, Mkt], parts: dict, eqs: dict, w: Window,
                 rate_ann: float = M.FUNDING_STRESS_ANN) -> dict[tuple, np.ndarray]:
    """The funding stress per sub-sleeve on the calendar: every open position pays rate_ann of its notional per
    calendar day, long or short. It is charged on each trade date for the calendar days since the market's
    previous trade date (Mondays carry 3), as |notional / equity| at the previous mark x rate x days / 365, and
    subtracted from the sub-sleeve's daily return (not compounded into position sizes)."""
    cal = _cal_ms(w)
    out = {}
    for key, sb in subs.items():
        m = mkts[sb.sym]
        _, tr = parts[key]
        lev, pos, on = exposure_path(m, tr, eqs[key], cal)
        days = np.r_[1, np.diff(m.bar_ts) // DAY_MS]
        f = np.zeros(len(cal))
        slots = pos[on]
        prev = np.r_[0.0, np.abs(lev[:-1])]
        f[slots] = prev[slots] * rate_ann * days[on] / 365.0
        out[key] = f
    return out


# ---------------------------------------------------------------- shift null

def _active_starts(subs: dict, mkts: dict[str, Mkt], cal: np.ndarray) -> dict:
    out = {}
    for key, sb in subs.items():
        bt = mkts[sb.sym].bar_ts
        out[key] = int(np.searchsorted(cal, bt[sb.ready])) if sb.ready < len(bt) else len(cal)
    return out


@dataclass
class NullInput:
    lev: np.ndarray      # signed notional / equity at each daily mark of the base run, forward-filled on the calendar
    r: np.ndarray        # the market's close-to-close return on its own trade dates (0 on other calendar days)
    cost: np.ndarray     # the rest of the base run's daily return (fees + slippage on fill days): d = lev[t-1] r[t] - cost
    a: int               # first calendar slot of the sub-sleeve's active span (its first ready day)


def null_inputs(subs: dict, mkts: dict[str, Mkt], parts: dict, eqs: dict, w: Window,
                liq_rows: dict | None = None, tol: float = 1e-9) -> dict[tuple, NullInput]:
    """Decompose each base-run sub-sleeve's calendar daily return exactly into exposure x market return minus
    costs. Weekends and holidays carry the last mark's exposure and have no return. A liquidated sub-sleeve
    has no exposure after its liquidation mark. Checked at run time: on every calendar day without a fill (or
    liquidation) the exposure x return reproduces the engine's daily return, and fill days carry a cost >= 0."""
    cal = _cal_ms(w)
    a = _active_starts(subs, mkts, cal)
    liq_rows = liq_rows or {}
    out = {}
    for key, sb in subs.items():
        m = mkts[sb.sym]
        mk = m.mk
        d, tr = parts[key]
        lev, pos, on = exposure_path(m, tr, eqs[key], cal)
        r_row = np.r_[0.0, mk.c[1:] / mk.c[:-1] - 1.0]
        r = np.zeros(len(cal))
        r[pos[on]] = r_row[on]
        cost = np.r_[0.0, lev[:-1]] * r - d.to_numpy()
        event_rows = np.r_[tr["entry_i"].to_numpy(), tr["exit_i"].to_numpy(),
                           [liq_rows[key]] if key in liq_rows else []].astype(np.int64)
        event_rows = event_rows[event_rows < len(mk.ts)]
        ev_slot = np.isin(cal, mk.ts[event_rows] - E.ROW_OFFSET_MS)
        liq_slot = np.isin(cal, mk.ts[[liq_rows[key]]] - E.ROW_OFFSET_MS) if key in liq_rows else np.zeros(len(cal), bool)
        if np.abs(cost[~ev_slot]).max(initial=0.0) > tol or (cost[ev_slot & ~liq_slot] < -tol).any():
            raise AssertionError(f"{key}: exposure x return does not reproduce the engine's daily returns")
        out[key] = NullInput(lev, r, cost, a[key])
    return out


def shifted_exposure(x: NullInput, k: int) -> np.ndarray:
    """The exposure path rolled circularly by k calendar days within the sub-sleeve's own active span."""
    if k == 0 or x.a >= len(x.lev):
        return x.lev
    return np.concatenate([x.lev[:x.a], np.roll(x.lev[x.a:], k)])


def shift_null(inputs: dict[tuple, NullInput], actual: float, n_draws: int = N_DRAWS, seed: int = NULL_SEED) -> dict:
    """Circular-shift null on exposure paths. Each draw rolls every sub-sleeve's base-run exposure path (signed
    notional / equity, forward-filled over non-trading days) by ONE common k ~ U{30 .. S_min - 30} calendar
    days within its own active span (S_min = the shortest span), against the unshifted market returns. Time in
    market, sizes, the leverage path and the fees and slippage paid are exactly the base run's; only their
    timing against returns changes. Statistic: mean daily net account return (all9, base costs, no funding);
    k = 0 reproduces the engine's value exactly. p = (1 + #{null >= actual}) / (1 + draws)."""
    rng = np.random.default_rng(seed)
    xs = list(inputs.values())
    n = len(xs[0].lev)
    spans = [n - x.a for x in xs if x.a < n]
    s_min = int(min(spans))
    if s_min < 61:
        raise ValueError("an active span is too short for the shift null")
    slots = len(E.MARKETS) * len(M.CONFIGS[M.PRIMARY])
    cost = float(sum(x.cost.sum() for x in xs))     # a roll keeps each span's total cost

    def stat(k: int) -> float:
        return (sum(float(np.dot(shifted_exposure(x, k)[:-1], x.r[1:])) for x in xs) - cost) / (slots * n)

    s0 = stat(0)                                    # a float sanity check; null_inputs checks the decomposition
    if not np.isclose(s0, actual, rtol=1e-8, atol=1e-13):
        raise AssertionError(f"the exposure decomposition does not reproduce the actual statistic ({s0} vs {actual})")
    null = np.empty(n_draws)
    ks = np.empty(n_draws, np.int64)
    for j in range(n_draws):
        ks[j] = int(rng.integers(30, s_min - 30 + 1))
        null[j] = stat(int(ks[j]))
    return {"actual_mean_daily": actual, "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
            "null_p95": float(np.quantile(null, 0.95)),
            "p_value": float((1 + np.sum(null >= actual)) / (1 + len(null))), "s_min": s_min, "draws": n_draws,
            "seed": seed, "dist": null, "k": ks}


# ---------------------------------------------------------------- benchmarks and reporting helpers

def ew_benchmark(mkts: dict[str, Mkt], w: Window) -> pd.Series:
    """Daily-rebalanced equal-weight long of the available markets: each market's close-to-close return is
    put on its own trade date; each UTC day is the mean over the markets with a return that day; days with
    none are 0. No costs, no funding."""
    df = pd.DataFrame({s: m.closes.pct_change() for s, m in mkts.items()})
    df.index = pd.to_datetime(df.index, unit="ms", utc=True).as_unit("ms")
    return df.reindex(w.calendar).mean(axis=1).fillna(0.0)


def bh_benchmark(mkts: dict[str, Mkt], w: Window) -> pd.Series:
    """Information: a literal equal-weight buy-and-hold (no rebalancing) of the markets live in the
    window's first week."""
    t0 = int((w.start + 7 * DAY).value // 10**6)
    live = {s: m.closes for s, m in mkts.items() if int(m.closes.index[0]) <= t0}
    px = pd.DataFrame(live)
    px.index = pd.to_datetime(px.index, unit="ms", utc=True).as_unit("ms")
    px = px.reindex(px.index.union(w.calendar)).ffill().reindex(w.calendar)
    val = (px / px.bfill().iloc[0]).mean(axis=1)
    r = val.pct_change().fillna(0.0)
    return r


def summary(res: engine.Result) -> dict:
    """metrics.summarize without its minute-row fields (exposure, avg_hold_h)."""
    s = metrics.summarize(res)
    s.pop("exposure", None)
    s.pop("avg_hold_h", None)
    t = res.trades
    if len(t):
        s["avg_hold_days"] = float((t["exit_time"] - t["entry_time"]).dt.total_seconds().mean() / 86400)
        s["avg_hold_rows"] = float((t["exit_i"] - t["entry_i"]).mean())
    return s


def full_book_day(subs: dict, mkts: dict[str, Mkt]) -> pd.Timestamp:
    """First trade date on which every sub-sleeve can hold a position."""
    last = max(int(mkts[sb.sym].bar_ts[min(sb.ready, len(sb.state) - 1)]) for sb in subs.values())
    return pd.Timestamp(last, unit="ms", tz="UTC")


def activity(parts: dict, mkts: dict[str, Mkt], since: pd.Timestamp, w: Window) -> dict:
    """Information: netted market orders. A market needs one order on a trade date when any of its 9
    sub-sleeves enters or exits there (sizes differ, so they rarely cancel); end-of-window closes excluded."""
    t0, t1 = int(since.value // 10**6), int(w.end.value // 10**6)
    order_dates, days, n_sub = {}, set(), 0
    for s, m in mkts.items():
        rows = set()
        for (sym, L), (_, tr) in parts.items():
            if sym != s or L not in M.CONFIGS[M.PRIMARY]:
                continue
            rows.update(int(i) for i in tr["entry_i"])
            rows.update(int(i) for i in tr.loc[tr["reason"] != "end", "exit_i"])
            ed = m.mk.ts[tr["entry_i"].to_numpy(np.int64)] - E.ROW_OFFSET_MS
            n_sub += int(((ed >= t0) & (ed < t1)).sum())
        d = m.mk.ts[sorted(rows)] - E.ROW_OFFSET_MS if rows else np.zeros(0, np.int64)
        order_dates[s] = set(int(x) for x in d if t0 <= x < t1)
        days |= set(int(x) for x in m.bar_ts if t0 <= x < t1)
    n_orders = sum(len(v) for v in order_dates.values())
    any_day = set().union(*order_dates.values()) if order_dates else set()
    nd = max(len(days), 1)
    return {"since": str(since.date()), "trading_days": len(days), "orders_per_trading_day": n_orders / nd,
            "share_days_with_order": len(any_day) / nd,
            "orders_per_market_per_day": {s: len(v) / nd for s, v in order_dates.items()},
            "sub_entries_per_week": n_sub / max((t1 - t0) / (7 * DAY_MS), 1e-9)}


def exposure(subs: dict, mkts: dict[str, Mkt], parts: dict, eqs: dict, w: Window, since: pd.Timestamp) -> dict:
    """Information: account gross and net exposure (notional / equity at each mark, sized at entry), all9."""
    cal = _cal_ms(w)
    gross = np.zeros(len(cal))
    net = np.zeros(len(cal))
    for key, sb in subs.items():
        if key[1] not in M.CONFIGS[M.PRIMARY]:
            continue
        lev, _, _ = exposure_path(mkts[sb.sym], parts[key][1], eqs[key], cal)
        gross += np.abs(lev)
        net += lev
    keep = w.calendar >= since
    g, n = gross[keep] / SLOTS, net[keep] / SLOTS
    return {"since": str(since.date()), "mean_gross": float(g.mean()), "mean_net": float(n.mean()),
            "gross_p5_p50_p95": [float(v) for v in np.quantile(g, [0.05, 0.5, 0.95])],
            "max_gross": float(g.max())}


# ---------------------------------------------------------------- evaluation

def evaluate(mkts: dict[str, Mkt], w: Window, with_null: bool = True, n_draws: int = N_DRAWS,
             light: bool = False) -> dict:
    """Everything the gates and the report need for one window. Integrity errors raise HardError before any
    gate is computed. Liquidations are outcomes (closed at max(equity, 0), flat after): booked in every run and
    reported. light=True (information-only variants): base run and headline numbers only."""
    t0 = time.time()
    subs = build(mkts)
    parts, info = run_all(subs, mkts, w)
    res, daily = assemble(parts, w)
    if light:
        return {"summary": summary(res), "weekly": stats.weekly_t(daily), "mean_daily": float(daily.mean()),
                "liquidated": info["ruins"], "daily": daily}
    mm = mismatches(subs, parts, mkts, w, info["liq_row"])
    integrity = hard_errors(mm)
    if integrity:
        raise HardError("; ".join(integrity))
    drag = funding_drag(subs, mkts, parts, info["eq"], w)
    fparts = {k: ((d - drag[k]).clip(lower=-1.0), t) for k, (d, t) in parts.items()}
    fres, daily_f = assemble(fparts, w)
    sparts, sinfo = run_all(subs, mkts, w, mult=2.0)
    sres, daily2 = assemble(sparts, w)
    liq = {"base": sorted(info["ruins"]), "stress_2x": sorted(sinfo["ruins"])}
    drag2 = funding_drag(subs, mkts, sparts, sinfo["eq"], w)
    daily_2f = assemble({k: ((d - drag2[k]).clip(lower=-1.0), t) for k, (d, t) in sparts.items()}, w)[1]
    cfg_daily = {c: assemble(parts, w, c)[1] for c in M.CONFIGS}
    cfg_trades = {c: int(sum(len(v[1]) for (s, L), v in parts.items() if L in M.CONFIGS[c])) for c in M.CONFIGS}
    cfg_tot = {c: float((1 + d).prod() - 1) for c, d in cfg_daily.items()}
    pbo, logits = stats.pbo_cscv(pd.concat(cfg_daily, axis=1).fillna(0.0).to_numpy())
    primary = {k: v for k, v in parts.items() if k[1] in M.CONFIGS[M.PRIMARY]}
    contrib = _coin_contrib(primary, lambda k: k[0], SLOTS)
    bench = ew_benchmark(mkts, w)
    fb = full_book_day(subs, mkts)
    since = max(fb, w.start)
    t = res.trades
    side = {("long" if d == 1 else "short"): float(t.loc[t["dir"] == d, "ret"].sum()) for d in (1, -1)}
    mdaily = pd.DataFrame({s: sum(v[0] for k, v in primary.items() if k[0] == s) / SLOTS for s in E.MARKETS})
    out = {"summary": summary(res), "weekly": stats.weekly_t(daily), "mean_daily": float(daily.mean()),
           "stress_2x_mean_daily": float(daily2.mean()), "stress_2x": summary(sres),
           "funding_mean_daily": float(daily_f.mean()), "funding": summary(fres),
           "funding_cost_ann": float((daily.mean() - daily_f.mean()) * 365),
           "year_share": stats.year_share_daily(daily),
           "market_contrib": contrib, "markets_positive": int(sum(v > 0 for v in contrib.values())),
           "lookback_contrib": {int(L): float(sum(v[0].sum() for k, v in parts.items() if k[1] == L) / SLOTS)
                                for L in M.LOOKBACKS},
           "year_returns": {int(y): float(v) for y, v in daily.groupby(daily.index.year).sum().items()},
           "side_contrib": side,
           "pbo": pbo, "plateau": _plateau([cfg_tot[c] for c in M.CONFIGS if c != M.PRIMARY]),
           "config_totals": cfg_tot,
           "alpha_vs_ew": stats.beta_alpha(daily, bench),
           "alpha_vs_bh_info": stats.beta_alpha(daily, bh_benchmark(mkts, w)),
           "full_book_day": str(fb.date()), "floors": info["floors"],
           "stress_2x_plus_funding_mean_daily": float(daily_2f.mean()),
           "hard_checks": mm, "liquidations": liq,
           "activity": activity(parts, mkts, since, w), "exposure": exposure(subs, mkts, parts, info["eq"], w, since),
           "daily": daily, "res": res, "parts": parts, "cfg_daily": cfg_daily, "cfg_trades": cfg_trades,
           "pbo_logits": logits, "benchmark_daily": bench, "market_daily": mdaily,
           "stress_trades": sres.trades, "funding_daily": daily_f}
    if with_null:
        out["null"] = shift_null(null_inputs(subs, mkts, parts, info["eq"], w, info["liq_row"]), float(daily.mean()),
                                 n_draws)
    out["runtime_s"] = time.time() - t0
    return out


def gates_dev(ev: dict) -> dict:
    return {
        "weekly t>=2": ev["weekly"]["t"] >= 2.0,
        "2x-cost mean>0": ev["stress_2x_mean_daily"] > 0,
        "funding-stress mean>0": ev["funding_mean_daily"] > 0,
        "year share<=50%": ev["year_share"] <= 0.5,
        ">=7/11 markets positive": ev["markets_positive"] >= MIN_MARKETS,
        "shift null p<0.05": ev["null"]["p_value"] < 0.05,
        "PBO<=0.5": ev["pbo"] <= 0.5,
        "plateau (both neighbours positive)": ev["plateau"] >= 1.0,
        "alpha PSR vs EW>=0.95": ev["alpha_vs_ew"]["psr_alpha"] >= 0.95,
    }


def boot_source(dev_daily: pd.Series, full_book: str) -> pd.Series:
    """Dev daily returns from the first full-book day to the end of dev (the bootstrap source window)."""
    return dev_daily[dev_daily.index >= pd.Timestamp(full_book, tz="UTC")]


def gates_holdout(evh: dict, dev_sharpe: float, boot_src: pd.Series) -> dict:
    hd = evh["daily"]
    _, mdd_dist = stats.block_bootstrap(boot_src.to_numpy(), BOOT_N, block=BOOT_BLOCK, n_out=len(hd), seed=BOOT_SEED)
    mdd = metrics.max_drawdown(np.r_[1.0, (1.0 + hd).cumprod().to_numpy()])
    evh["daily_mdd"], evh["boot_mdd_p95"] = mdd, float(np.quantile(mdd_dist, 0.95))
    evh["f1"] = f1_tests(hd)
    return {
        "holdout mean>0": float(hd.mean()) > 0,
        "holdout Sharpe>=50% dev": evh["summary"].get("sharpe", 0.0) >= 0.5 * dev_sharpe,
        "holdout MDD<=dev boot p95": mdd <= evh["boot_mdd_p95"],
        ">=7/11 markets positive (holdout)": evh["markets_positive"] >= MIN_MARKETS,
        "weekly corr with F1 (2020-23)<=0.3": evh["f1"]["weekly_corr"] <= 0.3,
    }


def n_dsr(n_eff: float) -> float:
    return float(max(n_eff, NEFF_FLOOR))


def headline(dev_pass: bool, hold_pass: bool, daily_ab: pd.Series, n: float) -> dict:
    return headline_r5(dev_pass, hold_pass, daily_ab, n)


# ---------------------------------------------------------------- data report

def _longest_run(b) -> int:
    best = run = 0
    for x in b:
        run = run + 1 if x else 0
        best = max(best, run)
    return best


def _s(x) -> str:
    """A manifest cell as text ('' for an empty CSV cell)."""
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)


def _date_strs(ms) -> np.ndarray:
    return pd.to_datetime(np.asarray(ms, np.int64), unit="ms", utc=True).strftime("%Y-%m-%d").to_numpy()


def uk_bank_holidays(y0: int = 1987, y1: int = 2026) -> set[str]:
    """England & Wales bank holidays on weekdays (substitute days included): Brent spot has no assessment on
    them, so they are not missing reference days for BZUSDT."""
    from datetime import date, timedelta
    from dateutil.easter import easter

    def last_monday(y, mth):
        d = date(y, mth + 1, 1) - timedelta(days=1)
        return d - timedelta(days=d.weekday())

    def first_monday(y, mth):
        d = date(y, mth, 1)
        return d + timedelta(days=(7 - d.weekday()) % 7)

    out = set()
    for y in range(y0, y1 + 1):
        d = date(y, 1, 1)
        out.add(d + timedelta(days=(7 - d.weekday()) % 7) if d.weekday() >= 5 else d)
        e = easter(y)
        out |= {e - timedelta(days=2), e + timedelta(days=1)}
        out.add({1995: date(1995, 5, 8), 2020: date(2020, 5, 8)}.get(y, first_monday(y, 5)))
        out |= {2002: {date(2002, 6, 3), date(2002, 6, 4)}, 2012: {date(2012, 6, 4), date(2012, 6, 5)},
                2022: {date(2022, 6, 2), date(2022, 6, 3)}}.get(y, {last_monday(y, 5)})
        out.add(last_monday(y, 8))
        c = date(y, 12, 25)
        out |= {5: {date(y, 12, 27), date(y, 12, 28)}, 6: {date(y, 12, 26), date(y, 12, 27)},
                4: {c, date(y, 12, 28)}}.get(c.weekday(), {c, date(y, 12, 26)})
    out |= {date(1999, 12, 31), date(2011, 4, 29), date(2022, 9, 19), date(2023, 5, 8)}
    return {d.isoformat() for d in out if d.weekday() < 5}


def closed_days(m: str) -> set[str]:
    """Days a market is known to be closed although SPY trades (and pre-1993 closures)."""
    return set(KNOWN_CLOSURES) | (uk_bank_holidays() if m == "BZUSDT" else set())


def market_report(m: str, f: pd.DataFrame, info: dict, ref: np.ndarray, nulls: list[str] | None = None) -> dict:
    """Data facts for one market. Holdout rows are used only for dates, counts, price signs and whether
    consecutive closes are equal; anything built on returns uses dev rows only."""
    src = E.SOURCES[m]
    cut = int(E.R6_HOLDOUT_START.value // 10**6)
    ts = f["ts"].to_numpy(np.int64)
    c = f["close"].to_numpy(np.float64)
    ds = _date_strs(ts)
    dev = ts < cut
    wd = pd.to_datetime(ts, unit="ms", utc=True).weekday
    closures = np.array(sorted(closed_days(m)))
    # missing reference days (SPY's trade dates) inside the market's span
    r = ref[(ref >= ts[0]) & (ref <= ts[-1])]
    r = r[~np.isin(_date_strs(r), closures)]
    miss = ~np.isin(r, ts)
    yrs = pd.Series(pd.to_datetime(r, unit="ms", utc=True).year)
    by_year = pd.Series(miss).groupby(yrs).agg(["mean", "size"])
    full = by_year[by_year["size"] >= MIN_YEAR_REF_DAYS]
    worst_year = float(full["mean"].max()) if len(full) else 0.0
    # weekdays missing between consecutive rows over the whole series (closures excluded): this also covers
    # SPY itself, the years before SPY starts and the hand-over to the reference
    gap = 0
    if len(ts) > 1:
        d0 = np.array(pd.to_datetime(ts[:-1], unit="ms", utc=True).date, "datetime64[D]")
        d1 = np.array(pd.to_datetime(ts[1:], unit="ms", utc=True).date, "datetime64[D]")
        gap = int(np.max(np.busday_count(d0 + 1, d1, holidays=closures.astype("datetime64[D]"))))
    eq_run = _longest_run(c[1:] == c[:-1]) + 1 if len(c) > 1 else 1
    lr = np.diff(np.log(c[dev])) if dev.sum() > 1 else np.zeros(0)
    big = np.flatnonzero(np.abs(lr) > 0.2)
    top = np.argsort(-np.abs(lr))[:5] if len(lr) else []
    spikes = [i for i in range(len(lr) - 1) if abs(lr[i]) > 0.15 and abs(lr[i] + lr[i + 1]) < 0.2 * abs(lr[i])]
    split_dates = [s for s in _s(info.get("split_dates")).split(";") if s and s < "2016-01-01"]
    split_lr, split_missing = [], []
    for s in split_dates:
        j = int(np.searchsorted(ds[dev], s))
        if 0 < j < dev.sum() and ds[j] == s:
            split_lr.append(abs(float(lr[j - 1])))
        elif ds[0] < s:
            split_missing.append(s)
    n_null = int(info.get("nulls_dropped") or 0)
    ref_days = set(_date_strs(r))
    null_ref = sum(1 for d in (nulls or []) if d in ref_days)      # Amendment 1: nulls on reference days only
    rows = int(len(ts))
    return {"market": m, "source": src.source, "ticker": src.ticker, "rows": rows,
            "rows_dev": int(dev.sum()), "rows_holdout": int((~dev & (ts < int(E.R6_END.value // 10**6))).sum()),
            "first_date": ds[0], "last_date": ds[-1], "first_expected": src.first_expected,
            "first_late_days": int((pd.Timestamp(ds[0]) - pd.Timestamp(src.first_expected)).days),
            "weekend_rows": int((wd >= 5).sum()),
            "nonpositive_dropped": _s(info.get("nonpositive_dropped")),
            "nonpositive_unexpected": _s(info.get("nonpositive_unexpected")),
            "ref_days": int(len(r)), "missing_ref_days": int(miss.sum()),
            "longest_missing_run": int(max(_longest_run(miss), gap)), "max_weekday_gap": gap,
            "worst_year_missing_share": worst_year,
            "extra_days_vs_ref": int((~np.isin(ts[ts >= ref[0]], ref)).sum()),
            "longest_equal_run": int(eq_run), "equal_share": float(np.mean(c[1:] == c[:-1])) if len(c) > 1 else 0.0,
            "nulls_dropped": n_null, "nulls_on_ref_days": null_ref, "null_share": null_ref / max(len(r), 1),
            "dups_dropped": int(info.get("dups_dropped") or 0),
            "dev_abs_lr_gt_20pct": int(len(big)),
            "dev_top5": ";".join(f"{ds[dev][i + 1]}:{lr[i]:+.3f}" for i in top),
            "dev_reverting_spikes": int(len(spikes)),
            "dev_spike_dates": ";".join(ds[dev][i + 1] for i in spikes[:10]),
            "dev_split_dates": ";".join(split_dates), "dev_split_dates_without_row": ";".join(split_missing),
            "dev_split_max_abs_lr": float(max(split_lr)) if split_lr else 0.0,
            "dev_max_abs_lr": float(np.abs(lr).max()) if len(lr) else 0.0}


def red_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-registered red flags; any one stops the round before any strategy run."""
    df = df.copy()
    checks = {
        "first date >30d late": df["first_late_days"] > 30,
        "last date before 2026-09-29": df["last_date"].astype(str) < LAST_DATE_MIN,
        "weekend rows": df["weekend_rows"] > 0,
        "non-positive close not allowlisted": df["nonpositive_unexpected"].fillna("").astype(str).str.len() > 0,
        "missing run >3": df["longest_missing_run"] > 3,
        "year missing >3%": df["worst_year_missing_share"] > 0.03,
        ">=5 equal closes": df["longest_equal_run"] >= 5,
        "nulls >1%": df["null_share"] > 0.01,
        "reverting spike (Yahoo, dev)": (df["source"] == "yahoo") & (df["dev_reverting_spikes"] > 0),
        "unadjusted split (dev)": df["dev_split_max_abs_lr"] > np.log(1.8),
        "split date without a row (dev)": df["dev_split_dates_without_row"].fillna("").astype(str).str.len() > 0,
        "|log return| > ln 1.8 (Yahoo, dev)": (df["source"] == "yahoo") & (df["dev_max_abs_lr"] > np.log(1.8)),
    }
    flags = pd.DataFrame(checks)
    df["red_flag_reasons"] = flags.apply(lambda r: ";".join(k for k, v in r.items() if v), axis=1)
    df["red_flag"] = flags.any(axis=1)
    return df


def nyse_holidays(y0: int = 1993, y1: int = 2026) -> set[str]:
    """NYSE full-day holidays by rule (weekdays only): New Year (Sunday -> Monday; Saturday not observed), MLK
    (from 1998), Presidents, Good Friday, Memorial, Juneteenth (from 2022), Independence, Labor, Thanksgiving,
    Christmas (Saturday -> Friday, Sunday -> Monday). Special closures are in KNOWN_CLOSURES."""
    from datetime import date, timedelta
    from dateutil.easter import easter

    def nth(y, mth, wd, n):
        d = date(y, mth, 1) + timedelta(days=(wd - date(y, mth, 1).weekday()) % 7)
        return d + timedelta(weeks=n - 1)

    def last(y, mth, wd):
        d = date(y, mth + 1, 1) - timedelta(days=1)
        return d - timedelta(days=(d.weekday() - wd) % 7)

    def observed(d):
        return d - timedelta(days=1) if d.weekday() == 5 else d + timedelta(days=1) if d.weekday() == 6 else d

    out = set()
    for y in range(y0, y1 + 1):
        nyd = date(y, 1, 1)
        if nyd.weekday() != 5:
            out.add(observed(nyd))
        if y >= 1998:
            out.add(nth(y, 1, 0, 3))
        out |= {nth(y, 2, 0, 3), easter(y) - timedelta(days=2), last(y, 5, 0), observed(date(y, 7, 4)),
                nth(y, 9, 0, 1), nth(y, 11, 3, 4), observed(date(y, 12, 25))}
        if y >= 2022:
            out.add(observed(date(y, 6, 19)))
    return {d.isoformat() for d in out if d.weekday() < 5}


def reference_dates(frames: dict[str, pd.DataFrame]) -> np.ndarray:
    """US trade dates from SPY's first to its last date: SPY's own dates, the NYSE calendar by rule (weekdays
    minus nyse_holidays and KNOWN_CLOSURES), and any date at least 2 of the other US-listed ETFs have. A hole in
    SPY itself is therefore a missing reference day for SPY in every year."""
    spy = frames["SPYUSDT"]["ts"].to_numpy(np.int64)
    peers = pd.Series(np.concatenate([np.zeros(0, np.int64)] + [frames[m]["ts"].to_numpy(np.int64)
                                                               for m in PEER_ETFS if m in frames]))
    agreed = peers.value_counts()
    first, last = pd.Timestamp(int(spy[0]), unit="ms", tz="UTC"), pd.Timestamp(int(spy[-1]), unit="ms", tz="UTC")
    wd = pd.bdate_range(first, last, tz="UTC")
    closed = nyse_holidays(first.year, last.year) | KNOWN_CLOSURES
    nyse = wd[~wd.strftime("%Y-%m-%d").isin(closed)].as_unit("ms").asi8
    return np.union1d(np.union1d(spy, agreed[agreed >= 2].index.to_numpy(np.int64)), nyse)


def data_report() -> pd.DataFrame:
    man = pd.read_csv(E.MANIFEST).set_index("market")
    frames = {m: E._verified_frame(m) for m in E.MARKETS}        # raises on any SHA256 mismatch
    ref = reference_dates(frames)
    return pd.DataFrame([market_report(m, frames[m], man.loc[m].to_dict(), ref, E.null_dates(m)) for m in E.MARKETS])


# ---------------------------------------------------------------- guards, logging

def _log_run(kind: str, **extra) -> None:
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    rec = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "kind": kind,
           "git": _git("rev-parse", "HEAD"), "dirty": bool(_git("status", "--porcelain", "--", "src"))} | extra
    with RUN_LOG.open("a", newline="\n") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def _r6_section(txt: str) -> str:
    i = txt.find("## Round 6")
    if i < 0:
        return ""
    j = txt.find("\n## ", i + 1)
    return txt[i:] if j < 0 else txt[i:j]


def _require_prereg() -> None:
    from btcq.research.round4 import _require_committed
    _require_committed([PREREG, POWER_CSV])
    sec = _r6_section(PREREG.read_text(encoding="utf-8"))
    if "Power table" not in sec or re.search(r"\{[A-Z_]+\}", sec):
        raise SystemExit("PREREGISTRATION.md has no complete committed Round 6 section (power table, no placeholders)")


def _require_data_report() -> None:
    from btcq.research.round4 import _require_committed
    _require_committed([DATA_REPORT, E.MANIFEST])
    if pd.read_csv(DATA_REPORT)["red_flag"].astype(str).eq("True").any():
        raise SystemExit("the Round 6 data report has red flags")


def _hard_error_versions() -> set[int]:
    texts = ([RUN_LOG.read_text()] if RUN_LOG.exists() else []) + _committed_run_logs()
    recs = [json.loads(x) for t in texts for x in t.splitlines() if x.strip()]
    return {int(r["version"]) for r in recs if r.get("kind") == "r6-sweep" and r.get("hard_error")}


def _sha_file(p) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _sha_text(p) -> str:
    """SHA256 of a text file with CRLF normalised (git may rewrite line endings on checkout)."""
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _starts_in(text: str, version: int) -> list[dict]:
    recs = [json.loads(x) for x in text.splitlines() if x.strip()]
    return [r for r in recs if r.get("kind") == "r6-sweep-start" and r.get("version") == version]


def _sweep_starts(version: int) -> list[dict]:
    return _starts_in(RUN_LOG.read_text(), version) if RUN_LOG.exists() else []


def _committed_run_logs() -> list[str]:
    """Every committed version of the run log, on any branch: a record once committed can't be erased by a
    checkout, a forward commit that restores an older log, or a branch cut before it."""
    try:
        rel = str(RUN_LOG.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return []
    revs = _git("log", "--all", "--format=%H", "--", rel).split()
    return [t for t in (_git("show", f"{c}:{rel}") for c in revs) if t]


def code_sha() -> str:
    """SHA256 over the Round 6 code paths (line endings normalised), stored at the sweep, checked at validate."""
    h = hashlib.sha256()
    for f in CODE_FILES:
        h.update(f.encode())
        h.update((ROOT / f).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def _holdout_touched(r: dict) -> list[str]:
    out = []
    if "holdout" in r:
        out.append("M1 holdout already evaluated")
    if E.r6_unlock_records():
        out.append("the holdout log already has a Round 6 unlock")
    return out


# ---------------------------------------------------------------- commands

def ramp_weights(days_per_year: int = 252, dev_years: int = 30) -> np.ndarray:
    """Share of the 99 sub-sleeves live on each dev trading day, from the pre-registered source start dates
    (source facts, not prices): a sub-sleeve is live from its market's first date + max(L, 90) trading days."""
    n = days_per_year * dev_years
    w = np.zeros(n)
    for src in E.SOURCES.values():
        y0 = (pd.Timestamp(src.first_expected, tz="UTC") - DEV.start).days / 365.25
        for L in M.LOOKBACKS:
            j0 = int(np.ceil(y0 * days_per_year)) + max(int(L), M.VOL_DAYS)
            w[max(j0, 0):] += 1
    return w / SLOTS


def cmd_power() -> list[dict]:
    w = ramp_weights()
    rows = stats.power_table_td() + stats.power_table_td(ramp=w)
    U.write_csv(pd.DataFrame(rows), POWER_CSV)
    print(f"ramp: {w.sum() / 252 * len(E.MARKETS):.0f} market-years = {w.sum() / 252:.1f} full-book years of 30")
    for r in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    return rows


def cmd_download(resume: bool = False) -> pd.DataFrame:
    _require_prereg()
    _clean_src()
    try:
        E.download_all(resume=resume)
    except FileExistsError as e:
        raise SystemExit(f"{e} (r6-download runs once; --resume only fills missing markets)")
    except Exception as e:  # noqa: BLE001 - a failed download or parse is a red flag: stop here
        _log_run("r6-download", error=f"{type(e).__name__}: {e}")
        raise SystemExit(f"download or parse failed (a red flag; fix and amend, then --resume): {e}")
    df = red_flags(data_report())
    U.write_csv(df, DATA_REPORT)
    _log_run("r6-download", red_flags=int(df["red_flag"].sum()), manifest_sha256=_sha_file(E.MANIFEST))
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(df.to_string())
    if df["red_flag"].any():
        raise SystemExit("data red flags: fix the data and write a dated amendment before any strategy run")
    return df


def _trial_rows(ev: dict, version: int) -> list[dict]:
    rows = []
    for c, d in ev["cfg_daily"].items():
        sr, T, sk, ku = stats.sharpe_moments(d)
        rows.append({"strategy": CODE, "version": version,
                     "params": {"set": c, "lookbacks": list(M.CONFIGS[c]), "vol_target": M.VOL_TARGET, "cap": M.LEV_CAP},
                     "period": "dev", "sr": sr, "T": T, "skew": sk, "kurt": ku, "trades": int(ev["cfg_trades"][c]),
                     "sharpe_ann": float(sr * np.sqrt(365))})
    return rows


def r6_trial_returns() -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in sorted(R6_RESULTS.glob("*_trial_returns.parquet"))]
    return pd.concat(frames, axis=1).fillna(0.0)


_SKIP = {"daily", "res", "parts", "cfg_daily", "cfg_trades", "pbo_logits", "benchmark_daily", "market_daily",
         "stress_trades", "funding_daily"}


def _public(ev: dict) -> dict:
    out = {k: v for k, v in ev.items() if k not in _SKIP}
    if "null" in out:
        out["null"] = {k: v for k, v in out["null"].items() if k not in ("dist", "k")}
    return out


def _write(tag: str, ev: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ev["res"].trades.to_parquet(RESULTS / f"{CODE}_{tag}_trades.parquet")
    ev["res"].equity.rename("equity").to_frame().to_parquet(RESULTS / f"{CODE}_{tag}_equity.parquet")
    ev["daily"].rename("ret").to_frame().to_parquet(RESULTS / f"{CODE}_{tag}_daily.parquet")
    ev["benchmark_daily"].rename("ret").to_frame().to_parquet(RESULTS / f"{CODE}_{tag}_benchmark.parquet")
    ev["market_daily"].to_parquet(RESULTS / f"{CODE}_{tag}_market_daily.parquet")
    ev["stress_trades"].to_parquet(RESULTS / f"{CODE}_{tag}_stress_2x_trades.parquet")
    ev["funding_daily"].rename("ret").to_frame().to_parquet(RESULTS / f"{CODE}_{tag}_funding_daily.parquet")
    if tag == "dev":
        pd.DataFrame([{"cfg": c, "total_return": ev["config_totals"][c], "trades": ev["cfg_trades"][c],
                       "sharpe": float(stats.sharpe_moments(d)[0] * np.sqrt(365))}
                      for c, d in ev["cfg_daily"].items()]).to_csv(RESULTS / f"{CODE}_configs.csv", index=False)
        src = boot_source(ev["daily"], ev["full_book_day"])
        n_out = len(HOLD.calendar)
        _, boot_mdd = stats.block_bootstrap(src.to_numpy(), BOOT_N, block=BOOT_BLOCK, n_out=n_out, seed=BOOT_SEED)
        nl = ev.get("null", {})
        np.savez_compressed(RESULTS / f"{CODE}_dev_dists.npz", null=nl.get("dist", np.zeros(1)),
                            null_k=nl.get("k", np.zeros(1)), boot_mdd=boot_mdd, pbo_logits=np.asarray(ev["pbo_logits"]))


def _info(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except Exception as e:  # noqa: BLE001 - information only; never affects a gate
        return {"error": f"{type(e).__name__}: {e}"}


def cmd_sweep(rerun: bool = False, n_draws: int = N_DRAWS) -> dict:
    _require_prereg()
    _require_data_report()
    _clean_src()
    meta = M.MacroTrend()
    p = RESULTS / f"{CODE}.json"
    closed = _holdout_touched(json.loads(p.read_text()) if p.exists() else {})
    if closed:
        raise SystemExit("Round 6 is closed once its holdout is touched (no reruns): " + "; ".join(closed))
    if p.exists():
        if not rerun:
            raise SystemExit(f"{CODE}.json exists; the dev sweep runs once (--rerun needs a version bump)")
        if meta.version <= json.loads(p.read_text()).get("version", 0):
            raise SystemExit("--rerun needs a MacroTrend.version bump (a rerun counts as new trials)")
    if meta.version in _hard_error_versions():
        raise SystemExit(f"version {meta.version} stopped on a hard error; a fix needs a version bump")
    sha = code_sha()
    man = _sha_text(E.MANIFEST) if E.MANIFEST.exists() else None
    committed = [r for t in _committed_run_logs() for r in _starts_in(t, meta.version)]
    starts = committed + _sweep_starts(meta.version)
    if any(r.get("code_sha256") != sha for r in starts):
        raise SystemExit(f"version {meta.version} already started with different code; a change needs a version bump")
    if any(r.get("manifest_sha256") not in (None, man) for r in starts):
        raise SystemExit(f"version {meta.version} already started on different data; a change needs a version bump")
    if not committed:                               # phase 1: a durable start record before any result exists
        if not _sweep_starts(meta.version):
            _log_run("r6-sweep-start", version=meta.version, code_sha256=sha, manifest_sha256=man)
        raise SystemExit("logged the Round 6 sweep start record: commit research/round6/runs.jsonl, then run "
                         "r6-sweep again")
    mkts = load_markets(DEV)
    print("M1 dev ...", flush=True)
    try:
        ev = evaluate(mkts, DEV, n_draws=n_draws)
    except HardError as e:
        _log_run("r6-sweep", version=meta.version, hard_error=str(e))
        raise SystemExit(f"hard error, logged: commit research/round6/runs.jsonl; a fix needs a MacroTrend.version "
                         f"bump: {e}")
    g = gates_dev(ev)
    v = meta.version
    registry.record(_trial_rows(ev, v), path=TRIALS)
    R6_RESULTS.mkdir(parents=True, exist_ok=True)
    name = f"{CODE}_trial_returns.parquet" if v == 1 else f"{CODE}v{v}_trial_returns.parquet"
    pd.concat({(f"{CODE}_{c}" if v == 1 else f"{CODE}_v{v}_{c}"): d for c, d in ev["cfg_daily"].items()}, axis=1) \
        .to_parquet(R6_RESULTS / name)
    trial = r6_trial_returns()
    n_eff = registry.ratcheted_effective_trials(stats.effective_trials(trial.to_numpy()), trial.shape[1],
                                                note=f"round 6 (M1 v{v})", log=NEFF_LOG)
    n = n_dsr(n_eff)
    ev["info_dividend_adjusted"] = _info(
        lambda: {k: x for k, x in evaluate(load_markets(DEV, adjusted=True), DEV, light=True).items() if k != "daily"})
    _write("dev", ev)
    r = {"code": CODE, "name": meta.name, "tf": "1d", "version": v, "round": 6, "hypothesis": meta.hypothesis,
         "rules": meta.rules, "params": {"set": M.PRIMARY, "lookbacks": list(M.LOOKBACKS), "vol_target": M.VOL_TARGET,
                                          "cap": M.LEV_CAP, "vol_days": M.VOL_DAYS, "ann": M.ANN},
         "markets": E.MARKETS, "holdout_scope": "clean (general-knowledge disclosure)",
         "start": str(DEV.start.date()), "end": str((DEV.end - DAY).date()), "full_book_day": ev["full_book_day"],
         "dev": ev["summary"], "dev_detail": _public(ev), "gates": g, "dev_pass": all(g.values()),
         "dsr_dev": _dsr(ev["daily"], n), "dsr_dev_own_neff": _dsr(ev["daily"], max(n_eff, 2.0)),
         "dsr_dev_n3": _dsr(ev["daily"], 3.0), "n_dsr": n, "n_eff": n_eff,
         "n_trials": registry.multiple_testing_inputs("dev", path=TRIALS)[0],
         "manifest_sha256": _sha_text(E.MANIFEST), "git_commit": _git("rev-parse", "HEAD"), "code_sha256": sha,
         "f1_daily_sha256": _sha_file(RESULTS / "F1_dev_daily.parquet"),
         "holdout_bootstrap": {"n": BOOT_N, "block": BOOT_BLOCK, "seed": BOOT_SEED}}
    _save_json(p, r)
    failed = [k for k, x in g.items() if not x]
    s = ev["summary"]
    print(f"M1 dev: sub-trades={s['trades']} sharpe={s.get('sharpe', 0):.2f} weekly t={ev['weekly']['t']:.2f} "
          f"funding mean/day={ev['funding_mean_daily']:.2e} null p={ev['null']['p_value']:.3f} -> "
          f"{'PASS' if r['dev_pass'] else 'FAIL: ' + ', '.join(failed)}", flush=True)
    _log_run("r6-sweep", version=v, n_eff=n_eff, n_dsr=n, verdict=r["dev_pass"], failed=failed,
             hard_checks=ev["hard_checks"], runtime_s=ev["runtime_s"])
    return r


def preflight_holdout() -> tuple[list[str], bool]:
    """Everything checked before the unlock, from timestamps, hashes and committed files only."""
    probs = []
    p = RESULTS / f"{CODE}.json"
    r = json.loads(p.read_text()) if p.exists() else {}
    if not r.get("dev_pass"):
        probs.append("M1 did not pass dev")
    if r and r.get("version") != M.MacroTrend().version:
        probs.append(f"MacroTrend.version {M.MacroTrend().version} is not the swept version {r.get('version')}")
    if r and r.get("code_sha256") != code_sha():
        probs.append("Round 6 code changed since the dev sweep (code_sha256 differs)")
    try:
        from btcq.research.round4 import _require_committed
        _require_committed([p, RESULTS / f"{CODE}_dev_daily.parquet", RESULTS / "F1_dev_daily.parquet", TRIALS, RUN_LOG])
    except SystemExit as e:
        probs.append(str(e))
    if r and E.MANIFEST.exists() and r.get("manifest_sha256") != _sha_text(E.MANIFEST):
        probs.append("the data manifest changed since the dev sweep")
    f1p = RESULTS / "F1_dev_daily.parquet"
    if r and f1p.exists() and r.get("f1_daily_sha256") != _sha_file(f1p):
        probs.append("F1_dev_daily.parquet changed since the dev sweep")
    done = "holdout" in r
    if done:
        probs.append("M1 holdout already evaluated")
    if r and "full_book_day" not in r:
        probs.append("M1.json has no full_book_day")
    for f in (f"{CODE}_dev_daily.parquet", f"{CODE}_dev_trades.parquet", "F1_dev_daily.parquet"):
        if not (RESULTS / f).exists():
            probs.append(f"{f} missing")
    try:
        _require_data_report()
    except SystemExit as e:
        probs.append(str(e))
    for m in E.MARKETS:
        try:
            d = E.load_dates(m)
            if _date_strs(d[-1:])[0] < LAST_DATE_MIN:
                probs.append(f"{m}: data ends before {LAST_DATE_MIN}")
        except Exception as e:  # noqa: BLE001 - reported as a problem
            probs.append(f"{m}: {type(e).__name__}: {e}")
    recs = E.r6_unlock_records()
    unlocks = [x for x in recs if x["kind"] == "unlock"]
    resumes = [x for x in recs if x["kind"] == "resume"]
    if recs and recs[0]["kind"] != "unlock":
        probs.append("a Round 6 resume record has no earlier unlock")
    resume = len(unlocks) > 0 and not done
    if len(unlocks) > 1 or len(resumes) > 1:
        probs.append("holdout_log already has more than one round 6 unlock")
    elif resume and resumes:
        probs.append("the round 6 unlock was already resumed once")
    elif resume and unlocks[0].get("version") != r.get("version"):
        probs.append(f"the logged unlock was for version {unlocks[0].get('version')}, not {r.get('version')}")
    return probs, resume


def cl_neutralised(evh: dict) -> dict:
    """Information: CL sub-sleeves' P&L over the dropped 2020-04-17 -> 2020-04-21 interval set to zero."""
    day = pd.Timestamp(CL_NEG[1], tz="UTC")
    parts = {k: ((d.where(d.index != day, 0.0), t) if k[0] == CL_NEG[0] else (d, t)) for k, (d, t) in evh["parts"].items()}
    _, daily = assemble(parts, HOLD)
    sr = stats.sharpe_moments(daily)[0] * np.sqrt(365)
    return {"mean_daily": float(daily.mean()), "sharpe": float(sr),
            "cl_day_contrib": float(sum(d.get(day, 0.0) for k, (d, _) in evh["parts"].items() if k[0] == CL_NEG[0]) / SLOTS)}


def cmd_validate(reason: str) -> dict:
    probs, resume = preflight_holdout()
    if probs:
        raise SystemExit("Round 6 holdout stays locked: " + "; ".join(probs))
    _clean_src()
    r = json.loads((RESULTS / f"{CODE}.json").read_text())
    v = int(r["version"])
    _log_run("r6-validate-start", version=v, resume=resume, code_sha256=r["code_sha256"])
    E.unlock_r6(reason, kind="resume" if resume else "unlock", version=v)
    mkts = load_markets(HOLD)
    try:
        evh = evaluate(mkts, HOLD, with_null=False)
    except HardError as e:                          # pre-registered: an integrity error in the holdout is a FAIL
        r |= {"holdout": {"integrity_error": str(e)}, "holdout_gates": {"no integrity error": False},
              "holdout_status": "integrity error: FAIL (pre-registered, no fix or rerun)",
              "headline": {"REPLICATED": False, "STRICT": False, "reason": "integrity error in the holdout run"},
              "final_pass": False}
        _save_json(RESULTS / f"{CODE}.json", r)
        _log_run("r6-validate", version=v, hard_error=str(e))
        raise SystemExit(f"holdout integrity error, recorded as a FAIL: {e}")
    dev_daily = pd.read_parquet(RESULTS / f"{CODE}_dev_daily.parquet")["ret"]
    hg = gates_holdout(evh, float(r["dev"].get("sharpe", 0.0)), boot_source(dev_daily, r["full_book_day"]))
    daily_ab = pd.concat([dev_daily, evh["daily"]]).sort_index()
    _write("holdout", evh)
    r |= {"holdout": _public(evh), "holdout_gates": hg, "holdout_status": "clean (general-knowledge disclosure)",
          "headline": headline(bool(r["dev_pass"]), all(hg.values()), daily_ab, float(r["n_dsr"]))}
    r["final_pass"] = r["headline"]["REPLICATED"]
    _save_json(RESULTS / f"{CODE}.json", r)          # the gated verdict is saved before anything optional runs
    f1 = evh["f1"]
    r["holdout_info"] = {
        "cl_2020_04_neutralised": _info(cl_neutralised, evh),
        "dividend_adjusted": _info(lambda: {k: x for k, x in evaluate(load_markets(HOLD, adjusted=True), HOLD,
                                                                      light=True).items() if k != "daily"}),
        "f1_hac": f1.get("hac"), "f1_t_alpha_sensitivity": f1.get("t_alpha_sensitivity"),
        "dsr_own_neff": _dsr(daily_ab, max(float(r["n_eff"]), 2.0)), "dsr_n3": _dsr(daily_ab, 3.0),
        "perp_vs_proxy": "run separately: Binance TradFi perp data is reachable only from the VPS"}
    _save_json(RESULTS / f"{CODE}.json", r)
    h = r["headline"]
    print(f"M1 holdout: sharpe={evh['summary'].get('sharpe', 0):.2f} gates={hg} | REPLICATED={h['REPLICATED']} "
          f"STRICT={h['STRICT']} (DSR {h['dsr']:.3f} at N {h['n']:.2f})")
    _log_run("r6-validate", version=v, headline=h, gates=hg, liquidations=evh["liquidations"])
    return r
