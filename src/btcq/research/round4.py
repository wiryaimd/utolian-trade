"""Round 4: F1 frozen on a point-in-time top-20 universe of Binance USD-M crypto perps.

    python -m btcq.research.run r4-symbols --exchange-info research/round4/exchange_info_um_<date>.json
    python -m btcq.research.run r4-universe
    python -m btcq.research.run r4-download --stage A
    python -m btcq.research.run r4-sweep
    python -m btcq.research.run r4-download --stage B     # only after Stage A passed
    python -m btcq.research.run r4-validate --unlock-holdout

Portfolio: one sleeve per symbol worth 1/SLOTS of the account (risk ACCOUNT_RISK x SLOTS of the
sleeve, notional <= 3x the sleeve). The account return is the sum of sleeve returns / SLOTS, and at
most SLOTS positions may be open at once (cap_pass). Entries only in months the symbol is a member;
after it leaves, open positions run to their normal exit. See research/PREREGISTRATION.md, Round 4.
"""
from __future__ import annotations

import json
import subprocess
import time
import zlib
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from btcq.backtest import engine, metrics, stats
from btcq.config import MINUTE_MS, ROOT, SYMBOLS, TF_MS, Costs, Risk
from btcq.data import store
from btcq.data import universe as U
from btcq.research import registry
from btcq.research.portfolio import assemble_slots, combine_null, sleeve_risk
from btcq.research.run import GATES, N_BOOT, N_PERM, RESULTS, _save_json, trade_equity_mdd
from btcq.strategies import BY_CODE

CODES = ["U1", "U1L"]
PRIMARY = "U1"
PARAMS = {"n": 40, "k": 3.5, "exit": "channel"}
UNIVERSE_ID = "um-top20-pit-v1"
SYMBOLS_CSV = U.ROUND4_DIR / "symbols.csv"
OVERRIDES_CSV = U.ROUND4_DIR / "symbol_overrides.csv"
UNIVERSE_CSV = U.ROUND4_DIR / "universe.csv"
UNIVERSE_META = U.ROUND4_DIR / "universe_meta.json"
RUN_LOG = U.ROUND4_DIR / "runs.jsonl"
PREREG = ROOT / "research" / "PREREGISTRATION.md"
MIN_FREQ = 52
MIN_SYM_TRADES = 10
MIN_CONSISTENCY = 0.6
DAY = pd.Timedelta(days=1)


@dataclass(frozen=True)
class Stage:
    name: str
    period: str
    start: pd.Timestamp
    end: pd.Timestamp            # exclusive

    @property
    def months(self) -> list[str]:
        return [m.strftime("%Y-%m") for m in pd.date_range(self.start, self.end - DAY, freq="MS")]

    @property
    def calendar(self) -> pd.DatetimeIndex:
        return pd.date_range(self.start, self.end - DAY, freq="D").as_unit("ms")


STAGES = {"A": Stage("A", "dev", pd.Timestamp("2020-04-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC")),
          "B": Stage("B", "full", pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2026-10-01", tz="UTC"))}


@dataclass(frozen=True)
class Spec:
    """Portfolio rules. The defaults are the pre-registered round 4 values; tests vary them."""
    slots: int = 20
    account_risk: float = 0.003
    max_lev: float = 3.0
    latency_min: int = 1
    taker: float = 0.0005
    maker: float = 0.0002
    tier_slip: tuple = (0.0002, 0.0003)      # rank 1-10, rank 11-20
    nonmember_slip: float = 0.0005
    delist_penalty: float = 0.01
    clean_data: bool = True                  # trading_span + funding-gap fill
    slip_by_symbol: dict = field(default_factory=dict)   # tests: fixed slippage per symbol
    null_seed: int | None = None             # tests: one seed for every symbol (round 3 used 11)

    def risk(self) -> Risk:
        return sleeve_risk(self.slots, Risk(self.account_risk, self.max_lev, Risk().initial_equity))

    def slip(self, sym: str, rank) -> float:
        if sym in self.slip_by_symbol:
            return self.slip_by_symbol[sym]
        if rank is None:
            return self.nonmember_slip
        return self.tier_slip[0] if rank <= 10 else self.tier_slip[1]

    def seed(self, sym: str) -> int:
        return self.null_seed if self.null_seed is not None else 11 + zlib.crc32(sym.encode()) % 65536


SPEC = Spec()


# ---------------------------------------------------------------- sleeves

class SlimMarket:
    """The arrays engine.run needs, cut to start at a sleeve's first possible entry (memory)."""

    def __init__(self, ts, o, h, l, c, fund, day_marks, days, ex, tf, latency_min):
        self.ts, self.o, self.h, self.l, self.c, self.fund = ts, o, h, l, c, fund
        self.day_marks, self.days = day_marks, days
        self._ex, self._tf, self._lat = ex, tf, latency_min

    def exec_idx(self, tf: str, latency_min: int = 0) -> np.ndarray:
        assert (tf, latency_min) == (self._tf, self._lat)
        return self._ex

    def index_at(self, ts: pd.Timestamp) -> int:
        return int(np.searchsorted(self.ts, ts.value // 10**6))


@dataclass
class Sleeve:
    symbol: str
    mk: SlimMarket
    sigs: dict                 # code -> tuple of signal arrays (entry, stop, target, trail, xl, xs, limit, size)
    max_bars: int
    member_bar: np.ndarray     # bool per (cut) strategy bar: decision month is a member month of this stage
    bar_rank: np.ndarray       # rank in the decision month (99 = not a member)
    bar_time: np.ndarray       # decision (bar close) time, ms
    slip: np.ndarray           # per-minute slippage at 1x
    force_exit: np.ndarray
    delisted: bool
    is_major: bool
    is_meme: bool
    member_months: int


def membership(universe: pd.DataFrame, stage: Stage) -> dict[str, dict[str, int]]:
    u = universe[universe["month"].isin(stage.months)]
    return {s: dict(zip(g["month"], g["rank"])) for s, g in u.groupby("symbol")}


def load_market(sym: str, stage: Stage, spec: Spec) -> tuple[engine.Market, dict]:
    m1 = store.load_1m(stage.period, sym)
    m1 = m1[m1["ts"] < int(stage.end.value // 10**6)].reset_index(drop=True)
    fund = store.load_funding(stage.period, sym)
    flags = {"funding_filled_h": 0.0}
    if spec.clean_data:
        m1 = store.trading_span(m1)
        if not len(m1):
            return None, flags
        fund, flags["funding_filled_h"] = store.fill_funding_gaps(
            fund, int(m1["ts"].iloc[0]), int(m1["ts"].iloc[-1]), Costs.synthetic_funding)
    mk = engine.Market(m1, fund, None, synthetic_funding=Costs.synthetic_funding)
    flags["last_ts"] = int(mk.ts[-1])
    return mk, flags


def make_sleeve(sym: str, stage: Stage, months: dict[str, int], cat_row: dict | None, spec: Spec,
                codes=CODES) -> Sleeve | None:
    mk, flags = load_market(sym, stage, spec)
    if mk is None:
        return None
    tf = BY_CODE[codes[0]].tf
    sigs_full = {c: BY_CODE[c].compute(mk, **PARAMS) for c in codes}
    ex = mk.exec_idx(tf, spec.latency_min)
    bars = mk.bars(tf)
    bar_time = bars["ts"].to_numpy(np.int64) + TF_MS[tf]
    bar_month = pd.to_datetime(bar_time, unit="ms", utc=True).strftime("%Y-%m").to_numpy()
    in_stage = (bar_time >= stage.start.value // 10**6) & (bar_time < stage.end.value // 10**6)
    rank_full = np.array([months.get(m, 99) for m in bar_month], dtype=np.int64)
    member_full = in_stage & (rank_full < 99)
    possible = np.flatnonzero(member_full & (ex < len(mk.ts)))
    if not len(possible):
        return None
    s0 = int(ex[possible[0]])
    b0 = int(np.searchsorted(ex, s0, side="left"))
    cut = slice(s0, None)
    marks = mk.day_marks[mk.day_marks >= s0]
    slim = SlimMarket(np.ascontiguousarray(mk.ts[cut]), np.ascontiguousarray(mk.o[cut]),
                      np.ascontiguousarray(mk.h[cut]), np.ascontiguousarray(mk.l[cut]),
                      np.ascontiguousarray(mk.c[cut]), np.ascontiguousarray(mk.fund[cut]),
                      (marks - s0).astype(np.int64), mk.days[mk.day_marks >= s0],
                      np.ascontiguousarray(ex[b0:] - s0), tf, spec.latency_min)
    n_tf = len(ex)
    sigs = {}
    for c, sg in sigs_full.items():
        sigs[c] = tuple(np.ascontiguousarray(a[b0:]) for a in sg.arrays(n_tf))
    minute_month = pd.to_datetime(slim.ts, unit="ms", utc=True).strftime("%Y-%m").to_numpy()
    um, inv = np.unique(minute_month, return_inverse=True)
    slip = np.array([spec.slip(sym, months.get(m)) for m in um])[inv]
    status = (cat_row or {}).get("status", "")
    delisted = bool(spec.clean_data and flags["last_ts"] < int((stage.end - DAY).value // 10**6)
                    and status != "TRADING")
    if spec.clean_data and flags["last_ts"] < int((stage.end - DAY).value // 10**6) and not delisted:
        raise RuntimeError(f"{sym}: data ends {pd.Timestamp(flags['last_ts'], unit='ms', tz='UTC')} "
                           f"but the snapshot says {status!r}; fix the data before running")
    fx = np.zeros(len(slim.ts), np.bool_)
    if delisted:
        fx[-1] = True
    return Sleeve(sym, slim, sigs, int(sigs_full[codes[0]].max_bars), np.ascontiguousarray(member_full[b0:]),
                  np.ascontiguousarray(rank_full[b0:]), np.ascontiguousarray(bar_time[b0:]), slip, fx,
                  delisted, sym in SYMBOLS, bool((cat_row or {}).get("is_meme", False)), len(months))


def run_sleeve(sl: Sleeve, code: str, masked, spec: Spec, mult: float = 1.0,
               stop_mode: int = 0) -> engine.Result:
    entry, stop, target, trail, xl, xs, limit, size = sl.sigs[code]
    entry = entry.copy()
    entry[~sl.member_bar] = 0
    if len(masked):
        entry[np.fromiter(masked, np.int64)] = 0
    sig = engine.Signals(sl.mk._tf, entry, stop, target, trail, xl, xs, limit, sl.max_bars, size)
    costs = Costs(taker=spec.taker * mult, maker=spec.maker * mult, slippage=0.0)
    res = engine.run(sl.mk, sig, costs, spec.risk(), latency_min=spec.latency_min,
                     slippage=sl.slip * mult, stop_mode=stop_mode, force_exit=sl.force_exit,
                     force_penalty=spec.delist_penalty)
    res.trades["symbol"] = sl.symbol
    res.trades["rank"] = sl.bar_rank[res.trades["entry_tf"].to_numpy()] if len(res.trades) else []
    return res


def sleeve_daily(res: engine.Result, calendar: pd.DatetimeIndex) -> pd.Series:
    """Daily returns on the stage calendar; days before the sleeve starts or after its data ends are 0."""
    eq = res.equity.copy()
    eq.index = eq.index.as_unit("ms")
    eq = eq.reindex(calendar).ffill().fillna(res.initial_equity)
    prev = eq.shift(1)
    prev.iloc[0] = res.initial_equity
    return eq / prev - 1.0


# ---------------------------------------------------------------- the position cap

_AT_OPEN = {"exit_signal", "time", "reverse"}   # these exits fill at the open, before new entries


def _slot_events(results: dict[str, engine.Result], sleeves: dict[str, Sleeve]) -> pd.DataFrame:
    rows = []
    for sym, r in results.items():
        t = r.trades
        if not len(t):
            continue
        ts = sleeves[sym].mk.ts
        te = ts[t["entry_i"].to_numpy()]
        xe = ts[t["exit_i"].to_numpy()] + np.where(t["reason"].isin(_AT_OPEN).to_numpy(), 0, MINUTE_MS)
        rows.append(pd.DataFrame({"symbol": sym, "te": te, "xe": xe, "bar": t["entry_tf"].to_numpy(),
                                  "rank": t["rank"].to_numpy(), "dir": t["dir"].to_numpy(),
                                  "R": t["R"].to_numpy()}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["symbol", "te", "xe", "bar", "rank", "dir", "R"])


def first_violation(ev: pd.DataFrame, slots: int):
    """Earliest entry minute at which more positions would open than there are free slots.
    A slot is busy over [entry, effective exit)."""
    if not len(ev):
        return None
    te = np.sort(ev["te"].to_numpy())
    xe = np.sort(ev["xe"].to_numpy())
    T, n_at = np.unique(te, return_counts=True)
    open_ = np.searchsorted(te, T, side="left") - np.searchsorted(xe, T, side="right")
    bad = np.flatnonzero(n_at > slots - open_)
    if not len(bad):
        return None
    k = bad[0]
    return int(T[k]), int(slots - open_[k])


def cap_pass(sleeves: dict[str, Sleeve], code: str, spec: Spec, max_iter: int = 5000):
    """Exact greedy cap: walk forward to the first minute where entries exceed the free slots,
    admit the best-ranked (then alphabetical) ones, mask the rest and re-run only those sleeves.
    Masks only grow and each sleeve's trades before a mask are unchanged by it, so the fixed
    point equals a step-by-step simulation of one shared account with `slots` positions."""
    masks = {s: set() for s in sleeves}
    results = {s: run_sleeve(sl, code, masks[s], spec) for s, sl in sleeves.items()}
    skipped = []
    it = 0
    while True:
        ev = _slot_events(results, sleeves)
        v = first_violation(ev, spec.slots)
        if v is None:
            break
        it += 1
        if it > max_iter:
            raise RuntimeError("cap pass did not converge")
        t, free = v
        at = ev[ev["te"] == t].sort_values(["rank", "symbol"])
        drop = at.iloc[max(free, 0):]
        for _, d in drop.iterrows():
            masks[d["symbol"]].add(int(d["bar"]))
            skipped.append({"symbol": d["symbol"], "decision_exec": pd.Timestamp(int(t), unit="ms", tz="UTC"),
                            "dir": int(d["dir"]), "rank": int(d["rank"]), "free_slots": free,
                            "R_if_taken": float(d["R"])})
        for s in drop["symbol"].unique():
            results[s] = run_sleeve(sleeves[s], code, masks[s], spec)
    return results, masks, pd.DataFrame(skipped), it


def concurrency(ev: pd.DataFrame, slots: int, start, end) -> dict:
    if not len(ev):
        return {"max": 0, "mean": 0.0, "share_at_cap": 0.0}
    pts = np.r_[ev["te"].to_numpy(), ev["xe"].to_numpy()]
    dlt = np.r_[np.ones(len(ev)), -np.ones(len(ev))]
    o = np.lexsort((dlt, pts))           # exits before entries at the same minute
    pts, dlt = pts[o], dlt[o]
    level = np.cumsum(dlt)
    t0, t1 = int(start.value // 10**6), int(end.value // 10**6)
    edges = np.r_[t0, np.clip(pts, t0, t1), t1]
    lv = np.r_[0, level]
    dur = np.diff(edges)
    total = max(t1 - t0, 1)
    return {"max": int(level.max()), "mean": float((lv * dur).sum() / total),
            "share_at_cap": float(dur[lv >= slots].sum() / total)}


# ---------------------------------------------------------------- evaluation

def _null_for(sl: Sleeve, trades: pd.DataFrame, spec: Spec) -> np.ndarray:
    if not len(trades):
        return np.zeros(N_PERM)
    ex = sl.mk._ex
    cand = ex[sl.member_bar & (ex < len(sl.mk.ts) - 1)]
    cost = 2 * spec.taker + float((sl.slip[trades["entry_i"].to_numpy()] + sl.slip[trades["exit_i"].to_numpy()]).mean())
    return stats.random_entry_null(sl.mk.o, cand, (trades["exit_i"] - trades["entry_i"]).to_numpy(),
                                   trades["dir"].to_numpy(), cost, N_PERM, seed=spec.seed(sl.symbol))


def ew_benchmark(sleeves: dict[str, Sleeve], universe: pd.DataFrame, stage: Stage) -> pd.Series:
    """Equal-weight buy-and-hold of the account's symbols that are members in each day's month."""
    cal = stage.calendar
    mem = membership(universe, stage)
    rets = {}
    for s, sl in sleeves.items():
        c = pd.Series(sl.mk.c[sl.mk.day_marks], index=sl.mk.days.as_unit("ms"))
        r = c.pct_change().reindex(cal)
        month = cal.strftime("%Y-%m")
        r[~np.isin(month, list(mem.get(s, {})))] = np.nan
        rets[s] = r
    if not rets:
        return pd.Series(0.0, index=cal)
    return pd.DataFrame(rets).mean(axis=1).fillna(0.0)


def _sym_table(trades: pd.DataFrame, sleeves: dict[str, Sleeve], daily_by: dict[str, pd.Series]) -> list[dict]:
    rows = []
    for s, sl in sleeves.items():
        t = trades[trades["symbol"] == s]
        d = daily_by.get(s)
        rows.append({"symbol": s, "member_months": sl.member_months, "trades": len(t),
                     "avg_R": float(t["R"].mean()) if len(t) else 0.0,
                     "sum_R": float(t["R"].sum()) if len(t) else 0.0,
                     "win_rate": float((t["R"] > 0).mean()) if len(t) else 0.0,
                     "long_avg_R": float(t.loc[t["dir"] == 1, "R"].mean()) if (t["dir"] == 1).any() else 0.0,
                     "short_avg_R": float(t.loc[t["dir"] == -1, "R"].mean()) if (t["dir"] == -1).any() else 0.0,
                     "delist_exits": int((t["reason"] == "delist").sum()),
                     "is_meme": sl.is_meme, "is_major": sl.is_major, "delisted": sl.delisted,
                     "sharpe": float(d.mean() / d.std() * np.sqrt(365)) if d is not None and d.std() > 0 else 0.0})
    return rows


def _subset(trades: pd.DataFrame, mask) -> dict:
    t = trades[mask]
    tot = trades["R"].sum()
    return {"trades": int(len(t)), "avg_R": float(t["R"].mean()) if len(t) else 0.0,
            "sum_R": float(t["R"].sum()), "share_of_R": float(t["R"].sum() / tot) if tot else None}


def evaluate(code: str, stage: Stage, sleeves: dict[str, Sleeve], universe: pd.DataFrame,
             spec: Spec = SPEC, with_stress: bool = True) -> dict:
    """One account (a set of sleeves) for one code and stage: cap pass, stress runs, statistics."""
    t0 = time.time()
    cal = stage.calendar
    results, masks, skipped, iters = cap_pass(sleeves, code, spec)
    parts, daily_by = [], {}
    for s, r in results.items():
        d = sleeve_daily(r, cal)
        daily_by[s] = d
        parts.append((d, r.trades))
    res, daily = assemble_slots(parts, spec.slots, cal)
    out = {"summary": metrics.summarize(res), "iterations": iters, "masks": {s: sorted(m) for s, m in masks.items() if m}}
    t = res.trades
    if with_stress:
        for name, kw in (("stress_2x", {"mult": 2.0}), ("worst_minute", {"stop_mode": 1})):
            st = {s: run_sleeve(sleeves[s], code, masks[s], spec, **kw) for s in sleeves}
            same = all(np.array_equal(st[s].trades[["entry_i", "exit_i"]].to_numpy(),
                                      results[s].trades[["entry_i", "exit_i"]].to_numpy()) for s in sleeves)
            sres, _ = assemble_slots([(sleeve_daily(r, cal), r.trades) for r in st.values()], spec.slots, cal)
            out[name] = {k: metrics.summarize(sres).get(k) for k in ("trades", "avg_R", "pf", "total_return",
                                                                     "sharpe")} | {"timing_identical": same}
            out[name + "_trades"] = sres.trades
    nulls, counts = [], []
    for s, sl in sleeves.items():
        tr = results[s].trades
        nulls.append(_null_for(sl, tr, spec))
        counts.append(len(tr))
    null = combine_null(nulls, counts)
    actual = float(((t["gross"] - t["fees"]) / (t["qty"] * t["entry_px"])).mean()) if len(t) else 0.0
    boot_tot, boot_mdd = stats.block_bootstrap(t["ret"].to_numpy(), N_BOOT) if len(t) > 1 \
        else (np.zeros(1), np.zeros(1))
    sym_rows = _sym_table(t, sleeves, daily_by)
    elig = [r for r in sym_rows if r["trades"] >= MIN_SYM_TRADES]
    meme = t["symbol"].isin([s for s, sl in sleeves.items() if sl.is_meme])
    dlst = t["symbol"].isin([s for s, sl in sleeves.items() if sl.delisted])
    major = t["symbol"].isin(SYMBOLS)
    bench = ew_benchmark(sleeves, universe, stage)
    out |= {
        "daily": daily, "trades": t, "res": res, "null_dist": null, "boot_tot": boot_tot, "boot_mdd": boot_mdd,
        "null_counts": np.asarray(counts), "benchmark_daily": bench,
        "sr_moments": stats.sharpe_moments(daily),
        "weekly": stats.weekly_t(daily),
        "null": {"actual_mean_ret": actual, "null_mean": float(null.mean()),
                 "null_p95": float(np.quantile(null, 0.95)),
                 "p_value": float((1 + np.sum(null >= actual)) / (1 + len(null)))},
        "bootstrap": {"ret_p5": float(np.quantile(boot_tot, 0.05)), "ret_p50": float(np.quantile(boot_tot, 0.5)),
                      "mdd_p50": float(np.quantile(boot_mdd, 0.5)), "mdd_p95": float(np.quantile(boot_mdd, 0.95))},
        "symbol_table": sym_rows,
        "consistency": {"eligible": len(elig), "positive": sum(r["avg_R"] > 0 for r in elig),
                        "share": (sum(r["avg_R"] > 0 for r in elig) / len(elig)) if elig else 0.0},
        "meme_info": _subset(t, meme) if len(t) else {},
        "delisted_info": _subset(t, dlst) | {"forced_exits": int((t["reason"] == "delist").sum())} if len(t) else {},
        "majors_info": _subset(t, major) if len(t) else {},
        "years": metrics.year_table(res).reset_index(names="year").to_dict("records") if len(t) else [],
        "concurrency": concurrency(_slot_events(results, sleeves), spec.slots, stage.start, stage.end),
        "skipped": skipped,
        "skipped_info": {"entries": int(len(skipped)),
                         "R_if_taken_mean": float(skipped["R_if_taken"].mean()) if len(skipped) else None},
        "benchmark": _bench_summary(bench),
        "n_symbols": len(sleeves), "runtime_s": time.time() - t0,
    }
    return out


def _bench_summary(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    return {"total_return": float(eq.iloc[-1] - 1) if len(eq) else 0.0,
            "sharpe": float(r.mean() / r.std() * np.sqrt(365)) if r.std() > 0 else 0.0,
            "max_dd": metrics.max_drawdown(np.r_[1.0, eq.to_numpy()]) if len(eq) else 0.0}


# ---------------------------------------------------------------- gates and verdicts

def gates_a(ev: dict, s1: dict) -> dict:
    d = ev["summary"]
    return {
        "trades>=100": d["trades"] >= GATES["min_trades"],
        "trades/yr>=52": d.get("trades_per_year", 0) >= MIN_FREQ,
        "PF>=1.2": d.get("pf", 0) >= GATES["min_pf"],
        "weekly t>=2": ev["weekly"]["t"] >= GATES["min_t"],
        "2x-cost avg_R>0": (ev["stress_2x"].get("avg_R") or 0) > 0,
        "worst-minute stops avg_R>0": (ev["worst_minute"].get("avg_R") or 0) > 0,
        "null p<0.05": ev["null"]["p_value"] < GATES["max_null_p"],
        "year share<=50%": (d.get("max_year_share") or np.inf) <= GATES["max_year_share"],
        ">=60% symbols (10+ trades) R>0": ev["consistency"]["share"] >= MIN_CONSISTENCY,
        "PBO<=0.5 (S1)": s1["pbo"] <= GATES["max_pbo"],
        "plateau>=70% (S1)": s1["plateau"] >= GATES["min_plateau"],
    }


def gates_b(evb: dict, eva: dict) -> dict:
    b, a = evb["summary"], eva["summary"]
    n_b = max(b["trades"], 2)
    _, mdd_dist = stats.block_bootstrap(eva["trades"]["ret"].to_numpy(), N_BOOT, n_out=n_b)
    mdd = trade_equity_mdd(evb["trades"]["ret"].to_numpy()) if b["trades"] else 0.0
    evb["trade_mdd"], evb["boot_mdd_p95_from_A"] = mdd, float(np.quantile(mdd_dist, 0.95))
    return {
        "B trades/yr>=52": b.get("trades_per_year", 0) >= MIN_FREQ,
        "B avg_R>0": b.get("avg_R", 0) > 0,
        "B Sharpe>=50% of A": b.get("sharpe", 0) >= GATES["holdout_sharpe_frac"] * a.get("sharpe", 0),
        "B trade MDD<=A boot p95": mdd <= evb["boot_mdd_p95_from_A"],
        "B weekly Sharpe>0": evb["weekly"]["sharpe_ann"] > 0,
    }


def headline(daily_ab: pd.Series, n_eff: float, a_pass: bool, b_pass: bool) -> dict:
    sr, T, sk, ku = stats.sharpe_moments(daily_ab)
    psr0 = stats.psr(sr, T, sk, ku)
    dsr = stats.dsr(sr, T, sk, ku, stats.null_sr_variance(T), max(int(round(n_eff)), 2))
    replicated = bool(a_pass and b_pass and psr0 >= GATES["min_dsr"])
    return {"window": [str(daily_ab.index[0].date()), str(daily_ab.index[-1].date())], "T": T,
            "sharpe_ann": float(sr * np.sqrt(365)), "psr0": psr0, "dsr": dsr, "n_eff": n_eff,
            "sharpe_needed_for_dsr_ann": float(stats.expected_max_sr(stats.null_sr_variance(T),
                                                                     max(int(round(n_eff)), 2)) * np.sqrt(365)),
            "REPLICATED": replicated, "STRICT": bool(replicated and dsr >= GATES["min_dsr"])}


def u1l_rule(u1: dict, u1l: dict) -> dict:
    a1, a2 = u1["alpha"], u1l["alpha"]
    ok = bool(u1l["headline"]["REPLICATED"] and a2["psr_alpha"] >= GATES["min_dsr"]
              and a2["sharpe_alpha_ann"] > a1["sharpe_alpha_ann"])
    return {"U1L_replaces_U1": ok, "U1_alpha_sharpe": a1["sharpe_alpha_ann"],
            "U1L_alpha_sharpe": a2["sharpe_alpha_ann"], "U1L_alpha_psr": a2["psr_alpha"]}


# ---------------------------------------------------------------- outputs

def _public(ev: dict) -> dict:
    keep = ("summary", "iterations", "stress_2x", "worst_minute", "weekly", "null", "bootstrap", "symbol_table",
            "consistency", "meme_info", "delisted_info", "majors_info", "years", "concurrency", "skipped_info",
            "benchmark", "n_symbols", "runtime_s", "trade_mdd", "boot_mdd_p95_from_A")
    out = {k: ev[k] for k in keep if k in ev}
    sr, T, sk, ku = ev["sr_moments"]
    out |= {"sr_daily": sr, "T": T, "skew": sk, "kurt": ku}
    return out


def _write_stage(code: str, tag: str, ev: dict) -> None:
    """tag: 'dev' (Stage A gated), 'full_dev' (Stage A full universe), 'holdout', 'holdout_full'."""
    ev["trades"].to_parquet(RESULTS / f"{code}_{tag}_trades.parquet")
    ev["res"].equity.rename("equity").to_frame().to_parquet(RESULTS / f"{code}_{tag}_equity.parquet")
    ev["daily"].rename("ret").to_frame().to_parquet(RESULTS / f"{code}_{tag}_daily.parquet")
    if len(ev["skipped"]):
        ev["skipped"].to_parquet(RESULTS / f"{code}_{tag}_skipped.parquet")
    for k in ("stress_2x_trades", "worst_minute_trades"):
        if k in ev:
            ev[k].to_parquet(RESULTS / f"{code}_{tag}_{k.removesuffix('_trades')}_trades.parquet")


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _log_run(kind: str, **extra) -> None:
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    rec = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "kind": kind,
           "git": _git("rev-parse", "HEAD"), "dirty": bool(_git("status", "--porcelain", "--", "src"))} | extra
    with RUN_LOG.open("a", newline="\n") as f:
        f.write(json.dumps(rec) + "\n")


def _require_committed(paths) -> None:
    rel = [str(p.relative_to(ROOT)).replace("\\", "/") for p in paths]
    missing = [r for r in rel if not (ROOT / r).exists()]
    if missing:
        raise SystemExit(f"missing: {missing}")
    if _git("status", "--porcelain", "--", *rel) or not _git("ls-files", "--", *rel):
        raise SystemExit(f"commit these first (pre-registration): {rel}")


def _exchange_info_path():
    snaps = sorted(U.ROUND4_DIR.glob("exchange_info_um_*.json"))
    if not snaps:
        raise SystemExit("no exchangeInfo snapshot in research/round4/")
    return snaps[-1]


def _catalogue() -> pd.DataFrame:
    return pd.read_csv(SYMBOLS_CSV, keep_default_na=False)


# ---------------------------------------------------------------- commands

def cmd_symbols(exinfo_path=None) -> pd.DataFrame:
    exinfo_path = exinfo_path or _exchange_info_path()
    cat = U.catalogue(U.bv.list_um_symbols(), U.load_exchange_info(exinfo_path), U.load_overrides(OVERRIDES_CSV))
    U.write_csv(cat, SYMBOLS_CSV)
    kept = cat[~cat["excluded"]]
    print(f"{len(cat)} archived symbols; kept {len(kept)}; excluded by reason:")
    print(cat[cat["excluded"]]["exclude_reason"].str.replace(r" \(.*", "", regex=True).value_counts().to_string())
    absent = kept[~kept["in_exchange_info"]]["symbol"].tolist()
    print(f"{len(absent)} kept symbols are not in the exchangeInfo snapshot (review for overrides):")
    print(" ".join(absent))
    return cat


def cmd_universe() -> pd.DataFrame:
    _require_committed([PREREG, SYMBOLS_CSV, OVERRIDES_CSV, _exchange_info_path()])
    if "## Round 4" not in PREREG.read_text(encoding="utf-8"):
        raise SystemExit("PREREGISTRATION.md has no Round 4 section")
    cat = _catalogue()
    kept = cat[cat["excluded"].astype(str) == "False"]["symbol"].tolist()
    qv = pd.read_parquet(U.QV_PATH) if U.QV_PATH.exists() else U.download_daily_qv(kept)
    cat["excluded"] = cat["excluded"].astype(str) == "True"
    cat["is_meme"] = cat["is_meme"].astype(str) == "True"
    cat["is_major"] = cat["is_major"].astype(str) == "True"
    uni = U.build_universe(qv, cat, U.month_starts())
    U.write_csv(uni, UNIVERSE_CSV)
    import hashlib
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    meta = {"universe_id": UNIVERSE_ID, "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git": _git("rev-parse", "HEAD"), "params": {"top_n": 20, "lookback_days": 90, "min_age_days": 90,
                                                         "recent_days": 7, "months": ["2020-04", "2026-09"]},
            "inputs_sha256": {p.name: sha(p) for p in (SYMBOLS_CSV, OVERRIDES_CSV, _exchange_info_path())},
            "universe_sha256": sha(UNIVERSE_CSV), "qv_symbols": int(qv["symbol"].nunique()),
            "qv_rows": int(len(qv))}
    for st in STAGES.values():
        mem = uni[uni["month"].isin(st.months)]
        meta[f"stage_{st.name}"] = {"symbols": sorted(mem["symbol"].unique()),
                                    "gated_symbols": int(mem[mem["gated"]]["symbol"].nunique()),
                                    "min_members": int(mem.groupby("month").size().min())}
    UNIVERSE_META.write_text(json.dumps(meta, indent=1) + "\n", newline="\n")
    print(f"universe: {uni['month'].nunique()} months, {uni['symbol'].nunique()} distinct symbols; "
          f"Stage A {len(meta['stage_A']['symbols'])} symbols, Stage B {len(meta['stage_B']['symbols'])}")
    return uni


def _member_minutes_coverage(sym: str, months: list[str]) -> tuple[float, float]:
    """(worst 1m coverage over the symbol's member months, share of member hours with a >9h funding gap)."""
    m1 = pd.read_parquet(store.symbol_dir(sym) / "perp_1m.parquet", columns=["ts", "volume"])
    f = pd.read_parquet(store.symbol_dir(sym) / "funding.parquet")["ts"].to_numpy(np.int64)
    ts = m1["ts"].to_numpy(np.int64)
    last = int(ts[np.flatnonzero(m1["volume"].to_numpy() > 0)[-1]])
    worst, gap_h, tot_h = 1.0, 0.0, 0.0
    for m in months:
        a = pd.Timestamp(m + "-01", tz="UTC")
        b = a + pd.offsets.MonthBegin(1)
        a_ms, b_ms = int(a.value // 10**6), min(int(b.value // 10**6), last + MINUTE_MS)
        if b_ms <= a_ms:
            continue
        n = np.searchsorted(ts, b_ms) - np.searchsorted(ts, a_ms)
        worst = min(worst, n / ((b_ms - a_ms) / MINUTE_MS))
        e = np.r_[a_ms, f[(f > a_ms) & (f < b_ms)], b_ms]
        d = np.diff(e)
        gap_h += d[d > 9 * 3_600_000].sum() / 3_600_000
        tot_h += (b_ms - a_ms) / 3_600_000
    return worst, (gap_h / tot_h if tot_h else 0.0)


def cmd_download(stage_name: str, keep_raw: bool = False) -> pd.DataFrame:
    st = STAGES[stage_name]
    if stage_name == "B":
        a = RESULTS / f"{PRIMARY}.json"
        if not (a.exists() and json.loads(a.read_text()).get("dev_pass")):
            raise SystemExit("Stage B data is only downloaded after U1 passed Stage A")
    uni = U.load_universe(UNIVERSE_CSV)
    cat = _catalogue().set_index("symbol")
    mem = membership(uni, st)
    rows = []
    for i, sym in enumerate(sorted(mem), 1):
        rep_p = store.symbol_dir(sym) / "build_report.json"
        if sym in SYMBOLS and (store.symbol_dir(sym) / "perp_1m.parquet").exists():
            rep = {"symbol": sym, "status": "ok (round 1-3 store)"}
        elif rep_p.exists():
            rep = json.loads(rep_p.read_text())
        else:
            print(f"[{i}/{len(mem)}] building {sym}", flush=True)
            rep = store.build_perp_only(sym, keep_raw=keep_raw)
        months = sorted(mem[sym])
        cov, fgap = _member_minutes_coverage(sym, months) if rep.get("status", "").startswith("ok") else (0.0, 1.0)
        status = cat.loc[sym, "status"] if sym in cat.index else ""
        last = rep.get("last_traded") or rep.get("last")
        ends_early = bool(last and pd.Timestamp(last) < st.end - 2 * DAY and status == "TRADING")
        rows.append({"symbol": sym, "member_months": len(months), "first_month": months[0], "last_month": months[-1],
                     "status": rep.get("status"), "snapshot_status": status, "last_traded": last,
                     "min_month_coverage": round(cov, 4), "funding_gap_share": round(fgap, 4),
                     "ends_while_trading": ends_early,
                     **{k: rep.get(k) for k in ("rows", "gaps_over_24h", "max_gap_h", "extreme_bars",
                                                "ohlc_violations", "bars_4h_mismatch_share", "zero_volume_tail_min")}})
    df = pd.DataFrame(rows)
    df["red_flag"] = ((df["min_month_coverage"] < 0.9) | (df["funding_gap_share"] > 0.05)
                      | df["ends_while_trading"] | ~df["status"].astype(str).str.startswith("ok"))
    U.write_csv(df, U.ROUND4_DIR / f"data_report_{stage_name}.csv")
    bad = df[df["red_flag"]]
    print(f"Stage {stage_name}: {len(df)} symbols built; {len(bad)} red flags")
    if len(bad):
        print(bad[["symbol", "status", "min_month_coverage", "funding_gap_share", "ends_while_trading"]].to_string())
        raise SystemExit("data red flags: fix the data before any strategy run")
    return df


def _stage_sleeves(st: Stage, uni: pd.DataFrame, spec: Spec = SPEC) -> dict[str, Sleeve]:
    cat = _catalogue().set_index("symbol")
    cat["is_meme"] = cat["is_meme"].astype(str) == "True"
    mem = membership(uni, st)
    out = {}
    for sym in sorted(mem):
        row = cat.loc[sym].to_dict() if sym in cat.index else None
        sl = make_sleeve(sym, st, mem[sym], row, spec)
        if sl is not None:
            out[sym] = sl
    return out


def cmd_sweep(codes=CODES, rerun: bool = False) -> None:
    _require_committed([PREREG, SYMBOLS_CSV, OVERRIDES_CSV, UNIVERSE_CSV, U.ROUND4_DIR / "data_report_A.csv"])
    for c in codes:
        if (RESULTS / f"{c}.json").exists() and not rerun:
            raise SystemExit(f"{c}.json exists; Stage A runs once (use --rerun only with a version bump)")
    st = STAGES["A"]
    uni = U.load_universe(UNIVERSE_CSV)
    sleeves = _stage_sleeves(st, uni)
    gated = {s: sl for s, sl in sleeves.items() if not sl.is_major}
    print(f"Stage A: {len(sleeves)} sleeves ({len(gated)} gated)", flush=True)
    s1 = json.loads((RESULTS / "S1.json").read_text())
    rows, out = [], {}
    for code in codes:
        strat = BY_CODE[code]
        ev = evaluate(code, st, gated, uni)
        evf = evaluate(code, st, sleeves, uni, with_stress=False)
        g = gates_a(ev, s1)
        sr, T, sk, ku = ev["sr_moments"]
        r = {"code": code, "name": strat.name, "tf": strat.tf, "version": strat.version, "round": 4,
             "hypothesis": strat.hypothesis, "rules": strat.rules, "grid": strat.grid, "n_configs": 1,
             "chosen_cfg": 0, "params": PARAMS, "universe_id": UNIVERSE_ID, "spec": SPEC.__dict__ | {"slip_by_symbol": {}},
             "holdout_scope": "gated", "latency_min": SPEC.latency_min,
             "start": str(st.start.date()), "end": str((st.end - DAY).date()),
             "dev": ev["summary"], "dev_detail": _public(ev), "dev_full": _public(evf),
             "stress_2x": ev["stress_2x"], "worst_minute": ev["worst_minute"],
             "sr_daily": sr, "T": T, "skew": sk, "kurt": ku, "pbo": s1["pbo"], "plateau": s1["plateau"],
             "inherited_gates": ["PBO", "plateau"], "null": ev["null"], "bootstrap": ev["bootstrap"],
             "weekly": ev["weekly"], "consistency": ev["consistency"], "gates": g, "dev_pass": all(g.values())}
        out[code] = (r, ev, evf)
        rows.append({"strategy": code, "version": strat.version,
                     "params": PARAMS | {"universe": UNIVERSE_ID, "slots": SPEC.slots, "risk": SPEC.account_risk},
                     "period": "dev", "sr": sr, "T": T, "skew": sk, "kurt": ku, "trades": ev["summary"]["trades"],
                     "sharpe_ann": ev["summary"].get("sharpe", 0.0)})
        ev["daily"].rename(f"{code}_0").to_frame().to_parquet(RESULTS / f"{code}_trial_returns.parquet")
    registry.record(rows)
    from btcq.research.run import all_trial_returns, cmd_gates
    trial = all_trial_returns()
    n_trials, _ = registry.multiple_testing_inputs("dev")
    n_eff = registry.ratcheted_effective_trials(stats.effective_trials(trial.to_numpy()), trial.shape[1],
                                                note="round 4 (U1, U1L)")
    for code, (r, ev, evf) in out.items():
        sr, T, sk, ku = ev["sr_moments"]
        r["n_trials"], r["n_eff"] = n_trials, n_eff
        r["psr0_dev"] = stats.psr(sr, T, sk, ku)
        r["dsr_dev"] = stats.dsr(sr, T, sk, ku, stats.null_sr_variance(T), max(int(round(n_eff)), 2))
        _write_stage(code, "dev", ev)
        _write_stage(code, "full_dev", evf)
        ev["benchmark_daily"].rename("ret").to_frame().to_parquet(RESULTS / f"{code}_dev_benchmark.parquet")
        ev["trades"].to_parquet(RESULTS / f"{code}_dev_scope_trades.parquet")
        pd.DataFrame([{"cfg": 0, **PARAMS, **ev["summary"], "sr": r["sr_daily"], "nbhd_sharpe": ev["summary"].get("sharpe")}]) \
            .to_csv(RESULTS / f"{code}_configs.csv", index=False)
        np.savez_compressed(RESULTS / f"{code}_dev_dists.npz", null=ev["null_dist"], boot_tot=ev["boot_tot"],
                            boot_mdd=ev["boot_mdd"], pbo_logits=np.array([]), null_counts=ev["null_counts"])
        _save_json(RESULTS / f"{code}.json", r)
        d = ev["summary"]
        failed = [k for k, v in r["gates"].items() if not v]
        print(f"{code} Stage A (gated, {ev['n_symbols']} symbols): trades={d['trades']} "
              f"({d.get('trades_per_year', 0):.0f}/yr) avgR={d.get('avg_R', 0):.3f} PF={d.get('pf', 0):.2f} "
              f"sharpe={d.get('sharpe', 0):.2f} weekly t={ev['weekly']['t']:.2f} null p={ev['null']['p_value']:.3f} "
              f"-> {'PASS' if r['dev_pass'] else 'FAIL: ' + ', '.join(failed)}", flush=True)
    _log_run("r4-sweep", codes=list(codes), n_eff=n_eff, n_trials=n_trials,
             verdict={c: out[c][0]["dev_pass"] for c in out})
    cmd_gates()


def preflight_b() -> list[str]:
    """Reasons Stage B may not be unlocked (empty = ok). Runs before anything touches the holdout."""
    probs = []
    p = RESULTS / f"{PRIMARY}.json"
    r = json.loads(p.read_text()) if p.exists() else {}
    if not r.get("dev_pass"):
        probs.append("U1 did not pass Stage A")
    for c in CODES:
        q = RESULTS / f"{c}.json"
        if q.exists() and "holdout" in json.loads(q.read_text()):
            probs.append(f"{c} holdout already evaluated")
    log = store.HOLDOUT_LOG.read_text() if store.HOLDOUT_LOG.exists() else ""
    if "[R4" in log:
        probs.append("holdout_log already has a round 4 unlock")
    rep = U.ROUND4_DIR / "data_report_B.csv"
    if not rep.exists():
        probs.append("Stage B data report missing (run r4-download --stage B)")
    else:
        df = pd.read_csv(rep)
        if df["red_flag"].astype(str).eq("True").any():
            probs.append("Stage B data report has red flags")
        uni = U.load_universe(UNIVERSE_CSV)
        need = set(membership(uni, STAGES["B"]))
        if need - set(df["symbol"]):
            probs.append(f"Stage B symbols not built: {sorted(need - set(df['symbol']))[:5]}")
    return probs


def cmd_validate(reason: str) -> None:
    probs = preflight_b()
    if probs:
        raise SystemExit("holdout stays locked: " + "; ".join(probs))
    store.unlock_holdout(f"{reason} [R4: {','.join(CODES)}]")
    st = STAGES["B"]
    uni = U.load_universe(UNIVERSE_CSV)
    sleeves = _stage_sleeves(st, uni)
    gated = {s: sl for s, sl in sleeves.items() if not sl.is_major}
    res = {}
    for code in CODES:
        r = json.loads((RESULTS / f"{code}.json").read_text())
        evb = evaluate(code, st, gated, uni)
        evbf = evaluate(code, st, sleeves, uni, with_stress=False)
        _write_stage(code, "holdout", evb)
        _write_stage(code, "holdout_full", evbf)
        eva = {"summary": r["dev"], "trades": pd.read_parquet(RESULTS / f"{code}_dev_trades.parquet")}
        hg = gates_b(evb, eva)
        daily_a = pd.read_parquet(RESULTS / f"{code}_dev_daily.parquet")["ret"]
        daily_ab = pd.concat([daily_a, evb["daily"]]).sort_index()
        r["holdout"] = _public(evb)
        r["holdout_full"] = _public(evbf)
        r["holdout_gates"] = hg
        r["holdout_status"] = "clean (gated set; majors info only)"
        r["headline"] = headline(daily_ab, r["n_eff"], bool(r["dev_pass"]), all(hg.values()))
        bench_a = pd.read_parquet(RESULTS / f"{code}_dev_benchmark.parquet")["ret"]
        r["alpha"] = stats.beta_alpha(daily_ab, pd.concat([bench_a, evb["benchmark_daily"]]).sort_index())
        r["final_pass"] = r["headline"]["REPLICATED"]
        evb["benchmark_daily"].rename("ret").to_frame().to_parquet(RESULTS / f"{code}_holdout_benchmark.parquet")
        res[code] = {"r": r}
    res["U1L"]["r"]["u1l_rule"] = u1l_rule(res["U1"]["r"], res["U1L"]["r"])
    for code in CODES:
        _save_json(RESULTS / f"{code}.json", res[code]["r"])
        h, hd = res[code]["r"]["holdout"], res[code]["r"]["headline"]
        print(f"{code} Stage B: trades={h['summary']['trades']} avgR={h['summary'].get('avg_R', 0):.3f} "
              f"sharpe={h['summary'].get('sharpe', 0):.2f} gates={res[code]['r']['holdout_gates']} | "
              f"PSR0={hd['psr0']:.3f} DSR={hd['dsr']:.3f} REPLICATED={hd['REPLICATED']} STRICT={hd['STRICT']}")
    _log_run("r4-validate", headline={c: res[c]["r"]["headline"] for c in CODES})
