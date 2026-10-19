"""Execution study X1: maker-first ("post, then cross") execution of F1's entries and signal exits, and generic
maker cost curves for faster strategies, on dev data only (research/PREREGISTRATION.md, Execution study X1).

    python -m btcq.research.run x1-start                # twice: the start record, then a check that it is committed
    python -m btcq.research.run x1-stage-a              # once: F1 under every policy and 1m model, plus Q3 step 1
    python -m btcq.research.run x1-download --set B1    # tick archives (B2 only after a committed strict pass)
    python -m btcq.research.run x1-stage-b --set B1     # B1: the cost model X1-cost-v1; B2: check (f)
    python -m btcq.research.run x1-report

Nothing here loads data from HOLDOUT_START on: Market.load("dev") cuts it, every output is checked against it,
and the tick archives go through btcq.data.ticks' dev-only key and timestamp guards.
"""
from __future__ import annotations

import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from btcq.backtest import engine, metrics, queue_sim, stats
from btcq.backtest import maker_engine as ME
from btcq.config import HOLDOUT_START, MINUTE_MS, PERP_START, ROOT, SLIPPAGE, SYMBOLS, TF_MS, Costs
from btcq.data import ticks
from btcq.research import guards, registry
from btcq.research import run as research_run
from btcq.research.portfolio import assemble, sleeve_risk
from btcq.research.run import RESULTS, _save_json, aligned_daily
from btcq.strategies.r3_trend import S1Frozen

CODE = "F1X"
VERSION = 1
X1_DIR = ROOT / "research" / "x1"
RUN_LOG = X1_DIR / "runs.jsonl"
PREREG = ROOT / "research" / "PREREGISTRATION.md"
SECTION = "## Execution study X1"
REQUIRED_PHRASES = ("post, then cross", "T = 15", "strict", "Decision rules", "+0.5%/yr")
T_GRID = (5, 15, 60)
PRIMARY_T = 15
NOFALLBACK_T = (15, 60)
MODELS = ("strict", "central", "optimistic")
F1_PARAMS = {"n": 40, "k": 3.5, "exit": "channel"}
LATENCY_MIN = 1
START = PERP_START
MIN_GAIN = 0.005
BOOT_N = 10_000
BOOT_BLOCK = 30
BOOT_SEED = 11
SHARPE_TOL = 0.02
DD_TOL = 0.01
MIN_YEARS = 3
MIN_COINS = 4
F_TOL_BP = 0.5
Q3_SEED = 1102
Q3_RANDOM_PER_COIN = 8000
B1_SEED = 1101
EXCHANGE_INFO = ROOT / "research" / "round4" / "exchange_info_um_2026-10-16.json"
F1_DEV_DAILY = RESULTS / "F1_dev_daily.parquet"
CODE_FILES = [f"src/btcq/{p}" for p in (
    "backtest/maker_engine.py", "backtest/queue_sim.py", "data/ticks.py", "research/x1_execution.py",
    "backtest/engine.py", "backtest/stats.py", "backtest/metrics.py", "data/store.py", "data/bardata.py",
    "data/binance_vision.py", "config.py", "research/run.py", "research/registry.py", "research/guards.py",
    "research/portfolio.py", "research/round3.py", "strategies/s1_donchian.py", "strategies/r3_trend.py",
    "strategies/base.py", "features/ta.py")]

YEARS = (2020, 2021, 2022, 2023)
TAKER = "taker"
PRIMARY = f"strict_T{PRIMARY_T}"
PASS_DECISION = "B2 eligible (needs the owner's OK to download)"
FAIL_DECISION = "F1 NO-GO"
DECISION_EXITS = ("exit_signal", "time", "reverse")   # the exits a maker order may take; stops stay taker
TICK_SIG_DIGITS = 10
TICK_TOL = 1e-6                  # a gap is a whole multiple of the tick within this many ticks
TICK_CONSISTENT = 0.99
PARITY_TOL = 1e-12
SANITY_TOL = 1e-9
STRICT_FILL_SAVING_BP = 3.0      # a strict maker fill is at the taker price: only the 5 - 2 bp fee changes
CLASSES = ("random", "momentum", "reversion")
MOM_BARS = 24                    # 1h bars
REV_BARS = 720                   # 1h bars = 30 days
REV_Z = 2.0
MARKOUTS = (1, 5, 15, 60)        # 1m rows after the fill
Z95 = 1.96
SCREEN = {"S4": "momentum", "S5": "momentum", "F2": "momentum", "F3": "momentum", "F4": "momentum",
          "S7": "reversion", "S8": "reversion", "F5": "reversion", "F6": "reversion"}
SCREEN_FROM = pd.Timestamp("2020-01-01", tz="UTC")
BOOK_FROM = pd.Timestamp("2023-05-16", tz="UTC")   # the first bookTicker day of the archive
B2_PRE_MS = 600_000              # B2 windows: 10 min of book before the order goes live ...
B2_POST_MS = 7_200_000           # ... and 2 h after (T = 15 plus the 60 min markout, with room)
B1_PRE_MS = B2_PRE_MS            # B1 reads each leg's book from 10 min before it goes live, as B2, to past its
B1_POST_MS = (max(T_GRID) + max(MARKOUTS) + 1) * MINUTE_MS   # longest rest plus the 60 min markout
QUEUES = tuple(queue_sim.QUEUE)
Q3_NOTIONAL = 1386.0             # a generic tick leg's size: F1's mean order on a $10k account (USDT)
TICK_SETS = ("B1", "B2")
ORDER_OUTCOMES = ("entry_maker_fills", "exit_maker_fills", "entry_fallbacks", "exit_fallbacks", "entries_missed",
                  "exits_cancelled", "exits_preempted_by_stop")    # every posted order ends in exactly one
FEE_TOL = 1e-9                   # relative: the fees a trade paid against its legs' maker/taker flags
HARD_KINDS = ("x1-stage-a", "x1-download", "x1-stage-b")       # a hard_error in any of them closes the version

# output files, resolved against X1_DIR or RESULTS at call time (with a _v<VERSION> suffix from version 2 on)
TICK_CSV = "tick_sizes.csv"
LEGS_CSV = "stage_a_f1_legs.csv"
Q3_CSV = "stage_a_q3_costs.csv"
SCREEN_CSV = "stage_a_screen.csv"
COST_CSV = "x1_cost_v1.csv"
STAGE_A_JSON = "X1_stage_a.json"
STAGE_B_JSON = "X1_stage_b.json"
Q3_LEGS = "X1_q3_legs.parquet"
B1_LEGS = "X1_stage_b1_legs.parquet"
B2_LEGS_CSV = "stage_b2_legs.csv"
STAGE_B_OUTPUTS = {"B1": (COST_CSV,), "B2": (B2_LEGS_CSV,)}     # under X1_DIR


class HardError(RuntimeError):
    """An integrity check failed (a bug or a data problem, never a result)."""


def versioned(name: str, version: int | None = None) -> str:
    """An output's file name for a version: unchanged for version 1, '<stem>_v<n>.<ext>' after it."""
    v = VERSION if version is None else version
    stem, _, ext = name.rpartition(".")
    return name if v == 1 else f"{stem}_v{v}.{ext}"


def x1_path(name: str) -> Path:
    return X1_DIR / versioned(name)


def result_path(name: str) -> Path:
    return RESULTS / versioned(name)


def trial_file(version: int | None = None) -> Path:
    v = VERSION if version is None else version
    return RESULTS / (f"{CODE}_trial_returns.parquet" if v == 1 else f"{CODE}v{v}_trial_returns.parquet")


def trial_column(i: int, version: int | None = None) -> str:
    v = VERSION if version is None else version
    return f"{CODE}_{i}" if v == 1 else f"{CODE}_v{v}_{i}"


def screen_inputs() -> list[Path]:
    """The committed dev trades of the screened families (read, never written, by X1)."""
    return [RESULTS / f"{code}_dev_trades.parquet" for code in SCREEN]


def _sha(p: Path) -> str:
    """Text files hash with CRLF normalised (git may rewrite their line endings), others byte for byte."""
    return guards.sha_text(p) if Path(p).suffix in (".csv", ".json", ".jsonl") else guards.sha_file(p)


def manifest_path(tick_set: str) -> Path:
    return X1_DIR / f"tick_manifest_{tick_set}.csv"


def _ms(ts) -> int:
    return int(pd.Timestamp(ts).value // 10**6)


def _log_run(kind: str, **extra) -> None:
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    rec = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "kind": kind,
           "git": guards.git("rev-parse", "HEAD"), "dirty": bool(guards.git("status", "--porcelain", "--", "src"))} \
        | extra
    with RUN_LOG.open("a", newline="\n") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def code_sha() -> str:
    return guards.code_sha(CODE_FILES)


def lib_versions() -> dict:
    import numba
    import pyarrow
    import scipy
    return {"numpy_version": np.__version__, "numba_version": numba.__version__, "pandas_version": pd.__version__,
            "scipy_version": scipy.__version__, "pyarrow_version": pyarrow.__version__,
            "python_version": platform.python_version()}


def start_pins(sec_sha: str) -> dict:
    return {"code_sha256": code_sha(), "prereg_section_sha256": sec_sha,
            "f1_daily_sha256": guards.sha_file(F1_DEV_DAILY), "exchange_info_sha256": guards.sha_text(EXCHANGE_INFO),
            **lib_versions()}


def run_key(model: str, T: int, fallback: bool = True) -> str:
    return f"{model}_T{T}" + ("" if fallback else "_nofb")


def policies() -> dict[str, ME.MakerPolicy]:
    out = {TAKER: ME.MakerPolicy(0)}
    out |= {run_key(m, T): ME.MakerPolicy(T, m, True) for m in MODELS for T in T_GRID}
    out |= {run_key("strict", T, False): ME.MakerPolicy(T, "strict", False) for T in NOFALLBACK_T}
    return out


TRIAL_KEYS = [run_key("strict", T) for T in T_GRID] + [run_key("strict", T, False) for T in NOFALLBACK_T]


# ---------------------------------------------------------------- tick sizes

def round_sig(x, digits: int = TICK_SIG_DIGITS) -> np.ndarray:
    x = np.asarray(x, np.float64)
    out = x.copy()
    nz = np.isfinite(x) & (x != 0)
    mag = 10.0 ** (digits - 1 - np.floor(np.log10(np.abs(x[nz]))))
    out[nz] = np.round(x[nz] * mag) / mag
    return out


def snap_tick(d: float) -> float:
    """One significant digit, then the nearest of {1, 2, 5} x 10^k (on a log scale)."""
    if not (np.isfinite(d) and d > 0):
        return float("nan")
    x = float(f"{d:.0e}")
    k = math.floor(math.log10(x))
    m = x / 10.0 ** k
    g = min((1.0, 2.0, 5.0, 10.0), key=lambda v: abs(math.log(m / v)))
    return float(f"{g * 10.0 ** k:.0e}")


def infer_tick(prices) -> dict:
    """The price grid of a set of 1m prices: the candidate tick (the smallest gap between distinct prices, snapped)
    and the share of gaps that are whole multiples of it."""
    p = np.asarray(prices, np.float64)
    p = np.unique(round_sig(p[np.isfinite(p)]))
    gaps = np.diff(p)
    gaps = gaps[gaps > 0]
    if not len(gaps):
        return {"candidate": float("nan"), "share": 0.0, "n_prices": int(len(p))}
    cand = snap_tick(float(gaps.min()))
    r = gaps / cand
    k = np.round(r)
    ok = (k >= 1) & (np.abs(r - k) <= TICK_TOL)
    return {"candidate": cand, "share": float(ok.mean()), "n_prices": int(len(p))}


def exchange_info_ticks(path: Path | None = None) -> dict[str, float]:
    info = json.loads(Path(path or EXCHANGE_INFO).read_text(encoding="utf-8"))
    out = {}
    for s in info.get("symbols", []):
        for f in s.get("filters", []):
            if f.get("filterType") == "PRICE_FILTER":
                out[s["symbol"]] = float(f["tickSize"])
    return out


def _months(ts: np.ndarray) -> np.ndarray:
    return np.asarray(ts, np.int64).astype("datetime64[ms]").astype("datetime64[M]").astype(str)


def tick_table(m1: pd.DataFrame, symbol: str, exinfo_tick: float) -> pd.DataFrame:
    """Per month of perp rows in [START, HOLDOUT_START): the tick inferred from the 1m OHLC grid when at least 99%
    of the gaps are multiples of it, else the exchangeInfo tick; with the tick in bp of the month's median close."""
    ts = m1["ts"].to_numpy(np.int64)
    perp = m1["is_perp"].to_numpy(bool) if "is_perp" in m1 else np.ones(len(m1), bool)
    keep = perp & (ts >= _ms(START)) & (ts < _ms(HOLDOUT_START))
    px = m1.loc[keep, ["open", "high", "low", "close"]].to_numpy(np.float64)
    month = _months(ts[keep])
    rows = []
    for mon in np.unique(month):
        g = px[month == mon]
        inf = infer_tick(g.ravel())
        ok = bool(np.isfinite(inf["candidate"]) and inf["share"] >= TICK_CONSISTENT)
        tick = inf["candidate"] if ok else float(exinfo_tick)
        med = float(np.median(g[:, 3]))
        rows.append({"symbol": symbol, "month": str(mon), "rows": int(len(g)), "n_prices": inf["n_prices"],
                     "inferred_tick": inf["candidate"], "multiple_share": inf["share"], "consistent": ok,
                     "source": "1m" if ok else "exchangeInfo", "exchange_info_tick": float(exinfo_tick),
                     "tick": tick, "median_close": med, "tick_bp": tick / med * 1e4})
    return pd.DataFrame(rows, columns=["symbol", "month", "rows", "n_prices", "inferred_tick", "multiple_share",
                                       "consistent", "source", "exchange_info_tick", "tick", "median_close", "tick_bp"])


# ---------------------------------------------------------------- F1 runs and legs

def leg_table(trades: pd.DataFrame, ts: np.ndarray, costs: Costs) -> pd.DataFrame:
    """One row per maker-eligible leg of a maker_engine run: every entry and the exits with a decision reason.
    saving_bp = (side*(ref - px)/ref + (taker fee - fee paid)) * 1e4; drift_bp = side*(px - ref)/ref * 1e4."""
    t = trades
    sym = t["symbol"] if "symbol" in t else pd.Series("", index=t.index)
    ent = pd.DataFrame({"symbol": sym, "leg": "entry", "reason": t["reason"], "entry_tf": t["entry_tf"],
                        "dir": t["dir"], "side": t["dir"], "post_i": t["entry_post_i"], "fill_i": t["entry_i"],
                        "ref_px": t["entry_ref_px"], "px": t["entry_px"], "maker": t["maker"], "qty": t["qty"]})
    x = t[t["reason"].isin(DECISION_EXITS)]
    ext = pd.DataFrame({"symbol": sym[x.index], "leg": "exit", "reason": x["reason"], "entry_tf": x["entry_tf"],
                        "dir": x["dir"], "side": -x["dir"], "post_i": x["exit_post_i"], "fill_i": x["exit_i"],
                        "ref_px": x["exit_ref_px"], "px": x["exit_px"], "maker": x["exit_maker"], "qty": x["qty"]})
    legs = pd.concat([ent, ext], ignore_index=True)
    cols = ["symbol", "leg", "reason", "entry_tf", "dir", "side", "post_i", "fill_i", "post_ms", "fill_ms",
            "post_time", "wait_min", "ref_px", "px", "maker", "fee", "qty", "saving_bp", "drift_bp"]
    if not len(legs):
        return pd.DataFrame(columns=cols)
    for c in ("entry_tf", "dir", "side", "post_i", "fill_i", "maker"):
        legs[c] = legs[c].astype(np.int64)
    if (legs["post_i"] < 0).any() or legs["ref_px"].isna().any():
        raise HardError("a maker-eligible leg has no post minute or reference price")
    ts = np.asarray(ts, np.int64)
    legs["post_ms"] = ts[legs["post_i"].to_numpy()]
    legs["fill_ms"] = ts[legs["fill_i"].to_numpy()]
    legs["post_time"] = pd.to_datetime(legs["post_ms"], unit="ms", utc=True)
    legs["wait_min"] = (legs["fill_ms"] - legs["post_ms"]) / MINUTE_MS
    legs["fee"] = np.where(legs["maker"] == 1, costs.maker, costs.taker)
    s, ref, px = legs["side"].to_numpy(np.float64), legs["ref_px"].to_numpy(), legs["px"].to_numpy()
    legs["saving_bp"] = (s * (ref - px) / ref + (costs.taker - legs["fee"].to_numpy())) * 1e4
    legs["drift_bp"] = s * (px - ref) / ref * 1e4
    return legs[cols].sort_values(["post_ms", "symbol", "leg"], kind="stable").reset_index(drop=True)


def _ratio(a: int, b: int) -> float:
    return a / b if b else float("nan")


def leg_summary(legs: pd.DataFrame, info: dict | None = None) -> dict:
    """The executed legs (leg_table): leg_fill_rate, savings, drift of the legs that fell back. With the run's
    info, fill_rate is maker fills / orders posted, so orders that ended without a leg (entries a post-and-cancel
    run missed, exits it cancelled, exits a stop pre-empted) count as unfilled; without it, the leg rates."""
    f = legs["maker"].to_numpy() == 1 if len(legs) else np.zeros(0, bool)
    s = legs["saving_bp"].to_numpy(np.float64) if len(legs) else np.zeros(0)

    def mean(x):
        return float(np.mean(x)) if len(x) else float("nan")
    ent = legs["leg"].to_numpy() == "entry" if len(legs) else np.zeros(0, bool)
    out = {"legs": int(len(legs)), "entries": int(ent.sum()), "exits": int((~ent).sum()), "maker_fills": int(f.sum()),
           "leg_fill_rate": mean(f), "entry_leg_fill_rate": mean(f[ent]), "exit_leg_fill_rate": mean(f[~ent]),
           "mean_saving_bp": mean(s), "mean_saving_filled_bp": mean(s[f]), "mean_saving_unfilled_bp": mean(s[~f]),
           "mean_drift_unfilled_bp": mean(legs["drift_bp"].to_numpy()[~f]) if len(legs) else float("nan"),
           "mean_wait_filled_min": mean(legs["wait_min"].to_numpy()[f]) if len(legs) else float("nan"),
           "sum_saving_bp": float(s.sum())}
    if info is None:
        return out | {"fill_rate": out["leg_fill_rate"], "entry_fill_rate": out["entry_leg_fill_rate"],
                      "exit_fill_rate": out["exit_leg_fill_rate"]}
    n = {k: int(info.get(k, 0)) for k in ("orders_posted", *ORDER_OUTCOMES)}
    entry_orders = n["entry_maker_fills"] + n["entry_fallbacks"] + n["entries_missed"]
    exit_orders = n["exit_maker_fills"] + n["exit_fallbacks"] + n["exits_cancelled"] + n["exits_preempted_by_stop"]
    return out | {"orders_posted": n["orders_posted"], "entry_orders": entry_orders, "exit_orders": exit_orders,
                  "orders_without_leg": n["entries_missed"] + n["exits_cancelled"] + n["exits_preempted_by_stop"],
                  "entries_missed": n["entries_missed"], "exits_cancelled": n["exits_cancelled"],
                  "exits_preempted_by_stop": n["exits_preempted_by_stop"],
                  "fill_rate": _ratio(n["entry_maker_fills"] + n["exit_maker_fills"], n["orders_posted"]),
                  "entry_fill_rate": _ratio(n["entry_maker_fills"], entry_orders),
                  "exit_fill_rate": _ratio(n["exit_maker_fills"], exit_orders)}


def run_problems(trades: pd.DataFrame, legs: pd.DataFrame, info: dict, costs: Costs, posts: bool) -> list[str]:
    """Cross-checks of a maker_engine run against its own accounts: every trade's fees equal its legs' maker/taker
    fees at its recorded prices; every posted order ends in exactly one outcome; the maker and fallback legs match
    the engine's counts (posts: the policy posts orders, T > 0)."""
    probs = []
    t = trades
    if len(t):
        fe = np.where(t["maker"].to_numpy() == 1, costs.maker, costs.taker)
        fx = np.where(t["exit_maker"].to_numpy() == 1, costs.maker, costs.taker)
        q = t["qty"].to_numpy(np.float64)
        want = q * t["entry_px"].to_numpy(np.float64) * fe + q * t["exit_px"].to_numpy(np.float64) * fx
        got = t["fees"].to_numpy(np.float64)
        bad = ~(np.abs(got - want) <= FEE_TOL * np.abs(want))
        if bad.any():
            probs.append(f"{int(bad.sum())} trade(s) paid fees other than their legs' maker/taker fees "
                         f"(first: {got[bad][0]} vs {want[bad][0]})")
        if ((t["exit_maker"] == 1) & ~t["reason"].isin(DECISION_EXITS)).any():
            probs.append("a stop, target, end or forced exit is flagged maker")
    n = {k: int(info.get(k, 0)) for k in ("orders_posted", *ORDER_OUTCOMES)}
    if n["orders_posted"] != sum(n[k] for k in ORDER_OUTCOMES):
        probs.append(f"{n['orders_posted']} orders posted, {sum(n[k] for k in ORDER_OUTCOMES)} order outcomes")
    if len(legs):
        ent = legs["leg"].to_numpy() == "entry"
        mk = legs["maker"].to_numpy() == 1
        seen = {"entry_maker_fills": int((ent & mk).sum()), "exit_maker_fills": int((~ent & mk).sum()),
                "entry_fallbacks": int((ent & ~mk).sum()), "exit_fallbacks": int((~ent & ~mk).sum())}
    else:
        seen = dict.fromkeys(("entry_maker_fills", "exit_maker_fills", "entry_fallbacks", "exit_fallbacks"), 0)
    if not posts:
        seen = {k: v if "maker" in k else 0 for k, v in seen.items()}    # taker legs are no fallbacks
        if n["orders_posted"]:
            probs.append(f"{n['orders_posted']} orders posted by a taker run")
    off = {k: (v, n[k]) for k, v in seen.items() if v != n[k]}
    if off:
        probs.append(f"legs vs the engine's order counts (legs, engine): {off}")
    return probs


def sanity_problems(summ: dict, strict: bool) -> list[str]:
    """mean saving = p_fill * mean(filled) + (1 - p_fill) * mean(unfilled) over the executed legs; a strict fill
    saves exactly 3 bp (that its fee is the maker fee is run_problems' check)."""
    if not summ["legs"]:
        return []
    p = summ["leg_fill_rate"]
    mf = summ["mean_saving_filled_bp"] if summ["maker_fills"] else 0.0
    mu = summ["mean_saving_unfilled_bp"] if summ["maker_fills"] < summ["legs"] else 0.0
    probs = []
    if not abs(summ["mean_saving_bp"] - (p * mf + (1 - p) * mu)) <= SANITY_TOL:
        probs.append(f"saving identity off: {summ['mean_saving_bp']} vs {p * mf + (1 - p) * mu}")
    if strict and summ["maker_fills"] and not abs(mf - STRICT_FILL_SAVING_BP) <= SANITY_TOL:
        probs.append(f"a strict maker fill saved {mf} bp on average, not {STRICT_FILL_SAVING_BP}")
    return probs


def f1_symbol_runs(mk: engine.Market, sym: str, tick: np.ndarray) -> dict[str, dict]:
    """F1 on one symbol under every policy: daily sleeve returns, trades, the engine's order counts and the legs."""
    sig = S1Frozen().compute(mk, **F1_PARAMS)
    costs = Costs(slippage=SLIPPAGE[sym])
    risk = sleeve_risk(len(SYMBOLS))
    out = {}
    for key, pol in policies().items():
        res = ME.run(mk, sig, costs, risk, start=START, latency_min=LATENCY_MIN, policy=pol, tick=tick)
        tr = res.trades.assign(symbol=sym)
        legs = leg_table(tr, mk.ts, costs)
        probs = run_problems(tr, legs, res.info, costs, posts=pol.T > 0)
        if probs:
            raise HardError(f"{sym} {key}: " + "; ".join(probs))
        out[key] = {"daily": aligned_daily(res, mk), "trades": tr, "info": res.info, "legs": legs}
    return out


def combine(runs: dict[str, dict[str, dict]], symbols=None) -> dict[str, dict]:
    """Per policy: the equal-capital portfolio of the sleeves (portfolio.assemble) and the pooled trades, legs and
    order counts."""
    symbols = list(symbols or runs)
    out = {}
    for key in runs[symbols[0]]:
        parts = [(runs[s][key]["daily"], runs[s][key]["trades"]) for s in symbols]
        _, daily = assemble(parts)
        info = {}
        for s in symbols:
            for k, v in runs[s][key]["info"].items():
                if k != "missed_entry_tf":
                    info[k] = info.get(k, 0) + int(v)
        out[key] = {"daily": daily, "sleeves": {s: runs[s][key]["daily"] for s in symbols},
                    "trades": pd.concat([runs[s][key]["trades"] for s in symbols], ignore_index=True),
                    "legs": pd.concat([runs[s][key]["legs"] for s in symbols], ignore_index=True)
                    .sort_values(["post_ms", "symbol", "leg"], kind="stable").reset_index(drop=True),
                    "info": info,
                    "missed": {s: np.asarray(runs[s][key]["info"].get("missed_entry_tf", []), np.int64)
                               for s in symbols}}
    return out


def parity_problem(daily: pd.Series, ref: pd.Series) -> str | None:
    """None when two daily series have the same days and agree to PARITY_TOL everywhere."""
    if len(daily) != len(ref) or not (pd.DatetimeIndex(daily.index).as_unit("ns")
                                      == pd.DatetimeIndex(ref.index).as_unit("ns")).all():
        return f"the days differ ({len(daily)} vs {len(ref)})"
    diff = np.abs(daily.to_numpy(np.float64) - ref.to_numpy(np.float64))
    if not (diff <= PARITY_TOL).all():
        return f"max |difference| {np.nanmax(diff) if np.isfinite(diff).any() else 'nan'}"
    return None


def check_runs(by_policy: dict[str, dict], f1_daily: pd.Series) -> None:
    hold = HOLDOUT_START
    for key, r in by_policy.items():
        if r["info"].get("funding_in_window", 0) > 0:
            raise HardError(f"{key}: {r['info']['funding_in_window']} funding settlements inside a resting window")
        t = r["trades"]
        if len(t) and (t["entry_time"].max() >= hold or t["exit_time"].max() >= hold):
            raise HardError(f"{key}: a trade reaches the holdout")
        if len(r["daily"]) and r["daily"].index.max() >= hold:
            raise HardError(f"{key}: a daily return reaches the holdout")
        for s, d in r["sleeves"].items():
            if len(d) and d.index.max() >= hold:
                raise HardError(f"{key} {s}: a daily return reaches the holdout")
    prob = parity_problem(by_policy[TAKER]["daily"], f1_daily)
    if prob:
        raise HardError(f"the taker run does not reproduce F1_dev_daily.parquet: {prob}")


# ---------------------------------------------------------------- gates

def sharpe(d: pd.Series) -> float:
    sd = float(d.std(ddof=1))
    return float(d.mean() / sd * np.sqrt(365)) if sd > 0 else 0.0


def max_dd(d: pd.Series) -> float:
    return metrics.max_drawdown(np.r_[1.0, np.cumprod(1.0 + d.to_numpy(np.float64))])


def daily_stats(d: pd.Series) -> dict:
    yrs = len(d) / 365
    total = float(np.prod(1.0 + d.to_numpy(np.float64)) - 1.0)
    return {"days": int(len(d)), "mean_daily": float(d.mean()), "ann_mean": float(d.mean() * 365),
            "sharpe": sharpe(d), "max_dd": max_dd(d), "total_return": total,
            "cagr": float((1 + total) ** (1 / yrs) - 1) if total > -1 and yrs > 0 else -1.0}


def _diff(a: pd.Series, b: pd.Series) -> pd.Series:
    if len(a) != len(b) or not a.index.equals(b.index):
        raise HardError("daily series to compare have different days")
    return a - b


def gate_values(taker: pd.Series, maker: pd.Series, neighbours: dict[int, pd.Series],
                taker_sleeves: dict[str, pd.Series], maker_sleeves: dict[str, pd.Series]) -> dict:
    """The quantities behind gates (a)-(e) for one maker run against the taker run (portfolio daily returns)."""
    d = _diff(maker, taker)
    boot = stats.block_bootstrap_means(d.to_numpy(np.float64), BOOT_N, BOOT_BLOCK, BOOT_SEED)
    yr = d.groupby(d.index.year).sum()
    years = {int(y): float(yr.get(y, 0.0)) for y in YEARS}
    coins = {s: float(_diff(maker_sleeves[s], taker_sleeves[s]).sum()) for s in taker_sleeves}
    return {"days": int(len(d)), "mean_daily_diff": float(d.mean()), "gain_ann": float(d.mean() * 365),
            "boot_p5": float(np.quantile(boot, 0.05)), "boot_mean": float(boot.mean()),
            "neighbour_mean_daily_diff": {int(T): float(_diff(x, taker).mean()) for T, x in neighbours.items()},
            "year_sum_diff": years, "years_positive": int(sum(v > 0 for v in years.values())),
            "coin_sum_diff": coins, "coins_positive": int(sum(v > 0 for v in coins.values())),
            "sharpe_maker": sharpe(maker), "sharpe_taker": sharpe(taker),
            "max_dd_maker": max_dd(maker), "max_dd_taker": max_dd(taker)}


def gates_from_values(v: dict) -> dict[str, bool]:
    nb = v["neighbour_mean_daily_diff"]
    want = [T for T in T_GRID if T != PRIMARY_T]
    return {"(a) gain >= +0.5%/yr": v["gain_ann"] >= MIN_GAIN,
            "(b) bootstrap 5% bound > 0": v["boot_p5"] > 0,
            "(c) plateau: T=5 and T=60 gain > 0": all(T in nb and nb[T] > 0 for T in want),
            "(d) >=3 of 4 years and >=4 of 5 coins positive": v["years_positive"] >= MIN_YEARS
            and v["coins_positive"] >= MIN_COINS,
            "(e) Sharpe at most -0.02, max drawdown at most +1pp": v["sharpe_maker"] >= v["sharpe_taker"] - SHARPE_TOL
            and v["max_dd_maker"] <= v["max_dd_taker"] + DD_TOL}


def model_gates(by_policy: dict[str, dict], model: str) -> tuple[dict, dict]:
    tk = by_policy[TAKER]
    mk = by_policy[run_key(model, PRIMARY_T)]
    nb = {T: by_policy[run_key(model, T)]["daily"] for T in T_GRID if T != PRIMARY_T}
    v = gate_values(tk["daily"], mk["daily"], nb, tk["sleeves"], mk["sleeves"])
    return v, gates_from_values(v)


def missed_info(taker_trades: pd.DataFrame, missed: dict[str, np.ndarray], nofb_trades: pd.DataFrame) -> dict:
    """Entries a post-and-cancel run missed, matched by symbol and entry_tf to the taker run's trades."""
    keys = {(s, int(tf)) for s, arr in missed.items() for tf in np.asarray(arr).tolist()}
    t = taker_trades
    hit = np.array([(s, int(tf)) in keys for s, tf in zip(t["symbol"], t["entry_tf"])], bool) if len(t) \
        else np.zeros(0, bool)
    m, k = t[hit], t[~hit]

    def mr(x):
        return float(x["R"].mean()) if len(x) else float("nan")
    return {"missed_entries": len(keys), "matched_to_taker_trades": int(hit.sum()),
            "unmatched": len(keys) - int(hit.sum()), "mean_R_missed": mr(m), "mean_R_taken": mr(k),
            "taker_trades": int(len(t)), "run_trades": int(len(nofb_trades)), "run_mean_R": mr(nofb_trades)}


# ---------------------------------------------------------------- Q3: generic legs on the 1m path

def momentum_events(bars: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(bar index, side): a 1h close crossing above the prior 24-bar high is a buy, below the prior low a sell."""
    c = bars["close"].astype(np.float64)
    hi = bars["high"].astype(np.float64).rolling(MOM_BARS).max().shift(1)
    lo = bars["low"].astype(np.float64).rolling(MOM_BARS).min().shift(1)
    up = ((c > hi) & (c.shift(1) <= hi.shift(1))).to_numpy()
    dn = ((c < lo) & (c.shift(1) >= lo.shift(1))).to_numpy()
    side = np.where(up, 1, np.where(dn, -1, 0))
    idx = np.flatnonzero(side != 0)
    return idx, side[idx].astype(np.int64)


def reversion_events(bars: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(bar index, side): a 1h log return beyond 2 sigma of the prior 720 bars, traded against."""
    c = bars["close"].astype(np.float64)
    r = np.log(c / c.shift(1))
    sd = r.rolling(REV_BARS).std().shift(1)
    side = np.where(r > REV_Z * sd, -1, np.where(r < -REV_Z * sd, 1, 0))
    idx = np.flatnonzero(side != 0)
    return idx, side[idx].astype(np.int64)


def q3_legs(mk: engine.Market, sym: str, rng: np.random.Generator) -> pd.DataFrame:
    """The generic legs of one symbol: random rows of the perp path from START (rows, then sides, from rng), and the
    momentum and reversion events executed at the 1h bar close + LATENCY_MIN."""
    ts = mk.ts
    perp = mk.m1["is_perp"].to_numpy(bool) if "is_perp" in mk.m1 else np.ones(len(ts), bool)
    lo = _ms(START)
    elig = np.flatnonzero(perp & (ts >= lo))
    if not len(elig):
        raise HardError(f"{sym}: no perp rows from {START.date()}")
    rows = elig[rng.integers(0, len(elig), Q3_RANDOM_PER_COIN)]
    sides = rng.integers(0, 2, Q3_RANDOM_PER_COIN) * 2 - 1
    frames = [pd.DataFrame({"class": "random", "m0": rows, "side": sides})]
    bars = mk.bars("1h")
    ex = mk.exec_idx("1h", LATENCY_MIN)
    for cls, fn in (("momentum", momentum_events), ("reversion", reversion_events)):
        idx, side = fn(bars)
        m0 = ex[idx]
        ok = m0 < len(ts)
        m0, side = m0[ok], side[ok]
        ok = (ts[m0] >= lo) & perp[m0]
        frames.append(pd.DataFrame({"class": cls, "m0": m0[ok], "side": side[ok]}))
    out = pd.concat(frames, ignore_index=True)
    out.insert(1, "symbol", sym)
    out["m0"] = out["m0"].astype(np.int64)
    out["side"] = out["side"].astype(np.int64)
    return out


def q3_simulate(mk: engine.Market, legs: pd.DataFrame, tick: np.ndarray, slip: float) -> pd.DataFrame:
    """maker_engine.simulate_legs for every T x model (fallback on), one wide row per leg. A leg counts only if it is
    complete at the largest T, so every T and model is measured on the same legs."""
    out = legs.copy()
    m0 = legs["m0"].to_numpy(np.int64)
    side = legs["side"].to_numpy(np.int64)
    out["m0_ms"] = mk.ts[m0]
    out["open_px"] = mk.o[m0]
    out["slip"] = float(slip)
    for T in T_GRID:
        for model in MODELS:
            r = ME.simulate_legs(mk.o, mk.h, mk.l, mk.c, mk.ts, m0, side, tick, slip, T, model, True,
                                 markouts=MARKOUTS)
            p = f"{model}_T{T}"
            out[f"saving_{p}"] = r["saving_bp"]
            out[f"filled_{p}"] = r["filled"]
            out[f"wait_{p}"] = r["wait_min"]
            for k in MARKOUTS:
                out[f"markout{k}_{p}"] = r[f"markout_bp_{k}"]
        out[f"complete_T{T}"] = r["complete"]
    out["complete"] = out[f"complete_T{max(T_GRID)}"]
    return out


def _cluster_se(x: np.ndarray, day: np.ndarray) -> tuple[int, float]:
    """(day count G, the day-clustered SE of the leg mean) = sqrt(G/(G-1) * sum_d (n_d * (mean_d - mean))^2) / n
    from the per-UTC-day means; with equal legs per day it is std(day means) / sqrt(G)."""
    x = np.asarray(x, np.float64)
    dev = pd.Series(x - x.mean()).groupby(np.asarray(day)).sum().to_numpy()
    g = len(dev)
    return g, float(np.sqrt(g / (g - 1) * np.sum(dev ** 2)) / len(x)) if g > 1 else float("nan")


def q3_costs(legs: pd.DataFrame) -> pd.DataFrame:
    """Per class, symbol (and ALL), T and model: fill rate, mean saving with a day-clustered 95% CI, waits,
    markouts, and c_eff = taker cost per side (fee + slippage) - mean saving, in bp."""
    taker = Costs().taker
    rows = []
    for cls in CLASSES:
        for sym in [*SYMBOLS, "ALL"]:
            g = legs[(legs["class"] == cls) & legs["complete"].astype(bool)]
            if sym != "ALL":
                g = g[g["symbol"] == sym]
            day = g["m0_ms"].to_numpy(np.int64) // TF_MS["1d"]
            cost = float(np.mean((taker + g["slip"].to_numpy()) * 1e4)) if len(g) else float("nan")
            for T in T_GRID:
                for model in MODELS:
                    p = f"{model}_T{T}"
                    s = g[f"saving_{p}"].to_numpy(np.float64)
                    f = g[f"filled_{p}"].to_numpy(bool)
                    n = len(s)
                    mean = float(np.mean(s)) if n else float("nan")
                    days, se = _cluster_se(s, day) if n else (0, float("nan"))
                    row = {"class": cls, "symbol": sym, "T": T, "model": model, "n": n, "days": days,
                           "fill_rate": float(f.mean()) if n else float("nan"), "mean_saving_bp": mean,
                           "se_bp": se, "ci_lo": mean - Z95 * se, "ci_hi": mean + Z95 * se,
                           "mean_wait_min": float(g[f"wait_{p}"].mean()) if n else float("nan"),
                           "mean_wait_filled_min": float(g.loc[f, f"wait_{p}"].mean()) if f.any() else float("nan")}
                    for k in MARKOUTS:
                        row[f"markout_bp_{k}"] = float(g.loc[f, f"markout{k}_{p}"].mean()) if f.any() else float("nan")
                    row["taker_cost_bp"] = cost
                    row["c_eff_bp"] = cost - mean
                    rows.append(row)
    return pd.DataFrame(rows)


def q3_legs_file(legs: pd.DataFrame) -> pd.DataFrame:
    """The compact per-leg file Stage B reads (the 1m savings it calibrates against)."""
    cols = ["class", "symbol", "m0_ms", "side", "open_px", "slip", "complete"]
    cols += [f"{w}_{m}_T{T}" for T in T_GRID for m in MODELS for w in ("saving", "filled")]
    out = legs[cols].reset_index(drop=True)
    out.insert(0, "leg_id", np.arange(len(out), dtype=np.int64))
    return out


# ---------------------------------------------------------------- screen of the committed fast families

def _ceff_lookup(q3c: pd.DataFrame, T: int = PRIMARY_T) -> dict:
    g = q3c[q3c["T"] == T]
    return {(c, s, m): float(v) for c, s, m, v in zip(g["class"], g["symbol"], g["model"], g["c_eff_bp"])}


def screen_family(code: str, trades: pd.DataFrame, ceff: dict) -> list[dict]:
    """Gross edge per trade (bp of entry notional, slippage added back on every taker leg) against the taker round
    trip and the class's maker-first round trip per model; all trades and entries from 2020 on."""
    cls = SCREEN[code]
    t = trades
    if len(t) and (t["entry_time"].max() >= HOLDOUT_START or t["exit_time"].max() >= HOLDOUT_START):
        raise HardError(f"{code}: a committed dev trade reaches the holdout")
    sym = t["symbol"].astype(str) if "symbol" in t else pd.Series("BTCUSDT", index=t.index)
    slip = sym.map(SLIPPAGE).to_numpy(np.float64)
    if np.isnan(slip).any():
        raise HardError(f"{code}: a trade's symbol has no registered slippage")
    notional = (t["qty"] * t["entry_px"]).to_numpy(np.float64)
    taker_entry = (t["maker"].to_numpy() != 1).astype(np.float64)
    gross = t["gross"].to_numpy(np.float64) + slip * (notional * taker_entry
                                                       + (t["qty"] * t["exit_px"]).to_numpy(np.float64))
    edge = gross / notional * 1e4
    side_cost = (Costs().taker + slip) * 1e4
    nonstop = t["reason"].isin(DECISION_EXITS).to_numpy()
    maker_rt = {}
    for m in MODELS:
        ce = np.array([ceff.get((cls, s, m), np.nan) for s in sym])
        cr = np.array([ceff.get(("random", s, m), np.nan) for s in sym])
        maker_rt[m] = ce + np.where(nonstop, cr, side_cost)
    rows = []
    for period, keep in (("all", np.ones(len(t), bool)), ("2020+", (t["entry_time"] >= SCREEN_FROM).to_numpy())):
        n = int(keep.sum())

        def mean(x):
            return float(np.mean(x[keep])) if n else float("nan")
        row = {"family": code, "class": cls, "period": period, "trades": n, "gross_bp": mean(edge),
               "taker_rt_bp": mean(2 * side_cost), "share_nonstop_exits": mean(nonstop.astype(float))}
        for m in MODELS:
            row[f"maker_rt_bp_{m}"] = mean(maker_rt[m])
        for m in MODELS:
            row[f"execution_sensitive_{m}"] = bool(row[f"maker_rt_bp_{m}"] < row["gross_bp"] < row["taker_rt_bp"])
        rows.append(row)
    return rows


def screen_table(q3c: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """The screen and the SHA256 of each committed dev trades file it read."""
    ceff = _ceff_lookup(q3c)
    rows, shas = [], {}
    for code, p in zip(SCREEN, screen_inputs()):
        shas[p.name] = guards.sha_file(p)
        rows += screen_family(code, pd.read_parquet(p), ceff)
    return pd.DataFrame(rows), shas


# ---------------------------------------------------------------- Stage A

def run_summary(r: dict) -> dict:
    t = r["trades"]
    return daily_stats(r["daily"]) | {"trades": int(len(t)), "fees": float(t["fees"].sum()) if len(t) else 0.0,
                                      "funding": float(t["funding"].sum()) if len(t) else 0.0,
                                      "info": dict(r["info"])}


def stage_a() -> dict:
    """Steps 1-7: tick table, F1 under every policy, gates, legs, post-and-cancel info, Q3 step 1 and the screen.
    One symbol in memory at a time. Raises HardError on any integrity failure."""
    exi = exchange_info_ticks()
    missing = [s for s in SYMBOLS if s not in exi]
    if missing:
        raise HardError(f"no exchangeInfo tick for {missing}")
    f1_ref = pd.read_parquet(F1_DEV_DAILY)["ret"]
    rng = np.random.default_rng(Q3_SEED)
    hold_ms = _ms(HOLDOUT_START)
    tick_rows, runs, q3 = [], {}, []
    for sym in SYMBOLS:
        print(f"X1 Stage A: {sym} ...", flush=True)
        mk = engine.Market.load("dev", sym, with_metrics=False)
        if len(mk.ts) and int(mk.ts[-1]) >= hold_ms:
            raise HardError(f"{sym}: the dev path reaches the holdout")
        tt = tick_table(mk.m1, sym, exi[sym])
        tick = ME.tick_array(mk, dict(zip(tt["month"], tt["tick"])))
        runs[sym] = f1_symbol_runs(mk, sym, tick)
        q3.append(q3_simulate(mk, q3_legs(mk, sym, rng), tick, SLIPPAGE[sym]))
        tick_rows.append(tt)
        del mk
    by = combine(runs, SYMBOLS)
    check_runs(by, f1_ref)
    legs = {k: leg_summary(r["legs"], r["info"]) for k, r in by.items()}
    probs = [f"{k}: {p}" for k, s in legs.items()
             for p in sanity_problems(s, strict=policies()[k].model == "strict" and policies()[k].T > 0)]
    if probs:
        raise HardError("Stage A sanity identity: " + "; ".join(probs))
    gv, gates = {}, {}
    for m in MODELS:
        gv[m], gates[m] = model_gates(by, m)
    strict_pass = bool(all(gates["strict"].values()))
    nofb = {k: missed_info(by[TAKER]["trades"], by[k]["missed"], by[k]["trades"])
            for k in policies() if k.endswith("_nofb")}
    q3_all = pd.concat(q3, ignore_index=True)
    if len(q3_all) and int(q3_all["m0_ms"].max()) >= hold_ms:
        raise HardError("a Q3 leg reaches the holdout")
    q3c = q3_costs(q3_all)
    screen, screen_sha = screen_table(q3c)
    return {"ticks": pd.concat(tick_rows, ignore_index=True), "by_policy": by, "leg_summaries": legs,
            "gate_values": gv, "gates": gates, "strict_pass": strict_pass,
            "decision": PASS_DECISION if strict_pass else FAIL_DECISION, "no_fallback": nofb,
            "q3_legs": q3_all, "q3_costs": q3c, "screen": screen, "screen_inputs_sha256": screen_sha}


def stage_a_result(a: dict, pins: dict) -> dict:
    by = a["by_policy"]
    tt = a["ticks"]
    return {"study": "X1", "stage": "A", "code": CODE, "version": VERSION, "decision": a["decision"],
            "strict_pass": a["strict_pass"], "gates": a["gates"], "gate_values": a["gate_values"],
            "policies": {k: {"T": p.T, "model": p.model, "fallback": p.fallback} for k, p in policies().items()},
            "primary": PRIMARY, "runs": {k: run_summary(r) for k, r in by.items()},
            "legs": a["leg_summaries"], "no_fallback": a["no_fallback"],
            "tick_table": {"months": int(len(tt)), "exchange_info_months": int((tt["source"] == "exchangeInfo").sum()),
                           "tick_bp_max": float(tt["tick_bp"].max()) if len(tt) else None},
            "q3": {"legs": int(len(a["q3_legs"])), "complete": int(a["q3_legs"]["complete"].sum())},
            "screen": a["screen"].to_dict("records"), "screen_inputs_sha256": a["screen_inputs_sha256"],
            "symbols": SYMBOLS, "start": str(START.date()), "latency_min": LATENCY_MIN, "f1_params": F1_PARAMS,
            "bootstrap": {"n": BOOT_N, "block": BOOT_BLOCK, "seed": BOOT_SEED},
            "git_commit": guards.git("rev-parse", "HEAD"), **pins}


def stage_a_outputs() -> dict[str, Path]:
    """The Stage A files later stages read (or report), by their base name: pinned by SHA256 in the x1-stage-a
    record, so a file changed after Stage A is refused."""
    return {n: x1_path(n) for n in (TICK_CSV, LEGS_CSV, Q3_CSV, SCREEN_CSV)} | {Q3_LEGS: result_path(Q3_LEGS),
                                                                              "trial_returns": trial_file()}


def family_prior(version: int | None = None) -> pd.DataFrame | None:
    """Earlier F1X versions' trial returns, part of the family for N_eff (as Round 7's Amendment 1): every earlier
    version with a logged Stage A decision. Their files must be committed."""
    v = VERSION if version is None else version
    done = sorted({int(r["version"]) for r in guards.log_records(RUN_LOG, "x1-stage-a")
                   if "decision" in r and int(r["version"]) < v})
    if not done:
        return None
    files = [trial_file(x) for x in done]
    guards.require_committed(files)
    return pd.concat([pd.read_parquet(f) for f in files], axis=1)


def write_stage_a(a: dict, pins: dict, prior: pd.DataFrame | None = None) -> dict:
    """Step 8. The gated result is saved before anything else."""
    X1_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    r = stage_a_result(a, pins)
    _save_json(result_path(STAGE_A_JSON), r)
    a["ticks"].to_csv(x1_path(TICK_CSV), index=False, lineterminator="\n")
    a["by_policy"][PRIMARY]["legs"].to_csv(x1_path(LEGS_CSV), index=False, lineterminator="\n")
    a["q3_costs"].to_csv(x1_path(Q3_CSV), index=False, lineterminator="\n")
    a["screen"].to_csv(x1_path(SCREEN_CSV), index=False, lineterminator="\n")
    q3_legs_file(a["q3_legs"]).to_parquet(result_path(Q3_LEGS))
    by = a["by_policy"]
    trial = pd.concat({trial_column(i): by[k]["daily"] for i, k in enumerate(TRIAL_KEYS)}, axis=1)
    trial.to_parquet(trial_file())
    rows = []
    for i, k in enumerate(TRIAL_KEYS):
        p = policies()[k]
        sr, T, sk, ku = stats.sharpe_moments(by[k]["daily"])
        rows.append({"strategy": CODE, "version": VERSION, "params": {"T": p.T, "fallback": p.fallback,
                                                                     "model": p.model},
                     "period": "dev", "sr": sr, "T": T, "skew": sk, "kurt": ku, "trades": int(len(by[k]["trades"])),
                     "sharpe_ann": float(sr * np.sqrt(365))})
    registry.record(rows)
    allt = research_run.all_trial_returns()
    fam = trial if prior is None else pd.concat([prior, trial], axis=1, join="inner")
    r["n_eff_family"] = stats.effective_trials(fam.to_numpy())
    r["family_columns"] = [str(c) for c in fam.columns]
    r["n_eff_project"] = registry.ratcheted_effective_trials(stats.effective_trials(allt.to_numpy()), allt.shape[1],
                                                             note=f"X1 ({CODE} v{VERSION})")
    r["trial_columns"] = {trial_column(i): k for i, k in enumerate(TRIAL_KEYS)}
    r["outputs_sha256"] = {n: _sha(p) for n, p in stage_a_outputs().items()}
    _save_json(result_path(STAGE_A_JSON), r)
    r["stage_a_json_sha256"] = _sha(result_path(STAGE_A_JSON))
    return r


def _fmt(x, spec: str, width: int = 0) -> str:
    """A number for the summary; NaN (None once saved to JSON) prints as n/a."""
    ok = x is not None and not (isinstance(x, float) and math.isnan(x))
    return format(format(x, spec) if ok else "n/a", f">{width}")


def format_stage_a(r: dict) -> str:
    tk = r["runs"][TAKER]
    lines = [f"X1 Stage A ({r['code']} v{r['version']}): taker F1 CAGR {_fmt(tk['cagr'], '.1%')} Sharpe "
             f"{_fmt(tk['sharpe'], '.2f')} maxDD {_fmt(tk['max_dd'], '.1%')} trades {tk['trades']}",
             "fill: maker fills / orders posted; bp/leg: per executed leg; no leg: orders cancelled or pre-empted",
             f"{'run':<18}{'gain %/yr':>10}{'Sharpe':>8}{'maxDD':>8}{'orders':>8}{'fill':>7}{'bp/leg':>8}"
             f"{'no leg':>8}"]
    for k, s in r["runs"].items():
        if k == TAKER:
            continue
        gain = (s["mean_daily"] - tk["mean_daily"]) * 365
        lg = r["legs"][k]
        lines.append(f"{k:<18}{_fmt(gain, '.3%', 10)}{_fmt(s['sharpe'], '.2f', 8)}{_fmt(s['max_dd'], '.1%', 8)}"
                     f"{lg.get('orders_posted', 0):>8}{_fmt(lg['fill_rate'], '.1%', 7)}"
                     f"{_fmt(lg['mean_saving_bp'], '.2f', 8)}{lg.get('orders_without_leg', 0):>8}")
    for m in MODELS:
        v, g = r["gate_values"][m], r["gates"][m]
        lines.append(f"gates {m} T={PRIMARY_T}: gain {_fmt(v['gain_ann'], '.3%')}/yr, boot p5 "
                     f"{_fmt(v['boot_p5'], '.2e')}, years+ {v['years_positive']}/4, coins+ {v['coins_positive']}/5, "
                     f"Sharpe {_fmt(v['sharpe_maker'], '.3f')} vs {_fmt(v['sharpe_taker'], '.3f')}, maxDD "
                     f"{_fmt(v['max_dd_maker'], '.2%')} vs {_fmt(v['max_dd_taker'], '.2%')} -> "
                     + ("PASS" if all(g.values()) else "fail " + ", ".join(x[:3] for x, ok in g.items() if not ok)))
    for k, x in r["no_fallback"].items():
        lines.append(f"{k}: {x['missed_entries']} entries missed, mean R missed {_fmt(x['mean_R_missed'], '.3f')} "
                     f"vs taken {_fmt(x['mean_R_taken'], '.3f')}")
    sens = [f"{x['family']} ({x['period']})" for x in r["screen"] if x.get("execution_sensitive_strict")]
    lines.append(f"screen: execution-sensitive under strict: {', '.join(sens) or 'none'}")
    if "n_eff_family" in r:
        lines.append(f"F1X family N_eff {_fmt(r['n_eff_family'], '.2f')}; project N_eff "
                     f"{_fmt(r['n_eff_project'], '.2f')}")
    lines.append(f"DECISION: {r['decision']}")
    return "\n".join(lines)


# ---------------------------------------------------------------- guards and commands

def _check_pins(records: list[dict], pins: dict) -> None:
    for k, v in pins.items():
        if any(r.get(k) != v for r in records):
            raise SystemExit(f"X1 version {VERSION} already started with a different {k}; a change needs a version bump")


def _section_sha() -> str:
    return guards.sha_str(guards.require_section(PREREG, SECTION, REQUIRED_PHRASES))


def cmd_start() -> dict:
    """First call: log the start record (pins) and stop, asking for the commit. Later calls check it."""
    sec = _section_sha()
    guards.clean_src()
    pins = start_pins(sec)
    starts = guards.log_records(RUN_LOG, "x1-start", version=VERSION)
    _check_pins(starts, pins)
    if not guards.log_records(RUN_LOG, "x1-start", committed_only=True, version=VERSION):
        if not starts:
            _log_run("x1-start", version=VERSION, **pins)
        raise SystemExit("logged the X1 start record: commit research/x1/runs.jsonl, then run x1-stage-a")
    print(f"X1 start record for version {VERSION} is committed and matches the current code, registration and "
          f"libraries")
    return pins


def _require_start() -> dict:
    """A committed x1-start record whose pins equal the current ones, a committed section, clean src."""
    sec = _section_sha()
    guards.clean_src()
    pins = start_pins(sec)
    _check_pins(guards.log_records(RUN_LOG, "x1-start", version=VERSION), pins)
    if not guards.log_records(RUN_LOG, "x1-start", committed_only=True, version=VERSION):
        raise SystemExit("no committed X1 start record: run x1-start, commit research/x1/runs.jsonl, then run this "
                         "again")
    return pins


def _require_open() -> None:
    """A hard error logged by any stage (Stage A, a download's guard, Stage B) closes the version."""
    errs = [x for k in HARD_KINDS for x in guards.log_records(RUN_LOG, k, version=VERSION) if "hard_error" in x]
    if errs:
        raise SystemExit(f"X1 version {VERSION} stopped on a hard error: the version is closed; a fix needs a version "
                         f"bump ({errs[0]['kind']}: {errs[0]['hard_error']})")


def _stage_a_done() -> list[str]:
    """What shows that this version's Stage A ran, besides its run record (which a restored working copy of an
    uncommitted log would lose): its result file, its trial returns, its registry rows."""
    out = [str(p.name) for p in (result_path(STAGE_A_JSON), trial_file()) if p.exists()]
    reg = registry.load()
    if len(reg) and ((reg["strategy"] == CODE) & (pd.to_numeric(reg["version"], errors="coerce") == VERSION)).any():
        out.append(f"{CODE} v{VERSION} rows in the trial registry")
    return out


def cmd_stage_a() -> dict:
    pins = _require_start()
    if guards.log_records(RUN_LOG, "x1-stage-a-run", version=VERSION):
        raise SystemExit(f"X1 Stage A already ran for version {VERSION} (logged); it runs once")
    done = _stage_a_done()
    if done:
        raise SystemExit(f"X1 Stage A already ran for version {VERSION} ({', '.join(done)}); it runs once")
    _require_open()
    guards.require_committed(screen_inputs())
    prior = family_prior()                                      # checked before the run is logged
    _log_run("x1-stage-a-run", version=VERSION, **pins)        # from here on this version has run, whatever happens
    try:
        a = stage_a()
        r = write_stage_a(a, pins, prior)
    except Exception as e:  # noqa: BLE001 - any failure before the logged result is a hard error
        _log_run("x1-stage-a", version=VERSION, hard_error=f"{type(e).__name__}: {e}")
        raise SystemExit(f"hard error, logged: commit research/x1/runs.jsonl; a fix needs a version bump: "
                         f"{type(e).__name__}: {e}")
    _log_run("x1-stage-a", version=VERSION, decision=r["decision"], strict_pass=r["strict_pass"],
             gates=r["gates"], n_eff_family=r["n_eff_family"], n_eff_project=r["n_eff_project"],
             outputs_sha256=r["outputs_sha256"], stage_a_json_sha256=r["stage_a_json_sha256"],
             screen_inputs_sha256=r["screen_inputs_sha256"], **pins)
    print(format_stage_a(r))
    return r


def stage_a_record() -> dict:
    """The committed Stage A result of this version (refused once the version is closed)."""
    _require_open()
    done = [x for x in guards.log_records(RUN_LOG, "x1-stage-a", committed_only=True, version=VERSION)
            if "decision" in x]
    if not done:
        raise SystemExit("no committed X1 Stage A result: run x1-stage-a and commit research/x1/runs.jsonl")
    return done[-1]


def require_stage_a_inputs(rec: dict, names) -> dict[str, str]:
    """Stage A outputs a later stage reads: committed, and byte-identical to what the logged Stage A wrote."""
    out = stage_a_outputs()
    paths = [out[n] for n in names]
    guards.require_committed(paths)
    want = rec.get("outputs_sha256") or {}
    got = {n: _sha(p) for n, p in zip(names, paths)}
    bad = [n for n in names if want.get(n) != got[n]]
    if bad:
        raise SystemExit(f"Stage A outputs differ from the ones the logged Stage A result wrote: {bad}")
    return got


def _caching_lister(lister):
    base = lister or ticks._list_sizes
    cache: dict[str, list] = {}

    def get(prefix: str):
        if prefix not in cache:
            cache[prefix] = list(base(prefix))
        return cache[prefix]
    return get, cache


def b2_legs(legs: pd.DataFrame) -> pd.DataFrame:
    """The primary strict run's legs posted inside the bookTicker archive's dev span."""
    p = legs["post_ms"].astype(np.int64)
    return legs[(p >= _ms(BOOK_FROM)) & (p < _ms(HOLDOUT_START))].copy()


def download_jobs(tick_set: str, lister=None) -> tuple[pd.DataFrame, dict]:
    """build() jobs for B1 (seeded whole days) or B2 (windows around F1's legs), keys listed on the archive."""
    get, cache = _caching_lister(lister)
    report: dict = {"tick_set": tick_set}
    if tick_set == "B1":
        avail = ticks.dev_days(SYMBOLS, lister=get)
        pairs = ticks.select_b1_days(avail, seed=B1_SEED)
        windows = ticks.full_day_windows(pairs)
        report["days"] = [list(x) for x in pairs]
    elif tick_set == "B2":
        legs = b2_legs(pd.read_csv(x1_path(LEGS_CSV)))
        avail = ticks.dev_days(sorted(set(legs["symbol"])), lister=get)
        orders = legs.drop_duplicates(["symbol", "post_ms"])[["symbol", "post_ms"]].rename(
            columns={"post_ms": "t_live_ms"}).reset_index(drop=True)
        windows = ticks.leg_windows(orders, pre_ms=B2_PRE_MS, post_ms=B2_POST_MS, avail=avail)
        report |= {"legs": int(len(legs)), "orders": int(len(orders)), "dropped": windows.attrs.get("dropped", []),
                   "missing": windows.attrs.get("missing", [])}
    else:
        raise SystemExit(f"unknown tick set {tick_set!r}; expected one of {TICK_SETS}")
    jobs = ticks.jobs_from_windows(windows)
    sizes = {k: int(s) for rows in cache.values() for k, s in rows}
    keys = [ticks.daily_key(ds, sym, day) for ds, sym, day in zip(jobs["dataset"], jobs["symbol"], jobs["day"])]
    ticks.require_dev_keys(keys)
    report |= {"jobs": int(len(jobs)), "bytes": int(sum(sizes.get(k, 0) for k in keys)),
               "unlisted": [k for k in keys if k not in sizes]}
    return jobs, report


def cmd_download(tick_set: str, getter=None, lister=None, dry_run: bool = False, workers: int = 2) -> pd.DataFrame:
    if tick_set not in TICK_SETS:
        raise SystemExit(f"unknown tick set {tick_set!r}; expected one of {TICK_SETS}")
    pins = _require_start()
    rec = stage_a_record()
    if tick_set == "B2":
        if rec.get("strict_pass") is not True:
            raise SystemExit("B2 is downloaded only after a committed Stage A strict pass")
        require_stage_a_inputs(rec, [LEGS_CSV])
    jobs, rep = download_jobs(tick_set, lister)
    print(f"X1 {tick_set}: {rep['jobs']} archives, {rep['bytes'] / 1e9:.2f} GB listed"
          + (f", {len(rep['unlisted'])} not listed" if rep["unlisted"] else ""), flush=True)
    if rep["unlisted"]:
        raise SystemExit(f"archives not on the listing: {rep['unlisted'][:5]}")
    if dry_run:
        return jobs
    _log_run("x1-download-start", version=VERSION, tick_set=tick_set, jobs=rep["jobs"], bytes=rep["bytes"],
             code_sha256=pins["code_sha256"])
    try:
        man = ticks.build(jobs, manifest_path(tick_set), getter=getter, workers=workers)
        if len(man) and (pd.to_numeric(man["last_ts"], errors="coerce").max() >= _ms(HOLDOUT_START)):
            raise HardError("a manifest row reaches the holdout")
    except (HardError, ticks.HoldoutError) as e:   # a holdout guard tripped (not any OSError, e.g. a locked file)
        _log_run("x1-download", version=VERSION, tick_set=tick_set, hard_error=f"{type(e).__name__}: {e}")
        raise SystemExit(f"hard error, logged: commit research/x1/runs.jsonl; the version is closed: "
                         f"{type(e).__name__}: {e}")
    except Exception as e:  # noqa: BLE001 - a failed archive (network, checksum) is resumable, not a hard error
        _log_run("x1-download", version=VERSION, tick_set=tick_set, error=f"{type(e).__name__}: {e}")
        raise SystemExit(f"download incomplete (logged): {type(e).__name__}: {e}; run x1-download --set {tick_set} "
                         f"again to resume")
    _log_run("x1-download", version=VERSION, tick_set=tick_set, keys=int(len(man)),
             bytes=int(pd.to_numeric(man["bytes"], errors="coerce").sum()),
             manifest_sha256=guards.sha_text(manifest_path(tick_set)),
             **{k: rep[k] for k in ("dropped", "missing", "days") if k in rep})
    print(f"X1 {tick_set}: manifest {manifest_path(tick_set)} ({len(man)} archives)")
    return man


# ---------------------------------------------------------------- Stage B (tick model)

BOOK_COLS = ["transaction_time", "best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty"]
TAPE_COLS = ["transact_time", "price", "quantity", "is_buyer_maker"]


def _windows_of(path: Path) -> list[list[int]]:
    import pyarrow.parquet as pq
    md = pq.read_schema(path).metadata or {}
    return json.loads(md[b"x1_windows"])


def _read_rows(path: Path, cols: list[str], tcol: str, span: tuple[int, int] | None) -> pd.DataFrame:
    """One extracted file's columns in file order; with a span (start_ms, end_ms) only its rows in [start, end),
    filtered batch by batch (row groups outside it skipped), so a whole BTC day is never held at once."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    if span is None:
        return pd.read_parquet(path, columns=cols)
    a, e = int(span[0]), int(span[1])
    pf = pq.ParquetFile(path)
    ci = pf.schema_arrow.get_field_index(tcol)
    groups = []
    for g in range(pf.metadata.num_row_groups):
        st = pf.metadata.row_group(g).column(ci).statistics
        if st is not None and st.has_min_max and (st.max < a or st.min >= e):
            continue
        groups.append(g)
    schema = pa.schema([pf.schema_arrow.field(c) for c in cols])
    parts = []
    if groups:
        for batch in pf.iter_batches(batch_size=1 << 20, row_groups=groups, columns=cols):
            ts = batch.column(tcol)
            parts.append(batch.filter(pc.and_(pc.greater_equal(ts, a), pc.less(ts, e))).select(cols))
    return pa.Table.from_batches(parts, schema=schema).to_pandas()


def _read_stream(files: list[Path], cols: list[str], tcol: str, span) -> tuple[pd.DataFrame, int]:
    """Files concatenated in the given order, rows with missing or non-positive values dropped (counted), then a
    stable sort by time (file order kept at equal ms) when the rows are not already in time order."""
    parts = [_read_rows(p, cols, tcol, span) for p in files]
    df = parts[0] if len(parts) == 1 else pd.concat(parts, ignore_index=True)
    ok = np.ones(len(df), bool)
    for c in cols:
        if c != tcol and df[c].dtype.kind == "f":
            v = df[c].to_numpy(np.float64)
            ok &= np.isfinite(v) & (v > 0)
    dropped = int((~ok).sum())
    if dropped:
        df = df[ok]
    ts = df[tcol].to_numpy(np.int64)
    if len(ts) > 1 and (ts[1:] < ts[:-1]).any():
        df = df.sort_values(tcol, kind="stable")
    return df, dropped


def load_streams(files_book: list[Path], files_tape: list[Path], span: tuple[int, int] | None = None
                 ) -> tuple[queue_sim.Book, queue_sim.Tape, list, dict]:
    """Extracted bookTicker / aggTrades files as queue_sim streams (see _read_stream), with the extraction windows
    of the book files. span (start_ms, end_ms): only the rows inside it, and the windows cut to it."""
    b, b_drop = _read_stream(files_book, BOOK_COLS, "transaction_time", span)
    t, t_drop = _read_stream(files_tape, TAPE_COLS, "transact_time", span)
    hold = _ms(HOLDOUT_START)
    if (len(b) and b["transaction_time"].max() >= hold) or (len(t) and t["transact_time"].max() >= hold):
        raise HardError("a tick row reaches the holdout")
    book = queue_sim.Book(b["transaction_time"].to_numpy(np.int64), b["best_bid_price"].to_numpy(np.float64),
                          b["best_bid_qty"].to_numpy(np.float64), b["best_ask_price"].to_numpy(np.float64),
                          b["best_ask_qty"].to_numpy(np.float64))
    tape = queue_sim.Tape(t["transact_time"].to_numpy(np.int64), t["price"].to_numpy(np.float64),
                          t["quantity"].to_numpy(np.float64), t["is_buyer_maker"].to_numpy(bool))
    del b, t
    windows = [w for p in files_book for w in _windows_of(p)]
    if span is not None:
        a, e = int(span[0]), int(span[1])
        windows = [[max(int(w0), a), min(int(w1), e)] for w0, w1 in windows if min(int(w1), e) > max(int(w0), a)]
    return book, tape, windows, {"book_dropped": b_drop, "tape_dropped": t_drop}


def leg_spans(t_ms: np.ndarray, pre_ms: int, post_ms: int) -> list[tuple[int, int, np.ndarray]]:
    """[t - pre, t + post) per leg, overlapping ones merged: (start_ms, end_ms, positions of its legs in t_ms)."""
    t_ms = np.asarray(t_ms, np.int64)
    out: list[list] = []
    for k in np.argsort(t_ms, kind="stable"):
        a, e = int(t_ms[k]) - pre_ms, int(t_ms[k]) + post_ms
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
            out[-1][2].append(int(k))
        else:
            out.append([a, e, [int(k)]])
    return [(a, e, np.array(sorted(ks), np.int64)) for a, e, ks in out]


def _manifest_files(man: pd.DataFrame, symbol: str, days) -> tuple[list[Path], list[Path]] | None:
    out = {}
    for ds in ticks.DATASETS:
        rows = man[(man["dataset"] == ds) & (man["symbol"] == symbol) & man["day"].isin(list(days))]
        if len(rows) != len(set(days)):
            return None
        out[ds] = [ticks._resolve(f) for f in rows.sort_values("day")["out_file"]]
    return out["bookTicker"], out["aggTrades"]


def b1_simulate(man: pd.DataFrame, legs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """The Q3 generic legs inside each B1 day through the tick model, every T and both queues. Each day is read
    one span of legs at a time ([m0 - B1_PRE_MS, m0 + B1_POST_MS), overlaps merged), never as a whole file: a leg
    joins the book from at most 10 min before it goes live, as in B2."""
    rows, info = [], {"days": 0, "days_missing": [], "book_dropped": 0, "tape_dropped": 0, "spans": 0}
    mks = tuple(k * MINUTE_MS for k in MARKOUTS)
    pairs = man[["symbol", "day"]].drop_duplicates().sort_values(["symbol", "day"]).itertuples(index=False)
    for sym, day in pairs:
        files = _manifest_files(man, sym, [day])
        if files is None:
            info["days_missing"].append([sym, day])
            continue
        d0 = _ms(pd.Timestamp(day, tz="UTC"))
        g = legs[(legs["symbol"] == sym) & (legs["m0_ms"] >= d0) & (legs["m0_ms"] < d0 + TF_MS["1d"])
                 & legs["complete"].astype(bool)].reset_index(drop=True)
        if not len(g):
            continue
        info["days"] += 1
        m0 = g["m0_ms"].to_numpy(np.int64)
        side = g["side"].to_numpy()
        qty = Q3_NOTIONAL / g["open_px"].to_numpy(np.float64)
        res: dict[tuple, list[pd.DataFrame]] = {(T, q): [] for T in T_GRID for q in QUEUES}
        for a, e, ix in leg_spans(m0, B1_PRE_MS, B1_POST_MS):
            book, tape, windows, dropped = load_streams(*files, span=(a, e))
            info["spans"] += 1
            info["book_dropped"] += dropped["book_dropped"]
            info["tape_dropped"] += dropped["tape_dropped"]
            for T in T_GRID:
                for q in QUEUES:
                    r = queue_sim.simulate_legs(book, tape, m0[ix], side[ix], qty[ix], T * MINUTE_MS, queue=q,
                                                markouts_ms=mks, windows_ms=windows)
                    res[(T, q)].append(r.set_index(pd.Index(ix)))
            del book, tape
        for T in T_GRID:
            for q in QUEUES:
                r = pd.concat(res[(T, q)]).sort_index()
                out = g[["leg_id", "class", "symbol", "m0_ms", "side"]].assign(T=T, queue=q)
                for src, dst in (("status", "status"), ("complete", "complete_tick"), ("saving_bp", "saving_bp"),
                                 ("maker_share", "maker_share"), ("book_crossed", "book_crossed")):
                    out[dst] = r[src].to_numpy()
                for m in ("strict", "central"):
                    out[f"m1_{m}_saving_bp"] = g[f"saving_{m}_T{T}"].to_numpy()
                rows.append(out)
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), info


def cost_model(b1: pd.DataFrame, q3c: pd.DataFrame) -> pd.DataFrame:
    """X1-cost-v1: per class, symbol (and ALL) and T, the tick-model saving (conservative and pro-rata queues) on
    the B1 legs, the correction = tick conservative - 1m mean saving on the same legs, and the corrected c_eff."""
    ok = b1[(b1["status"] == "ok") & b1["complete_tick"].astype(bool)]
    base = {(c, s, T, m): float(v) for c, s, T, m, v in zip(q3c["class"], q3c["symbol"], q3c["T"], q3c["model"],
                                                            q3c["c_eff_bp"])}
    rows = []
    for cls in CLASSES:
        for sym in [*SYMBOLS, "ALL"]:
            for T in T_GRID:
                g = ok[(ok["class"] == cls) & (ok["T"] == T)]
                if sym != "ALL":
                    g = g[g["symbol"] == sym]
                cons = g[g["queue"] == "conservative"]
                pr = g[g["queue"] == "prorata"]
                both = cons.merge(pr[["leg_id", "saving_bp"]], on="leg_id", suffixes=("", "_prorata"))
                n = len(both)

                def mean(c):
                    return float(both[c].astype(float).mean()) if n else float("nan")
                tick_c, m1s, m1c = mean("saving_bp"), mean("m1_strict_saving_bp"), mean("m1_central_saving_bp")
                rows.append({"class": cls, "symbol": sym, "T": T, "n": n, "tick_conservative_bp": tick_c,
                             "tick_prorata_bp": mean("saving_bp_prorata"), "maker_share": mean("maker_share"),
                             "m1_strict_bp": m1s, "m1_central_bp": m1c,
                             "corr_strict_bp": tick_c - m1s, "corr_central_bp": tick_c - m1c,
                             "c_eff_1m_strict_bp": base.get((cls, sym, T, "strict"), np.nan),
                             "c_eff_1m_central_bp": base.get((cls, sym, T, "central"), np.nan),
                             "c_eff_v1_strict_bp": base.get((cls, sym, T, "strict"), np.nan) - (tick_c - m1s),
                             "c_eff_v1_central_bp": base.get((cls, sym, T, "central"), np.nan) - (tick_c - m1c)})
    return pd.DataFrame(rows)


def merged_split(r: pd.Series, side: float, q_exit: float, q_entry: float, fee_t: float, fee_m: float) -> list[dict]:
    """A merged reversal order (exit + entry at one price) is filled exit first: (exit, entry) maker share, average
    price and saving against the same taker price. A whole order maker-filled (a post price and no fallback price)
    fills both parts, whatever float dust filled_qty carries; a part within queue_sim.DUST of filled or empty is
    snapped, so a fallback price is mixed in only for a real partial fill."""
    full = bool(np.isfinite(r["post_px"]) and not np.isfinite(r["fallback_px"]))
    got = float(r["filled_qty"]) if np.isfinite(r["filled_qty"]) else 0.0
    out = []
    for q, f in ((q_exit, min(got, q_exit)), (q_entry, min(max(got - q_exit, 0.0), q_entry))):
        share = 1.0 if full else f / q
        share = 1.0 if share >= 1.0 - queue_sim.DUST else 0.0 if share <= queue_sim.DUST else share
        if share == 1.0:
            avg = r["post_px"]
        elif share == 0.0:
            avg = r["fallback_px"]
        else:
            avg = share * r["post_px"] + (1 - share) * r["fallback_px"]
        out.append({"maker_share": share, "avg_px": avg,
                    "saving_bp": (side * (r["taker_px"] - avg) / r["taker_px"] + share * (fee_t - fee_m)) * 1e4})
    return out


def b2_simulate(man: pd.DataFrame, legs: pd.DataFrame, costs: Costs = Costs()) -> tuple[pd.DataFrame, dict]:
    """F1's primary legs through the tick model at T = PRIMARY_T; a reversal (exit and entry posted at one minute
    on one side) is one order of both quantities, filled exit first."""
    legs = legs.reset_index(drop=True)
    out_rows, info = [], {"orders": 0, "no_files": 0}
    for (sym, post_ms, side), g in legs.groupby(["symbol", "post_ms", "side"], sort=True):
        a, b = int(post_ms) - B2_PRE_MS, int(post_ms) + B2_POST_MS
        days = [str(np.datetime64(d * TF_MS["1d"], "ms").astype("datetime64[D]"))
                for d in range(a // TF_MS["1d"], (b - 1) // TF_MS["1d"] + 1)]
        files = _manifest_files(man, sym, days)
        base = {"symbol": sym, "post_ms": int(post_ms), "side": int(side)}
        if files is None:
            info["no_files"] += len(g)
            out_rows += [base | {"leg": x, "status": "no_files"} for x in g["leg"]]
            continue
        book, tape, windows, _ = load_streams(*files, span=(a, b))
        info["orders"] += 1
        ex = g[g["leg"] == "exit"]
        en = g[g["leg"] == "entry"]
        q_x = float(ex["qty"].sum())
        q_n = float(en["qty"].sum())
        res = {}
        for q in QUEUES:
            res[q] = queue_sim.simulate_legs(book, tape, int(post_ms), int(side), q_x + q_n, PRIMARY_T * MINUTE_MS,
                                             queue=q, fee_t=costs.taker, fee_m=costs.maker,
                                             windows_ms=windows).iloc[0]
        for _, leg in g.iterrows():
            row = base | {"leg": leg["leg"], "qty": float(leg["qty"]), "merged": bool(len(ex) and len(en)),
                          "m1_strict_saving_bp": float(leg["saving_bp"])}
            for q in QUEUES:
                r = res[q]
                parts = merged_split(r, float(side), q_x, q_n, costs.taker, costs.maker) if len(ex) and len(en) \
                    else None
                one = (parts[0] if leg["leg"] == "exit" else parts[1]) if parts else \
                    {"maker_share": float(r["maker_share"]), "saving_bp": float(r["saving_bp"])}
                row |= {f"{q}_saving_bp": float(one["saving_bp"]), f"{q}_maker_share": float(one["maker_share"])}
            row |= {"status": res["conservative"]["status"], "complete": bool(res["conservative"]["complete"]),
                    "book_crossed": bool(res["conservative"]["book_crossed"])}
            out_rows.append(row)
        del book, tape
    return pd.DataFrame(out_rows), info


def check_f(b2: pd.DataFrame) -> dict:
    """(f): on the usable B2 legs the conservative tick model saves at least the strict 1m saving - 0.5 bp."""
    if len(b2) and {"status", "complete"} <= set(b2.columns):
        ok = (b2["status"] == "ok") & b2["complete"].eq(True)
    else:
        ok = pd.Series(False, index=b2.index)
    g = b2[ok]
    cols = [f"{q}_{c}" for q in QUEUES for c in ("saving_bp", "maker_share")] + ["m1_strict_saving_bp"]
    if len(g) and not np.isfinite(g[cols].to_numpy(np.float64)).all():
        raise HardError("a usable B2 leg has no tick or 1m saving: the means of (f) would cover different legs")
    tick_c = float(g["conservative_saving_bp"].mean()) if len(g) else float("nan")
    m1 = float(g["m1_strict_saving_bp"].mean()) if len(g) else float("nan")
    return {"legs": int(len(b2)), "legs_used": int(len(g)), "excluded": int(len(b2) - len(g)),
            "tick_conservative_mean_bp": tick_c,
            "tick_prorata_mean_bp": float(g["prorata_saving_bp"].mean()) if len(g) else float("nan"),
            "strict_1m_mean_bp": m1, "tolerance_bp": F_TOL_BP,
            "f_pass": bool(len(g) > 0 and tick_c >= m1 - F_TOL_BP)}


def cmd_stage_b(tick_set: str) -> dict:
    if tick_set not in TICK_SETS:
        raise SystemExit(f"unknown tick set {tick_set!r}; expected one of {TICK_SETS}")
    pins = _require_start()
    rec = stage_a_record()
    if tick_set == "B2" and rec.get("strict_pass") is not True:
        raise SystemExit("B2 is evaluated only after a committed Stage A strict pass")
    if guards.log_records(RUN_LOG, "x1-stage-b-run", version=VERSION, tick_set=tick_set):
        raise SystemExit(f"X1 Stage B {tick_set} already ran for version {VERSION} (logged); it runs once")
    p = result_path(STAGE_B_JSON)
    out = json.loads(p.read_text()) if p.exists() else {"study": "X1", "stage": "B", "code": CODE, "version": VERSION}
    done = ([tick_set + " in " + p.name] if tick_set in out else []) + \
        [x1_path(n).name for n in STAGE_B_OUTPUTS[tick_set] if x1_path(n).exists()]
    if done:
        raise SystemExit(f"X1 Stage B {tick_set} already ran for version {VERSION} ({', '.join(done)}); it runs once")
    man_p = manifest_path(tick_set)
    guards.require_committed([man_p])
    man_sha = guards.sha_text(man_p)
    if not any(x.get("manifest_sha256") == man_sha and not ({"error", "hard_error"} & set(x)) for x in
               guards.log_records(RUN_LOG, "x1-download", committed_only=True, version=VERSION, tick_set=tick_set)):
        raise SystemExit(f"{man_p.name} is not the manifest of a completed, committed x1-download --set {tick_set}")
    inputs = require_stage_a_inputs(rec, [Q3_LEGS, Q3_CSV] if tick_set == "B1" else [LEGS_CSV])
    _log_run("x1-stage-b-run", version=VERSION, tick_set=tick_set, manifest_sha256=man_sha, inputs_sha256=inputs,
             **pins)
    try:
        man = ticks.read_manifest(man_p)
        if tick_set == "B1":
            b1, info = b1_simulate(man, pd.read_parquet(result_path(Q3_LEGS)))
            if not len(b1):
                raise HardError("no B1 leg could be simulated")
            cm = cost_model(b1, pd.read_csv(x1_path(Q3_CSV)))
            X1_DIR.mkdir(parents=True, exist_ok=True)
            cm.to_csv(x1_path(COST_CSV), index=False, lineterminator="\n")
            b1.to_parquet(result_path(B1_LEGS))
            res = {"info": info, "cost_model": x1_path(COST_CSV).name, "cost_sha256": _sha(x1_path(COST_CSV)),
                   "legs": int(len(b1))}
        else:
            b2, info = b2_simulate(man, b2_legs(pd.read_csv(x1_path(LEGS_CSV))))
            f = check_f(b2)
            b2.to_csv(x1_path(B2_LEGS_CSV), index=False, lineterminator="\n")
            go = bool(rec.get("strict_pass") is True and f["f_pass"])
            res = {"info": info, "f": f, "go_shadow": go, "decision": "GO-SHADOW" if go else "NO-GO"}
        out[tick_set] = res | {"manifest_sha256": man_sha, "inputs_sha256": inputs, **pins}
        _save_json(p, out)
    except Exception as e:  # noqa: BLE001
        _log_run("x1-stage-b", version=VERSION, tick_set=tick_set, hard_error=f"{type(e).__name__}: {e}")
        raise SystemExit(f"hard error, logged: commit research/x1/runs.jsonl; a fix needs a version bump: "
                         f"{type(e).__name__}: {e}")
    _log_run("x1-stage-b", version=VERSION, tick_set=tick_set, **{k: v for k, v in res.items() if k != "info"},
             manifest_sha256=man_sha, inputs_sha256=inputs)
    print(format_stage_b(out))
    return out


def format_stage_b(r: dict) -> str:
    lines = [f"X1 Stage B ({r.get('code')} v{r.get('version')})"]
    if "B1" in r:
        lines.append(f"B1: {r['B1']['legs']} tick-leg runs, cost model {r['B1']['cost_model']} "
                     f"sha256 {r['B1']['cost_sha256'][:12]}")
    if "B2" in r:
        f = r["B2"]["f"]
        lines.append(f"B2: {f['legs_used']}/{f['legs']} legs, tick conservative "
                     f"{_fmt(f['tick_conservative_mean_bp'], '.2f')} bp vs strict 1m "
                     f"{_fmt(f['strict_1m_mean_bp'], '.2f')} bp (tolerance {f['tolerance_bp']}) -> (f) "
                     f"{'pass' if f['f_pass'] else 'FAIL'}; DECISION: {r['B2']['decision']}")
    return "\n".join(lines)


def cmd_report() -> None:
    for name, fmt in ((STAGE_A_JSON, format_stage_a), (STAGE_B_JSON, format_stage_b)):
        p = result_path(name)
        print(fmt(json.loads(p.read_text())) if p.exists() else f"{name}: not run yet")
