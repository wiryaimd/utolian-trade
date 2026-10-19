"""Round 8 (IBS): daily IBS mean reversion (Pagonidis 2014) on the SPY, QQQ and EWY proxies of Binance TradFi perps,
two co-primaries (IBS-L, IBS-LS), judged under the practical rulebook plus the round's own gates (alpha, and the
null at p < 0.01 per primary). See research/PREREGISTRATION.md, Round 8.

    python -m btcq.research.run r8-calibrate                     # synthetic, before any data
    python -m btcq.research.run r8-power                         # synthetic, before any data
    python -m btcq.research.run r8-data                          # OHLC from Round 6's raw files, blind report
    python -m btcq.research.run r8-sweep                         # twice: the start record, then the dev run
    python -m btcq.research.run r8-validate --unlock-holdout     # once, only after a dev pass and the owner's OK

Round 8 keeps its own trial registry and N_eff history (research/round8/), its results in reports/results/round8/
(run.all_trial_returns never sees them) and its own holdout lock (btcq.data.external_ohlc, tag [R8:). It never calls
run.cmd_gates and never unlocks the crypto 2024+ data or Round 6's other markets.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from btcq.backtest import daily_sim as DS
from btcq.backtest import metrics, stats
from btcq.config import ROOT, Risk
from btcq.data import external_ohlc as X
from btcq.data import store
from btcq.research import guards, registry
from btcq.research.round5 import Window, _dsr
from btcq.research.round8_calib import (CAL_DRAWS, CAL_GATE_P, CAL_MAX_P01, CAL_MIN_DETECT, CAL_ROWS,  # noqa: F401
                                        CALIB_CSV, calib_sha, cmd_calibrate, lib_versions_calib)
from btcq.research.run import RESULTS, _save_json
from btcq.strategies import r8_ibs as R

CODE = "IBS"
R8_DIR = X.R8_DIR
R8_RESULTS = RESULTS / "round8"          # a subfolder: run.all_trial_returns (non-recursive glob) never sees it
TRIALS = R8_DIR / "trials.csv"
NEFF_LOG = R8_DIR / "neff_history.jsonl"
RUN_LOG = R8_DIR / "runs.jsonl"
DATA_REPORT = R8_DIR / "data_report.csv"
POWER_CSV = R8_DIR / "power_table.csv"
PREREG = ROOT / "research" / "PREREGISTRATION.md"
SECTION = "## Round 8 (IBS)"         # outcomes go in separate top-level "## Round 8 ... outcome" sections
REQUIRED_PHRASES = ("IBS-L", "IBS-LS", "Null calibration", "Power table", "Dev gates", "Holdout gates", "[R8:")
F1_DEV_DAILY = RESULTS / "F1_dev_daily.parquet"
F1_HOLD_DAILY = RESULTS / "F1_holdout_sym_daily.parquet"
M1_TRIALS = RESULTS / "round6" / "M1_trial_returns.parquet"
VERDICT = R8_RESULTS / f"{CODE}.json"
CODE_FILES = ["src/btcq/data/external_ohlc.py", "src/btcq/data/external.py", "src/btcq/strategies/r8_ibs.py",
              "src/btcq/strategies/r6_macro.py", "src/btcq/strategies/r5_d1.py", "src/btcq/strategies/base.py",
              "src/btcq/backtest/daily_sim.py", "src/btcq/backtest/stats.py", "src/btcq/backtest/metrics.py",
              "src/btcq/research/round8.py", "src/btcq/research/round8_calib.py", "src/btcq/research/round5.py",
              "src/btcq/research/guards.py", "src/btcq/research/registry.py", "src/btcq/research/run.py",
              "src/btcq/config.py", "src/btcq/features/ta.py", "src/btcq/data/store.py"]

DAY = pd.Timedelta(days=1)
DEV = Window("dev", "dev", pd.Timestamp("1993-07-01", tz="UTC"), X.HOLDOUT_START)
HOLD = Window("holdout", "full", X.HOLDOUT_START, X.END)
EQ0 = Risk().initial_equity
SLOTS = len(R.MARKETS)
SLEEVE_EQ0 = EQ0 / SLOTS
N_DRAWS = 1000
NULL_SEED = 8008
BOOT_N = 10_000
BOOT_BLOCK = 30
BOOT_SEED = 7
MIN_TRADES = 100
MAX_YEAR_SHARE = 0.65
MIN_DSR = 0.95
MIN_ALPHA_PSR = 0.95
MAX_NULL_P = CAL_GATE_P                  # 0.01: owner decision 2026-10-20, after the first calibration failed
MIN_MARKETS_POSITIVE = 2
HOLDOUT_SHARPE_FRAC = 0.5
HOLDOUT_ALPHA_FRAC = 0.5
FUNDING_STRESS_ANN = 0.05
LIQ_LEV = 20.0
NEFF_FLOOR = 14.849570528171945 + 5      # the strict tier only: global max N_eff + one per post-round-4 family
# Maker-first (information only): X1-cost-v1 c_eff per side at T = 15 (research/x1/x1_cost_v1.csv, ALL rows) against
# X1's 6.6 bp taker: entries use the reversion class (6.18 bp), exits the random class (4.89 bp).
X1_TAKER_BP = 6.6
MAKER_ENTRY_SAVING = (X1_TAKER_BP - 6.17928965576635) * 1e-4
MAKER_EXIT_SAVING = (X1_TAKER_BP - 4.891326848761046) * 1e-4
MAKER_LABEL = ("X1-cost-v1, calibrated on Binance crypto majors in 2023, not TradFi books; a resting order also moves "
               "the fill off the close (information only)")
ROLL_ERAS = (("1993-2000", "1993-01-01", "2001-01-01"), ("2001-2008", "2001-01-01", "2009-01-01"),
             ("2009-2015", "2009-01-01", "2016-01-01"))
VERIFY_TAG = X.VERIFY_TAG


class HardError(RuntimeError):
    """An integrity check failed (a bug or a data problem, never a result)."""


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
    import platform

    import numba
    import pyarrow
    import scipy
    return {"numpy_version": np.__version__, "numba_version": numba.__version__, "pandas_version": pd.__version__,
            "scipy_version": scipy.__version__, "pyarrow_version": pyarrow.__version__,
            "python_version": platform.python_version()}


def _info(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except Exception as e:  # noqa: BLE001 - information only; never affects a gate
        return {"error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------- data

def rows_from_frame(sym: str, df: pd.DataFrame) -> R.MarketRows:
    return R.MarketRows(sym, df["ts"].to_numpy(np.int64), df["open"].to_numpy(np.float64),
                        df["high"].to_numpy(np.float64), df["low"].to_numpy(np.float64),
                        df["close"].to_numpy(np.float64))


def load_markets(w: Window) -> list[R.MarketRows]:
    """The 3 markets' OHLC up to the window's end (dev: rows before 2016; holdout: needs the Round 8 unlock)."""
    out = []
    for s in R.MARKETS:
        df = X.load_ohlc(s, "dev" if w.period == "dev" else "full")
        if len(df) and int(df["ts"].iloc[-1]) >= _ms(w.end):
            raise HardError(f"{s}: rows at or after the window's end were loaded")
        if not np.isfinite(df["close"].to_numpy()).all() or (df["close"] <= 0).any():
            raise HardError(f"{s}: missing or non-positive closes")
        out.append(rows_from_frame(s, df))
    return out


def window_rows(m: R.MarketRows, w: Window) -> tuple[int, int]:
    """(first, last) own rows inside the window; first > last when the market has none."""
    ts = np.asarray(m.ts, np.int64)
    return int(np.searchsorted(ts, _ms(w.start))), int(np.searchsorted(ts, _ms(w.end))) - 1


def full_book_day(mkts: list[R.MarketRows]) -> pd.Timestamp:
    """The first trade date on which every market's sigma exists."""
    days = []
    for m in mkts:
        fin = np.flatnonzero(np.isfinite(R.vol_size_td(m.close)))
        if not len(fin):
            raise HardError(f"{m.sym}: sigma never exists")
        days.append(int(m.ts[fin[0]]))
    return pd.Timestamp(max(days), unit="ms", tz="UTC")


# ---------------------------------------------------------------- sleeves and the account

class Sleeves:
    """One config's 3 sleeves (each $10,000 / 3, compounding its own equity) and the account built from them:
    account daily return = the sum of the sleeves' daily returns / 3 on the UTC calendar (0 on days without trading
    and for a market that is not ready)."""

    def __init__(self, mkts, res: dict, w: Window, sdaily: dict[str, pd.Series]):
        self.mkts, self.res, self.w, self.sdaily = mkts, res, w, sdaily
        self.daily = sum(sdaily.values()) / SLOTS
        self.mdaily = pd.DataFrame({s: d / SLOTS for s, d in sdaily.items()})

    def trades(self) -> pd.DataFrame:
        parts = []
        for m in self.mkts:
            r = self.res.get(m.sym)
            if r is None or not len(r.trades):
                continue
            t = r.trades.copy()
            t["symbol"] = m.sym
            t["entry_date"] = pd.to_datetime(m.ts[t["entry_i"].to_numpy()], unit="ms", utc=True)
            t["exit_date"] = pd.to_datetime(m.ts[t["exit_i"].to_numpy()], unit="ms", utc=True)
            parts.append(t)
        cols = DS.TRADE_COLS + ["ret", "symbol", "entry_date", "exit_date"]
        return (pd.concat(parts, ignore_index=True).sort_values(["entry_date", "symbol"]).reset_index(drop=True)
                if parts else pd.DataFrame(columns=cols))

    def funded(self, rate_ann: float = FUNDING_STRESS_ANN) -> pd.Series:
        """The account under the funding stress: each sleeve's drag (daily_sim.funding_drag) is subtracted from
        its daily return (floored at -100%)."""
        out = []
        for m in self.mkts:
            d = self.sdaily[m.sym]
            r = self.res.get(m.sym)
            if r is not None:
                d = (d - DS.funding_drag(m.ts, r.lev, self.w.calendar, rate_ann)).clip(lower=-1.0)
            out.append(d)
        return sum(out) / SLOTS

    def grid_activity(self) -> dict:
        """Orders and time in market on the window's trade dates (union of the markets' dates)."""
        lo, hi = _ms(self.w.start), _ms(self.w.end)
        dates = np.unique(np.concatenate([m.ts[(m.ts >= lo) & (m.ts < hi)] for m in self.mkts]))
        orders = pd.Series(0, index=dates)
        held = pd.Series(False, index=dates)
        fees = slipc = turn = 0.0
        for m in self.mkts:
            r = self.res.get(m.sym)
            if r is None:
                continue
            sl = slice(r.first, r.last + 1)
            ts = m.ts[sl]
            orders.loc[ts] += (r.turnover[sl] > 0).astype(int)
            held.loc[ts] |= r.lev[sl] != 0
            fees += float(r.fees.sum())
            slipc += float(r.slip_cost.sum())
            turn += float(r.turnover.sum())
        n = max(len(dates), 1)
        yrs = len(self.w.calendar) / 365
        return {"trading_days": int(len(dates)), "orders_per_trading_day": float(orders.sum() / n),
                "share_days_with_order": float((orders > 0).mean()) if len(dates) else 0.0,
                "share_days_in_market": float(held.mean()) if len(dates) else 0.0,
                "fees_total": fees, "slippage_total": slipc, "turnover_per_year": turn / yrs}


def run_config(mkts: list[R.MarketRows], ci: int, w: Window, *, mult: float = 1.0, lag: int = 0,
               fill: str = "close", maker: bool = False, cache: dict | None = None) -> Sleeves:
    cfg = R.CONFIGS[ci]
    res, sd = {}, {}
    for m in mkts:
        key = (m.sym, ci)
        if cache is not None and key in cache:
            pos, size = cache[key]
        else:
            pos, size = R.positions(m.high, m.low, m.close, cfg)
            if cache is not None:
                cache[key] = (pos, size)
        first, last = window_rows(m, w)
        if first > last:
            res[m.sym] = None
            sd[m.sym] = pd.Series(0.0, index=w.calendar)
            continue
        if fill == "close":
            px = m.close
        elif fill == "open":
            px = np.where(np.isfinite(m.open), m.open, m.close)     # information only: a missing open fills at C
        else:
            raise ValueError(fill)
        fee = R.TAKER * mult
        fin, fout = (fee - MAKER_ENTRY_SAVING, fee - MAKER_EXIT_SAVING) if maker else (fee, fee)
        r = DS.simulate(m.close, px, pos, size, first, last, lag=lag, slip=R.SLIP[m.sym] * mult, eq0=SLEEVE_EQ0,
                        fee_in=fin, fee_out=fout, lev_cap=R.LEV_CAP, liq_lev=LIQ_LEV)
        res[m.sym] = r
        sd[m.sym] = DS.calendar_daily(m.ts, r.equity, w.calendar, SLEEVE_EQ0)
    return Sleeves(mkts, res, w, sd)


def ew_benchmark(mkts: list[R.MarketRows], w: Window) -> pd.Series:
    """Buy-and-hold scaled like the account: each market's close-to-close return on its own trade date, summed over
    the markets with a return that day and divided by 3 (days with none are 0). No costs, no funding."""
    cal = w.calendar
    tot = pd.Series(0.0, index=cal)
    for m in mkts:
        r = m.close[1:] / m.close[:-1] - 1.0
        s = pd.Series(r, index=pd.to_datetime(m.ts[1:], unit="ms", utc=True).as_unit("ms"))
        s = s[s.index.isin(cal)]
        tot.loc[s.index] += s.to_numpy()
    return tot / SLOTS


def _max_dd(daily: pd.Series) -> float:
    return metrics.max_drawdown(np.r_[1.0, (1.0 + daily).cumprod().to_numpy()])


def _sharpe(daily: pd.Series) -> float:
    sd = daily.std(ddof=1)
    return float(daily.mean() / sd * np.sqrt(365)) if sd > 0 else 0.0


def summary(sl: Sleeves) -> dict:
    d = sl.daily
    t = sl.trades()
    yrs = len(d) / 365
    total = float((1 + d).prod() - 1)
    held = (t["exit_date"] - t["entry_date"]).dt.days if len(t) else pd.Series(dtype=float)
    out = {"trades": int(len(t)), "trades_per_year": float(len(t) / yrs), "total_return": total,
           "cagr": float((1 + total) ** (1 / yrs) - 1) if total > -1 else -1.0, "sharpe": _sharpe(d),
           "vol_ann": float(d.std(ddof=1) * np.sqrt(365)), "mean_daily": float(d.mean()), "max_dd": _max_dd(d),
           "win_rate": float((t["net"] > 0).mean()) if len(t) else 0.0,
           "avg_trade_ret": float(t["ret"].mean()) if len(t) else 0.0,
           "avg_days_held": float(held.mean()) if len(t) else 0.0,
           "long_trades": int((t["dir"] == 1).sum()) if len(t) else 0,
           "short_trades": int((t["dir"] == -1).sum()) if len(t) else 0,
           "long_net": float(t.loc[t["dir"] == 1, "net"].sum()) if len(t) else 0.0,
           "short_net": float(t.loc[t["dir"] == -1, "net"].sum()) if len(t) else 0.0,
           "liquidations": [s for s, r in sl.res.items() if r is not None and r.liq_row >= 0],
           "days": int(len(d))}
    return out | sl.grid_activity()


def _brief(d: pd.Series) -> dict:
    return {"mean_daily": float(d.mean()), "sharpe": _sharpe(d), "total_return": float((1 + d).prod() - 1),
            "max_dd": _max_dd(d)}


def roll_spread(mkts: list[R.MarketRows], w: Window) -> dict:
    """Information: Roll's implied spread 2 sqrt(-cov(r_t, r_t-1)) in bp, by market and era (window rows only;
    None when the autocovariance is not negative)."""
    out = {}
    for m in mkts:
        r = pd.Series(m.close[1:] / m.close[:-1] - 1.0, index=m.ts[1:])
        r = r[(r.index >= _ms(w.start)) & (r.index < _ms(w.end))]
        row = {}
        for name, a, b in ROLL_ERAS:
            x = r[(r.index >= _ms(pd.Timestamp(a, tz="UTC"))) & (r.index < _ms(pd.Timestamp(b, tz="UTC")))].to_numpy()
            if len(x) < 30:
                row[name] = None
                continue
            c = float(np.cov(x[1:], x[:-1], ddof=1)[0, 1])
            row[name] = float(2 * np.sqrt(-c) * 1e4) if c < 0 else None
        out[m.sym] = row
    return out


def breakdowns(sp: R.StrataPanel, v: np.ndarray, daily: pd.Series) -> dict:
    """Information: the null statistic's gross sized P&L per held market-day by the decision date's volatility
    tercile, gap (a weekend or holiday follows) and weekday; the account's mean daily return on Mondays and on
    other trading days."""
    held = (v != 0)
    contrib = np.where(held, v * np.nan_to_num(sp.Y), 0.0)
    wd = pd.to_datetime(sp.dates, unit="ms", utc=True).weekday.to_numpy()

    def by(mask):
        n = int(held[mask].sum())
        return {"market_days": n, "mean": float(contrib[mask].sum() / n) if n else None}

    vol = {f"tercile_{g}": by(sp.vol_bin == g) for g in (0, 1, 2)} | {"warmup": by(sp.vol_bin == R.WARMUP)}
    acc_wd = daily.index.weekday
    return {"by_vol_tercile": vol, "gap": by(sp.gap == 1), "no_gap": by((sp.gap == 0) & sp.in_window),
            "by_weekday": {str(k): by(wd == k) for k in range(5)},
            "account_monday_mean": float(daily[acc_wd == 0].mean()),
            "account_tue_fri_mean": float(daily[(acc_wd >= 1) & (acc_wd <= 4)].mean())}


def evaluate(mkts: list[R.MarketRows], w: Window, with_null: bool = True, n_draws: int = N_DRAWS) -> dict:
    """Everything the gates and the report need for one window, for both primaries."""
    t0 = time.time()
    cache: dict = {}
    base = {i: run_config(mkts, i, w, cache=cache) for i in range(len(R.CONFIGS))}
    cfg_daily = {i: s.daily for i, s in base.items()}
    for d in cfg_daily.values():
        if d.index[0] != w.start or d.index[-1] != w.end - DAY:
            raise HardError("the daily returns do not span the window")
    F = pd.concat(cfg_daily, axis=1)
    n_eff = stats.effective_trials(F.to_numpy())
    pbo, logits = stats.pbo_cscv(F.to_numpy())
    totals = {i: float((1 + d).prod() - 1) for i, d in cfg_daily.items()}
    bench = ew_benchmark(mkts, w)
    sp = R.strata_panel(mkts, _ms(w.start), _ms(w.end))
    prim = {}
    for name, ci in R.PRIMARIES.items():
        b = base[ci]
        daily = b.daily
        s2 = run_config(mkts, ci, w, mult=2.0, cache=cache)
        fund = b.funded()
        contrib = {s: float(x) for s, x in b.mdaily.sum().items()}
        v = R.held_vectors(sp, mkts, R.CONFIGS[ci])
        rec = {"config": R.CONFIGS[ci], "summary": summary(b), "weekly": stats.weekly_t(daily),
               "stress_2x_mean_daily": float(s2.daily.mean()), "stress_2x": _brief(s2.daily),
               "funding_mean_daily": float(fund.mean()), "funding": _brief(fund),
               "funding_cost_ann": float((daily.mean() - fund.mean()) * 365),
               "stress_2x_plus_funding_mean_daily": float(s2.funded().mean()),
               "maker_first": _brief(run_config(mkts, ci, w, maker=True, cache=cache).daily) | {"label": MAKER_LABEL},
               "fill_next_close": _brief(run_config(mkts, ci, w, lag=1, cache=cache).daily),
               "fill_next_open": _brief(run_config(mkts, ci, w, lag=1, fill="open", cache=cache).daily),
               "year_share": stats.year_share_daily(daily),
               "year_returns": {int(y): float(x) for y, x in daily.groupby(daily.index.year).sum().items()},
               "market_contrib": contrib, "markets_positive": int(sum(x > 0 for x in contrib.values())),
               "market_trades": {s: int((b.trades()["symbol"] == s).sum()) for s in R.MARKETS},
               "plateau": {str(j): totals[j] > 0 for j in R.NEIGHBOURS[name]},
               "dsr_family": _dsr(daily, n_eff), "dsr_raw12": _dsr(daily, float(len(R.CONFIGS))),
               "alpha_vs_ew": stats.beta_alpha(daily, bench),
               "breakdowns": _info(breakdowns, sp, v, daily),
               "daily": daily, "sleeves": b, "stress_sleeves": s2, "funding_daily": fund}
        if with_null:
            try:
                rec["null"] = R.null_test(sp, v, n_draws, NULL_SEED)
            except ValueError as e:
                raise HardError(str(e)) from e
        prim[name] = rec
    return {"primaries": prim, "n_eff_family": n_eff, "pbo": pbo, "pbo_logits": logits,
            "config_totals": {str(i): x for i, x in totals.items()},
            "config_sharpes": {str(i): _sharpe(d) for i, d in cfg_daily.items()},
            "config_trades": {str(i): int(len(s.trades())) for i, s in base.items()},
            "cfg_daily": cfg_daily, "benchmark_daily": bench,
            "benchmark": _brief(bench) | {"sharpe": _sharpe(bench)},
            "roll_spread": _info(roll_spread, mkts, w),
            "window": [str(w.start.date()), str((w.end - DAY).date())], "runtime_s": time.time() - t0}


# ---------------------------------------------------------------- gates and verdict

def gates_dev(rec: dict) -> dict:
    return {k: bool(v) for k, v in _gates_dev(rec).items()}


def _gates_dev(rec: dict) -> dict:
    return {"trades>=100": rec["summary"]["trades"] >= MIN_TRADES,
            "2x-cost mean daily>0": rec["stress_2x_mean_daily"] > 0,
            "funding-stress mean daily>0": rec["funding_mean_daily"] > 0,
            "null p<0.01": rec["null"]["p_value"] < MAX_NULL_P,
            "DSR>=0.95 at the family N_eff": rec["dsr_family"]["dsr"] >= MIN_DSR,
            "alpha PSR>=0.95 vs the account-scaled hold": rec["alpha_vs_ew"]["psr_alpha"] >= MIN_ALPHA_PSR,
            "best year<=65%": rec["year_share"] <= MAX_YEAR_SHARE,
            "plateau: 3 neighbours positive": all(rec["plateau"].values()),
            ">=2 of 3 markets positive": rec["markets_positive"] >= MIN_MARKETS_POSITIVE}


def candidate(dev: dict) -> str | None:
    """The primary that can pass: the one that passed every dev gate; if both did, the higher dev alpha PSR, then (PSR
    saturates at 1) the higher dev alpha Sharpe, then IBS-L. Fixed in the dev verdict, before any unlock."""
    passed = [n for n in R.PRIMARIES if dev[n]["dev_pass"]]
    if not passed:
        return None
    return max(passed, key=lambda n: (dev[n]["alpha_psr"], dev[n]["alpha_sharpe"], -list(R.PRIMARIES).index(n)))


def boot_source(dev_daily: pd.Series, full_book: str) -> pd.Series:
    return dev_daily[dev_daily.index >= pd.Timestamp(full_book, tz="UTC")]


def holdout_dd_limit(dev_daily: pd.Series, full_book: str) -> float:
    src = boot_source(dev_daily, full_book)
    _, mdd = stats.block_bootstrap(src.to_numpy(), BOOT_N, block=BOOT_BLOCK, n_out=len(HOLD.calendar),
                                   seed=BOOT_SEED)
    return float(np.quantile(mdd, 0.95))


def gates_holdout(rec: dict, dev: dict, dd_limit: float) -> dict:
    s, a = rec["summary"], rec["alpha_vs_ew"]
    return {k: bool(v) for k, v in {
            "holdout mean daily>0": s["mean_daily"] > 0,
            "holdout Sharpe>=50% of dev": s["sharpe"] >= HOLDOUT_SHARPE_FRAC * dev["sharpe"],
            "holdout worst drawdown<=dev bootstrap p95": s["max_dd"] <= dd_limit,
            "holdout alpha mean>0": a["alpha_ann"] > 0,
            "holdout alpha Sharpe>=50% of dev": a["sharpe_alpha_ann"] >= HOLDOUT_ALPHA_FRAC * dev["alpha_sharpe"]}.items()}


def strict_n() -> float:
    """The strict tier's N: max(the project's ratcheted N_eff, Round 8's own ratchet, 14.85 + 5)."""
    vals = [NEFF_FLOOR]
    for log in (registry.NEFF_LOG, NEFF_LOG):
        if log.exists():
            vals += [json.loads(x)["pr"] for x in log.read_text().splitlines() if x.strip()]
    return float(max(vals))


# ---------------------------------------------------------------- data report (holdout rows: counts only)

def market_report(sym: str, df: pd.DataFrame) -> dict:
    after = int((df["ts"] >= _ms(X.END)).sum())
    df = df[df["ts"] < _ms(X.END)]                      # rows after 2026-09-30 belong to no window: counted only
    ts = df["ts"].to_numpy(np.int64)
    o, h, lo, c = (df[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
    cut, end = _ms(X.HOLDOUT_START), _ms(X.END)
    dev, hold = ts < cut, (ts >= cut) & (ts < end)
    tol = 1e-6 * np.abs(c)
    lo_ok, hi_ok = np.isfinite(lo), np.isfinite(h)
    mn = np.where(np.isfinite(o), np.minimum(o, c), c)
    mx = np.where(np.isfinite(o), np.maximum(o, c), c)
    viol = (lo_ok & (lo - mn > tol)) | (hi_ok & (mx - h > tol)) | (lo_ok & hi_ok & (lo - h > tol))
    flat = lo_ok & hi_ok & (h == lo)
    null_hl = ~lo_ok | ~hi_ok
    rng = np.where(lo_ok & hi_ok, (h - lo) / c, np.nan)
    rd = rng[dev]
    dates = pd.to_datetime(ts, unit="ms", utc=True).strftime("%Y-%m-%d").to_numpy()
    mid = dev & (rng > 0.15) & (rng <= 0.30)
    big = dev & (rng > 0.30)
    out = {"market": sym, "rows": int(len(ts)), "rows_dev": int(dev.sum()), "rows_holdout": int(hold.sum()),
           "rows_after_holdout": after,
           "first_date": dates[0], "last_date": dates[-1]}
    for name, m in (("dev", dev), ("holdout", hold)):
        out |= {f"{name}_null_open": int((~np.isfinite(o) & m).sum()), f"{name}_null_high": int((~hi_ok & m).sum()),
                f"{name}_null_low": int((~lo_ok & m).sum()), f"{name}_ohlc_violations": int((viol & m).sum()),
                f"{name}_high_eq_low": int((flat & m).sum())}
    n = len(ts)
    out |= {"null_high_or_low_share": float(null_hl.sum() / n), "ohlc_violation_share": float(viol.sum() / n),
            "dev_high_eq_low_share": float(flat[dev].sum() / max(dev.sum(), 1)),
            "dev_max_range": float(np.nanmax(rd)) if np.isfinite(rd).any() else None,
            "dev_range_15_30_dates": ";".join(dates[mid]), "dev_range_gt_30_dates": ";".join(dates[big]),
            "dev_range_gt_30": int(big.sum()), "closes_match_round6": True}
    return out


def red_flags(df: pd.DataFrame) -> pd.DataFrame:
    """The registered red flags (1, a SHA256 or close mismatch, is raised while building)."""
    df = df.copy()
    checks = {"null H or L >0.5% of rows": df["null_high_or_low_share"] > 0.005,
              "OHLC inconsistent on >0.1% of rows": df["ohlc_violation_share"] > 0.001,
              "H = L on >1% of dev rows": df["dev_high_eq_low_share"] > 0.01,
              "dev range (H-L)/C >30%": df["dev_range_gt_30"] > 0,
              "closes differ from Round 6": ~df["closes_match_round6"].astype(str).eq("True")}
    flags = pd.DataFrame(checks)
    df["red_flag_reasons"] = flags.apply(lambda r: ";".join(k for k, x in r.items() if x), axis=1)
    df["red_flag"] = flags.any(axis=1)
    return df


def data_report() -> pd.DataFrame:
    return red_flags(pd.DataFrame([market_report(s, X._verified_frame(s)) for s in R.MARKETS]))


# ---------------------------------------------------------------- guards

def calibration_problems(cal: pd.DataFrame) -> list[str]:
    """The committed calibration must be the registered one, run by the current calibration code, and pass."""
    if not {"scenario", "kind", "rule", "planted_edge", "reps", "draws", "reject_01", "ok",
            "calib_sha256"} <= set(cal.columns):
        return ["the calibration file is not in the registered format"]
    probs = []
    want = [(sc, kind, rule, float(edge), reps) for sc, edge, reps, kind in CAL_ROWS for rule in R.PRIMARIES]
    have = list(zip(cal["scenario"], cal["kind"], cal["rule"], cal["planted_edge"].astype(float), cal["reps"]))
    if have != want or not cal["draws"].eq(CAL_DRAWS).all():
        probs.append("the calibration does not have the registered scenarios, rules, replications and draws")
    if not cal["calib_sha256"].eq(calib_sha()).all():
        probs.append("the calibration was run by other calibration code")
    no_edge = cal[cal["kind"] == "no-edge"]
    within = no_edge["reject_01"] <= CAL_MAX_P01
    if not len(no_edge) or not no_edge["ok"].astype(str).eq("True").all() or not within.all():
        probs.append("a no-edge calibration row is above its limit")
    det = cal[(cal["kind"] == "planted edge") & (cal["planted_edge"].astype(float) == max(e for _, e, _, _ in CAL_ROWS))]
    if not len(det) or not (det["reject_01"] >= CAL_MIN_DETECT).all():
        probs.append("the planted edge (kappa 0.10) is not detected")
    vers = lib_versions_calib()
    if any(k not in cal or not cal[k].astype(str).eq(v).all() for k, v in vers.items()):
        probs.append("the calibration ran under other library versions")
    return probs


def r6_touched() -> list[dict]:
    """Round 6 holdout records (any committed version or the working copy): one would contaminate these markets'
    2016+ holdout."""
    return [r for r in guards.all_records(store.HOLDOUT_LOG)
            if r.get("round") == 6 or "[R6" in str(r.get("reason", r.get("raw", "")))]


def _require_clean_holdout() -> None:
    if r6_touched():
        raise SystemExit("the holdout log has a Round 6 record: SPY/QQQ/EWY's 2016+ holdout is not clean")


def _require_prereg() -> str:
    """The committed Round 8 section, power table and passing calibration. Returns the section's SHA256."""
    _require_clean_holdout()
    guards.require_committed([PREREG, POWER_CSV, CALIB_CSV])
    probs = calibration_problems(pd.read_csv(CALIB_CSV))
    if probs:
        raise SystemExit("; ".join(probs))
    return guards.sha_str(guards.require_section(PREREG, SECTION, REQUIRED_PHRASES))


def _require_data_report() -> None:
    guards.require_committed([DATA_REPORT, X.MANIFEST])
    if red_flags(pd.read_csv(DATA_REPORT))["red_flag"].any():
        raise SystemExit("the Round 8 data report has red flags")
    X.check_files()                             # raises unless every file matches both manifests


def holdout_touches() -> list[dict]:
    """Every sign that the holdout was opened: Round 8 records of the holdout log (unlock or verify, any committed
    version or the working copy) and validate-start records of the run log (written just before the unlock), so
    restoring one log can't hide an unlock."""
    return X.r8_unlock_records() + guards.log_records(RUN_LOG, "r8-validate-start")


def _hard_error_versions() -> set[int]:
    return {int(r["version"]) for r in guards.log_records(RUN_LOG, "r8-sweep") if r.get("hard_error")}


def _holdout_touched(r: dict) -> list[str]:
    out = []
    if "holdout" in r:
        out.append("the Round 8 holdout was already evaluated")
    if holdout_touches():
        out.append("a Round 8 holdout unlock (or a load) was already logged")
    return out


def verify_unlock(note: str) -> None:
    """Independent checks after the holdout verdict load the window again; each load is logged ([R8 verify)."""
    if not (VERDICT.exists() and "holdout_gates" in json.loads(VERDICT.read_text())):
        raise PermissionError("no Round 8 holdout verdict yet: checks come after it")
    guards.clean_src()
    X.unlock_r8(f"Round 8 independent check {VERIFY_TAG} {note}]", kind="verify")


# ---------------------------------------------------------------- power

def ready_dates() -> dict[str, pd.Timestamp]:
    """The day each market's sigma exists, from the registered source first dates (facts, not prices): its 91st
    NYSE trading day (90 returns), on the rule calendar of Round 6."""
    from btcq.data import external as E
    from btcq.research.round6 import KNOWN_CLOSURES, nyse_holidays
    hol = np.array(sorted(nyse_holidays(1993, 2026) | KNOWN_CLOSURES), dtype="datetime64[D]")
    out = {}
    for s in R.MARKETS:
        d0 = np.datetime64(E.SOURCES[s].first_expected, "D")
        out[s] = pd.Timestamp(str(np.busday_offset(d0, R.VOL_DAYS, roll="forward", holidays=hol)), tz="UTC")
    return out


def ramp_weights(w: Window = DEV) -> np.ndarray:
    """Share of the 3 slots live on each calendar day of the window (from ready_dates)."""
    cal = w.calendar
    rd = ready_dates()
    return np.sum([(cal >= rd[s]).astype(float) for s in R.MARKETS], axis=0) / SLOTS


def _groups(w: np.ndarray) -> list[tuple[float, int]]:
    vals, counts = np.unique(np.asarray(w, np.float64), return_counts=True)
    return [(float(v), int(c)) for v, c in zip(vals, counts) if v > 0] + \
        ([(0.0, int(counts[vals == 0].sum()))] if (vals == 0).any() else [])


def _uni(rng, m: int, groups, mu: float):
    """Sample mean and sample s.d. (ddof 1) of m series of independent days, day t ~ N(w mu, w) (unit full-book
    variance), from per-group sufficient statistics (exact for Gaussian days)."""
    T = sum(n for _, n in groups)
    means, ss = [], np.zeros(m)
    for wg, ng in groups:
        if wg == 0:
            means.append(np.zeros(m))
            continue
        means.append(rng.normal(wg * mu, np.sqrt(wg / ng), m))
        ss += wg * rng.chisquare(ng - 1, m) if ng > 1 else 0.0
    mean = sum(ng * mg for (_, ng), mg in zip(groups, means)) / T
    ss = ss + sum(ng * (mg - mean) ** 2 for (_, ng), mg in zip(groups, means))
    return mean, np.sqrt(ss / (T - 1))


def _bi(rng, m: int, groups, mu_b: float, mu_a: float, beta: float, sd_b: float):
    """Sufficient statistics of (benchmark b, strategy r = beta b + a) over independent Gaussian days: day t has
    b ~ N(w mu_b, w sd_b^2) and a ~ N(w mu_a, w) independent. Returns (mean r, sd r, alpha mean, alpha sd) with the
    alpha from an OLS beta fitted on the sample, as stats.beta_alpha."""
    T = sum(n for _, n in groups)
    mb, ma, Sbb, Sba, Saa = [], [], np.zeros(m), np.zeros(m), np.zeros(m)
    for wg, ng in groups:
        if wg == 0:
            mb.append(np.zeros(m))
            ma.append(np.zeros(m))
            continue
        mb.append(rng.normal(wg * mu_b, sd_b * np.sqrt(wg / ng), m))
        ma.append(rng.normal(wg * mu_a, np.sqrt(wg / ng), m))
        c1 = np.sqrt(rng.chisquare(ng - 1, m))           # Bartlett decomposition of Wishart(ng - 1, w diag(sd_b^2, 1))
        c2 = np.sqrt(rng.chisquare(ng - 2, m))
        z = rng.standard_normal(m)
        Sbb += wg * sd_b ** 2 * c1 ** 2
        Sba += wg * sd_b * c1 * z
        Saa += wg * (z ** 2 + c2 ** 2)
    Mb = sum(ng * x for (_, ng), x in zip(groups, mb)) / T
    Ma = sum(ng * x for (_, ng), x in zip(groups, ma)) / T
    for (_, ng), xb, xa in zip(groups, mb, ma):
        Sbb = Sbb + ng * (xb - Mb) ** 2
        Sba = Sba + ng * (xb - Mb) * (xa - Ma)
        Saa = Saa + ng * (xa - Ma) ** 2
    Mr = beta * Mb + Ma
    Srr = beta ** 2 * Sbb + 2 * beta * Sba + Saa
    Srb = beta * Sbb + Sba
    bhat = Srb / Sbb
    a_mean = Mr - bhat * Mb
    a_sd = np.sqrt(np.maximum(Srr - Srb ** 2 / Sbb, 0.0) / (T - 1))
    return Mr, np.sqrt(Srr / (T - 1)), a_mean, a_sd


def _psr_z(sr, T):
    return sr * np.sqrt(T - 1) / np.sqrt(1 + 0.5 * sr ** 2)


ALPHA_BETA_SHARE = 0.5      # beta x sd(benchmark) / sd(alpha): the benchmark explains 20% of the strategy's variance
ALPHA_BENCH_SHARPE = 0.45   # the account-scaled hold's annual Sharpe (calendar days, sqrt 365)


def power_table(sharpes=(0.0, 0.25, 0.5, 0.75, 1.0), n_sims: int = 200_000, dev_days: int | None = None,
                hold_days: int | None = None, n_effs=(1.0, 2.0, 3.0, 4.0), rhos=(1.0, 0.5, 0.0),
                variants=("constant", "ramp", "ramp+alpha"), seed: int = 8, chunk: int = 50_000) -> list[dict]:
    """Round 8 protocol power under iid Gaussian calendar-day returns (sqrt 365), from exact sufficient statistics.

    sharpe_ann is the true full-book dev Sharpe (for ramp+alpha: the true alpha Sharpe); the holdout's true Sharpe is
    rho x that. p_dev = P(dev mean > 0 and DSR >= 0.95 at n_eff, null variance 1/(T-1)); p_pass adds holdout mean > 0
    and holdout Sharpe >= 50% of dev. ramp: the dev mean scales with the live share w of the 3 slots and the variance
    with w (ramp_weights). ramp+alpha: the strategy is beta x b + a with the account-scaled hold b (annual Sharpe
    0.45, beta sd(b) = 0.5 sd(a)); dev also needs alpha PSR >= 0.95 and the holdout alpha mean > 0 and alpha Sharpe
    >= 50% of dev's, the alpha from an OLS beta fitted in each window. The null, 2x-cost, funding, year-share,
    plateau, breadth and holdout drawdown gates are not modelled (each can only lower these numbers)."""
    from scipy.stats import norm
    dev_days = dev_days or len(DEV.calendar)
    hold_days = hold_days or len(HOLD.calendar)
    w_ramp = ramp_weights()[:dev_days] if any(v != "constant" for v in variants) else None
    v0 = stats.null_sr_variance(dev_days)
    z95 = norm.ppf(0.95)
    out = []
    for vi, var in enumerate(variants):
        g_dev = [(1.0, dev_days)] if var == "constant" else _groups(w_ramp)
        g_hold = [(1.0, hold_days)]
        fb_years = sum(wg * ng for wg, ng in g_dev) / 365
        for si, s in enumerate(sharpes):
            mu = s / np.sqrt(365)
            k = {(n, r): [0, 0] for n in n_effs for r in rhos}
            rng = np.random.default_rng([seed, vi, si])
            for st in range(0, n_sims, chunk):
                m = min(chunk, n_sims - st)
                if var == "ramp+alpha":
                    mu_b = ALPHA_BENCH_SHARPE / np.sqrt(365) * ALPHA_BETA_SHARE
                    md, sdd, amd, asd = _bi(rng, m, g_dev, mu_b, mu, 1.0, ALPHA_BETA_SHARE)
                    a_ok = norm.cdf(_psr_z(amd / asd, dev_days)) >= 0.95
                    a_sr_d = amd / asd
                else:
                    md, sdd = _uni(rng, m, g_dev, mu)
                    a_ok, a_sr_d = True, None
                sr_d = md / sdd
                hold = {}
                for r in rhos:
                    if var == "ramp+alpha":
                        mh, sdh, amh, ash = _bi(rng, m, g_hold, mu_b, r * mu, 1.0, ALPHA_BETA_SHARE)
                        hold[r] = (mh > 0) & (mh / sdh >= 0.5 * sr_d) & (amh > 0) & (amh / ash >= 0.5 * a_sr_d)
                    else:
                        mh, sdh = _uni(rng, m, g_hold, r * mu)
                        hold[r] = (mh > 0) & (mh / sdh >= 0.5 * sr_d)
                for n in n_effs:
                    star = stats.expected_max_sr(v0, n)
                    dev = (md > 0) & (norm.cdf((sr_d - star) * np.sqrt(dev_days - 1) / np.sqrt(1 + 0.5 * sr_d ** 2))
                                      >= 0.95) & a_ok
                    for r in rhos:
                        k[(n, r)][0] += int(dev.sum())
                        k[(n, r)][1] += int((dev & hold[r]).sum())
            for n in n_effs:
                star = stats.expected_max_sr(v0, n)
                for r in rhos:
                    out.append({"variant": var, "sharpe_ann": s, "n_eff": n, "rho": r,
                                "p_dev": k[(n, r)][0] / n_sims, "p_pass": k[(n, r)][1] / n_sims,
                                "sharpe_needed_ann": float((star + z95 / np.sqrt(dev_days - 1)) * np.sqrt(365)),
                                "full_book_years_dev": float(fb_years), "dev_days": dev_days,
                                "hold_days": hold_days, "n_sims": n_sims})
    return out


# ---------------------------------------------------------------- commands

def cmd_power() -> pd.DataFrame:
    rd = ready_dates()
    df = pd.DataFrame(power_table())
    R8_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(POWER_CSV, index=False, lineterminator="\n")
    w = ramp_weights()
    print("ready dates: " + ", ".join(f"{s} {d.date()}" for s, d in rd.items()) +
          f" | ramp: {w.sum() / 365:.1f} full-book years of {len(w) / 365:.1f}")
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(df.drop(columns=["dev_days", "hold_days", "n_sims"]).round(4).to_string(index=False))
    return df


def cmd_data() -> pd.DataFrame:
    """Step 3 of the run order: OHLC from Round 6's raw files, the blind data report and the manifest."""
    _require_prereg()
    guards.clean_src()
    if DATA_REPORT.exists() or X.MANIFEST.exists():
        raise SystemExit("the Round 8 data was already built (r8-data runs once; a rebuild after a red flag needs a "
                         "dated amendment)")
    try:
        X.build_all()
    except (X.DataError, RuntimeError) as e:          # a raw-file or close mismatch with Round 6: red flag 1
        _log_run("r8-data", error=f"red flag 1: {type(e).__name__}: {e}")
        raise SystemExit(f"red flag 1 (raw SHA256 or close mismatch with Round 6), logged: {e}")
    except FileExistsError as e:
        raise SystemExit(f"the Round 8 data was already built (r8-data runs once): {e}")
    except Exception as e:  # noqa: BLE001 - a failed build stops the round like a red flag
        _log_run("r8-data", error=f"build failed: {type(e).__name__}: {e}")
        raise SystemExit(f"the Round 8 build failed (it stops the round until a dated amendment), logged: {e}")
    df = data_report()
    R8_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_REPORT, index=False, lineterminator="\n")
    bad = df[df["red_flag"]]
    _log_run("r8-data", red_flags=bad["market"].tolist(), manifest_sha256=guards.sha_text(X.MANIFEST),
             data_report_sha256=guards.sha_text(DATA_REPORT), code_sha256=code_sha())
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        print(df.to_string(index=False))
    if len(bad):
        raise SystemExit("data red flags: a dated amendment is needed before any strategy run")
    return df


def _trial_rows(ev: dict, version: int) -> list[dict]:
    rows = []
    for i, d in ev["cfg_daily"].items():
        sr, T, sk, ku = stats.sharpe_moments(d)
        rows.append({"strategy": CODE, "version": version, "params": R.CONFIGS[i], "period": "dev", "sr": sr, "T": T,
                     "skew": sk, "kurt": ku, "trades": int(ev["config_trades"][str(i)]),
                     "sharpe_ann": float(sr * np.sqrt(365))})
    return rows


def trial_file(version: int):
    return R8_RESULTS / (f"{CODE}_trial_returns.parquet" if version == 1 else f"{CODE}v{version}_trial_returns.parquet")


def sweep_pins(sec_sha: str) -> dict:
    return {"code_sha256": code_sha(), "manifest_sha256": guards.sha_text(X.MANIFEST),
            "r6_manifest_sha256": guards.sha_text(X.R6_MANIFEST), "data_report_sha256": guards.sha_text(DATA_REPORT),
            "f1_daily_sha256": guards.sha_file(F1_DEV_DAILY), "prereg_section_sha256": sec_sha, **lib_versions()}


YARDSTICKS = ("dev_pass", "gates", "sharpe", "alpha_psr", "alpha_sharpe", "holdout_dd_limit")


def yardsticks(dev: dict) -> dict:
    """What the holdout gates and the candidate choice depend on, per primary (logged in the committed r8-sweep
    record and compared with IBS.json before the unlock)."""
    return {n: {k: d.get(k) for k in YARDSTICKS} for n, d in dev.items()}


_SKIP = {"daily", "sleeves", "stress_sleeves", "funding_daily"}


def _public_rec(rec: dict) -> dict:
    out = {k: v for k, v in rec.items() if k not in _SKIP}
    if "null" in out:
        out["null"] = {k: v for k, v in out["null"].items() if k != "dist"}
    return out


def _public(ev: dict) -> dict:
    out = {k: v for k, v in ev.items() if k not in ("primaries", "cfg_daily", "benchmark_daily", "pbo_logits")}
    out["primaries"] = {n: _public_rec(r) for n, r in ev["primaries"].items()}
    return out


def _write(tag: str, ev: dict) -> None:
    R8_RESULTS.mkdir(parents=True, exist_ok=True)
    for name, rec in ev["primaries"].items():
        stem = f"{name}_{tag}"
        rec["daily"].rename("ret").to_frame().to_parquet(R8_RESULTS / f"{stem}_daily.parquet")
        rec["sleeves"].trades().to_parquet(R8_RESULTS / f"{stem}_trades.parquet")
        rec["stress_sleeves"].trades().to_parquet(R8_RESULTS / f"{stem}_stress_2x_trades.parquet")
        rec["sleeves"].mdaily.to_parquet(R8_RESULTS / f"{stem}_market_daily.parquet")
        rec["funding_daily"].rename("ret").to_frame().to_parquet(R8_RESULTS / f"{stem}_funding_daily.parquet")
        if "null" in rec:
            np.save(R8_RESULTS / f"{stem}_null_dist.npy", rec["null"]["dist"])
    ev["benchmark_daily"].rename("ret").to_frame().to_parquet(R8_RESULTS / f"{CODE}_{tag}_benchmark.parquet")
    if "cfg_daily" in ev:
        pd.concat({str(i): d for i, d in ev["cfg_daily"].items()}, axis=1).to_parquet(
            R8_RESULTS / f"{CODE}_{tag}_config_daily.parquet")


def _verdicts() -> list[dict]:
    """Every logged dev verdict (any version, committed or in the working copy)."""
    return [r for r in guards.log_records(RUN_LOG, "r8-sweep") if "verdict" in r]


def cmd_sweep(rerun: bool = False, n_draws: int = N_DRAWS) -> dict:
    """The dev sweep runs once. A version bump may rerun it only after a logged hard error (no verdict): once any
    version has a dev verdict (PASS or FAIL) the rule is never re-run or varied (plan section 6)."""
    sec_sha = _require_prereg()
    _require_data_report()
    guards.clean_src()
    meta = R.IBS()
    v = meta.version
    old = json.loads(VERDICT.read_text()) if VERDICT.exists() else {}
    closed = _holdout_touched(old)
    if closed:
        raise SystemExit("Round 8 is closed once its holdout is touched (no reruns): " + "; ".join(closed))
    if rerun:
        raise SystemExit("Round 8 has no reruns: the dev sweep runs once (a fix after a hard error is a version bump)")
    if old or _verdicts():
        raise SystemExit("Round 8 has a dev verdict: the dev sweep runs once, with no reruns or variants")
    started = {int(x["version"]) for kind in ("r8-sweep-start", "r8-sweep-run")
               for x in guards.log_records(RUN_LOG, kind)}
    stopped = {int(x["version"]) for x in guards.log_records(RUN_LOG, "r8-sweep", committed_only=True)
               if x.get("hard_error")}
    early = sorted(x for x in started if x < v and x not in stopped)
    if early:
        raise SystemExit(f"earlier version(s) {early} started without a committed hard error: a version bump is "
                         "allowed only after a committed hard error")
    if v in _hard_error_versions():
        raise SystemExit(f"version {v} stopped on a hard error; a fix needs a version bump")
    if guards.log_records(RUN_LOG, "r8-sweep", version=v) or guards.log_records(RUN_LOG, "r8-sweep-run", version=v):
        raise SystemExit(f"version {v} already ran its dev sweep (logged); a fix needs a version bump")
    pins = sweep_pins(sec_sha)
    starts = guards.log_records(RUN_LOG, "r8-sweep-start", version=v)
    for k, val in pins.items():
        if any(r.get(k) != val for r in starts):
            raise SystemExit(f"version {v} already started with a different {k}; a change needs a version bump")
    if not guards.log_records(RUN_LOG, "r8-sweep-start", committed_only=True, version=v):
        if not starts:
            _log_run("r8-sweep-start", version=v, **pins)
        raise SystemExit("logged the Round 8 sweep start record: commit research/round8/runs.jsonl, then run "
                         "r8-sweep again")
    _log_run("r8-sweep-run", version=v, **pins)       # from here on this version has run, whatever happens
    try:
        mkts = load_markets(DEV)
        fb = str(full_book_day(mkts).date())
        ev = evaluate(mkts, DEV, n_draws=n_draws)
        dev, gates = {}, {}
        for name, rec in ev["primaries"].items():
            if rec["daily"].index.max() >= HOLD.start or rec["daily"].index.min() < DEV.start:
                raise HardError("dev output leaves the dev window")
            g = gates_dev(rec)
            gates[name] = g
            a = rec["alpha_vs_ew"]
            dev[name] = {"dev_pass": bool(all(g.values())), "gates": g, "sharpe": float(rec["summary"]["sharpe"]),
                         "alpha_psr": float(a["psr_alpha"]), "alpha_sharpe": float(a["sharpe_alpha_ann"]),
                         "holdout_dd_limit": holdout_dd_limit(rec["daily"], fb)}
        cand = candidate(dev)
        r = {"code": CODE, "name": meta.name, "tf": "1d", "version": v, "round": 8, "hypothesis": meta.hypothesis,
             "rules": meta.rules, "configs": R.CONFIGS, "primaries": R.PRIMARIES, "neighbours": R.NEIGHBOURS,
             "markets": R.MARKETS, "holdout_scope": "clean (general-knowledge and planning-snippet disclosures)",
             "rulebook": "practical (Amendments 1 and 2) + Round 8 gates (alpha, null p < 0.01)",
             "start": str(DEV.start.date()), "end": str((DEV.end - DAY).date()), "full_book_day": fb,
             "dev": dev, "candidate": cand, "dev_pass_any": cand is not None, "n_eff_family": ev["n_eff_family"],
             "holdout_bootstrap": {"n": BOOT_N, "block": BOOT_BLOCK, "seed": BOOT_SEED, "n_out": len(HOLD.calendar),
                                   "source_from": fb},
             "dev_detail": _public(ev), "git_commit": guards.git("rev-parse", "HEAD"), **pins}
        _save_json(VERDICT, r)                   # the gated verdict is saved before anything else is written
    except Exception as e:  # noqa: BLE001 - any failure before the verdict is a logged hard error
        _log_run("r8-sweep", version=v, hard_error=f"{type(e).__name__}: {e}")
        raise SystemExit(f"hard error, logged: commit research/round8/runs.jsonl; a fix needs a version bump: "
                         f"{type(e).__name__}: {e}")
    _log_run("r8-sweep", version=v, verdict={n: d["dev_pass"] for n, d in dev.items()}, candidate=cand, gates=gates,
             yardsticks=yardsticks(dev), n_eff_family=ev["n_eff_family"], full_book_day=fb,
             runtime_s=ev["runtime_s"], **pins)
    for name, rec in ev["primaries"].items():
        s, nl = rec["summary"], rec["null"]
        failed = [k for k, x in gates[name].items() if not x]
        print(f"{name} dev: trades={s['trades']} sharpe={s['sharpe']:.2f} total={s['total_return']:.2%} "
              f"null p={nl['p_value']:.3f} DSR(N_eff {ev['n_eff_family']:.2f})={rec['dsr_family']['dsr']:.3f} "
              f"alpha PSR={rec['alpha_vs_ew']['psr_alpha']:.3f} -> "
              f"{'PASS' if dev[name]['dev_pass'] else 'FAIL: ' + ', '.join(failed)}", flush=True)
    print(f"candidate for the holdout: {cand}"
          + (" (stop here: the unlock needs the owner's OK)" if cand else " (no candidate: Round 8 FAILS dev)"))
    registry.record(_trial_rows(ev, v), path=TRIALS)
    R8_RESULTS.mkdir(parents=True, exist_ok=True)
    pd.concat({(f"{CODE}_{i}" if v == 1 else f"{CODE}_v{v}_{i}"): d for i, d in ev["cfg_daily"].items()}, axis=1) \
        .to_parquet(trial_file(v))
    trial = pd.concat([pd.read_parquet(p) for p in sorted(R8_RESULTS.glob("*_trial_returns.parquet"))], axis=1)
    n_own = registry.ratcheted_effective_trials(stats.effective_trials(trial.fillna(0.0).to_numpy()), trial.shape[1],
                                                note=f"round 8 (IBS v{v})", log=NEFF_LOG)
    n_strict = strict_n()
    r |= {"n_eff_own_ratchet": n_own, "n_strict": n_strict,
          "dsr_dev_strict_n": {n: _dsr(rec["daily"], n_strict) for n, rec in ev["primaries"].items()},
          "n_trials": registry.multiple_testing_inputs("dev", path=TRIALS)[0]}
    _save_json(VERDICT, r)
    _write("dev", ev)
    r["dev_info"] = {"dsr_with_m1": _info(dsr_with_m1, ev)}
    _save_json(VERDICT, r)
    _log_run("r8-sweep-info", version=v, n_eff_own_ratchet=n_own, n_strict=n_strict)
    return r


def dsr_with_m1(ev: dict) -> dict:
    """Information: N_eff of the 12 IBS configs together with M1's 3, over their common days, and each primary's
    DSR at it."""
    j = pd.concat([pd.concat(ev["cfg_daily"], axis=1), pd.read_parquet(M1_TRIALS)], axis=1, join="inner")
    n = stats.effective_trials(j.to_numpy())
    return {"n_eff": n, "days": int(len(j)), **{name: _dsr(rec["daily"], n) for name, rec in ev["primaries"].items()}}


PIN_KEYS = ("code_sha256", "manifest_sha256", "r6_manifest_sha256", "data_report_sha256", "f1_daily_sha256",
            "prereg_section_sha256", "numpy_version", "numba_version", "pandas_version", "scipy_version",
            "pyarrow_version", "python_version")


def preflight_holdout() -> list[str]:
    """Everything checked before the unlock, from hashes, timestamps and committed files only. Empty = go."""
    probs = []
    r = json.loads(VERDICT.read_text()) if VERDICT.exists() else {}
    dev = r.get("dev", {})
    cand = r.get("candidate")
    if not cand:
        probs.append("no Round 8 primary passed dev")
    for n, d in dev.items():
        if bool(d.get("dev_pass")) != bool(d.get("gates") and all(d["gates"].values())):
            probs.append(f"{n}: dev_pass does not match its gates")
    if r and cand != candidate(dev):
        probs.append("the recorded candidate is not the registered choice")
    if r and r.get("version") != R.IBS().version:
        probs.append(f"IBS.version {R.IBS().version} is not the swept version {r.get('version')}")
    if r and r.get("code_sha256") != code_sha():
        probs.append("Round 8 code changed since the dev sweep")
    if r and {k: r.get(k) for k in lib_versions()} != lib_versions():
        probs.append("numpy/numba/pandas/scipy/pyarrow/python versions changed since the dev sweep")
    for k, pth in (("manifest_sha256", X.MANIFEST), ("r6_manifest_sha256", X.R6_MANIFEST),
                   ("data_report_sha256", DATA_REPORT)):
        if r and r.get(k) != (guards.sha_text(pth) if pth.exists() else None):
            probs.append(f"{pth.name} changed since the dev sweep")
    if r and r.get("f1_daily_sha256") != (guards.sha_file(F1_DEV_DAILY) if F1_DEV_DAILY.exists() else None):
        probs.append("F1_dev_daily.parquet changed or is missing since the dev sweep")
    if r:
        try:
            sec = guards.sha_str(guards.committed_section(PREREG, SECTION))
        except SystemExit as e:
            sec = str(e)
        if r.get("prereg_section_sha256") != sec:
            probs.append("the committed Round 8 registration changed since the dev sweep")
        if guards.log_records(RUN_LOG, "r8-validate"):
            probs.append("Round 8 already has a holdout result or a recorded holdout failure")
        verdicts = [x for x in guards.log_records(RUN_LOG, "r8-sweep", committed_only=True, version=r.get("version"))
                    if "verdict" in x]
        if not any(x.get("candidate") == cand and x.get("yardsticks") == yardsticks(dev)
                   and all(x.get(k) == r.get(k) for k in PIN_KEYS) for x in verdicts):
            probs.append("no committed r8-sweep record matching IBS.json (candidate, yardsticks and pins)")
        if len(_verdicts()) != 1:
            probs.append("Round 8 must have exactly one logged dev verdict")
    if r6_touched():
        probs.append("the holdout log has a Round 6 record: the 2016+ holdout of these markets is not clean")
    try:
        _require_data_report()
    except (SystemExit, Exception) as e:  # noqa: BLE001
        probs.append(str(e))
    try:
        guards.require_committed([VERDICT, RUN_LOG, TRIALS, NEFF_LOG, DATA_REPORT, X.MANIFEST, CALIB_CSV, POWER_CSV,
                                  F1_DEV_DAILY, trial_file(int(r.get("version", 1)))]
                                 + [R8_RESULTS / f"{n}_dev_daily.parquet" for n in R.PRIMARIES])
    except SystemExit as e:
        probs.append(str(e))
    if "holdout" in r:
        probs.append("the Round 8 holdout was already evaluated")
    return probs


def _record_holdout_failure(r: dict, v: int, err: str) -> None:
    r |= {"holdout": {"integrity_error": err}, "holdout_gates": {"no error after the unlock": False},
          "holdout_status": "error after the unlock: FAIL (pre-registered, no fix or rerun)",
          "headline": {"candidate": r.get("candidate"), "PASSES_IN_PRACTICE": False, "STRICT_TIER": False},
          "final_pass": False}
    _save_json(VERDICT, r)
    _log_run("r8-validate", version=v, hard_error=err)


def f1_compare(daily: pd.Series, f1: pd.Series) -> dict:
    j = pd.concat([daily.rename("x"), f1.rename("f1")], axis=1, sort=True).dropna()
    wk = pd.concat([stats.weekly_returns(j["x"]), stats.weekly_returns(j["f1"])], axis=1).dropna()
    return {"weekly_corr": float(wk.corr().iloc[0, 1]), "days": int(len(j)),
            "hac": stats.hac_ols(j["x"].to_numpy(), j["f1"].to_numpy(), lag=7)}


def cmd_validate(reason: str) -> dict:
    """The one holdout evaluation (after a dev pass and the owner's OK). There is no resume: an unlock that left no
    result, and any error after the unlock, is recorded as a FAIL."""
    r = json.loads(VERDICT.read_text()) if VERDICT.exists() else {}
    if r and "holdout" not in r and not guards.log_records(RUN_LOG, "r8-validate") and holdout_touches():
        v = int(r.get("version", 0))
        _record_holdout_failure(r, v, "an earlier unlock attempt (or load) of the holdout left no result")
        raise SystemExit("an earlier Round 8 unlock left no result: recorded as a FAIL (pre-registered)")
    probs = preflight_holdout() + _holdout_touched(r)
    if probs:
        raise SystemExit("Round 8 holdout stays locked: " + "; ".join(dict.fromkeys(probs)))
    guards.clean_src()
    if "[R" in reason:
        raise SystemExit("the reason must not carry a round tag (the tag is added by the unlock)")
    v = int(r["version"])
    cand = r["candidate"]
    passed = [n for n in R.PRIMARIES if r["dev"][n]["dev_pass"]]
    _log_run("r8-validate-start", version=v, code_sha256=r["code_sha256"], candidate=cand)
    try:
        X.unlock_r8(reason, kind="unlock", version=v)
        mkts = load_markets(HOLD)
        evh = evaluate(mkts, HOLD, with_null=False)
        evh["primaries"] = {n: x for n, x in evh["primaries"].items() if n in passed}   # dev-failed: not evaluated
        for k in ("config_totals", "config_sharpes", "config_trades", "pbo", "pbo_logits", "n_eff_family", "cfg_daily"):
            evh.pop(k, None)                    # the grid is a dev diagnostic; a dev-failed primary's config is in it
        hg = {}
        for name, rec in evh["primaries"].items():
            if rec["daily"].index.min() < HOLD.start or rec["daily"].index.max() >= HOLD.end:
                raise HardError("holdout output leaves the holdout window")
            hg[name] = gates_holdout(rec, r["dev"][name], float(r["dev"][name]["holdout_dd_limit"]))
        cr = evh["primaries"][cand]
        dev_daily = pd.read_parquet(R8_RESULTS / f"{cand}_dev_daily.parquet")["ret"]
        daily_ab = pd.concat([dev_daily, cr["daily"]]).sort_index()
        ds = _dsr(daily_ab, float(r.get("n_strict", NEFF_FLOOR)))
        ok = bool(r["dev"][cand]["dev_pass"] and all(hg[cand].values()))
        head = {"candidate": cand, "PASSES_IN_PRACTICE": ok, "STRICT_TIER": bool(ok and ds["dsr"] >= MIN_DSR),
                "strict_dsr": ds["dsr"], "strict_n": ds["n"], "sharpe_ann_dev_plus_holdout": ds["sharpe_ann"]}
        r |= {"holdout": _public(evh), "holdout_gates": hg, "holdout_status": "clean",
              "holdout_window": [str(HOLD.start.date()), str((HOLD.end - DAY).date())], "headline": head,
              "final_pass": head["PASSES_IN_PRACTICE"],
              "non_candidate_note": "only the candidate can pass; another dev-passing primary's holdout result is "
                                    "information only"}
        _save_json(VERDICT, r)                   # the gated verdict is saved before anything else is written
        _log_run("r8-validate", version=v, headline=head, gates=hg)
    except BaseException as e:  # noqa: BLE001 - pre-registered: any error after the unlock is a FAIL
        err = f"{type(e).__name__}: {e}"
        r.pop("holdout", None)
        _record_holdout_failure(r, v, err)
        raise SystemExit(f"error after the holdout unlock, recorded as a FAIL: {err}")
    _write("holdout", evh)
    f1d = _info(lambda: pd.read_parquet(F1_DEV_DAILY)["ret"])
    f1h = _info(lambda: pd.read_parquet(F1_HOLD_DAILY).mean(axis=1))
    info = {}
    for name, rec in evh["primaries"].items():
        info[name] = {"f1_2020_23": _info(f1_compare, rec["daily"], f1d) if isinstance(f1d, pd.Series) else f1d,
                      "f1_2024_26": _info(f1_compare, rec["daily"], f1h) if isinstance(f1h, pd.Series) else f1h,
                      "alpha_psr_holdout": rec["alpha_vs_ew"]["psr_alpha"]}
    r["holdout_info"] = info
    _save_json(VERDICT, r)
    s = cr["summary"]
    print(f"{cand} holdout: total={s['total_return']:.2%} sharpe={s['sharpe']:.2f} maxDD={s['max_dd']:.2%} "
          f"gates={hg[cand]} | PASSES IN PRACTICE={head['PASSES_IN_PRACTICE']}")
    return r
