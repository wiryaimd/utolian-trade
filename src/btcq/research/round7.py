"""Round 7 (RT): daily top-3 rotation among the liquid non-major perps of the Round 4 universe, judged under the
practical rulebook (research/PREREGISTRATION.md, Round 7).

    python -m btcq.research.run r7-power
    python -m btcq.research.run r7-calibrate
    python -m btcq.research.run r7-download [--resume]
    python -m btcq.research.run r7-sweep                         # twice: the start record, then the dev run
    python -m btcq.research.run r7-validate --unlock-holdout     # only if RT passed dev
"""
from __future__ import annotations

import json
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from btcq.backtest import engine, metrics, stats
from btcq.backtest import rotation_sim as RS
from btcq.config import MINUTE_MS, ROOT, Costs, Risk
from btcq.data import store
from btcq.data import universe as U
from btcq.research import guards, registry, round4
from btcq.research.round5 import Window, _dsr
from btcq.research.run import RESULTS, _save_json
from btcq.research.round7_calib import (CAL_DRAWS, CAL_REPS, CAL_SCENARIOS, CALIB_CSV, calib_sha, cmd_calibrate,  # noqa: F401
                                        synthetic_panel)
from btcq.strategies import r7_rotation as R

CODE = "RT"
R7_DIR = ROOT / "research" / "round7"
RUN_LOG = R7_DIR / "runs.jsonl"
DATA_REPORT = R7_DIR / "data_report.csv"
MANIFEST = R7_DIR / "manifest.json"
POWER_CSV = R7_DIR / "power_table.csv"
PREREG = ROOT / "research" / "PREREGISTRATION.md"
SECTION = "## Round 7"
REQUIRED_PHRASES = ("um-rot15-pit-v1", "L = 14, H = 6", "Null", "Dev gates", "Holdout gates", "Power table",
                    "Null calibration")
UNIVERSE_ID = "um-rot15-pit-v1"
UNIVERSE_CSV = round4.UNIVERSE_CSV
UNIVERSE_SHA256 = "6d79828bb183fce140900b0e1cc52e5d4d1687d7720484abfad0945074dcf36f"
F1_DEV_DAILY = RESULTS / "F1_dev_daily.parquet"
F1_HOLD_DAILY = RESULTS / "F1_holdout_sym_daily.parquet"
XS_TRIALS = RESULTS / "XS_trial_returns.parquet"
CODE_FILES = ["src/btcq/backtest/rotation_sim.py", "src/btcq/strategies/r7_rotation.py", "src/btcq/research/round7.py",
              "src/btcq/research/guards.py", "src/btcq/backtest/engine.py", "src/btcq/backtest/stats.py",
              "src/btcq/backtest/metrics.py", "src/btcq/data/store.py", "src/btcq/data/bardata.py",
              "src/btcq/data/universe.py", "src/btcq/config.py", "src/btcq/research/round4.py",
              "src/btcq/research/round5.py", "src/btcq/research/run.py", "src/btcq/research/registry.py",
              "src/btcq/strategies/base.py", "src/btcq/research/round7_calib.py"]

DAY = pd.Timedelta(days=1)
DAY_MS = RS.DAY_MS
DEV = Window("dev", "dev", pd.Timestamp("2020-06-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC"))
HOLD = Window("holdout", "full", pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2026-10-01", tz="UTC"))
WARMUP_DAYS = 70                 # calendar days before a window (scores, the null's volatility); signals only
TAKER = 0.0005
PENALTY = round4.SPEC.delist_penalty
EQ0 = Risk().initial_equity
N_DRAWS = 1000
NULL_SEED = 7007
BOOT_N = 10_000
BOOT_BLOCK = 30
BOOT_SEED = 7
MIN_TRADES = 100
MAX_YEAR_SHARE = 0.65
MIN_DSR = 0.95
MAX_NULL_P = 0.05
MIN_COIN_TRADES = 5
MIN_COINS = 10
MIN_COIN_SHARE = 0.6
HOLDOUT_SHARPE_FRAC = 0.5
NEFF_FLOOR = 14.849570528171945 + 4      # the strict tier only: global max N_eff + one per post-round-4 family
MIN_COVERAGE = 0.9
MAX_FUNDING_GAP_SHARE = 0.05
VERIFY_TAG = "[R7 verify"


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
    """The null's random stream depends on numba's generator and the DSR on scipy: versions are pinned with the code."""
    import platform

    import numba
    import scipy
    return {"numpy_version": np.__version__, "numba_version": numba.__version__, "pandas_version": pd.__version__,
            "scipy_version": scipy.__version__, "python_version": platform.python_version()}


def _info(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except Exception as e:  # noqa: BLE001 - information only; never affects a gate
        return {"error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------- universe

def gated_universe() -> pd.DataFrame:
    """um-rot15-pit-v1: the gated (non-major) rows of the SHA-pinned Round 4 universe."""
    if guards.sha_text(UNIVERSE_CSV) != UNIVERSE_SHA256:
        raise SystemExit("research/round4/universe.csv does not match its registered SHA256")
    u = U.load_universe(UNIVERSE_CSV)
    g = u[u["gated"].astype(str) == "True"].copy()
    if g["is_major"].astype(str).eq("True").any():
        raise HardError("a major is gated")
    return g


def window_months(w: Window) -> list[str]:
    return [m.strftime("%Y-%m") for m in pd.date_range(w.start, w.end - DAY, freq="MS")]


def window_symbols(gu: pd.DataFrame, w: Window) -> list[str]:
    return sorted(gu[gu["month"].isin(window_months(w))]["symbol"].unique())


def membership_grid(gu: pd.DataFrame, symbols: list[str], days: pd.DatetimeIndex):
    """member / rank / slippage per (day, coin) from the day's month (a fill on day t uses t's month)."""
    look = {(m, s): int(r) for m, s, r in zip(gu["month"], gu["symbol"], gu["rank"])}
    month = days.strftime("%Y-%m")
    um, inv = np.unique(month, return_inverse=True)
    rk = np.array([[look.get((m, s), 99) for s in symbols] for m in um], np.int64)[inv]
    member = rk < 99
    tier = np.where(rk <= 10, round4.SPEC.tier_slip[0], round4.SPEC.tier_slip[1])
    slip = np.where(member, tier, round4.SPEC.nonmember_slip)
    return member, rk, slip


def meme_flags(symbols: list[str]) -> np.ndarray:
    cat = round4._catalogue().set_index("symbol")
    return np.array([str(cat.loc[s, "is_meme"]) == "True" if s in cat.index else False for s in symbols])


# ---------------------------------------------------------------- data -> panel

def _empty_cols(n: int) -> dict:
    nan, f = np.full(n, np.nan), np.zeros(n, np.bool_)
    return {"close": nan.copy(), "traded": f.copy(), "fill_px": nan.copy(), "fund_pre": np.zeros(n),
            "fund_post": np.zeros(n), "fx_pre": f.copy(), "fx_post": f.copy(), "fx_px": nan.copy(),
            "fund_fx": np.zeros(n), "blocked": f.copy(), "days_since_gap": np.full(n, np.inf)}


def coin_columns(sym: str, w: Window, cal_ms: np.ndarray) -> tuple[dict | None, dict]:
    """One coin's daily columns: 1m rows before the window end, trading span only, funding gaps filled at the
    synthetic rate, per-minute funding as in engine.Market, the delisting rule's gaps and forced exits."""
    end_ms = _ms(w.end)
    m1 = store.load_1m(w.period, sym)
    m1 = store.trading_span(m1[m1["ts"] < end_ms].reset_index(drop=True))
    info = {"symbol": sym, "rows": len(m1)}
    if not len(m1):
        return None, info
    fund, filled_h = store.fill_funding_gaps(store.load_funding(w.period, sym), int(m1["ts"].iloc[0]),
                                             int(m1["ts"].iloc[-1]), Costs.synthetic_funding)
    mk = engine.Market(m1, fund, None, synthetic_funding=Costs.synthetic_funding)
    vol = m1["volume"].to_numpy()
    fx, gaps = round4.no_trade_gaps(mk.ts, vol, end_ms)
    cols = RS.daily_columns(mk.ts, mk.o, mk.c, vol, mk.fund, cal_ms, fx, gaps, round4.COOLDOWN_MS)
    delisted = bool(gaps and gaps[-1][1] == end_ms)
    info |= {"first": pd.Timestamp(int(mk.ts[0]), unit="ms", tz="UTC").isoformat(),
             "funding_filled_h": float(filled_h), "delisted": delisted,
             "settlement_gaps": len(gaps) - int(delisted), "forced_exit_minutes": int(len(fx))}
    return cols, info


def build_panel(w: Window, symbols: list[str] | None = None) -> tuple[RS.Panel, pd.DataFrame]:
    gu = gated_universe()
    syms = window_symbols(gu, w) if symbols is None else sorted(symbols)
    days = pd.date_range(w.start - WARMUP_DAYS * DAY, w.end - DAY, freq="D").as_unit("ms")
    cal_ms = days.asi8.astype(np.int64)
    member, rank, slip = membership_grid(gu, syms, days)
    cols, infos, filled = {}, [], []
    for j, s in enumerate(syms):
        c, info = coin_columns(s, w, cal_ms)
        if c is None:
            if member[days >= w.start, j].any():
                raise HardError(f"{s}: a member in this window without data")
            c = _empty_cols(len(days))
        cols[s] = c
        infos.append(info)
        filled.append(info.get("funding_filled_h", 0.0))
    p = RS.stack(cols, days, member, rank, slip, meme_flags(syms), np.asarray(filled))
    return p, pd.DataFrame(infos)


def window_bounds(p: RS.Panel, w: Window) -> tuple[int, int]:
    return p.index_of(w.start), p.index_of(w.end - DAY)


def panel_frame(p: RS.Panel) -> pd.DataFrame:
    """Long form of a panel (one row per day and coin) for independent re-simulation."""
    n, k = p.shape
    out = {"day": np.repeat(p.days.asi8, k), "symbol": np.tile(np.asarray(p.symbols, dtype=object), n)}
    for f in ("close", "traded", "fill_px", "fund_pre", "fund_post", "fx_pre", "fx_post", "fx_px", "fund_fx",
              "blocked", "days_since_gap", "member", "rank", "slip"):
        out[f] = getattr(p, f).reshape(-1)
    df = pd.DataFrame(out)
    df["day"] = pd.to_datetime(df["day"], unit="ms", utc=True)
    df["is_meme"] = np.tile(p.is_meme, n)
    return df


# ---------------------------------------------------------------- evaluation

def run_config(p: RS.Panel, cfg: dict, first: int, last: int, mult: float = 1.0, zero: bool = False,
               resize: bool = True) -> RS.SimResult:
    W, cb, _ = R.path(p, cfg, first, last)
    m = 0.0 if zero else mult
    return RS.simulate(p, W, cb, first, last, taker=TAKER, slip_mult=m, fee_mult=m, penalty=PENALTY, eq0=EQ0,
                       resize=resize, K=R.K)


def ew_benchmark(p: RS.Panel, first: int, last: int) -> pd.Series:
    """Equal-weight daily-rebalanced hold of the coins that are members in each day's month (close to close)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = p.close[1:] / p.close[:-1] - 1.0
    r = np.vstack([np.full((1, p.shape[1]), np.nan), r])
    r = np.where(p.member & np.isfinite(r), r, np.nan)
    cnt = np.isfinite(r).sum(axis=1)
    mean = np.where(cnt > 0, np.nansum(np.nan_to_num(r, nan=0.0), axis=1) / np.maximum(cnt, 1), 0.0)
    return pd.Series(mean[first:last + 1], index=p.days[first:last + 1])


def _max_dd(eq: pd.Series) -> float:
    return metrics.max_drawdown(np.r_[EQ0, eq.to_numpy()])


def summary(sim: RS.SimResult) -> dict:
    d, ep, eq = sim.daily, sim.episodes, sim.equity
    yrs = len(d) / 365
    sd = d.std(ddof=1)
    total = float(eq.iloc[-1] / EQ0 - 1)
    traded_days = sim.turnover > 0
    return {"trades": int(len(ep)), "trades_per_year": float(len(ep) / yrs), "total_return": total,
            "cagr": float((1 + total) ** (1 / yrs) - 1) if total > -1 else -1.0,
            "sharpe": float(d.mean() / sd * np.sqrt(365)) if sd > 0 else 0.0, "vol_ann": float(sd * np.sqrt(365)),
            "mean_daily": float(d.mean()), "max_dd": _max_dd(eq),
            "avg_trade_ret": float(ep["ret"].mean()) if len(ep) else 0.0,
            "win_rate": float((ep["net"] > 0).mean()) if len(ep) else 0.0,
            "avg_days_held": float(ep["days_held"].mean()) if len(ep) else 0.0,
            "fees_total": float(sim.fees.sum()), "funding_total": float(sim.funding.sum()),
            "turnover_per_year": float(sim.turnover.sum() / yrs), "mean_held": float(sim.held.mean()),
            "share_days_invested": float((sim.held > 0).mean()), "share_days_fully_invested": float((sim.held >= R.K).mean()),
            "share_days_with_order": float(traded_days.mean()), "days": int(len(d))}


def year_table(daily: pd.Series, bench: pd.Series) -> list[dict]:
    rows = []
    for y, d in daily.groupby(daily.index.year):
        b = bench.loc[d.index]
        rows.append({"year": int(y), "days": int(len(d)), "return": float((1 + d).prod() - 1),
                     "sum_daily": float(d.sum()), "sharpe": float(d.mean() / d.std() * np.sqrt(365)) if d.std() > 0 else 0.0,
                     "bench_return": float((1 + b).prod() - 1)})
    return rows


def coin_table(sim: RS.SimResult, p: RS.Panel, first: int, last: int) -> list[dict]:
    ep = sim.episodes
    rows = []
    for j, s in enumerate(p.symbols):
        t = ep[ep["symbol"] == s]
        rows.append({"symbol": s, "member_days": int(p.member[first:last + 1, j].sum()), "trades": int(len(t)),
                     "sum_net": float(t["net"].sum()), "mean_ret": float(t["ret"].mean()) if len(t) else 0.0,
                     "win_rate": float((t["net"] > 0).mean()) if len(t) else 0.0,
                     "days_held": int(t["days_held"].sum()), "delist_exits": int((t["reason"] == "delist").sum()),
                     "is_meme": bool(p.is_meme[j]), "funding_filled_h": float(p.funding_filled_h[j])})
    return rows


def breadth(coins: list[dict]) -> dict:
    q = [c for c in coins if c["trades"] >= MIN_COIN_TRADES]
    pos = sum(c["sum_net"] > 0 for c in q)
    return {"qualifying": len(q), "positive": int(pos), "share": pos / len(q) if q else 0.0}


def meme_split(ep: pd.DataFrame, p: RS.Panel) -> dict:
    memes = {s for s, m in zip(p.symbols, p.is_meme) if m}
    is_m = ep["symbol"].isin(memes)
    tot = ep["net"].sum()
    return {"meme_trades": int(is_m.sum()), "meme_net": float(ep.loc[is_m, "net"].sum()),
            "other_net": float(ep.loc[~is_m, "net"].sum()),
            "meme_share_of_net": float(ep.loc[is_m, "net"].sum() / tot) if tot else None}


def evaluate(p: RS.Panel, w: Window, with_null: bool = True, n_draws: int = N_DRAWS,
             family_prior: pd.DataFrame | None = None) -> dict:
    """One window. family_prior: earlier RT versions' dev trial returns, counted in the family's N_eff."""
    t0 = time.time()
    first, last = window_bounds(p, w)
    cfg = R.CONFIGS[R.PRIMARY]
    try:
        sims = {i: run_config(p, c, first, last) for i, c in enumerate(R.CONFIGS)}
        prim = sims[R.PRIMARY]
        stress = run_config(p, cfg, first, last, mult=2.0)
        zero_mean = R.actual_stat(p, cfg, first, last, PENALTY, EQ0)
        no_resize = run_config(p, cfg, first, last, resize=False)
    except RS.SimError as e:
        raise HardError(str(e)) from e
    daily = prim.daily
    if daily.index[0] != w.start or daily.index[-1] != w.end - DAY:
        raise HardError("the daily returns do not span the window")
    cfg_daily = {i: s.daily for i, s in sims.items()}
    M = pd.concat(cfg_daily, axis=1).to_numpy()
    pbo, logits = stats.pbo_cscv(M)
    totals = {i: float(s.equity.iloc[-1] / EQ0 - 1) for i, s in sims.items()}
    fam = pd.concat(cfg_daily, axis=1)
    if family_prior is not None:
        fam = pd.concat([family_prior, fam], axis=1, join="inner")
    n_eff = stats.effective_trials(fam.to_numpy())
    coins = coin_table(prim, p, first, last)
    bench = ew_benchmark(p, first, last)
    ep = prim.episodes
    out = {
        "summary": summary(prim), "weekly": stats.weekly_t(daily),
        "stress_2x": summary(stress), "stress_2x_mean_daily": float(stress.daily.mean()),
        "zero_cost_mean_daily": zero_mean,
        "no_resize": {k: v for k, v in summary(no_resize).items() if k in ("total_return", "sharpe", "max_dd",
                                                                           "fees_total", "trades")},
        "year_share": stats.year_share_daily(daily), "years": year_table(daily, bench),
        "pbo": pbo, "config_totals": {str(i): v for i, v in totals.items()},
        "config_sharpes": {str(i): summary(s)["sharpe"] for i, s in sims.items()},
        "config_trades": {str(i): int(len(s.episodes)) for i, s in sims.items()},
        "plateau": {str(i): totals[i] > 0 for i in R.NEIGHBOURS},
        "n_eff_family": n_eff, "dsr_family": _dsr(daily, n_eff), "dsr_raw6": _dsr(daily, float(len(R.CONFIGS))),
        "coins": coins, "breadth": breadth(coins), "meme": meme_split(ep, p),
        "alpha_vs_ew": stats.beta_alpha(daily, bench),
        "benchmark": {"total_return": float((1 + bench).prod() - 1),
                      "sharpe": float(bench.mean() / bench.std() * np.sqrt(365)) if bench.std() > 0 else 0.0,
                      "max_dd": metrics.max_drawdown(np.r_[1.0, (1 + bench).cumprod().to_numpy()])},
        "exit_reasons": ep["reason"].value_counts().to_dict(),
        "n_coins": len(p.symbols), "window": [str(w.start.date()), str((w.end - DAY).date())],
        "daily": daily, "sim": prim, "stress_sim": stress, "cfg_daily": cfg_daily, "pbo_logits": logits,
        "benchmark_daily": bench,
    }
    if with_null:
        try:
            out["null"] = R.null_test(p, cfg, first, last, n_draws, NULL_SEED, PENALTY, EQ0)
        except RS.SimError as e:
            raise HardError(str(e)) from e
    out["runtime_s"] = time.time() - t0
    return out


def gates_dev(ev: dict) -> dict:
    b = ev["breadth"]
    return {"trades>=100": ev["summary"]["trades"] >= MIN_TRADES,
            "2x-cost mean daily>0": ev["stress_2x_mean_daily"] > 0,
            "null p<0.05": ev["null"]["p_value"] < MAX_NULL_P,
            "DSR>=0.95 at the family N_eff": ev["dsr_family"]["dsr"] >= MIN_DSR,
            "best year<=65%": ev["year_share"] <= MAX_YEAR_SHARE,
            "plateau: 3 neighbours positive": all(ev["plateau"].values()),
            ">=60% of coins (5+ trades) positive, 10+ coins": b["qualifying"] >= MIN_COINS and b["share"] >= MIN_COIN_SHARE}


def holdout_dd_limit(dev_daily: pd.Series) -> float:
    _, mdd = stats.block_bootstrap(dev_daily.to_numpy(), BOOT_N, block=BOOT_BLOCK, n_out=len(HOLD.calendar),
                                   seed=BOOT_SEED)
    return float(np.quantile(mdd, 0.95))


def gates_holdout(evh: dict, dev_sharpe: float, dd_limit: float) -> dict:
    s = evh["summary"]
    return {"holdout mean daily>0": s["mean_daily"] > 0,
            "holdout Sharpe>=50% of dev": s["sharpe"] >= HOLDOUT_SHARPE_FRAC * dev_sharpe,
            "holdout worst drawdown<=dev bootstrap p95": s["max_dd"] <= dd_limit}


def headline(dev_pass: bool, hold_pass: bool, daily_ab: pd.Series, n_strict: float) -> dict:
    d = _dsr(daily_ab, n_strict)
    return {"PASSES_IN_PRACTICE": bool(dev_pass and hold_pass), "STRICT_TIER": bool(dev_pass and hold_pass
                                                                                      and d["dsr"] >= MIN_DSR),
            "strict_dsr": d["dsr"], "strict_n": n_strict, "sharpe_ann": d["sharpe_ann"], "T": d["T"]}


def f1_compare(daily: pd.Series, f1: pd.Series) -> dict:
    j = pd.concat([daily.rename("x"), f1.rename("f1")], axis=1, sort=True).dropna()
    wk = pd.concat([stats.weekly_returns(j["x"]), stats.weekly_returns(j["f1"])], axis=1).dropna()
    return {"daily_corr": float(j.corr().iloc[0, 1]), "weekly_corr": float(wk.corr().iloc[0, 1]),
            "hac": stats.hac_ols(j["x"].to_numpy(), j["f1"].to_numpy(), lag=7), "days": int(len(j))}


def without_best_coin(p: RS.Panel, w: Window, coins: list[dict]) -> dict:
    best = max(coins, key=lambda c: c["sum_net"])["symbol"]
    q = p.drop(best)
    first, last = window_bounds(q, w)
    s = summary(run_config(q, R.CONFIGS[R.PRIMARY], first, last))
    return {"dropped": best, **{k: s[k] for k in ("total_return", "sharpe", "max_dd", "mean_daily", "trades")}}


_SKIP = {"daily", "sim", "stress_sim", "cfg_daily", "pbo_logits", "benchmark_daily"}


def _public(ev: dict) -> dict:
    out = {k: v for k, v in ev.items() if k not in _SKIP}
    if "null" in out:
        out["null"] = {k: v for k, v in out["null"].items() if k != "dist"}
    return out


def _write(tag: str, ev: dict, p: RS.Panel) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    sim = ev["sim"]
    sim.episodes.to_parquet(RESULTS / f"{CODE}_{tag}_trades.parquet")
    ev["stress_sim"].episodes.to_parquet(RESULTS / f"{CODE}_{tag}_stress_2x_trades.parquet")
    sim.equity.rename("equity").to_frame().to_parquet(RESULTS / f"{CODE}_{tag}_equity.parquet")
    ev["daily"].rename("ret").to_frame().to_parquet(RESULTS / f"{CODE}_{tag}_daily.parquet")
    ev["benchmark_daily"].rename("ret").to_frame().to_parquet(RESULTS / f"{CODE}_{tag}_benchmark.parquet")
    pd.DataFrame({"fees": sim.fees, "funding": sim.funding, "turnover": sim.turnover, "held": sim.held}) \
        .to_parquet(RESULTS / f"{CODE}_{tag}_activity.parquet")
    pd.concat({str(i): d for i, d in ev["cfg_daily"].items()}, axis=1).to_parquet(
        RESULTS / f"{CODE}_{tag}_config_daily.parquet")
    panel_frame(p).to_parquet(RESULTS / f"{CODE}_{tag}_panel.parquet")
    if tag == "dev":
        np.savez_compressed(RESULTS / f"{CODE}_dev_dists.npz", null=ev.get("null", {}).get("dist", np.zeros(1)),
                            pbo_logits=np.asarray(ev["pbo_logits"]))


# ---------------------------------------------------------------- data report (timestamps, volume, funding times only)

def _facts(sym: str, months: list[str], end_ms: int) -> dict:
    """Lock-free data facts from ts/volume of the 1m file and ts of the funding file (never prices or rates)."""
    d = store.symbol_dir(sym)
    m1 = pd.read_parquet(d / "perp_1m.parquet", columns=["ts", "volume"])
    f = pd.read_parquet(d / "funding.parquet", columns=["ts"])["ts"].to_numpy(np.int64)
    ts, vol = m1["ts"].to_numpy(np.int64), m1["volume"].to_numpy()
    cut = ts < end_ms
    live = np.flatnonzero(cut & (vol > 0))
    last = int(ts[live[-1]]) if len(live) else None
    _, gaps = round4.no_trade_gaps(ts[cut], vol[cut], end_ms)
    worst, gap_h, tot_h = 1.0, 0.0, 0.0
    for m in months:
        a = pd.Timestamp(m + "-01", tz="UTC")
        a_ms = _ms(a)
        b_ms = min(_ms(a + pd.offsets.MonthBegin(1)), (last + MINUTE_MS) if last else a_ms, end_ms)
        if b_ms <= a_ms:
            continue
        n = np.searchsorted(ts, b_ms) - np.searchsorted(ts, a_ms)
        worst = min(worst, n / ((b_ms - a_ms) / MINUTE_MS))
        e = np.r_[a_ms, f[(f > a_ms) & (f < b_ms)], b_ms]
        dd = np.diff(e)
        gap_h += dd[dd > 9 * 3_600_000].sum() / 3_600_000
        tot_h += (b_ms - a_ms) / 3_600_000
    first_member = _ms(pd.Timestamp(months[0] + "-01", tz="UTC"))
    return {"last_traded": pd.Timestamp(last, unit="ms", tz="UTC").isoformat() if last else None,
            "traded_in_member_months": bool(last is not None and last >= first_member),
            "ends_early": last is None or end_ms - last > DAY_MS,
            "settlement_gaps": len(gaps) - int(bool(gaps) and gaps[-1][1] == end_ms),
            "last_row": pd.Timestamp(int(ts[-1]), unit="ms", tz="UTC").isoformat() if len(ts) else None,
            "last_row_ms": int(ts[-1]) if len(ts) else None,
            "min_month_coverage": round(float(worst), 4), "funding_gap_share": round(gap_h / tot_h, 4) if tot_h else 0.0}


def flag_rows(df: pd.DataFrame) -> pd.Series:
    """The registered red flags, recomputed from the report's own columns."""
    flag = ~df["status"].astype(str).str.startswith("ok")
    for w in ("dev", "holdout"):
        has = df[f"{w}_member_months"].fillna(0) > 0
        col = lambda c, d: df[c] if c in df else pd.Series(d, index=df.index)  # noqa: E731
        cov = col(f"{w}_min_month_coverage", 0.0).fillna(0.0).astype(float)
        fg = col(f"{w}_funding_gap_share", 1.0).fillna(1.0).astype(float)
        sh = col(f"{w}_archive_short", False).astype(str).eq("True")
        traded = col(f"{w}_traded_in_member_months", False).astype(str).eq("True")
        flag |= has & ((cov < MIN_COVERAGE) | (fg > MAX_FUNDING_GAP_SHARE) | sh | ~traded)
    return flag


def data_report() -> pd.DataFrame:
    gu = gated_universe()
    cat = round4._catalogue().set_index("symbol")
    rows = []
    for sym in sorted(gu["symbol"].unique()):
        g = gu[gu["symbol"] == sym]
        row = {"symbol": sym, "is_meme": str(cat.loc[sym, "is_meme"]) == "True" if sym in cat.index else None,
               "snapshot_status": cat.loc[sym, "status"] if sym in cat.index else ""}
        rep_p = store.symbol_dir(sym) / "build_report.json"
        rep = json.loads(rep_p.read_text()) if rep_p.exists() else {"status": "missing"}
        blind = rep.get("extreme_bars_before") is not None
        row |= {"status": rep.get("status"), "blind_build": blind,
                **{k: rep.get(k) for k in ("rows", "gaps_over_24h", "max_gap_h", "ohlc_violations",
                                           "nonpositive_prices", "bars_4h_mismatch_share")},
                # a whole-file count (Round 4 builds) would summarise 2024+ moves: only the blind count is shown
                "extreme_bars_before_2024": rep.get("extreme_bars") if blind else None}
        for w in (DEV, HOLD):
            months = sorted(set(g["month"]) & set(window_months(w)))
            row[f"{w.name}_member_months"] = len(months)
            if not months:
                continue
            ok = str(rep.get("status", "")).startswith("ok") and (store.symbol_dir(sym) / "perp_1m.parquet").exists()
            facts = _facts(sym, months, _ms(w.end)) if ok else {"min_month_coverage": 0.0, "funding_gap_share": 1.0,
                                                                  "traded_in_member_months": False}
            short = bool(row["snapshot_status"] == "TRADING" and facts.get("last_row_ms") is not None
                         and facts["last_row_ms"] < _ms(w.end) - 3_600_000)
            row |= {f"{w.name}_{k}": v for k, v in facts.items() if k != "last_row_ms"}
            row[f"{w.name}_archive_short"] = short
        rows.append(row)
    df = pd.DataFrame(rows)
    df["red_flag"] = flag_rows(df)
    return df


def red_flags(df: pd.DataFrame) -> pd.DataFrame:
    return df[flag_rows(df) | df["red_flag"].astype(str).eq("True")]


def write_manifest(df: pd.DataFrame) -> dict:
    files = {}
    for sym in df["symbol"]:
        for name in ("perp_1m.parquet", "funding.parquet"):
            pth = store.symbol_dir(sym) / name
            if pth.exists():
                files[f"{sym}/{name}"] = guards.sha_file(pth)
    man = {"universe_id": UNIVERSE_ID, "universe_sha256": guards.sha_text(UNIVERSE_CSV), "files": files,
           "data_report_sha256": guards.sha_text(DATA_REPORT)}
    MANIFEST.write_text(json.dumps(man, indent=1, sort_keys=True) + "\n", newline="\n")
    return man


def verify_manifest() -> list[str]:
    if not MANIFEST.exists():
        return ["manifest missing"]
    man = json.loads(MANIFEST.read_text())
    probs = []
    bad = [k for k, v in man["files"].items() if not (store.PARQUET_DIR / k).exists()
           or guards.sha_file(store.PARQUET_DIR / k) != v]
    if bad:
        probs.append(f"data changed since the manifest: {bad[:5]}")
    if man.get("universe_sha256") != guards.sha_text(UNIVERSE_CSV):
        probs.append("the universe changed since the manifest")
    if not DATA_REPORT.exists() or man.get("data_report_sha256") != guards.sha_text(DATA_REPORT):
        probs.append("the data report changed since the manifest")
    return probs


# ---------------------------------------------------------------- guards

def calibration_problems(cal: pd.DataFrame) -> list[str]:
    """The committed calibration must be the registered one, run by the current calibration code, and pass."""
    probs = []
    if "kind" not in cal or "calib_sha256" not in cal:
        return ["the calibration file is not in the registered format"]
    want = [(sc, kind, reps) for sc, edge, reps, kind in CAL_SCENARIOS]
    have = list(zip(cal["scenario"], cal["kind"], cal["reps"]))
    if have != want or not cal["draws"].eq(CAL_DRAWS).all():
        probs.append("the calibration does not have the registered scenarios, replications and draws")
    if not cal["calib_sha256"].eq(calib_sha()).all():
        probs.append("the calibration was run by other calibration code")
    no_edge = cal[cal["kind"] == "no-edge"]
    if not len(no_edge) or not no_edge["ok"].astype(str).eq("True").all():
        probs.append("a no-edge calibration scenario is above its limit")
    return probs


def _require_prereg() -> str:
    """The committed Round 7 section (with the power table and the registered, passing calibration committed).
    Returns the section's SHA256."""
    guards.require_committed([PREREG, POWER_CSV, CALIB_CSV])
    probs = calibration_problems(pd.read_csv(CALIB_CSV))
    if probs:
        raise SystemExit("; ".join(probs))
    return guards.sha_str(guards.require_section(PREREG, SECTION, REQUIRED_PHRASES))


def _require_data_report() -> None:
    guards.require_committed([DATA_REPORT, MANIFEST])
    if len(red_flags(pd.read_csv(DATA_REPORT))):
        raise SystemExit("the Round 7 data report has red flags")
    probs = verify_manifest()
    if probs:
        raise SystemExit("; ".join(probs))


def r7_unlock_records() -> list[dict]:
    """Every Round 7 holdout record, over every committed version of the holdout log and its working copy:
    structured ones and any line whose reason carries an R7 tag."""
    out = []
    for r in guards.all_records(store.HOLDOUT_LOG):
        reason = str(r.get("reason", r.get("raw", "")))
        if r.get("round") == 7 or "[R7" in reason:
            kind = r.get("kind") or ("verify" if VERIFY_TAG in reason else
                                     "resume" if "[R7 resume" in reason else "unlock")
            out.append(r | {"kind": kind})
    return out


def unlock_attempts() -> tuple[list[dict], list[dict]]:
    """(unlocks, resumes): holdout-log records, and validate-start records in the run log (written just before
    an unlock), so restoring one log can't hide an unlock."""
    recs = r7_unlock_records()
    starts = guards.log_records(RUN_LOG, "r7-validate-start")
    unlocks = [x for x in recs if x["kind"] == "unlock"]
    resumes = [x for x in recs if x["kind"] == "resume"]
    s_unlock = [x for x in starts if not x.get("resume")]
    s_resume = [x for x in starts if x.get("resume")]
    return (unlocks if len(unlocks) >= len(s_unlock) else s_unlock,
            resumes if len(resumes) >= len(s_resume) else s_resume)


def _hard_error_versions() -> set[int]:
    return {int(r["version"]) for r in guards.log_records(RUN_LOG, "r7-sweep") if r.get("hard_error")}


def _holdout_touched(r: dict) -> list[str]:
    out = []
    if "holdout" in r:
        out.append("RT holdout already evaluated")
    if any(unlock_attempts()):
        out.append("a Round 7 holdout unlock was already logged")
    return out


def verify_unlock(note: str) -> None:
    """Independent checks after the holdout verdict load the window again; each load is logged."""
    p = RESULTS / f"{CODE}.json"
    if not (p.exists() and "holdout_gates" in json.loads(p.read_text())):
        raise PermissionError("no Round 7 holdout verdict yet: checks come after it")
    guards.clean_src()
    store.unlock_holdout(f"Round 7 independent check {VERIFY_TAG} {note}]", round=7, kind="verify")


# ---------------------------------------------------------------- commands

def cmd_power() -> pd.DataFrame:
    df = pd.DataFrame(stats.power_table_practical(dev_days=len(DEV.calendar), hold_days=len(HOLD.calendar)))
    R7_DIR.mkdir(parents=True, exist_ok=True)
    U.write_csv(df, POWER_CSV)
    print(df.round(3).to_string(index=False))
    return df


def cmd_download(resume: bool = False) -> pd.DataFrame:
    _require_prereg()
    guards.clean_src()
    if DATA_REPORT.exists() and not resume:
        raise SystemExit("the Round 7 data report exists; a rebuild after a red flag needs a dated amendment and "
                         "--resume")
    gu = gated_universe()
    hold_start = _ms(HOLD.start)
    built = []
    for i, sym in enumerate(sorted(gu["symbol"].unique()), 1):
        d = store.symbol_dir(sym)
        rep_p = d / "build_report.json"
        ok = rep_p.exists() and (d / "perp_1m.parquet").exists() and \
            str(json.loads(rep_p.read_text()).get("status", "")).startswith("ok")
        if ok or (rep_p.exists() and not resume):
            continue
        print(f"[{i}] building {sym} (blind: no return statistics from 2024 on)", flush=True)
        rep = store.build_perp_only(sym, returns_before_ms=hold_start)
        built.append({"symbol": sym, "status": rep.get("status")})
    df = data_report()
    R7_DIR.mkdir(parents=True, exist_ok=True)
    U.write_csv(df, DATA_REPORT)
    man = write_manifest(df)
    bad = red_flags(df)
    _log_run("r7-download", built=built, red_flags=bad["symbol"].tolist(), files=len(man["files"]), resume=resume,
             code_sha256=code_sha(), data_report_sha256=man["data_report_sha256"])
    print(f"Round 7 data: {len(df)} coins, built {len(built)} now, {len(bad)} red flags")
    if len(bad):
        cols = [c for c in df.columns if c.endswith(("coverage", "funding_gap_share", "archive_short",
                                                     "traded_in_member_months"))]
        print(bad[["symbol", "status"] + cols].to_string(index=False))
        raise SystemExit("data red flags: a dated amendment is needed before any strategy run")
    return df


def _trial_rows(ev: dict, version: int) -> list[dict]:
    rows = []
    for i, d in ev["cfg_daily"].items():
        sr, T, sk, ku = stats.sharpe_moments(d)
        rows.append({"strategy": CODE, "version": version, "params": R.CONFIGS[i] | {"K": R.K}, "period": "dev",
                     "sr": sr, "T": T, "skew": sk, "kurt": ku, "trades": int(ev["config_trades"][str(i)]),
                     "sharpe_ann": float(sr * np.sqrt(365))})
    return rows


def trial_file(version: int):
    return RESULTS / (f"{CODE}_trial_returns.parquet" if version == 1 else f"{CODE}v{version}_trial_returns.parquet")


def family_prior(version: int) -> pd.DataFrame | None:
    """Earlier RT versions' trial returns, part of the family for N_eff (Amendment 1): every earlier version
    with a logged dev verdict (a version that stopped on a hard error produced no trial returns). Their files
    must be committed."""
    done = sorted({int(r["version"]) for r in guards.log_records(RUN_LOG, "r7-sweep")
                   if "verdict" in r and int(r["version"]) < version})
    if not done:
        return None
    files = [trial_file(x) for x in done]
    guards.require_committed(files)
    return pd.concat([pd.read_parquet(f) for f in files], axis=1)


def sweep_pins(sec_sha: str) -> dict:
    return {"code_sha256": code_sha(), "manifest_sha256": guards.sha_text(MANIFEST),
            "universe_sha256": guards.sha_text(UNIVERSE_CSV), "f1_daily_sha256": guards.sha_file(F1_DEV_DAILY),
            "prereg_section_sha256": sec_sha, **lib_versions()}


def cmd_sweep(rerun: bool = False, n_draws: int = N_DRAWS) -> dict:
    sec_sha = _require_prereg()
    _require_data_report()
    guards.clean_src()
    meta = R.Rotation()
    v = meta.version
    p = RESULTS / f"{CODE}.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    closed = _holdout_touched(old)
    if closed:
        raise SystemExit("Round 7 is closed once its holdout is touched (no reruns): " + "; ".join(closed))
    if old:
        if not rerun:
            raise SystemExit(f"{CODE}.json exists; the dev sweep runs once (--rerun needs a version bump)")
        if v <= old.get("version", 0):
            raise SystemExit("--rerun needs a Rotation.version bump (a rerun counts as new trials)")
    if v in _hard_error_versions():
        raise SystemExit(f"version {v} stopped on a hard error; a fix needs a version bump")
    if guards.log_records(RUN_LOG, "r7-sweep", version=v) or guards.log_records(RUN_LOG, "r7-sweep-run", version=v):
        raise SystemExit(f"version {v} already ran its dev sweep (logged); a rerun needs a version bump")
    pins = sweep_pins(sec_sha)
    starts = guards.log_records(RUN_LOG, "r7-sweep-start", version=v)
    for k, val in pins.items():
        if any(r.get(k) != val for r in starts):
            raise SystemExit(f"version {v} already started with a different {k}; a change needs a version bump")
    if not guards.log_records(RUN_LOG, "r7-sweep-start", committed_only=True, version=v):
        if not starts:
            _log_run("r7-sweep-start", version=v, **pins)
        raise SystemExit("logged the Round 7 sweep start record: commit research/round7/runs.jsonl, then run "
                         "r7-sweep again")
    prior = family_prior(v)                           # checked before the run is logged
    _log_run("r7-sweep-run", version=v, **pins)       # from here on this version has run, whatever happens
    print("building the dev panel ...", flush=True)
    try:
        panel, infos = build_panel(DEV)
        print(f"RT dev: {len(panel.symbols)} coins ...", flush=True)
        ev = evaluate(panel, DEV, n_draws=n_draws, family_prior=prior)
        if ev["daily"].index.max() >= HOLD.start:
            raise HardError("dev output reaches the holdout")
        g = gates_dev(ev)
        dd_limit = holdout_dd_limit(ev["daily"])
    except Exception as e:  # noqa: BLE001 - any failure before the verdict is a logged hard error
        _log_run("r7-sweep", version=v, hard_error=f"{type(e).__name__}: {e}")
        raise SystemExit(f"hard error, logged: commit research/round7/runs.jsonl; a fix needs a version bump: "
                         f"{type(e).__name__}: {e}")
    r = {"code": CODE, "name": meta.name, "tf": "1d", "version": v, "round": 7, "hypothesis": meta.hypothesis,
         "rules": meta.rules, "params": R.CONFIGS[R.PRIMARY] | {"K": R.K}, "configs": R.CONFIGS,
         "universe_id": UNIVERSE_ID, "holdout_scope": "clean", "rulebook": "practical (Amendments 1 and 2)",
         "start": str(DEV.start.date()), "end": str((DEV.end - DAY).date()),
         "dev": ev["summary"], "dev_detail": _public(ev), "gates": g, "dev_pass": all(g.values()),
         "n_eff_family": ev["n_eff_family"], "dsr_family": ev["dsr_family"], "dsr_raw6": ev["dsr_raw6"],
         "holdout_dd_limit": dd_limit,
         "holdout_bootstrap": {"n": BOOT_N, "block": BOOT_BLOCK, "seed": BOOT_SEED, "n_out": len(HOLD.calendar)},
         "git_commit": guards.git("rev-parse", "HEAD"), **pins}
    _save_json(p, r)                                 # the gated verdict is saved before anything else is written
    failed = [k for k, x in g.items() if not x]
    _log_run("r7-sweep", version=v, verdict=r["dev_pass"], gates=g, failed=failed, n_eff_family=ev["n_eff_family"],
             dev_sharpe=ev["summary"]["sharpe"], holdout_dd_limit=dd_limit, runtime_s=ev["runtime_s"], **pins)
    s = ev["summary"]
    print(f"RT dev: trades={s['trades']} sharpe={s['sharpe']:.2f} total={s['total_return']:.2%} "
          f"maxDD={s['max_dd']:.2%} null p={ev['null']['p_value']:.3f} DSR(N_eff {ev['n_eff_family']:.2f})="
          f"{ev['dsr_family']['dsr']:.3f} -> {'PASS' if r['dev_pass'] else 'FAIL: ' + ', '.join(failed)}", flush=True)
    registry.record(_trial_rows(ev, v))
    pd.concat({(f"{CODE}_{i}" if v == 1 else f"{CODE}_v{v}_{i}"): d for i, d in ev["cfg_daily"].items()}, axis=1) \
        .to_parquet(trial_file(v))
    from btcq.research.run import all_trial_returns
    trial = all_trial_returns()
    n_proj = registry.ratcheted_effective_trials(stats.effective_trials(trial.to_numpy()), trial.shape[1],
                                                 note=f"round 7 (RT v{v})")
    n_strict = max(n_proj, NEFF_FLOOR)
    r |= {"n_eff_project": n_proj, "n_strict": n_strict, "dsr_dev_strict_n": _dsr(ev["daily"], n_strict),
          "n_trials": registry.multiple_testing_inputs("dev")[0]}
    _save_json(p, r)
    _write("dev", ev, panel)
    infos.to_csv(R7_DIR / "dev_coin_info.csv", index=False)
    r["dev_info"] = {
        "f1": _info(lambda: f1_compare(ev["daily"], pd.read_parquet(F1_DEV_DAILY)["ret"])),
        "without_best_coin": _info(without_best_coin, panel, DEV, ev["coins"]),
        "dsr_rt_plus_xs": _info(dsr_with_xs, ev)}
    _save_json(p, r)
    _log_run("r7-sweep-info", version=v, n_eff_project=n_proj, n_strict=n_strict)
    return r


def dsr_with_xs(ev: dict) -> dict:
    """Information: DSR with RT's 6 configs and XS's 6 as one family, over their common days."""
    j = pd.concat([pd.concat(ev["cfg_daily"], axis=1), pd.read_parquet(XS_TRIALS)], axis=1, join="inner")
    if len(j) < 365:
        raise ValueError(f"only {len(j)} common days with XS")
    n = stats.effective_trials(j.to_numpy())
    return _dsr(ev["daily"], n) | {"n_eff": n, "days": len(j)}


def preflight_holdout() -> tuple[list[str], bool]:
    """Everything checked before the unlock, from hashes, timestamps and committed files only."""
    probs = []
    p = RESULTS / f"{CODE}.json"
    r = json.loads(p.read_text()) if p.exists() else {}
    if not r.get("dev_pass"):
        probs.append("RT did not pass dev")
    if r and bool(r.get("dev_pass")) != bool(r.get("gates") and all(r["gates"].values())):
        probs.append("RT.json's dev_pass does not match its gates")
    if r and r.get("version") != R.Rotation().version:
        probs.append(f"Rotation.version {R.Rotation().version} is not the swept version {r.get('version')}")
    if r and r.get("code_sha256") != code_sha():
        probs.append("Round 7 code changed since the dev sweep")
    if r and {k: r.get(k) for k in lib_versions()} != lib_versions():
        probs.append("numpy/numba/pandas versions changed since the dev sweep")
    if r and r.get("manifest_sha256") != (guards.sha_text(MANIFEST) if MANIFEST.exists() else None):
        probs.append("the data manifest changed since the dev sweep")
    if r and r.get("universe_sha256") != guards.sha_text(UNIVERSE_CSV):
        probs.append("the universe changed since the dev sweep")
    if r and r.get("f1_daily_sha256") != (guards.sha_file(F1_DEV_DAILY) if F1_DEV_DAILY.exists() else None):
        probs.append("F1_dev_daily.parquet changed or is missing since the dev sweep")
    if r:
        try:
            sec = guards.sha_str(guards.committed_section(PREREG, SECTION))
        except SystemExit as e:
            sec = str(e)
        if r.get("prereg_section_sha256") != sec:
            probs.append("the committed Round 7 registration changed since the dev sweep")
        if guards.log_records(RUN_LOG, "r7-validate", version=r.get("version")):
            probs.append("this version already has a holdout result or a recorded holdout failure")
    probs += verify_manifest()
    try:
        guards.require_committed([p, RESULTS / f"{CODE}_dev_daily.parquet", RUN_LOG, registry.PATH, DATA_REPORT,
                                  MANIFEST, CALIB_CSV, POWER_CSV, F1_DEV_DAILY])
    except SystemExit as e:
        probs.append(str(e))
    if r:
        verdicts = [x for x in guards.log_records(RUN_LOG, "r7-sweep", committed_only=True, version=r.get("version"))
                    if "verdict" in x]
        if not any(x.get("verdict") is True and x.get("code_sha256") == r.get("code_sha256")
                   and x.get("holdout_dd_limit") == r.get("holdout_dd_limit") for x in verdicts):
            probs.append("no committed r7-sweep pass record matching RT.json (version, code, drawdown limit)")
    done = "holdout" in r
    if done:
        probs.append("RT holdout already evaluated")
    if r and "holdout_dd_limit" not in r:
        probs.append("RT.json has no holdout drawdown limit")
    unlocks, resumes = unlock_attempts()
    if resumes and not unlocks:
        probs.append("a Round 7 resume record has no earlier unlock")
    resume = len(unlocks) > 0 and not done
    if len(unlocks) > 1 or len(resumes) > 1:
        probs.append("Round 7 already has more than one unlock or resume")
    elif resume and resumes:
        probs.append("the Round 7 unlock was already resumed once")
    elif resume and unlocks[0].get("version") != r.get("version"):
        probs.append(f"the logged unlock was for version {unlocks[0].get('version')}, not {r.get('version')}")
    return probs, resume


def _record_holdout_failure(r: dict, p, v: int, err: str) -> None:
    r |= {"holdout": {"integrity_error": err}, "holdout_gates": {"no integrity error": False},
          "holdout_status": "error after the unlock: FAIL (pre-registered, no fix or rerun)",
          "headline": {"PASSES_IN_PRACTICE": False, "STRICT_TIER": False}, "final_pass": False}
    _save_json(p, r)
    _log_run("r7-validate", version=v, hard_error=err)


def cmd_validate(reason: str) -> dict:
    probs, resume = preflight_holdout()
    if probs:
        raise SystemExit("Round 7 holdout stays locked: " + "; ".join(probs))
    guards.clean_src()
    p = RESULTS / f"{CODE}.json"
    r = json.loads(p.read_text())
    v = int(r["version"])
    _log_run("r7-validate-start", version=v, resume=resume, code_sha256=r["code_sha256"])
    tag = f"[R7 resume RT v{v}]" if resume else f"[R7: RT v{v}]"
    store.unlock_holdout(f"{reason} {tag}", round=7, kind="resume" if resume else "unlock", version=v)
    try:
        panel, infos = build_panel(HOLD)
        evh = evaluate(panel, HOLD, with_null=False)
        if evh["daily"].index.min() < HOLD.start:
            raise HardError("holdout output starts before the holdout")
        hg = gates_holdout(evh, float(r["dev"]["sharpe"]), float(r["holdout_dd_limit"]))
        dev_daily = pd.read_parquet(RESULTS / f"{CODE}_dev_daily.parquet")["ret"]
        daily_ab = pd.concat([dev_daily, evh["daily"]]).sort_index()
        head = headline(bool(r["dev_pass"]), all(hg.values()), daily_ab, float(r.get("n_strict", NEFF_FLOOR)))
    except Exception as e:  # noqa: BLE001 - pre-registered: any error after the unlock is a FAIL
        err = f"{type(e).__name__}: {e}"
        _record_holdout_failure(r, p, v, err)
        raise SystemExit(f"error after the holdout unlock, recorded as a FAIL: {err}")
    r |= {"holdout": _public(evh), "holdout_gates": hg, "holdout_status": "clean",
          "holdout_window": [str(HOLD.start.date()), str((HOLD.end - DAY).date())], "headline": head,
          "final_pass": head["PASSES_IN_PRACTICE"]}
    _save_json(p, r)                                 # the gated verdict is saved before anything else is written
    _log_run("r7-validate", version=v, headline=head, gates=hg)
    _write("holdout", evh, panel)
    infos.to_csv(R7_DIR / "holdout_coin_info.csv", index=False)
    f1h = _info(lambda: pd.concat([pd.read_parquet(F1_DEV_DAILY)["ret"],
                                   pd.read_parquet(F1_HOLD_DAILY).mean(axis=1)]).sort_index())
    r["holdout_info"] = {
        "f1": _info(f1_compare, evh["daily"], f1h) if isinstance(f1h, pd.Series) else f1h,
        "without_best_coin": _info(without_best_coin, panel, HOLD, evh["coins"]),
        "dsr_family_dev_plus_holdout": _dsr(daily_ab, float(r["n_eff_family"])),
        "dsr_raw6_dev_plus_holdout": _dsr(daily_ab, float(len(R.CONFIGS)))}
    _save_json(p, r)
    s = evh["summary"]
    print(f"RT holdout: total={s['total_return']:.2%} sharpe={s['sharpe']:.2f} maxDD={s['max_dd']:.2%} gates={hg} "
          f"| PASSES IN PRACTICE={head['PASSES_IN_PRACTICE']} (strict tier DSR {head['strict_dsr']:.3f} at N "
          f"{head['strict_n']:.2f})")
    return r
