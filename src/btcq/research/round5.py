"""Round 5: daily-activity strategies on the 5 majors. D1 (daily long-only Donchian ensemble, shadow-only
if it passes) and XS (dollar-neutral cross-sectional momentum, confirmed by the one-shot 2024-26 holdout).

    python -m btcq.research.run r5-power
    python -m btcq.research.run r5-download-spot
    python -m btcq.research.run r5-sweep
    python -m btcq.research.run r5-validate --unlock-holdout     # only if XS passed dev

See research/PREREGISTRATION.md, Round 5. Protocol (user decision 2026-10-16): REPLICATED = dev gates
+ one-shot holdout gates; DSR only decides STRICT, at N = max(ratcheted N_eff, 14.85 + 2).
"""
from __future__ import annotations

import json
import subprocess
import time
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from btcq.backtest import engine, metrics, stats
from btcq.config import PERP_START, ROOT, SLIPPAGE, SYMBOLS, Costs, Risk
from btcq.data import store
from btcq.data import universe as U
from btcq.research import registry
from btcq.research.portfolio import assemble_slots
from btcq.research.round4 import SlimMarket, sleeve_daily
from btcq.research.run import GATES, N_BOOT, N_PERM, RESULTS, _save_json, trade_equity_mdd
from btcq.strategies import r5_d1 as D
from btcq.strategies import r5_xsmom as X

R5_DIR = ROOT / "research" / "round5"
RUN_LOG = R5_DIR / "runs.jsonl"
SPOT_REPORT = R5_DIR / "data_report_spot.csv"
POWER_CSV = R5_DIR / "power_table.csv"
PREREG = ROOT / "research" / "PREREGISTRATION.md"
CODES = ["D1", "XS"]
LATENCY_MIN = 1
NEFF_FLOOR = 14.849570528171945 + 2
N_DRAWS = 1000
DAY = pd.Timedelta(days=1)
SPOT_LAST_MONTH = {"ETHUSDT": "2020-03", "XRPUSDT": "2020-03", "BNBUSDT": "2020-04", "SOLUSDT": "2020-11"}


@dataclass(frozen=True)
class Window:
    name: str
    period: str
    start: pd.Timestamp
    end: pd.Timestamp        # exclusive

    @property
    def calendar(self) -> pd.DatetimeIndex:
        return pd.date_range(self.start, self.end - DAY, freq="D").as_unit("ms")


DEV = Window("dev", "dev", pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC"))
HOLD = Window("holdout", "full", pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2026-10-01", tz="UTC"))


# ---------------------------------------------------------------- data

@dataclass
class Coin:
    sym: str
    mk: SlimMarket
    bar_ts: np.ndarray           # perp daily bar open times (ms)
    perp_close: pd.Series        # perp daily closes, index = bar open ms


def load_coin(sym: str, w: Window) -> Coin:
    m1 = store.load_1m(w.period, sym)
    m1 = m1[(m1["ts"] >= int(PERP_START.value // 10**6)) & (m1["ts"] < int(w.end.value // 10**6))]
    return coin_from_m1(sym, m1.reset_index(drop=True), store.load_funding(w.period, sym))


def coin_from_m1(sym: str, m1: pd.DataFrame, funding: pd.DataFrame | None) -> Coin:
    mk = engine.Market(m1, funding, None, synthetic_funding=Costs.synthetic_funding)
    bars = mk.bars("1d")
    ex = mk.exec_idx("1d", LATENCY_MIN)
    slim = SlimMarket(mk.ts.copy(), mk.o.copy(), mk.h.copy(), mk.l.copy(), mk.c.copy(), mk.fund.copy(),
                      mk.day_marks.copy(), mk.days, ex.astype(np.int64), "1d", LATENCY_MIN)
    bt = bars["ts"].to_numpy(np.int64)
    return Coin(sym, slim, bt, pd.Series(bars["close"].to_numpy(np.float64), index=bt))


def load_coins(w: Window) -> dict[str, Coin]:
    return {s: load_coin(s, w) for s in SYMBOLS}


def stitched_closes(coin: Coin) -> pd.Series:
    return store.stitch_daily(store.spot_daily_close(coin.sym), coin.perp_close)


def _costs(sym: str, mult: float = 1.0, zero: bool = False) -> Costs:
    return Costs(taker=0.0, maker=0.0, slippage=0.0) if zero else Costs(slippage=SLIPPAGE[sym]).scaled(mult)


def _seed(base: int, sym: str) -> int:
    return base + zlib.crc32(sym.encode()) % 65536


# ---------------------------------------------------------------- D1

@dataclass
class D1Sub:
    sym: str
    L: int
    sig: engine.Signals
    trail: np.ndarray
    state: np.ndarray


def d1_build(coins: dict[str, Coin], closes_by: dict[str, pd.Series] | None = None) -> dict[tuple, D1Sub]:
    """closes_by: daily closes used for signals (default: spot-warmed stitched closes)."""
    out = {}
    for s, coin in coins.items():
        closes = closes_by[s] if closes_by is not None else stitched_closes(coin)
        for L in D.LOOKBACKS:
            sig, trail, state = D.to_engine(D.sub_signals(closes, L), coin.bar_ts)
            out[(s, L)] = D1Sub(s, L, sig, trail, state)
    return out


def _d1_risk() -> Risk:
    return Risk(risk_per_trade=0.01, max_leverage=D.LEV_CAP, initial_equity=Risk().initial_equity)


def d1_run(subs: dict, coins: dict[str, Coin], w: Window, mult: float = 1.0, sig_override: dict | None = None):
    """Run every sub-sleeve; returns {(sym, L): (daily, trades)} with R from the trailing level."""
    parts, floors = {}, 0
    for key, sb in subs.items():
        sig = sig_override[key] if sig_override else sb.sig
        res = engine.run(coins[sb.sym].mk, sig, _costs(sb.sym, mult), _d1_risk(), start=w.start,
                         latency_min=LATENCY_MIN)
        tr, nf = D.apply_r(res.trades, sb.trail)
        floors += nf
        parts[key] = (sleeve_daily(res, w.calendar), tr.assign(symbol=sb.sym, lookback=sb.L))
    return parts, floors


def d1_assemble(parts: dict, w: Window, cfg: str = D.PRIMARY):
    Ls = D.CONFIGS[cfg]
    sel = [v for (s, L), v in parts.items() if L in Ls]
    return assemble_slots(sel, len(SYMBOLS) * len(Ls), w.calendar)


def d1_mismatches(subs: dict, parts: dict, coins: dict[str, Coin], w: Window) -> int:
    """Bars where the engine's position differs from the state machine, over the decisions executed inside
    the window (the last bar is excluded), plus any stop exit (the sentinel must never fire)."""
    bad = 0
    for key, (_, tr) in parts.items():
        sb = subs[key]
        mk = coins[sb.sym].mk
        ex = mk._ex
        live = (ex >= mk.index_at(w.start)) & (ex < len(mk.ts))
        pos = np.zeros(len(ex), np.int8)
        for _, t in tr.iterrows():
            b0 = int(t["entry_tf"])
            b1 = int(np.searchsorted(ex, int(t["exit_i"]))) if t["reason"] != "end" else len(ex)
            pos[b0:b1] = 1
            bad += int(t["reason"] == "stop")
        idx = np.flatnonzero(live)[:-1]
        bad += int((pos[idx] != sb.state[idx]).sum())
    return bad


def _null_calendar(w: Window) -> np.ndarray:
    """Decision bars (day-open ms) whose execution falls inside the window, shared by all coins."""
    t0, t1 = int(w.start.value // 10**6), int(w.end.value // 10**6)
    return np.arange(t0 - 86_400_000, t1 - 86_400_000, 86_400_000, dtype=np.int64)


def _shift_sig(sig: engine.Signals, state: np.ndarray, bar_ts: np.ndarray, cal: np.ndarray, k: int) -> engine.Signals:
    """Roll a sub-signal's state and size by k days on the shared calendar (days before a coin's listing are
    flat), then map back to the coin's bars; bars outside the calendar are unchanged. Level exits."""
    st = state.astype(np.int8).copy()
    sz = np.asarray(sig.size, np.float64).copy()
    pos = np.searchsorted(cal, bar_ts)
    on = (pos < len(cal)) & (cal[np.minimum(pos, len(cal) - 1)] == bar_ts)
    st_c = np.zeros(len(cal), np.int8)
    sz_c = np.full(len(cal), np.nan)
    st_c[pos[on]] = st[on]
    sz_c[pos[on]] = sz[on]
    st[on] = np.roll(st_c, k)[pos[on]]
    sz[on] = np.roll(sz_c, k)[pos[on]]
    entry = ((st == 1) & np.isfinite(sz)).astype(np.int8)
    n = len(st)
    return engine.Signals("1d", entry, np.where(entry == 1, D.STOP_SENTINEL, np.nan), np.full(n, np.nan),
                          exit_long=(st == 0) | ~np.isfinite(sz) & (st == 1), size=np.where(entry == 1, sz, np.nan))


def d1_shift_null(subs: dict, coins: dict[str, Coin], w: Window, actual: float, n_draws: int = N_DRAWS) -> dict:
    """Circular-shift null: ONE random offset per draw, applied to every coin and lookback on the shared
    calendar, so the coins' joint timing structure is kept and only timing relative to returns is broken.
    k ~ U{30 .. n-30}; statistic = mean daily net portfolio return (primary config), exact re-simulation."""
    rng = np.random.default_rng(55)
    cal = _null_calendar(w)
    null = np.empty(n_draws)
    for j in range(n_draws):
        k = int(rng.integers(30, len(cal) - 30))
        sig = {key: _shift_sig(sb.sig, sb.state, coins[sb.sym].bar_ts, cal, k) for key, sb in subs.items()}
        parts, _ = d1_run(subs, coins, w, sig_override=sig)
        null[j] = float(d1_assemble(parts, w)[1].mean())
    return {"actual_mean_daily": actual, "null_mean": float(null.mean()), "null_p95": float(np.quantile(null, 0.95)),
            "p_value": float((1 + np.sum(null >= actual)) / (1 + len(null))), "dist": null}


# ---------------------------------------------------------------- XS

def xs_panel(coins: dict[str, Coin], exclude: str | None = None) -> pd.DataFrame:
    return X.panel({s: c.perp_close for s, c in coins.items() if s != exclude})


def xs_run(coins: dict[str, Coin], px: pd.DataFrame, cfg: dict, w: Window, mult: float = 1.0, zero: bool = False,
           stop_mode: int = 0, ranks_override=None):
    plan = X.leg_plan(px, cfg, ranks_override)
    parts = {}
    for s in px.columns:
        c = coins[s]
        sig = X.to_engine(plan, s, c.bar_ts)
        res = engine.run(c.mk, sig, _costs(s, mult, zero), Risk(0.01, X.GROSS_SIDE * X.SLOTS, Risk().initial_equity),
                         start=w.start, latency_min=LATENCY_MIN, stop_mode=stop_mode)
        parts[s] = (sleeve_daily(res, w.calendar), res.trades.assign(symbol=s))
    res, daily = assemble_slots(list(parts.values()), X.SLOTS, w.calendar)
    return res, daily, parts, plan


def xs_rank_null(coins: dict[str, Coin], px: pd.DataFrame, cfg: dict, w: Window, actual: float,
                 n_draws: int = N_DRAWS) -> dict:
    """Ranks randomly permuted among eligible coins on every date, pushed through the same legs/stops/engine;
    statistic = mean daily portfolio return with fees and slippage at 0 (funding kept)."""
    rng = np.random.default_rng(5005)
    score, sigma, _ = X.features(px, int(cfg["L"]))
    rk = X.ranks(score, X.eligible(px, score, sigma))
    null = np.empty(n_draws)
    for j in range(n_draws):
        _, daily, _, _ = xs_run(coins, px, cfg, w, zero=True, ranks_override=X.permute_ranks(rk, rng))
        null[j] = float(daily.mean())
    return {"actual_mean_daily_gross": actual, "null_mean": float(null.mean()),
            "null_p95": float(np.quantile(null, 0.95)),
            "p_value": float((1 + np.sum(null >= actual)) / (1 + len(null))), "dist": null}


def xs_loco(coins: dict[str, Coin], cfg: dict, w: Window) -> dict:
    """Leave one coin out: re-simulate the rule on each 4-coin universe (slots stay 5)."""
    out = {}
    for s in SYMBOLS:
        res, _, _, _ = xs_run(coins, xs_panel(coins, exclude=s), cfg, w)
        out[s] = float(res.trades["R"].mean()) if len(res.trades) else 0.0
    return out


def f1_tests(daily: pd.Series) -> dict:
    f1 = pd.read_parquet(RESULTS / "F1_dev_daily.parquet")["ret"]
    j = pd.concat([daily.rename("x"), f1.rename("f1")], axis=1).dropna()
    wk = pd.concat([stats.weekly_returns(j["x"]), stats.weekly_returns(j["f1"])], axis=1).dropna()
    h = stats.hac_ols(j["x"].to_numpy(), j["f1"].to_numpy(), lag=7)
    sens = {lag: stats.hac_ols(j["x"].to_numpy(), j["f1"].to_numpy(), lag=lag)["t_alpha"] for lag in (0, 14, 28)}
    return {"weekly_corr": float(wk.corr().iloc[0, 1]), "hac": h, "t_alpha_sensitivity": sens, "days": len(j)}


# ---------------------------------------------------------------- shared

def ew_benchmark(coins: dict[str, Coin], w: Window) -> pd.Series:
    r = {s: c.perp_close.pct_change() for s, c in coins.items()}
    df = pd.DataFrame(r)
    df.index = pd.to_datetime(df.index, unit="ms", utc=True).as_unit("ms")
    return df.reindex(w.calendar).mean(axis=1).fillna(0.0)


def n_dsr(n_eff: float) -> float:
    return float(max(n_eff, NEFF_FLOOR))


def _dsr(daily: pd.Series, n: float) -> dict:
    sr, T, sk, ku = stats.sharpe_moments(daily)
    v0 = stats.null_sr_variance(T)
    return {"sharpe_ann": float(sr * np.sqrt(365)), "T": T, "psr0": stats.psr(sr, T, sk, ku),
            "dsr": stats.psr(sr, T, sk, ku, stats.expected_max_sr(v0, n)), "n": n,
            "sharpe_needed_ann": float(stats.expected_max_sr(v0, n) * np.sqrt(365))}


def _coin_contrib(parts: dict, key_sym, slots: int) -> dict:
    tot = {}
    for k, (d, _) in parts.items():
        s = key_sym(k)
        tot[s] = tot.get(s, 0.0) + float(d.sum()) / slots
    return tot


def _plateau(totals: list[float]) -> float:
    return float(np.mean([t > 0 for t in totals])) if totals else 0.0


# ---------------------------------------------------------------- evaluation

def d1_evaluate(coins: dict[str, Coin], w: Window, with_null: bool = True,
                closes_by: dict[str, pd.Series] | None = None, n_draws: int = N_DRAWS) -> dict:
    t0 = time.time()
    subs = d1_build(coins, closes_by)
    parts, floors = d1_run(subs, coins, w)
    res, daily = d1_assemble(parts, w)
    stress, _ = d1_run(subs, coins, w, mult=2.0)
    _, daily2 = d1_assemble(stress, w)
    cfg_daily = {c: d1_assemble(parts, w, c)[1] for c in D.CONFIGS}
    cfg_trades = {c: int(sum(len(v[1]) for (s, L), v in parts.items() if L in D.CONFIGS[c])) for c in D.CONFIGS}
    M = pd.concat(cfg_daily, axis=1).fillna(0.0).to_numpy()
    pbo, logits = stats.pbo_cscv(M)
    contrib = _coin_contrib({k: v for k, v in parts.items() if k[1] in D.CONFIGS[D.PRIMARY]}, lambda k: k[0],
                            len(SYMBOLS) * len(D.LOOKBACKS))
    bench = ew_benchmark(coins, w)
    out = {"summary": metrics.summarize(res), "weekly": stats.weekly_t(daily),
           "stress_2x_mean_daily": float(daily2.mean()), "year_share": stats.year_share_daily(daily),
           "coin_contrib": contrib, "coins_positive": int(sum(v > 0 for v in contrib.values())),
           "pbo": pbo, "plateau": _plateau([float((1 + cfg_daily[c]).prod() - 1) for c in D.CONFIGS if c != D.PRIMARY]),
           "config_totals": {c: float((1 + d).prod() - 1) for c, d in cfg_daily.items()},
           "alpha_vs_ew": stats.beta_alpha(daily, bench), "floors": floors,
           "mismatches": d1_mismatches(subs, parts, coins, w), "exposure": d1_exposure(subs, parts, coins, w),
           "daily": daily, "res": res,
           "cfg_daily": cfg_daily, "cfg_trades": cfg_trades, "pbo_logits": logits, "benchmark_daily": bench,
           "activity": {"order_days_share": _order_day_share(res.trades, w),
                        "sub_trades_per_week": len(res.trades) / (len(w.calendar) / 7)}}
    if with_null:
        out["null"] = d1_shift_null(subs, coins, w, float(daily.mean()), n_draws)
    out["runtime_s"] = time.time() - t0
    return out


def d1_exposure(subs: dict, parts: dict, coins: dict[str, Coin], w: Window) -> dict:
    """Information: realized gross exposure (sized at entry) vs the paper's daily-resized target, primary config.
    Exposure of a sub-sleeve = notional / its own equity; the account's is their mean over 45 sub-sleeves."""
    cal = w.calendar
    real = pd.Series(0.0, index=cal)
    target = pd.Series(0.0, index=cal)
    for key, (d, tr) in parts.items():
        if key[1] not in D.CONFIGS[D.PRIMARY]:
            continue
        sb, coin = subs[key], coins[key[0]]
        close = pd.Series(coin.perp_close.to_numpy(), index=pd.to_datetime(coin.bar_ts, unit="ms", utc=True).as_unit("ms"))
        close = close.reindex(cal).ffill()
        eq = 1e4 * (1 + d).cumprod()
        q = pd.Series(0.0, index=cal)
        for _, t in tr.iterrows():
            a, b = t["entry_time"].floor("D"), t["exit_time"].floor("D")
            q[(cal >= a) & (cal < b)] += t["qty"]
        real = real.add((q * close / eq).fillna(0.0), fill_value=0.0)
        tgt = pd.Series(np.where(sb.state == 1, np.nan_to_num(np.asarray(sb.sig.size, float)), 0.0),
                        index=pd.to_datetime(coin.bar_ts + 86_400_000, unit="ms", utc=True).as_unit("ms"))
        target = target.add(tgt.reindex(cal).fillna(0.0), fill_value=0.0)
    ratio = (real / target).where(target > 0).dropna()
    return {"mean_ratio": float(ratio.mean()) if len(ratio) else None,
            "p5_p50_p95": [float(x) for x in ratio.quantile([0.05, 0.5, 0.95])] if len(ratio) else None,
            "share_outside_0.5_2": float(((ratio < 0.5) | (ratio > 2)).mean()) if len(ratio) else None,
            "mean_gross_exposure": float(real.mean() / (len(SYMBOLS) * len(D.LOOKBACKS)))}


def _order_day_share(trades: pd.DataFrame, w: Window) -> float:
    if not len(trades):
        return 0.0
    days = pd.concat([trades["entry_time"], trades["exit_time"]]).dt.floor("D").nunique()
    return float(days / len(w.calendar))


def gates_d1(ev: dict) -> dict:
    return {
        "weekly t>=2": ev["weekly"]["t"] >= 2.0,
        "2x-cost mean>0": ev["stress_2x_mean_daily"] > 0,
        "year share<=50%": ev["year_share"] <= 0.5,
        ">=3/5 coins positive": ev["coins_positive"] >= 3,
        "shift null p<0.05": ev["null"]["p_value"] < 0.05,
        "PBO<=0.5": ev["pbo"] <= 0.5,
        "plateau>=70%": ev["plateau"] >= 0.7,
        "alpha PSR vs EW hold>=0.95": ev["alpha_vs_ew"]["psr_alpha"] >= 0.95,
    }


def xs_evaluate(coins: dict[str, Coin], w: Window, with_null: bool = True, cfg_i: int = X.PRIMARY,
                n_draws: int = N_DRAWS) -> dict:
    t0 = time.time()
    px = xs_panel(coins)
    cfg = X.CONFIGS[cfg_i]
    res, daily, parts, plan = xs_run(coins, px, cfg, w)
    st, _, _, _ = xs_run(coins, px, cfg, w, mult=2.0)
    wm, _, _, _ = xs_run(coins, px, cfg, w, stop_mode=1)
    cfg_daily, cfg_tot, cfg_trades = {}, {}, {}
    for i, c in enumerate(X.CONFIGS):
        r_i, d_i, _, _ = (res, daily, None, None) if i == cfg_i else xs_run(coins, px, c, w)
        cfg_daily[i], cfg_tot[i], cfg_trades[i] = d_i, float((1 + d_i).prod() - 1), len(r_i.trades)
    pbo, logits = stats.pbo_cscv(pd.concat(cfg_daily, axis=1).fillna(0.0).to_numpy())
    t = res.trades
    sym_R = {s: float(t.loc[t["symbol"] == s, "R"].mean()) if (t["symbol"] == s).any() else 0.0 for s in SYMBOLS}
    out = {"summary": metrics.summarize(res), "weekly": stats.weekly_t(daily),
           "stress_2x": metrics.summarize(st), "worst_minute": metrics.summarize(wm),
           "sym_R": sym_R, "coins_positive": int(sum(v > 0 for v in sym_R.values())),
           "pbo": pbo, "plateau": _plateau([v for i, v in cfg_tot.items() if i != cfg_i]), "config_totals": cfg_tot,
           "f1": f1_tests(daily) if w.name == "dev" else None, "loco": xs_loco(coins, cfg, w),
           "alpha_vs_ew": stats.beta_alpha(daily, ew_benchmark(coins, w)),
           "daily": daily, "res": res, "cfg_daily": cfg_daily, "cfg_trades": cfg_trades, "pbo_logits": logits,
           "plan": plan,
           "stress_trades": st.trades, "activity": {"order_days_share": _order_day_share(t, w)}}
    if with_null:
        _, gross, _, _ = xs_run(coins, px, cfg, w, zero=True)
        out["null"] = xs_rank_null(coins, px, cfg, w, float(gross.mean()), n_draws)
    out["runtime_s"] = time.time() - t0
    return out


def gates_xs(ev: dict) -> dict:
    d = ev["summary"]
    return {
        "PF>=1.2": d.get("pf", 0) >= GATES["min_pf"],
        "weekly t>=2": ev["weekly"]["t"] >= 2.0,
        "2x-cost avg_R>0": (ev["stress_2x"].get("avg_R") or 0) > 0,
        "year share<=50%": (d.get("max_year_share") or np.inf) <= GATES["max_year_share"],
        ">=3/5 coins positive": ev["coins_positive"] >= 3,
        "PBO<=0.5": ev["pbo"] <= 0.5,
        "plateau>=70%": ev["plateau"] >= 0.7,
        "rank null p<0.05": ev["null"]["p_value"] < 0.05,
        "weekly corr with F1<=0.3": ev["f1"]["weekly_corr"] <= 0.3,
        "HAC alpha t vs F1>=2": ev["f1"]["hac"]["t_alpha"] >= 2.0,
        "leave-one-coin-out avg_R>0": all(v > 0 for v in ev["loco"].values()),
    }


def gates_xs_holdout(evh: dict, dev: dict, dev_trades: pd.DataFrame) -> dict:
    h = evh["summary"]
    n_h = max(h["trades"], 2)
    _, mdd_dist = stats.block_bootstrap(dev_trades["ret"].to_numpy(), N_BOOT, n_out=n_h)
    mdd = trade_equity_mdd(evh["res"].trades["ret"].to_numpy()) if h["trades"] else 0.0
    evh["trade_mdd"], evh["boot_mdd_p95"] = mdd, float(np.quantile(mdd_dist, 0.95))
    return {
        "holdout avg_R>0": h.get("avg_R", 0) > 0,
        "holdout Sharpe>=50% dev": h.get("sharpe", 0) >= 0.5 * dev["summary"].get("sharpe", 0),
        "holdout trade MDD<=dev boot p95": mdd <= evh["boot_mdd_p95"],
        "holdout trades/yr>=52": h.get("trades_per_year", 0) >= 52,
        "holdout leave-one-coin-out avg_R>0": all(v > 0 for v in evh["loco"].values()),
    }


def headline_r5(dev_pass: bool, hold_pass: bool, daily_ab: pd.Series, n: float) -> dict:
    d = _dsr(daily_ab, n)
    rep = bool(dev_pass and hold_pass)
    return d | {"REPLICATED": rep, "STRICT": bool(rep and d["dsr"] >= 0.95)}


# ---------------------------------------------------------------- guards, logging

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
        f.write(json.dumps(rec, default=str) + "\n")


def _require_prereg() -> None:
    from btcq.research.round4 import _require_committed
    _require_committed([PREREG])
    txt = PREREG.read_text(encoding="utf-8")
    if "## Round 5" not in txt or "Power table" not in txt:
        raise SystemExit("PREREGISTRATION.md has no committed Round 5 section with its power table")


def _clean_src() -> None:
    if _git("status", "--porcelain", "--", "src"):
        raise SystemExit("src/ has uncommitted changes: commit the code before a pre-registered run")


# ---------------------------------------------------------------- commands

def cmd_power() -> list[dict]:
    rows = stats.power_table()
    U.write_csv(pd.DataFrame(rows), POWER_CSV)
    for r in rows:
        print({k: round(v, 4) for k, v in r.items()})
    return rows


def cmd_download_spot() -> pd.DataFrame:
    _require_prereg()
    rows = []
    for sym in SYMBOLS:
        if sym == "BTCUSDT":
            rep = {"symbol": sym, "status": "ok (local spot 1m)"}
        elif (store.symbol_dir(sym) / "spot_1d.parquet").exists():
            rep = {"symbol": sym, "status": "ok (already built)"}
        else:
            rep = store.build_spot_1d(sym, SPOT_LAST_MONTH[sym])
        rows.append(rep | _spot_facts(sym))
    df = spot_red_flags(pd.DataFrame(rows))
    U.write_csv(df, SPOT_REPORT)
    _log_run("r5-download-spot", red_flags=int(df["red_flag"].sum()))
    print(df.to_string())
    if df["red_flag"].any():
        raise SystemExit("spot data red flags: fix the data and amend before any strategy run")
    return df


def spot_red_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-registered spot warm-up red flags. A missing stitch jump is a red flag (it must be measurable);
    the overlap basis is not checkable for BTC (local spot ends the day before the perp starts)."""
    df = df.copy()
    df["red_flag"] = (~df["status"].astype(str).str.startswith("ok") | (df["longest_gap_days"] > 2)
                      | (df["median_overlap_basis"].fillna(0) > 0.005) | df["stitch_jump"].isna()
                      | (df["stitch_jump"].abs().fillna(0) > 0.01) | (df["bad_prices"] > 0))
    return df


def _spot_facts(sym: str) -> dict:
    coin = load_coin(sym, DEV)
    spot = store.spot_daily_close(sym)
    perp = coin.perp_close
    both = spot.index.intersection(perp.index)
    basis = np.abs(np.log(spot[both] / perp[both])) if len(both) else pd.Series(dtype=float)
    s = store.stitch_daily(spot, perp, max_fill_days=10_000)
    pre = spot[spot.index < perp.index[0]]
    gaps = np.diff(np.r_[pre.index.to_numpy(), perp.index[0]]) // 86_400_000 - 1 if len(pre) else np.array([0])
    d0 = int(perp.index[0])
    if d0 in spot.index:                      # same-day basis at the stitch
        jump = float(np.log(perp.iloc[0] / spot[d0]))
    elif sym == "BTCUSDT":                    # adjacent minutes: spot 2019-12-31 23:59 close vs perp first open
        sp = pd.read_parquet(store.symbol_dir(sym) / "spot_1m.parquet", columns=["ts", "close"])
        sp = sp[sp["ts"] < int(PERP_START.value // 10**6)]
        jump = float(np.log(coin.mk.o[0] / sp["close"].iloc[-1]))
    else:
        jump = None
    return {"spot_first": str(pd.Timestamp(int(spot.index[0]), unit="ms", tz="UTC").date()) if len(spot) else None,
            "perp_first": str(pd.Timestamp(d0, unit="ms", tz="UTC").date()),
            "stitched_days": len(s), "longest_gap_days": int(gaps.max()) if len(gaps) else 0,
            "overlap_days": len(both), "median_overlap_basis": float(basis.median()) if len(basis) else None,
            "max_overlap_basis": float(basis.max()) if len(basis) else None, "stitch_jump": jump,
            "bad_prices": int((spot <= 0).sum() + (perp <= 0).sum())}


def _trial_rows(code: str, ev: dict, params_of, version: int = 1) -> list[dict]:
    rows = []
    for k, d in ev["cfg_daily"].items():
        sr, T, sk, ku = stats.sharpe_moments(d)
        rows.append({"strategy": code, "version": version, "params": params_of(k), "period": "dev", "sr": sr, "T": T,
                     "skew": sk, "kurt": ku, "trades": int(ev["cfg_trades"][k]), "sharpe_ann": float(sr * np.sqrt(365))})
    return rows


def cmd_sweep(rerun: bool = False) -> None:
    _require_prereg()
    from btcq.research.round4 import _require_committed
    _require_committed([SPOT_REPORT])
    if pd.read_csv(SPOT_REPORT)["red_flag"].astype(str).eq("True").any():
        raise SystemExit("spot data report has red flags")
    _clean_src()
    metas = {"D1": D.DonchianEnsemble(), "XS": X.XSMomentum()}
    for c in CODES:
        p = RESULTS / f"{c}.json"
        if p.exists():
            if not rerun:
                raise SystemExit(f"{c}.json exists; the dev sweep runs once (--rerun needs a version bump)")
            if metas[c].version <= json.loads(p.read_text()).get("version", 0):
                raise SystemExit(f"--rerun of {c} needs a version bump (a rerun counts as new trials)")
    coins = load_coins(DEV)
    print("D1 ...", flush=True)
    d1 = d1_evaluate(coins, DEV)
    print("XS ...", flush=True)
    xs = xs_evaluate(coins, DEV)
    g_d1, g_xs = gates_d1(d1), gates_xs(xs)
    vd, vx = metas["D1"].version, metas["XS"].version
    rows = _trial_rows("D1", d1, lambda c: {"set": c, "lookbacks": list(D.CONFIGS[c]),
                                             "vol_target": D.VOL_TARGET, "cap": D.LEV_CAP}, vd)
    rows += _trial_rows("XS", xs, lambda i: X.CONFIGS[i], vx)
    registry.record(rows)
    for code, v, cd in (("D1", vd, list(d1["cfg_daily"].values())), ("XS", vx, list(xs["cfg_daily"].values()))):
        name = f"{code}_trial_returns.parquet" if v == 1 else f"{code}v{v}_trial_returns.parquet"
        pd.concat({(f"{code}_{i}" if v == 1 else f"{code}_v{v}_{i}"): d for i, d in enumerate(cd)}, axis=1) \
            .to_parquet(RESULTS / name)
    from btcq.research.run import all_trial_returns, cmd_gates
    trial = all_trial_returns()
    n_eff = registry.ratcheted_effective_trials(stats.effective_trials(trial.to_numpy()), trial.shape[1],
                                                note="round 5 (D1, XS)")
    n = n_dsr(n_eff)
    for code, ev, g in (("D1", d1, g_d1), ("XS", xs, g_xs)):
        _write(code, "dev", ev)
        meta = metas[code]
        r = {"code": code, "name": meta.name, "tf": "1d", "version": meta.version, "round": 5,
             "hypothesis": meta.hypothesis, "rules": meta.rules,
             "params": {"set": D.PRIMARY} if code == "D1" else X.CONFIGS[X.PRIMARY],
             "holdout_scope": "information only (contaminated)" if code == "D1" else "clean",
             "start": str(DEV.start.date()), "end": str((DEV.end - DAY).date()),
             "dev": ev["summary"], "dev_detail": _public(ev), "gates": g, "dev_pass": all(g.values()),
             "dsr_dev": _dsr(ev["daily"], n), "n_dsr": n, "n_eff": n_eff, "n_trials": registry.multiple_testing_inputs("dev")[0]}
        _save_json(RESULTS / f"{code}.json", r)
        failed = [k for k, v in g.items() if not v]
        s = ev["summary"]
        print(f"{code} dev: trades={s['trades']} avgR={s.get('avg_R', 0):.3f} sharpe={s.get('sharpe', 0):.2f} "
              f"weekly t={ev['weekly']['t']:.2f} null p={ev['null']['p_value']:.3f} -> "
              f"{'PASS' if r['dev_pass'] else 'FAIL: ' + ', '.join(failed)}", flush=True)
    _log_run("r5-sweep", n_eff=n_eff, n_dsr=n, verdict={"D1": all(g_d1.values()), "XS": all(g_xs.values())},
             d1_floors=d1["floors"], d1_mismatches=d1["mismatches"], d1_exposure=d1["exposure"])
    cmd_gates()
    for c in ("U1", "U1L"):   # round 4 is not re-gated (it failed); refresh its multiple-testing fields as information
        p = RESULTS / f"{c}.json"
        if p.exists():
            r = json.loads(p.read_text())
            sr, T, sk, ku = r["sr_daily"], r["T"], r["skew"], r["kurt"]
            r |= {"n_trials": registry.multiple_testing_inputs("dev")[0], "n_eff": n_eff,
                  "dsr_dev": stats.dsr(sr, T, sk, ku, stats.null_sr_variance(T), max(int(round(n_eff)), 2)),
                  "refreshed_by": "round 5 (information only)"}
            _save_json(p, r)


def _public(ev: dict) -> dict:
    skip = {"daily", "res", "cfg_daily", "cfg_trades", "pbo_logits", "plan", "stress_trades", "benchmark_daily"}
    out = {k: v for k, v in ev.items() if k not in skip}
    if "null" in out:
        out["null"] = {k: v for k, v in out["null"].items() if k != "dist"}
    return out


def _write(code: str, tag: str, ev: dict) -> None:
    ev["res"].trades.to_parquet(RESULTS / f"{code}_{tag}_trades.parquet")
    ev["res"].equity.rename("equity").to_frame().to_parquet(RESULTS / f"{code}_{tag}_equity.parquet")
    ev["daily"].rename("ret").to_frame().to_parquet(RESULTS / f"{code}_{tag}_daily.parquet")
    if tag == "dev":
        pd.DataFrame([{"cfg": 0, **ev["summary"], "nbhd_sharpe": ev["summary"].get("sharpe")}]) \
            .to_csv(RESULTS / f"{code}_configs.csv", index=False)
        boot_tot, boot_mdd = stats.block_bootstrap(ev["res"].trades["ret"].to_numpy(), N_BOOT) \
            if len(ev["res"].trades) > 1 else (np.zeros(1), np.zeros(1))
        np.savez_compressed(RESULTS / f"{code}_dev_dists.npz", null=ev.get("null", {}).get("dist", np.zeros(1)),
                            boot_tot=boot_tot, boot_mdd=boot_mdd, pbo_logits=np.asarray(ev["pbo_logits"]))
        if "stress_trades" in ev:
            ev["stress_trades"].to_parquet(RESULTS / f"{code}_dev_stress_2x_trades.parquet")
    if "plan" in ev:
        pl = ev["plan"]
        pd.concat({"rank": pl["ranks"], "side": pl["side"], "weight": pl["weight"]}, axis=1) \
            .to_parquet(RESULTS / f"{code}_{tag}_ranks.parquet")


def preflight_holdout() -> tuple[list[str], bool]:
    probs = []
    xs = json.loads((RESULTS / "XS.json").read_text()) if (RESULTS / "XS.json").exists() else {}
    if not xs.get("dev_pass"):
        probs.append("XS did not pass dev")
    done = "holdout" in xs
    if done:
        probs.append("XS holdout already evaluated")
    for f in ("XS_dev_trades.parquet", "XS_dev_daily.parquet", "D1.json"):
        if not (RESULTS / f).exists():
            probs.append(f"{f} missing")
    log = store.HOLDOUT_LOG.read_text() if store.HOLDOUT_LOG.exists() else ""
    unlocks, resumes = log.count("[R5:"), log.count("[R5 resume")
    resume = unlocks > 0 and not done
    if unlocks > 1 or resumes > 1:
        probs.append("holdout_log already has more than one round 5 unlock")
    elif resume and resumes:
        probs.append("the round 5 unlock was already resumed once")
    end_ms = int((HOLD.end - DAY).value // 10**6)
    for s in SYMBOLS:
        d = store.symbol_dir(s)
        m = pd.read_parquet(d / "perp_1m.parquet", columns=["ts"])["ts"]
        f = pd.read_parquet(d / "funding.parquet", columns=["ts"])["ts"]
        if int(m.max()) < end_ms + 86_400_000 - 60_000 or int(f.max()) < end_ms:
            probs.append(f"{s}: 1m or funding data does not reach 2026-09-30")
    return probs, resume


def cmd_validate(reason: str) -> None:
    probs, resume = preflight_holdout()
    if probs:
        raise SystemExit("holdout stays locked: " + "; ".join(probs))
    _clean_src()
    tag = "[R5 resume" if resume else "[R5:"
    store.unlock_holdout(f"{reason} {tag} XS,D1]")
    coins = load_coins(HOLD)
    xs_r = json.loads((RESULTS / "XS.json").read_text())
    evh = xs_evaluate(coins, HOLD, with_null=False)
    dev_trades = pd.read_parquet(RESULTS / "XS_dev_trades.parquet")
    hg = gates_xs_holdout(evh, {"summary": xs_r["dev"]}, dev_trades)
    daily_ab = pd.concat([pd.read_parquet(RESULTS / "XS_dev_daily.parquet")["ret"], evh["daily"]]).sort_index()
    _write("XS", "holdout", evh)
    xs_r |= {"holdout": _public(evh), "holdout_gates": hg, "holdout_status": "clean (general-knowledge disclosure)",
             "headline": headline_r5(bool(xs_r["dev_pass"]), all(hg.values()), daily_ab, xs_r["n_dsr"])}
    xs_r["final_pass"] = xs_r["headline"]["REPLICATED"]
    _save_json(RESULTS / "XS.json", xs_r)          # the gated verdict is saved before anything optional runs
    d1_r = json.loads((RESULTS / "D1.json").read_text())
    if d1_r.get("dev_pass"):
        try:
            evd = d1_evaluate(coins, HOLD, with_null=False)
            _write("D1", "holdout", evd)
            d1_r |= {"holdout": _public(evd), "holdout_status": "contaminated: information only"}
        except Exception as e:  # noqa: BLE001 - information only; must never affect XS
            d1_r |= {"holdout_error": f"{type(e).__name__}: {e}"}
        _save_json(RESULTS / "D1.json", d1_r)
    h = xs_r["headline"]
    print(f"XS holdout: {evh['summary']} gates={hg} | REPLICATED={h['REPLICATED']} STRICT={h['STRICT']} "
          f"(DSR {h['dsr']:.3f} at N {h['n']:.2f})")
    _log_run("r5-validate", headline=h)
