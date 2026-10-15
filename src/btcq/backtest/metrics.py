"""Performance summary of a backtest Result."""
from __future__ import annotations

import numpy as np
import pandas as pd

from btcq.backtest.engine import Result


def daily_returns(res: Result) -> pd.Series:
    eq = res.equity
    prev = eq.shift(1)
    prev.iloc[0] = res.initial_equity
    return eq / prev - 1.0


def max_drawdown(eq: np.ndarray) -> float:
    eq = np.asarray(eq, dtype=np.float64)
    if eq.size == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    return float(np.max(1.0 - eq / peak))


def _dd_duration_days(eq: pd.Series) -> int:
    peak = eq.cummax()
    under = (eq < peak).to_numpy()
    best = run = 0
    for u in under:
        run = run + 1 if u else 0
        best = max(best, run)
    return best


def summarize(res: Result) -> dict:
    t = res.trades
    eq = res.equity
    n = len(t)
    out: dict = {"trades": n}
    if n == 0 or eq.empty:
        return out | {"sharpe": 0.0, "avg_R": 0.0, "t_stat": 0.0, "pf": 0.0, "total_return": 0.0,
                      "max_dd": 0.0, "cagr": 0.0}
    R = t["R"].to_numpy()
    rets = daily_returns(res).to_numpy()
    years = max(len(eq) / 365.0, 1e-9)
    eq_all = np.r_[res.initial_equity, eq.to_numpy()]
    final = eq_all[-1] / res.initial_equity
    sd = rets.std(ddof=1)
    downside = rets[rets < 0]
    mdd = max_drawdown(eq_all)
    cagr = final ** (1 / years) - 1 if final > 0 else -1.0
    gains, losses = R[R > 0].sum(), -R[R < 0].sum()
    by_year = t.groupby(t["entry_time"].dt.year)["R"].agg(["count", "sum", "mean"])
    tot_R = R.sum()
    minutes = (t["exit_i"] - t["entry_i"]).sum()
    span = (eq.index[-1] - res.start).total_seconds() / 60 + 1440
    longs, shorts = t[t["dir"] == 1], t[t["dir"] == -1]
    out |= {
        "win_rate": float((R > 0).mean()),
        "avg_R": float(R.mean()),
        "median_R": float(np.median(R)),
        "t_stat": float(R.mean() / R.std(ddof=1) * np.sqrt(n)) if n > 1 and R.std() > 0 else 0.0,
        "pf": float(gains / losses) if losses > 0 else float("inf"),
        "total_R": float(tot_R),
        "total_return": float(final - 1),
        "cagr": float(cagr),
        "sharpe": float(rets.mean() / sd * np.sqrt(365)) if sd > 0 else 0.0,
        "sortino": float(rets.mean() / downside.std(ddof=1) * np.sqrt(365)) if downside.size > 1 else 0.0,
        "max_dd": mdd,
        "calmar": float(cagr / mdd) if mdd > 0 else 0.0,
        "dd_days": _dd_duration_days(eq),
        "exposure": float(minutes / span) if span > 0 else 0.0,
        "avg_hold_h": float((t["exit_i"] - t["entry_i"]).mean() / 60),
        "trades_per_year": n / years,
        "long_n": len(longs), "long_avg_R": float(longs["R"].mean()) if len(longs) else 0.0,
        "short_n": len(shorts), "short_avg_R": float(shorts["R"].mean()) if len(shorts) else 0.0,
        "fees_R": float((t["fees"] / t["init_risk"]).mean()),
        "funding_R": float((t["funding"] / t["init_risk"]).mean()),
        "max_year_share": float(by_year["sum"].max() / tot_R) if tot_R > 0 else float("inf"),
        "years_positive": int((by_year["sum"] > 0).sum()), "years": int(len(by_year)),
    }
    return out


def year_table(res: Result) -> pd.DataFrame:
    t = res.trades
    eq = pd.concat([pd.Series([res.initial_equity], index=[res.start - pd.Timedelta(days=1)]),
                    res.equity])
    ye = eq.groupby(eq.index.year).last()
    yr_ret = ye / ye.shift(1) - 1
    yr_ret.iloc[0] = ye.iloc[0] / res.initial_equity - 1
    g = t.groupby(t["entry_time"].dt.year)
    tab = pd.DataFrame({"trades": g.size(), "win_rate": g["R"].apply(lambda r: (r > 0).mean()),
                        "avg_R": g["R"].mean(), "sum_R": g["R"].sum()})
    tab["return"] = yr_ret.reindex(tab.index)
    return tab


def monthly_returns(res: Result) -> pd.DataFrame:
    eq = pd.concat([pd.Series([res.initial_equity], index=[res.start - pd.Timedelta(days=1)]),
                    res.equity])
    me = eq.resample("ME").last()
    r = (me / me.shift(1) - 1).iloc[1:]
    return pd.DataFrame({"year": r.index.year, "month": r.index.month, "ret": r.to_numpy()}) \
        .pivot(index="year", columns="month", values="ret")
