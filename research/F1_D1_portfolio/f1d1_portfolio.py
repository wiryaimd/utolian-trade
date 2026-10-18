"""F1 + D1 as one portfolio, 2020-01-01 .. 2026-09-30, from the saved and verified daily results (no new data loaded).

F1: 5-coin portfolio (mean of the 5 equal-capital sleeves), dev daily + 2024-26 per-coin daily.
D1: 45 sub-sleeves, dev daily + 2024-26 (Deviation 2) daily.
Writes CSVs to research/F1_D1_portfolio/.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from btcq.backtest import stats

ROOT = Path("F:/trade2")
R = ROOT / "reports/results"
OUT = ROOT / "research/F1_D1_portfolio"
OUT.mkdir(parents=True, exist_ok=True)
END = pd.Timestamp("2026-09-30", tz="UTC")
DAY = pd.Timedelta(days=1)

f1 = pd.concat([pd.read_parquet(R / "F1_dev_daily.parquet")["ret"],
                pd.read_parquet(R / "F1_holdout_sym_daily.parquet").mean(axis=1)]).loc[:END]
d1 = pd.concat([pd.read_parquet(R / "D1_dev_daily.parquet")["ret"],
                pd.read_parquet(R / "D1_holdout_daily.parquet")["ret"]]).loc[:END]
cal = pd.date_range(pd.Timestamp("2020-01-01", tz="UTC"), END, freq="D").as_unit("ms")
assert f1.index.equals(cal) and d1.index.equals(cal), "calendars differ"
DEV_END = pd.Timestamp("2023-12-31", tz="UTC")


def drawdown(eq: pd.Series):
    peak = eq.cummax()
    dd = 1 - eq / peak
    t = dd.idxmax()
    p = eq.loc[:t].idxmax()
    rec = eq.loc[t:][eq.loc[t:] >= eq.loc[p]]
    under = (dd > 0).astype(int)
    longest = int(under.groupby((under == 0).cumsum()).sum().max())
    return float(dd.max()), p, t, (rec.index[0] if len(rec) else None), longest


def metrics(eq: pd.Series) -> dict:
    r = eq.pct_change().fillna(eq.iloc[0] - 1.0)
    yrs = len(r) / 365
    mdd, p, t, rec, longest = drawdown(eq)
    yearly = (1 + r).groupby(r.index.year).prod() - 1
    monthly = (1 + r).groupby([r.index.year, r.index.month]).prod() - 1
    return {"total": float(eq.iloc[-1] - 1), "cagr": float(eq.iloc[-1] ** (1 / yrs) - 1),
            "sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(365)), "vol": float(r.std(ddof=1) * np.sqrt(365)),
            "max_dd": mdd, "dd_peak": str(p.date()), "dd_trough": str(t.date()),
            "dd_recovered": str(rec.date()) if rec is not None else "not by 2026-09-30", "longest_under_water_days": longest,
            "worst_year": float(yearly.min()), "worst_year_is": int(yearly.idxmin()),
            "pos_months": float((monthly > 0).mean()), **{f"y{y}": float(v) for y, v in yearly.items()}}


def portfolio(w: float, r1: pd.Series, r2: pd.Series, how: str) -> pd.Series:
    """Equity (start 1.0) with w in F1 and 1-w in D1. how: 'monthly' rebalance to target, or 'never' (two pots)."""
    if how == "never":
        return w * (1 + r1).cumprod() + (1 - w) * (1 + r2).cumprod()
    eq, out = 1.0, []
    a, b = w, 1 - w
    month = None
    for t, x, y in zip(r1.index, r1.to_numpy(), r2.to_numpy()):
        if (t.year, t.month) != month:              # rebalance at the start of each month
            month, a, b = (t.year, t.month), w * eq, (1 - w) * eq
        a, b = a * (1 + x), b * (1 + y)
        eq = a + b
        out.append(eq)
    return pd.Series(out, index=r1.index)


rows = []
splits = [1.0, 0.75, 0.6, 0.5, 0.4, 0.25, 0.0]
vol_f1, vol_d1 = f1.loc[:DEV_END].std(), d1.loc[:DEV_END].std()
w_iv = float((1 / vol_f1) / (1 / vol_f1 + 1 / vol_d1))         # equal-risk split chosen from 2020-23 only
for how in ("monthly", "never"):
    for w in splits + [round(w_iv, 3)]:
        for period, sl in (("2020-2026", slice(None)), ("2020-2023", slice(None, DEV_END)),
                           ("2024-2026", slice(DEV_END + DAY, None))):
            eq = portfolio(w, f1.loc[sl], d1.loc[sl], how)
            rows.append({"rebalance": how, "w_F1": w, "w_D1": round(1 - w, 3), "period": period,
                         "equal_risk_from_dev": w == round(w_iv, 3), **metrics(eq)})
tab = pd.DataFrame(rows)
tab.to_csv(OUT / "splits.csv", index=False)

# correlation
wk = lambda s: (1 + s).resample("W-SUN").prod() - 1  # noqa: E731
corr = {p: {"daily": float(f1.loc[sl].corr(d1.loc[sl])), "weekly": float(wk(f1.loc[sl]).corr(wk(d1.loc[sl])))}
        for p, sl in (("2020-2026", slice(None)), ("2020-2023", slice(None, DEV_END)), ("2024-2026", slice(DEV_END + DAY, None)))}

# order days (fills); forced 'end' exits and D1's state-synchronous re-entries at the 2024 window start are artifacts
def ts(x):
    x = pd.to_datetime(x, utc=True) if not np.issubdtype(np.asarray(x).dtype, np.integer) else pd.to_datetime(x, unit="ms", utc=True)
    return x


orders = []
for code, files in (("F1", ["F1_dev_trades", "F1_holdout_parts"]), ("D1", ["D1_dev_trades", "D1_holdout_trades"])):
    for f in files:
        t = pd.read_parquet(R / f"{f}.parquet")
        e, x = ts(t["entry_time"]), ts(t["exit_time"])
        start_art = (e == pd.Timestamp("2024-01-01 00:01", tz="UTC")) if code == "D1" and "holdout" in f else pd.Series(False, index=t.index)
        for when, sym in zip(e[~start_art.to_numpy()], t["symbol"][~start_art.to_numpy()]):
            orders.append((code, sym, when))
        keep = (t["reason"] != "end").to_numpy()
        for when, sym in zip(x[keep], t["symbol"][keep]):
            orders.append((code, sym, when))
od = pd.DataFrame(orders, columns=["book", "symbol", "ts"])
od = od[od["ts"] <= END + DAY]
od["day"] = od["ts"].dt.floor("D")
days = pd.Series(1, index=cal)
act = {}
for name, sel in (("F1", od["book"] == "F1"), ("D1", od["book"] == "D1"), ("F1+D1", od["book"].notna())):
    dset = set(od.loc[sel, "day"])
    on = days.index.isin(list(dset))
    gaps = pd.Series((~on).astype(int), index=cal)
    longest_gap = int(gaps.groupby((gaps == 0).cumsum()).sum().max())
    per_year = pd.Series(on, index=cal).groupby(cal.year).mean()
    netted = od.loc[sel].drop_duplicates(["book", "symbol", "day"]) if name != "F1+D1" else od.loc[sel].drop_duplicates(["symbol", "day"])
    act[name] = {"share_days_with_order": float(on.mean()), "fills_per_week": float(sel.sum() / (len(cal) / 7)),
                 "netted_orders_per_week": float(len(netted) / (len(cal) / 7)), "longest_days_without_order": longest_gap,
                 **{f"y{y}": float(v) for y, v in per_year.items()}}
pd.DataFrame(act).T.to_csv(OUT / "order_days.csv")

# practical rulebook re-read (information only)
def dsr_family(daily: pd.Series, n: int) -> float:
    sr, T, sk, ku = stats.sharpe_moments(daily)
    return stats.psr(sr, T, sk, ku, stats.expected_max_sr(stats.null_sr_variance(T), n))


f1j, d1j = json.loads((R / "F1.json").read_text()), json.loads((R / "D1.json").read_text())
reread = {
    "F1": {"dsr_family(N=30, S1 grid)": dsr_family(f1.loc[:DEV_END], 30),
           "best_year_share": float(stats.year_share_daily(f1.loc[:DEV_END])),
           "2x_cost": f1j["gates"]["2x-cost avg_R>0"], "null p<0.05": f1j["gates"]["null p<0.05"],
           "trades": f1j["dev"]["trades"], "markets_positive": f1j.get("symbols_positive"),
           "clean_oos": "holdout passed all 5 gates (Deviation 1, clean)", "holdout_gates": f1j["holdout_gates"]},
    "D1": {"dsr_family(N=3)": dsr_family(d1.loc[:DEV_END], 3),
           "best_year_share": float(d1j["dev_detail"]["year_share"]), "2x_cost": d1j["gates"]["2x-cost mean>0"],
           "null p<0.05": d1j["gates"]["shift null p<0.05"], "trades": d1j["dev_detail"]["summary"]["trades"],
           "PBO": d1j["dev_detail"]["pbo"], "plateau": d1j["dev_detail"]["plateau"],
           "coins_positive": d1j["dev_detail"]["coins_positive"],
           "clean_oos": "none: its 2024-26 holdout is contaminated; needs a forward paper period"},
}
json.dump({"corr": corr, "w_equal_risk_from_dev": w_iv, "vol_dev": {"F1": float(vol_f1 * np.sqrt(365)),
           "D1": float(vol_d1 * np.sqrt(365))}, "activity": act, "practical_reread": reread},
          open(OUT / "summary.json", "w"), indent=2, default=str)

pd.set_option("display.width", 250)
cols = ["w_F1", "w_D1", "total", "cagr", "sharpe", "vol", "max_dd", "dd_trough", "worst_year", "worst_year_is", "pos_months"]
for how in ("monthly", "never"):
    for period in ("2020-2026", "2020-2023", "2024-2026"):
        print(f"\n== {how} rebalance, {period}")
        print(tab[(tab.rebalance == how) & (tab.period == period)][cols].round(3).to_string(index=False))
print("\nyearly (monthly rebalance, 2020-2026):")
print(tab[(tab.rebalance == "monthly") & (tab.period == "2020-2026")][["w_F1"] + [c for c in tab.columns if c.startswith("y20")]].round(3).to_string(index=False))
print("\ncorr", json.dumps(corr), "| equal-risk w_F1 from dev:", round(w_iv, 3), "| dev vol F1/D1:", round(vol_f1 * np.sqrt(365), 3), round(vol_d1 * np.sqrt(365), 3))
print("\nactivity:", json.dumps(act, indent=1))
print("\npractical re-read:", json.dumps(reread, indent=1, default=str))
