"""Builds the local parquet store and serves bars with the holdout lock enforced.

Layout: data/parquet/<SYMBOL>/<dataset>.parquet. BTCUSDT additionally keeps spot 1m
history from before the perp archive begins.
"""
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from btcq.config import (HOLDOUT_START, MINUTE_MS, OI_LAG_MS, PARQUET_DIR, PERP_START, RAW_DIR,
                         RESEARCH_DIR, SPOT_START, SYMBOL, TF_MS)
from btcq.data import binance_rest, binance_vision as bv

_HOLDOUT_UNLOCKED = False
HOLDOUT_LOG = RESEARCH_DIR / "holdout_log.jsonl"


def unlock_holdout(reason: str) -> None:
    """Allow loading data on/after HOLDOUT_START. Every unlock is appended to the log."""
    global _HOLDOUT_UNLOCKED
    HOLDOUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with HOLDOUT_LOG.open("a") as f:
        f.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "reason": reason}) + "\n")
    _HOLDOUT_UNLOCKED = True


def _period_filter(df: pd.DataFrame, period: str) -> pd.DataFrame:
    cut = int(HOLDOUT_START.value // 10**6)
    if period == "dev":
        return df[df["ts"] < cut].reset_index(drop=True)
    if period in ("full", "holdout"):
        if not _HOLDOUT_UNLOCKED:
            raise PermissionError("Holdout is locked. Call unlock_holdout(reason) first.")
        return df if period == "full" else df[df["ts"] >= cut].reset_index(drop=True)
    raise ValueError(period)


def symbol_dir(symbol: str = SYMBOL):
    return PARQUET_DIR / symbol


def _pq(name: str, symbol: str = SYMBOL) -> str:
    return str(symbol_dir(symbol) / f"{name}.parquet")


# ---------------------------------------------------------------- build

def build(symbol: str = SYMBOL, skip_metrics: bool = False) -> dict:
    """Download every archive for one symbol, verify checksums, write parquet. Idempotent."""
    out_dir = symbol_dir(symbol)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"symbol": symbol}

    def dl(name, **kw):
        return bv.download(name, symbol=symbol, **kw)

    if symbol == SYMBOL:
        perp_cut = PERP_START.strftime("%Y-%m")
        spot = bv.parse_klines(dl("spot_1m_monthly", keep=lambda k: bv.key_period(k) < perp_cut))
        spot.to_parquet(_pq("spot_1m", symbol))
        report["spot_1m_rows"] = len(spot)

    monthly = dl("perp_1m_monthly")
    last_month = max(bv.key_period(str(p).replace("\\", "/")) for p in monthly)
    daily = dl("perp_1m_daily", keep=lambda k: bv.key_period(k)[:7] > last_month)
    perp = _patch_holes(bv.parse_klines(monthly + daily), dl, report)
    tail =_try_rest(lambda: binance_rest.klines("1m", int(perp["ts"].iloc[-1]) + MINUTE_MS), report) \
        if symbol == SYMBOL else None
    if tail is not None:
        perp = pd.concat([perp, tail]).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    perp.to_parquet(_pq("perp_1m", symbol))
    report["perp_1m_rows"] = len(perp)

    for tf in ("15m", "4h") if symbol == SYMBOL else ("4h",):
        bv.parse_klines(dl(f"perp_{tf}_monthly")).to_parquet(_pq(f"perp_{tf}_binance", symbol))

    fund = bv.parse_funding(dl("funding_monthly"))
    if symbol == SYMBOL:
        early = _try_rest(lambda: binance_rest.funding_history(
            int(pd.Timestamp("2019-09-01", tz="UTC").value // 10**6), int(fund["ts"].iloc[0])), report)
        late = _try_rest(lambda: binance_rest.funding_history(int(fund["ts"].iloc[-1]) + 1), report)
        fund = pd.concat([early, fund, late]).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    fund.to_parquet(_pq("funding", symbol))
    report["funding_rows"] = len(fund)

    pm = dl("premium_1h_monthly")
    last_pm = max(bv.key_period(str(p).replace("\\", "/")) for p in pm)
    pdaily = dl("premium_1h_daily", keep=lambda k: bv.key_period(k)[:7] > last_pm)
    prem = bv.parse_klines(pm + pdaily)[["ts", "open", "high", "low", "close"]]
    prem.to_parquet(_pq("premium_1h", symbol))
    report["premium_rows"] = len(prem)

    if not skip_metrics:
        met = bv.parse_metrics(dl("metrics_daily", concurrency=16))
        met.to_parquet(_pq("metrics", symbol))
        report["metrics_rows"] = len(met)

    gaps = gap_report(load_1m_raw(symbol))
    gaps.to_csv(out_dir / "gaps_1m.csv", index=False)
    report["gaps_over_1m"] = len(gaps)
    report["missing_minutes"] = int(gaps["missing_minutes"].sum()) if len(gaps) else 0
    return report


def _patch_holes(perp: pd.DataFrame, dl, report: dict) -> pd.DataFrame:
    """Some monthly archives are truncated (e.g. SOL/XRP 2022-02 and 2022-03); patch the missing
    days from the daily archive."""
    holes = gap_report(perp)
    if len(holes):
        days = set()
        for _, g in holes.iterrows():
            for d in pd.date_range(g["after"].normalize(), g["before"].normalize(), freq="D"):
                days.add(d.strftime("%Y-%m-%d"))
        patch = dl("perp_1m_daily", keep=lambda k: bv.key_period(k) in days)
        if patch:
            perp = pd.concat([perp, bv.parse_klines(patch)]).drop_duplicates("ts") \
                .sort_values("ts").reset_index(drop=True)
        report["patched_days"] = sorted(days)
    return perp


def build_perp_only(symbol: str, keep_raw: bool = False) -> dict:
    """Perp 1m + 4h + funding for any USD-M symbol, tolerating datasets the archive lacks
    (new listings, delisted contracts). Writes build_report.json with data-quality checks.
    Unless keep_raw, the raw kline zips are deleted after the parquet is read back and verified."""
    out_dir = symbol_dir(symbol)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"symbol": symbol, "status": "ok"}

    def dl(name, **kw):
        return bv.download(name, symbol=symbol, **kw)

    monthly = dl("perp_1m_monthly")
    last_month = max((bv.key_period(str(p).replace("\\", "/")) for p in monthly), default="")
    daily = dl("perp_1m_daily", keep=lambda k: bv.key_period(k)[:7] > last_month)
    if not monthly and not daily:
        report["status"] = "no_1m"
        return report
    perp = _patch_holes(bv.parse_klines(monthly + daily), dl, report)
    perp.to_parquet(_pq("perp_1m", symbol))

    k4 = dl("perp_4h_monthly")
    bars4 = bv.parse_klines(k4) if k4 else None
    if bars4 is not None:
        bars4.to_parquet(_pq("perp_4h_binance", symbol))
    fk = dl("funding_monthly")
    fund = bv.parse_funding(fk) if fk else pd.DataFrame({"ts": pd.Series([], dtype="int64"),
                                                          "rate": pd.Series([], dtype="float64")})
    fund.to_parquet(_pq("funding", symbol))
    gaps = gap_report(perp)
    gaps.to_csv(out_dir / "gaps_1m.csv", index=False)

    report.update(_quality(perp, bars4, fund))
    back = pd.read_parquet(_pq("perp_1m", symbol))
    verified = (len(back) == len(perp) and int(back["ts"].iloc[0]) == int(perp["ts"].iloc[0])
                and int(back["ts"].iloc[-1]) == int(perp["ts"].iloc[-1]))
    report["parquet_verified"] = bool(verified)
    if not verified:
        report["status"] = "read_back_failed"
    elif not keep_raw:
        for sub in ("monthly/klines", "daily/klines"):
            _rmtree(RAW_DIR / "futures" / "um" / sub / symbol)
        report["raw_deleted"] = True
    (out_dir / "build_report.json").write_text(json.dumps(report, indent=1, default=str))
    return report


def _rmtree(path) -> None:
    """Windows can briefly hold file handles (antivirus, indexer): retry before giving up."""
    for attempt in range(5):
        if not path.exists():
            return
        try:
            shutil.rmtree(path)
            return
        except OSError:
            time.sleep(1 + attempt)
    shutil.rmtree(path)


def _quality(perp: pd.DataFrame, bars4: pd.DataFrame | None, fund: pd.DataFrame) -> dict:
    """Reported only; nothing is cleaned here."""
    ts = perp["ts"].to_numpy(np.int64)
    o, h, l, c = (perp[k].to_numpy() for k in ("open", "high", "low", "close"))
    vol = perp["volume"].to_numpy()
    d = np.diff(ts)
    live = np.flatnonzero(vol > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        extreme = (h / l - 1 > 0.5) | (np.abs(c / o - 1) > 0.3)
    q = {"rows": len(perp), "first": _iso(ts[0]), "last": _iso(ts[-1]),
         "last_traded": _iso(ts[live[-1]]) if len(live) else None,
         "zero_volume_tail_min": int(len(ts) - 1 - live[-1]) if len(live) else len(ts),
         "monotonic_minutes": bool((d > 0).all() and (ts % MINUTE_MS == 0).all()),
         "nonpositive_prices": int(((o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)).sum()),
         "ohlc_violations": int(((h < np.maximum(o, c)) | (l > np.minimum(o, c))).sum()),
         "gaps_over_24h": int((d > TF_MS["1d"]).sum()),
         "max_gap_h": round(float(d.max()) / 3_600_000, 2) if len(d) else 0.0,
         "extreme_bars": int(extreme.sum()),
         "funding_rows": len(fund)}
    if len(fund):
        f = fund["ts"].to_numpy(np.int64)
        f = f[(f >= ts[0]) & (f <= ts[-1])]
        fd = np.diff(np.r_[ts[0], f, ts[-1]])
        q["funding_first"] = _iso(f[0]) if len(f) else None
        q["funding_last"] = _iso(f[-1]) if len(f) else None
        q["funding_gap_hours_over_9h"] = round(float(fd[fd > 9 * 3_600_000].sum()) / 3_600_000, 1)
    if bars4 is not None and len(bars4):
        mine = resample(perp, "4h").set_index("ts")["close"]
        theirs = bars4.set_index("ts")["close"]
        both = mine.index.intersection(theirs.index)
        diff = (mine[both] - theirs[both]).abs() > 1e-9 * theirs[both].abs()
        q["bars_4h_mismatch_share"] = round(float(diff.mean()), 6) if len(both) else None
    return q


def _iso(ms) -> str:
    return pd.Timestamp(int(ms), unit="ms", tz="UTC").isoformat()


def trading_span(m1: pd.DataFrame) -> pd.DataFrame:
    """Drop the trailing stale minutes after the last bar with volume (e.g. a halted, delisted
    contract that keeps printing flat zero-volume bars)."""
    live = np.flatnonzero(m1["volume"].to_numpy() > 0)
    if not len(live):
        return m1.iloc[:0]
    return m1.iloc[: live[-1] + 1].reset_index(drop=True)


def fill_funding_gaps(fund: pd.DataFrame, first_ts: int, last_ts: int, rate: float,
                      max_gap_ms: int = 9 * 3_600_000) -> tuple[pd.DataFrame, float]:
    """Trim funding to [first_ts, last_ts] and fill every stretch longer than max_gap_ms without
    a settlement (head, interior and tail) with `rate` at the 00/08/16 UTC settlements inside
    it, at least 4h from its edges. Returns the frame and the hours of the filled stretches."""
    f = fund[(fund["ts"] >= first_ts) & (fund["ts"] <= last_ts)][["ts", "rate"]]
    edges = np.r_[first_ts, f["ts"].to_numpy(np.int64), last_ts]
    step = 8 * 3_600_000
    fill, hours = [], 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a > max_gap_ms:
            t = np.arange((a // step + 1) * step, b, step)
            # keep 4h clear of the gap's edges so a fill never doubles a real settlement
            fill.append(t[(t - a > step // 2) & (b - t > step // 2)])
            hours += (b - a) / 3_600_000
    if fill:
        syn = pd.DataFrame({"ts": np.concatenate(fill).astype(np.int64), "rate": rate})
        f = pd.concat([f, syn]).sort_values("ts")
    return f.reset_index(drop=True), round(hours, 1)


def _try_rest(fn, report: dict):
    """REST gap-fill is best effort: some networks block api hosts but not the archive."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        report.setdefault("rest_skipped", []).append(type(e).__name__)
        return None


def gap_report(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["ts"].to_numpy()
    d = np.diff(ts)
    idx = np.flatnonzero(d > MINUTE_MS)
    return pd.DataFrame({
        "after": pd.to_datetime(ts[idx], unit="ms", utc=True),
        "before": pd.to_datetime(ts[idx + 1], unit="ms", utc=True),
        "missing_minutes": (d[idx] // MINUTE_MS - 1).astype(int),
    })


# ---------------------------------------------------------------- load

def load_1m_raw(symbol: str = SYMBOL) -> pd.DataFrame:
    """Perp 1m; for BTCUSDT, spot history before PERP_START is stitched in front."""
    perp = pd.read_parquet(_pq("perp_1m", symbol))
    if symbol != SYMBOL:
        df = perp.assign(is_perp=True)
    else:
        cut = int(PERP_START.value // 10**6)
        spot = pd.read_parquet(_pq("spot_1m", symbol))
        spot = spot[(spot["ts"] >= SPOT_START.value // 10**6) & (spot["ts"] < cut)]
        perp = perp[perp["ts"] >= cut]
        df = pd.concat([spot.assign(is_perp=False), perp.assign(is_perp=True)], ignore_index=True)
    # A few 2017-2018 spot rows carry off-minute timestamps around maintenance; snap them.
    df["ts"] = df["ts"] // MINUTE_MS * MINUTE_MS
    return df.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)


def load_1m(period: str = "dev", symbol: str = SYMBOL) -> pd.DataFrame:
    return _period_filter(load_1m_raw(symbol), period)


def load_funding(period: str = "dev", symbol: str = SYMBOL) -> pd.DataFrame:
    return _period_filter(pd.read_parquet(_pq("funding", symbol)), period)


def load_metrics(period: str = "dev", symbol: str = SYMBOL) -> pd.DataFrame:
    """OI/positioning snapshots; "ts" is the availability time (snapshot + OI_LAG_MS)."""
    p = symbol_dir(symbol) / "metrics.parquet"
    if not p.exists():
        return pd.DataFrame({"ts": []})
    m = pd.read_parquet(p)
    m["ts"] = m["ts"] + OI_LAG_MS
    return _period_filter(m, period)


def load_premium(period: str = "dev", symbol: str = SYMBOL) -> pd.DataFrame:
    p = symbol_dir(symbol) / "premium_1h.parquet"
    if not p.exists():
        return pd.DataFrame({"ts": [], "close": []})
    return _period_filter(pd.read_parquet(p), period)


def load_binance_klines(tf: str, symbol: str = SYMBOL) -> pd.DataFrame:
    return pd.read_parquet(_pq(f"perp_{tf}_binance", symbol))


def resample(df1m: pd.DataFrame, tf: str) -> pd.DataFrame:
    """UTC-aligned OHLCV bars labelled by open time. Drops a trailing incomplete bar."""
    ms = TF_MS[tf]
    key = df1m["ts"] // ms * ms
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
           "quote_volume": "sum", "taker_buy_volume": "sum"}
    if "is_perp" in df1m:
        agg["is_perp"] = "max"
    bars = df1m.groupby(key, sort=True).agg(agg)
    bars["n_min"] = df1m.groupby(key, sort=True).size()
    bars.index.name = "ts"
    bars = bars.reset_index()
    last_close = int(df1m["ts"].iloc[-1]) + MINUTE_MS
    return bars[bars["ts"] + ms <= last_close].reset_index(drop=True)
