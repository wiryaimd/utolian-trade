"""Round 4 point-in-time universe of Binance USD-M crypto perps.

1. catalogue(): every archived symbol, with exclusion and meme flags from a frozen exchangeInfo
   snapshot plus hand-written overrides (for delisted symbols the snapshot no longer lists).
2. download_daily_qv(): daily quote volume from the 1d kline archive. Prices are never stored.
3. build_universe(): for each UTC month m, rank by data strictly before m and keep the top N.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from btcq.config import DATA_DIR, RESEARCH_DIR, SYMBOLS, TF_MS
from btcq.data import binance_vision as bv

ROUND4_DIR = RESEARCH_DIR / "round4"
QV_PATH = DATA_DIR / "round4" / "daily_qv.parquet"
DAY_MS = TF_MS["1d"]

# Frozen lists (round 4 pre-registration). Bases are compared after normalize_base().
STABLE_BASES = {"USDC", "BUSD", "TUSD", "USDP", "FDUSD", "DAI", "USDE", "USD1", "RLUSD", "PYUSD",
                "USDS", "FRAX", "LUSD", "GUSD", "EUR", "EURI", "AEUR", "XUSD", "USTC", "UST"}
INDEX_SYMBOLS = {"BTCDOMUSDT", "DEFIUSDT", "FOOTBALLUSDT", "BLUEBIRDUSDT"}
GOLD_BASES = {"PAXG", "XAUT"}
_PREFIX = re.compile(r"^(1000000|100000|10000|1000|1M)(?=[A-Z])")


def normalize_base(symbol: str) -> str:
    """1000SHIBUSDT -> SHIB, 1MBABYDOGEUSDT -> BABYDOGE, 1INCHUSDT -> 1INCH."""
    base = symbol[:-4] if symbol.endswith("USDT") else symbol
    return _PREFIX.sub("", base)


def _ex_map(exinfo: dict) -> dict[str, dict]:
    return {s["symbol"]: s for s in exinfo.get("symbols", [])}


def catalogue(symbols: list[str], exinfo: dict, overrides: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per archived symbol. A symbol is kept only if it is a USDT perpetual on a crypto
    underlying; the snapshot classifies, it never ranks."""
    ex = _ex_map(exinfo)
    ov = overrides.set_index("symbol") if overrides is not None and len(overrides) else None
    rows = []
    for sym in sorted(set(symbols)):
        e = ex.get(sym, {})
        base = normalize_base(sym)
        sub = e.get("underlyingSubType") or []
        reason = ""
        if not sym.endswith("USDT") or "_" in sym:
            reason = "not a USDT perpetual"
        elif base in STABLE_BASES:
            reason = "stablecoin"
        elif sym in INDEX_SYMBOLS or e.get("underlyingType") == "INDEX":
            reason = "index"
        elif e.get("contractType") == "TRADIFI_PERPETUAL" or e.get("underlyingType") not in (None, "COIN"):
            reason = f"non-crypto ({e.get('underlyingType') or e.get('contractType')})"
        elif e and e.get("contractType") != "PERPETUAL":
            reason = f"not perpetual ({e.get('contractType')})"
        elif base in GOLD_BASES:
            reason = "tokenized gold"
        is_meme, meme_src = "Meme" in sub, "exchangeInfo" if "Meme" in sub else ""
        if ov is not None and sym in ov.index:
            o = ov.loc[sym]
            if str(o.get("exclude", "")).strip() == "1" and not reason:
                reason = f"override: {o.get('reason', '')}"
            if str(o.get("is_meme", "")).strip() in ("0", "1"):
                is_meme, meme_src = str(o["is_meme"]).strip() == "1", "override"
        rows.append({"symbol": sym, "base_norm": base, "in_exchange_info": bool(e),
                     "status": e.get("status", ""), "contract_type": e.get("contractType", ""),
                     "underlying_type": e.get("underlyingType", ""), "underlying_sub_type": ",".join(sub),
                     "onboard_date": e.get("onboardDate", ""), "delivery_date": e.get("deliveryDate", ""),
                     "excluded": bool(reason), "exclude_reason": reason, "is_meme": bool(is_meme),
                     "meme_source": meme_src, "is_major": sym in SYMBOLS})
    return pd.DataFrame(rows)


def download_daily_qv(symbols: list[str], concurrency: int = 16) -> pd.DataFrame:
    """Daily quote volume (and trade count) per symbol from the 1d kline archive."""
    prefix = bv.DATASETS["perp_1d_monthly"]
    with ThreadPoolExecutor(concurrency) as ex:
        listed = dict(zip(symbols, ex.map(lambda s: bv.list_keys(prefix.format(sym=s)), symbols)))
    keys = [k for ks in listed.values() for k in ks]
    bv.download_keys(keys, concurrency)
    frames = []
    for sym, ks in listed.items():
        if ks:
            k = bv.parse_klines([bv._local_path(x) for x in ks])
            frames.append(pd.DataFrame({"symbol": sym, "ts": k["ts"], "quote_volume": k["quote_volume"],
                                        "count": k["count"]}))
    qv = pd.concat(frames, ignore_index=True)
    QV_PATH.parent.mkdir(parents=True, exist_ok=True)
    qv.to_parquet(QV_PATH)
    return qv


def month_starts(first: str = "2020-04", last: str = "2026-09") -> list[pd.Timestamp]:
    return list(pd.date_range(f"{first}-01", f"{last}-01", freq="MS", tz="UTC"))


def build_universe(qv: pd.DataFrame, cat: pd.DataFrame, months: list[pd.Timestamp], top_n: int = 20,
                   lookback: int = 90, min_age: int = 90, recent: int = 7) -> pd.DataFrame:
    """Top-N by median daily quote volume over the `lookback` days before each month (missing days
    count as 0). Eligible: first archived day at least `min_age` days before the month, volume on
    each of the last `recent` days, not excluded. Same normalized base: the higher volume wins.
    Uses only rows with ts < month start, so later data cannot change a month's members."""
    keep = cat[~cat["excluded"]].set_index("symbol")
    qv = qv[qv["symbol"].isin(keep.index)]
    out = []
    for m in months:
        ms = int(m.value // 10**6)
        hist = qv[qv["ts"] < ms]
        first = hist.groupby("symbol")["ts"].min()
        old = first.index[first <= ms - min_age * DAY_MS]
        win = hist[(hist["ts"] >= ms - lookback * DAY_MS) & hist["symbol"].isin(old)]
        rec = win[(win["ts"] >= ms - recent * DAY_MS) & (win["quote_volume"] > 0)]
        ok = rec.groupby("symbol")["ts"].nunique()
        ok = ok.index[ok >= recent]
        win = win[win["symbol"].isin(ok)]
        med = win.groupby("symbol")["quote_volume"].apply(
            lambda v: float(np.median(np.r_[v.to_numpy(), np.zeros(max(lookback - len(v), 0))])))
        df = pd.DataFrame({"symbol": med.index, "median_qv": med.to_numpy()})
        df["base_norm"] = keep.loc[df["symbol"], "base_norm"].to_numpy()
        df = df.sort_values(["median_qv", "symbol"], ascending=[False, True])
        df = df.drop_duplicates("base_norm", keep="first")
        n_elig = len(df)
        df = df.head(top_n).reset_index(drop=True)
        df.insert(0, "rank", np.arange(1, len(df) + 1))
        df.insert(0, "month", m.strftime("%Y-%m"))
        df["tier"] = np.where(df["rank"] <= 10, 1, 2)
        df["is_meme"] = keep.loc[df["symbol"], "is_meme"].to_numpy()
        df["is_major"] = keep.loc[df["symbol"], "is_major"].to_numpy()
        df["gated"] = ~df["is_major"]
        df["n_eligible"] = n_elig
        out.append(df)
    cols = ["month", "rank", "symbol", "base_norm", "median_qv", "tier", "is_meme", "is_major", "gated",
            "n_eligible"]
    return pd.concat(out, ignore_index=True)[cols]


def write_csv(df: pd.DataFrame, path) -> None:
    """LF line endings so file hashes are the same on every OS."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")


def load_universe(path=None) -> pd.DataFrame:
    return pd.read_csv(path or ROUND4_DIR / "universe.csv")


def load_exchange_info(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_overrides(path) -> pd.DataFrame:
    """symbol,exclude,is_meme,reason — exclude/is_meme are "1", "0" or blank (no override)."""
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "exclude", "is_meme", "reason"])
    return pd.read_csv(path, dtype=str, keep_default_na=False)
