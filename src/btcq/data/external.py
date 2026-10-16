"""Round 6 external daily data (Yahoo chart API, EIA spot spreadsheets) for the macro trend sleeve.

Raw downloads are kept under data/external/raw/<source>/, parsed closes under data/external/<source>/,
and a manifest with SHA256 hashes is committed under research/round6/. Loading enforces Round 6's own
holdout lock (2016-01-01) and checks every parquet against the manifest.
"""
from __future__ import annotations

import hashlib
import io
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from btcq.config import DATA_DIR, RESEARCH_DIR

EXT_DIR = DATA_DIR / "external"
R6_DIR = RESEARCH_DIR / "round6"
MANIFEST = R6_DIR / "data_manifest.csv"
R6_HOLDOUT_START = pd.Timestamp("2016-01-01", tz="UTC")
R6_END = pd.Timestamp("2026-10-01", tz="UTC")              # exclusive
DAY_MS = 86_400_000
ROW_OFFSET_MS = DAY_MS - 60_000                           # engine rows are stamped 23:59 UTC of the trade date
PARSER_VERSION = 1
ALLOWED_NONPOSITIVE = {("CLUSDT", "2020-04-20")}
_R6_UNLOCKED = False
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) btcq-research"}


@dataclass(frozen=True)
class Source:
    market: str
    source: str        # "yahoo" | "eia"
    ticker: str        # Yahoo symbol or EIA series id
    first_expected: str

    @property
    def url(self) -> str:
        if self.source == "yahoo":
            now = int(time.time())
            return (f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(self.ticker)}"
                    f"?period1=-2208988800&period2={now}&interval=1d&events=div%7Csplit")
        area = "ng" if self.ticker.startswith("RNG") else "pet"
        return f"https://www.eia.gov/dnav/{area}/hist_xls/{self.ticker}d.xls"

    @property
    def stem(self) -> str:
        return self.ticker.replace("=", "_").replace("^", "")


SOURCES = {s.market: s for s in [
    Source("XAUUSDT", "yahoo", "GLD", "2004-11-18"),
    Source("XAGUSDT", "yahoo", "SLV", "2006-04-28"),
    Source("XPTUSDT", "yahoo", "PPLT", "2010-01-08"),
    Source("XPDUSDT", "yahoo", "PALL", "2010-01-08"),
    Source("COPPERUSDT", "yahoo", "HG=F", "2000-08-30"),
    Source("CLUSDT", "eia", "RWTC", "1986-01-02"),
    Source("BZUSDT", "eia", "RBRTE", "1987-05-20"),
    Source("NATGASUSDT", "eia", "RNGWHHD", "1997-01-07"),
    Source("SPYUSDT", "yahoo", "SPY", "1993-01-29"),
    Source("QQQUSDT", "yahoo", "QQQ", "1999-03-10"),
    Source("EWYUSDT", "yahoo", "EWY", "2000-05-12"),
]}
MARKETS = list(SOURCES)


def fetch(url: str, timeout: int = 60, retries: int = 4) -> bytes:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except Exception:  # noqa: BLE001 - retried, then re-raised
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt * 2)
    raise RuntimeError("unreachable")


def _next_weekday(d):
    d = d + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def parse_yahoo(raw: bytes, download_utc: datetime) -> tuple[pd.DataFrame, dict]:
    """Daily closes on exchange-local trade dates. A bar stamped 17:00 local or later (an evening session)
    belongs to the next weekday; the unfinished bar of the download day, null rows and duplicates go."""
    res = json.loads(raw)["chart"]["result"][0]
    gran = res["meta"].get("dataGranularity", "1d")
    if gran != "1d":
        raise ValueError(f"Yahoo returned {gran} bars, not 1d")
    tz = ZoneInfo(res["meta"]["exchangeTimezoneName"])
    q = res["indicators"]["quote"][0]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose") or [None] * len(res["timestamp"])
    today = download_utc.astimezone(tz).date()
    rows, nulls, unfinished = [], 0, 0
    for t, c, a in zip(res["timestamp"], q["close"], adj):
        local = datetime.fromtimestamp(t, tz)
        d = _next_weekday(local.date()) if local.hour >= 17 else local.date()
        if d >= today:
            unfinished += 1
            continue
        if c is None:
            nulls += 1
            continue
        rows.append((pd.Timestamp(d, tz="UTC"), float(c), float(a) if a is not None else np.nan))
    df = pd.DataFrame(rows, columns=["date", "close", "adjclose"])
    n = len(df)
    df = df.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    splits = res.get("events", {}).get("splits", {})
    divs = res.get("events", {}).get("dividends", {})
    return df, {"tz": str(tz), "nulls_dropped": nulls, "unfinished_dropped": unfinished,
                "dups_dropped": n - len(df), "splits": len(splits),
                "split_dates": ";".join(sorted(pd.Timestamp(int(s["date"]), unit="s").strftime("%Y-%m-%d")
                                               for s in splits.values())),
                "dividends": len(divs)}


def parse_eia_xls(raw: bytes, series_id: str, download_utc: datetime) -> tuple[pd.DataFrame, dict]:
    import xlrd
    wb = xlrd.open_workbook(file_contents=raw)
    sh = wb.sheet_by_name("Data 1")
    key = str(sh.cell_value(1, 1)).strip()
    if key != series_id:
        raise ValueError(f"EIA sheet key {key!r} != {series_id!r}")
    today = download_utc.date()
    rows, nulls = [], 0
    for r in range(3, sh.nrows):
        dv, cv = sh.cell_value(r, 0), sh.cell_value(r, 1)
        if dv in ("", None):
            continue
        d = xlrd.xldate_as_datetime(dv, wb.datemode).date()
        if d >= today:
            continue
        if cv in ("", None):
            nulls += 1
            continue
        rows.append((pd.Timestamp(d, tz="UTC"), float(cv), np.nan))
    df = pd.DataFrame(rows, columns=["date", "close", "adjclose"])
    n = len(df)
    df = df.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    return df, {"tz": "America/New_York", "nulls_dropped": nulls, "unfinished_dropped": 0,
                "dups_dropped": n - len(df), "splits": 0, "split_dates": "", "dividends": 0}


def clean(df: pd.DataFrame, market: str) -> tuple[pd.DataFrame, dict]:
    """Drop non-positive closes; the allowlisted ones are expected, any other is reported for a red flag."""
    bad = df["close"] <= 0
    dates = df.loc[bad, "date"].dt.strftime("%Y-%m-%d").tolist()
    unexpected = [d for d in dates if (market, d) not in ALLOWED_NONPOSITIVE]
    return df[~bad].reset_index(drop=True), {"nonpositive_dropped": ";".join(dates),
                                             "nonpositive_unexpected": ";".join(unexpected)}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def parquet_path(src: Source):
    return EXT_DIR / src.source / f"{src.stem}.parquet"


def raw_files(src: Source) -> list:
    d = EXT_DIR / "raw" / src.source
    return sorted(d.glob(f"{src.stem}_*")) if d.exists() else []


def date_ms(dates: pd.Series) -> np.ndarray:
    """UTC midnight timestamps as int ms, whatever the datetime unit (pandas 3 builds datetime64[s] here)."""
    return dates.dt.as_unit("ms").astype("int64").to_numpy(np.int64)


MIN_TS = int(pd.Timestamp("1980-01-01", tz="UTC").value // 10**6)


def check_ts(ts: np.ndarray, what: str) -> None:
    """Every stored date must be a UTC midnight in ms after 1980, strictly increasing."""
    ts = np.asarray(ts, np.int64)
    if not len(ts) or (ts % DAY_MS).any() or ts.min() < MIN_TS or (np.diff(ts) <= 0).any():
        raise ValueError(f"{what}: stored dates are not strictly increasing UTC midnights in ms after 1980")


def download_all(markets: list[str] | None = None, resume: bool = False) -> pd.DataFrame:
    """Download, parse and store every market; returns (and writes) the manifest. The manifest is rewritten
    after each market, so a crash leaves a resumable state. Nothing is ever overwritten: without resume any
    existing manifest or parquet is refused; with resume, markets already in the manifest are kept as they
    are and a parquet with no manifest row is refused."""
    markets = markets or MARKETS
    old = pd.read_csv(MANIFEST) if MANIFEST.exists() else None
    if not resume and (old is not None or any(parquet_path(SOURCES[m]).exists() or raw_files(SOURCES[m])
                                              for m in markets)):
        raise FileExistsError("Round 6 data already exists; only a resume may fill missing markets")
    done = set(old["market"]) if old is not None else set()
    orphans = [m for m in markets if m not in done and parquet_path(SOURCES[m]).exists()]
    if orphans:
        raise FileExistsError(f"parquet without a manifest row (refusing to overwrite): {orphans}")
    rows = [r for r in old.to_dict("records")] if old is not None else []
    for m in markets:
        if m in done:
            continue
        src = SOURCES[m]
        now = datetime.now(timezone.utc)
        url = src.url
        raw = fetch(url)
        raw_dir = EXT_DIR / "raw" / src.source
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / f"{src.stem}_{now:%Y%m%dT%H%M%SZ}.{'json' if src.source == 'yahoo' else 'xls'}"
        if raw_path.exists():
            raise FileExistsError(f"{raw_path.name} exists (refusing to overwrite a raw download)")
        raw_path.write_bytes(raw)                  # kept even if parsing fails: evidence for an amendment
        df, info = parse_yahoo(raw, now) if src.source == "yahoo" else parse_eia_xls(raw, src.ticker, now)
        df, info2 = clean(df, m)
        out = pd.DataFrame({"ts": date_ms(df["date"]), "close": df["close"].astype(float),
                            "adjclose": df["adjclose"].astype(float)})
        check_ts(out["ts"].to_numpy(), m)
        if not (pd.to_datetime(out["ts"], unit="ms", utc=True).dt.strftime("%Y-%m-%d").to_numpy()
                == df["date"].dt.strftime("%Y-%m-%d").to_numpy()).all():
            raise ValueError(f"{m}: stored dates do not round-trip")
        pq = parquet_path(src)
        pq.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        out.to_parquet(buf, index=False)
        pq.write_bytes(buf.getvalue())
        rows.append({"market": m, "source": src.source, "ticker": src.ticker, "url": url.split("&period2")[0],
                     "download_utc": now.isoformat(timespec="seconds"), "raw_file": raw_path.name,
                     "raw_bytes": len(raw), "raw_sha256": _sha(raw), "parquet_sha256": _sha(buf.getvalue()),
                     "rows": len(out), "first_date": df["date"].iloc[0].strftime("%Y-%m-%d"),
                     "last_date": df["date"].iloc[-1].strftime("%Y-%m-%d"), **info, **info2,
                     "parser_version": PARSER_VERSION})
        _write_manifest(rows)
        time.sleep(1.0)
    return _write_manifest(rows)


def _write_manifest(rows: list[dict]) -> pd.DataFrame:
    order = {m: i for i, m in enumerate(MARKETS)}
    man = pd.DataFrame(rows).sort_values("market", key=lambda s: s.map(order)).reset_index(drop=True)
    R6_DIR.mkdir(parents=True, exist_ok=True)
    man.to_csv(MANIFEST, index=False, lineterminator="\n")
    return man


def _verified_frame(market: str) -> pd.DataFrame:
    man = pd.read_csv(MANIFEST).set_index("market")
    p = parquet_path(SOURCES[market])
    b = p.read_bytes()
    if _sha(b) != man.loc[market, "parquet_sha256"]:
        raise RuntimeError(f"{market}: {p.name} does not match the committed manifest")
    df = pd.read_parquet(io.BytesIO(b))
    check_ts(df["ts"].to_numpy(), market)
    return df


def load_dates(market: str) -> np.ndarray:
    """Trade dates (ms, 00:00 UTC) of the whole file: timestamps only, no prices, no lock."""
    return _verified_frame(market)["ts"].to_numpy(np.int64)


def load_closes(market: str, period: str = "dev", adjusted: bool = False) -> pd.Series:
    """Closes indexed by trade date (ms, 00:00 UTC). dev: before 2016-01-01; full: needs unlock_r6()."""
    df = _verified_frame(market)
    cut = int(R6_HOLDOUT_START.value // 10**6)
    if period == "dev":
        df = df[df["ts"] < cut]
    elif period == "full":
        if not _R6_UNLOCKED:
            raise PermissionError("Round 6 holdout is locked. Call unlock_r6(reason) first.")
        df = df[df["ts"] < int(R6_END.value // 10**6)]
    else:
        raise ValueError(period)
    col = "adjclose" if adjusted else "close"
    return pd.Series(df[col].to_numpy(np.float64), index=df["ts"].to_numpy(np.int64), name=market)


def unlock_r6(reason: str, kind: str = "unlock", version: int | None = None) -> None:
    """Unlock Round 6 data from 2016 on (never the crypto holdout). Every call writes a structured record to
    research/holdout_log.jsonl with round 6, its kind (unlock | resume) and the strategy version, and tags the
    reason itself ('[R6: M1 v1]' or '[R6 resume M1 v1]'), so no Round 6 unlock can go uncounted."""
    global _R6_UNLOCKED
    if kind not in ("unlock", "resume"):
        raise ValueError(kind)
    if kind == "resume" and not any(r["kind"] == "unlock" for r in r6_unlock_records()):
        raise PermissionError("a Round 6 resume needs an earlier unlock")
    from btcq.data import store
    tag = f"[R6: M1 v{version}]" if kind == "unlock" else f"[R6 resume M1 v{version}]"
    rec = {"utc": datetime.now(timezone.utc).isoformat(), "round": 6, "kind": kind, "version": version,
           "reason": f"{reason} {tag}"}
    store.HOLDOUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with store.HOLDOUT_LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    _R6_UNLOCKED = True


def r6_unlock_records() -> list[dict]:
    """Every Round 6 record in the holdout log: structured ones, and any line whose reason carries an R6 tag."""
    from btcq.data import store
    if not store.HOLDOUT_LOG.exists():
        return []
    out = []
    for line in store.HOLDOUT_LOG.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError:
            r = {"reason": line}
        reason = str(r.get("reason", ""))
        if r.get("round") == 6 or "[R6" in reason:
            kind = r.get("kind") or ("resume" if "[R6 resume" in reason else "unlock")
            out.append(r | {"kind": kind})
    return out


def rows_from_closes(closes: pd.Series) -> pd.DataFrame:
    """Engine rows: one per trade date, stamped 23:59 UTC, open = high = low = close (close-only series)."""
    c = closes.to_numpy(np.float64)
    n = len(c)
    return pd.DataFrame({"ts": closes.index.to_numpy(np.int64) + ROW_OFFSET_MS, "open": c, "high": c, "low": c,
                         "close": c, "volume": np.zeros(n), "quote_volume": np.zeros(n),
                         "taker_buy_volume": np.zeros(n)})
