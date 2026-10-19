"""Round 8 daily OHLC for SPY, QQQ and EWY, re-parsed from Round 6's raw Yahoo chart files (no download).

- The raw files are the ones named in Round 6's manifest (`research/round6/data_manifest.csv`); their SHA256 must
  still match it.
- `parse_yahoo_ohlc` applies `external.parse_yahoo`'s date rules (exchange-local trade dates, a bar stamped 17:00
  local or later belongs to the next weekday, the unfinished bar of the download day and null closes are dropped,
  of duplicate dates the last is kept) and also keeps open, high and low. It is a hard error unless its (date,
  close) rows equal `external.parse_yahoo`'s and, after Round 6's cleaning, Round 6's verified parquet row for row.
- Derived parquets go to data/external/ohlc/ (not committed); `research/round8/data_manifest.csv` (raw, Round 6 and
  derived SHA256s, row counts) is committed and checked on every load.
- Lock: dev loads return only rows dated before 2016-01-01. The holdout (2016-01-01 -> 2026-09-30) needs Round 8's
  own unlock, which writes a structured `round: 8` record tagged `[R8:` to research/holdout_log.jsonl. It never
  unlocks the crypto 2024+ data or Round 6's markets: only these 3 markets can be loaded here.
"""
from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from btcq.config import RESEARCH_DIR
from btcq.data import external as E

MARKETS = ["SPYUSDT", "QQQUSDT", "EWYUSDT"]
R8_DIR = RESEARCH_DIR / "round8"
OHLC_DIR = E.EXT_DIR / "ohlc"
MANIFEST = R8_DIR / "data_manifest.csv"
R6_MANIFEST = E.MANIFEST
HOLDOUT_START = E.R6_HOLDOUT_START          # 2016-01-01
END = E.R6_END                              # 2026-10-01, exclusive
PARSER_VERSION = 1
TAG, VERIFY_TAG = "[R8:", "[R8 verify"
_R8_UNLOCKED = False


class DataError(RuntimeError):
    """A Round 8 data red flag 1: a raw-file SHA256 or a close differs from Round 6."""


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _check_market(market: str) -> None:
    if market not in MARKETS:
        raise PermissionError(f"Round 8 loads only {MARKETS}, not {market}")


def parse_yahoo_ohlc(raw: bytes, download_utc: datetime) -> tuple[pd.DataFrame, dict]:
    """Daily OHLC on exchange-local trade dates, with external.parse_yahoo's date rules. Columns: date, open, high,
    low, close (open/high/low NaN when Yahoo gives null). Raises unless (date, close) equal parse_yahoo's."""
    res = json.loads(raw)["chart"]["result"][0]
    if res["meta"].get("dataGranularity", "1d") != "1d":
        raise ValueError("Yahoo returned non-daily bars")
    tz = ZoneInfo(res["meta"]["exchangeTimezoneName"])
    q = res["indicators"]["quote"][0]
    n = len(res["timestamp"])
    opens, highs, lows = (q.get(k) or [None] * n for k in ("open", "high", "low"))
    today = download_utc.astimezone(tz).date()
    rows = []
    for t, o, h, lo, c in zip(res["timestamp"], opens, highs, lows, q["close"]):
        local = datetime.fromtimestamp(t, tz)
        d = E._next_weekday(local.date()) if local.hour >= 17 else local.date()
        if d >= today or c is None:
            continue
        f = (lambda x: float(x) if x is not None else np.nan)
        rows.append((pd.Timestamp(d, tz="UTC"), f(o), f(h), f(lo), float(c)))
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close"])
    df = df.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    ref, info = E.parse_yahoo(raw, download_utc)
    if len(ref) != len(df) or not (ref["date"].to_numpy() == df["date"].to_numpy()).all() \
            or not np.array_equal(ref["close"].to_numpy(np.float64), df["close"].to_numpy(np.float64)):
        raise DataError("the OHLC parse does not reproduce external.parse_yahoo's (date, close) rows")
    return df, info


def ohlc_path(market: str):
    return OHLC_DIR / f"{E.SOURCES[market].stem}.parquet"


def r6_row(market: str) -> dict:
    return pd.read_csv(R6_MANIFEST).set_index("market").loc[market].to_dict()


def _build_market(market: str) -> tuple[pd.DataFrame, dict]:
    """Re-parse one market's Round 6 raw file into OHLC and check it against Round 6, row for row. Returns
    (frame with ts, open, high, low, close; manifest row). Nothing is written."""
    _check_market(market)
    src = E.SOURCES[market]
    row = r6_row(market)
    raw_path = E.EXT_DIR / "raw" / src.source / str(row["raw_file"])
    raw = raw_path.read_bytes()
    if _sha(raw) != row["raw_sha256"]:
        raise DataError(f"{market}: {raw_path.name} does not match Round 6's manifest")
    df, _ = parse_yahoo_ohlc(raw, datetime.fromisoformat(str(row["download_utc"])))
    df, _ = E.clean(df, market)
    r6 = E._verified_frame(market)                          # raises unless it matches Round 6's manifest
    out = pd.DataFrame({"ts": E.date_ms(df["date"]), "open": df["open"].astype(float),
                        "high": df["high"].astype(float), "low": df["low"].astype(float),
                        "close": df["close"].astype(float)})
    E.check_ts(out["ts"].to_numpy(), market)
    if not (np.array_equal(out["ts"].to_numpy(np.int64), r6["ts"].to_numpy(np.int64))
            and np.array_equal(out["close"].to_numpy(np.float64), r6["close"].to_numpy(np.float64))):
        raise DataError(f"{market}: the re-parsed (date, close) rows differ from Round 6's verified parquet")
    buf = io.BytesIO()
    out.to_parquet(buf, index=False)
    ds = pd.to_datetime(out["ts"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    cut = int(HOLDOUT_START.value // 10**6)
    man = {"market": market, "ticker": src.ticker, "raw_file": str(row["raw_file"]), "raw_sha256": row["raw_sha256"],
           "r6_parquet_sha256": row["parquet_sha256"], "ohlc_parquet_sha256": _sha(buf.getvalue()),
           "rows": int(len(out)), "rows_dev": int((out["ts"] < cut).sum()), "first_date": ds.iloc[0],
           "last_date": ds.iloc[-1], "parser_version": PARSER_VERSION}
    return out, man | {"_bytes": buf.getvalue()}


def build_all(overwrite: bool = False) -> pd.DataFrame:
    """Build the 3 markets' OHLC parquets and write the Round 8 manifest. Refuses to overwrite unless asked."""
    if MANIFEST.exists() and not overwrite:
        raise FileExistsError("the Round 8 data manifest exists (r8-data runs once)")
    rows = []
    for m in MARKETS:
        _, man = _build_market(m)
        p = ohlc_path(m)
        if p.exists() and not overwrite and _sha(p.read_bytes()) != man["ohlc_parquet_sha256"]:
            raise FileExistsError(f"{p} exists with other content (refusing to overwrite)")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(man.pop("_bytes"))
        rows.append(man)
    R8_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(MANIFEST, index=False, lineterminator="\n")
    return df


def _verified_frame(market: str) -> pd.DataFrame:
    """The whole derived file (holdout rows included: callers use only the dev rows, the lock or row counts), after
    checking it against the Round 8 manifest and Round 6's manifest pins."""
    _check_market(market)
    man = pd.read_csv(MANIFEST).set_index("market")
    if market not in man.index:
        raise DataError(f"{market} is not in the Round 8 manifest")
    r = man.loc[market]
    if r6_row(market)["parquet_sha256"] != r["r6_parquet_sha256"] or r6_row(market)["raw_sha256"] != r["raw_sha256"]:
        raise DataError(f"{market}: Round 6's manifest changed since the Round 8 build")
    b = ohlc_path(market).read_bytes()
    if _sha(b) != r["ohlc_parquet_sha256"]:
        raise DataError(f"{market}: {ohlc_path(market).name} does not match the Round 8 manifest")
    df = pd.read_parquet(io.BytesIO(b))
    E.check_ts(df["ts"].to_numpy(), market)
    if len(df) != int(r["rows"]):
        raise DataError(f"{market}: row count differs from the Round 8 manifest")
    return df


def check_files() -> None:
    """Raise unless every market's derived file matches both manifests (no prices are returned)."""
    for m in MARKETS:
        _verified_frame(m)


def load_ohlc(market: str, period: str = "dev") -> pd.DataFrame:
    """ts (ms, 00:00 UTC), open, high, low, close. dev: rows before 2016-01-01; full: rows before 2026-10-01, only
    after unlock_r8()."""
    df = _verified_frame(market)
    cut = int(HOLDOUT_START.value // 10**6)
    if period == "dev":
        df = df[df["ts"] < cut]
    elif period == "full":
        if not _R8_UNLOCKED:
            raise PermissionError("Round 8 holdout is locked. Call unlock_r8(reason) first.")
        df = df[df["ts"] < int(END.value // 10**6)]
    else:
        raise ValueError(period)
    return df.reset_index(drop=True)


def unlock_r8(reason: str, kind: str = "unlock", version: int | None = None) -> None:
    """Unlock the 3 markets from 2016-01-01 on (never the crypto holdout or Round 6's other markets). Every call
    writes a structured record (round 8, kind, version) to research/holdout_log.jsonl and tags the reason itself
    ('[R8: IBS v1]' or '[R8 verify ...]'), so no Round 8 unlock can go uncounted. There is one unlock: it is refused
    once any Round 8 record exists (there is no resume; an unlock without a result is a FAIL). A verify load (an
    independent check) is allowed only after the unlock."""
    global _R8_UNLOCKED
    if kind not in ("unlock", "verify"):
        raise ValueError(kind)
    recs = r8_unlock_records()
    if kind == "unlock":
        if recs:
            raise PermissionError("Round 8 was already unlocked once (there is no second unlock and no resume)")
        if "[R" in reason:
            raise ValueError("the reason must not carry a round tag (the tag is added here)")
        tag = f"{TAG} IBS v{version}]"
    else:
        if not any(r["kind"] == "unlock" for r in recs):
            raise PermissionError("a Round 8 verify load needs the earlier holdout unlock")
        if VERIFY_TAG not in reason:
            raise ValueError(f"a verify load must carry {VERIFY_TAG}")
        tag = ""
    from btcq.data import store
    rec = {"utc": datetime.now(timezone.utc).isoformat(), "round": 8, "kind": kind, "version": version,
           "reason": f"{reason} {tag}".strip()}
    store.HOLDOUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with store.HOLDOUT_LOG.open("a", newline="\n") as f:
        f.write(json.dumps(rec) + "\n")
    _R8_UNLOCKED = True


def r8_unlock_records() -> list[dict]:
    """Every Round 8 record of the holdout log, over every committed version and the working copy: structured ones
    and any line whose reason carries an R8 tag."""
    from btcq.data import store
    from btcq.research import guards
    out = []
    for r in guards.all_records(store.HOLDOUT_LOG):
        reason = str(r.get("reason", r.get("raw", "")))
        if r.get("round") == 8 or "[R8" in reason:
            kind = r.get("kind") or ("verify" if VERIFY_TAG in reason else "unlock")
            if kind == "resume":                       # no resume exists in Round 8: it counts as an unlock
                kind = "unlock"
            out.append(r | {"kind": kind})
    return out
