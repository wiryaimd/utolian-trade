"""Stream-and-discard tick archives (daily aggTrades and bookTicker) for the X1 execution study.

Each daily zip goes to a scratch folder, is checked against its .CHECKSUM, read batch by batch with pyarrow,
cut to the event windows (parquet) and deleted. Dev only: a key's period is parsed to its last day, which
must be on or before DEV_LAST_DAY ("2024-01" < "2024-01-01" as strings, so periods are never compared as
text). Every time column of every parsed row must be before HOLDOUT_START, and the kept rows are re-read and
checked again before the output is moved into place. Windows are half-open [start_ms, end_ms).
"""
from __future__ import annotations

import calendar
import hashlib
import json
import os
import re
import time
import zipfile
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from xml.etree import ElementTree

import httpx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from btcq.config import DATA_DIR, HOLDOUT_START, ROOT, TF_MS
from btcq.data import binance_vision as bv

if os.name == "nt":
    import msvcrt
else:
    import fcntl

DEV_LAST_DAY = date(2023, 12, 31)
HOLDOUT_MS = int(HOLDOUT_START.value // 10**6)
DAY_MS = TF_MS["1d"]
DATASETS = {"aggTrades": "data/futures/um/daily/aggTrades/{sym}/",
            "bookTicker": "data/futures/um/daily/bookTicker/{sym}/"}
TICK_DIR = DATA_DIR / "ticks"
SCRATCH = TICK_DIR / "_tmp"
COLS = {"aggTrades": ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "transact_time",
                      "is_buyer_maker"],
        "bookTicker": ["update_id", "best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty",
                       "transaction_time", "event_time"]}
TS_COL = {"aggTrades": "transact_time", "bookTicker": "transaction_time"}
TIME_COLS = {"aggTrades": ("transact_time",), "bookTicker": ("transaction_time", "event_time")}
_I, _F = pa.int64(), pa.float64()
SCHEMAS = {"aggTrades": pa.schema(list(zip(COLS["aggTrades"], [_I, _F, _F, _I, _I, _I, pa.bool_()]))),
           "bookTicker": pa.schema(list(zip(COLS["bookTicker"], [_I, _F, _F, _F, _F, _I, _I])))}
MANIFEST_COLS = ["dataset", "symbol", "day", "key", "bytes", "sha256", "rows_total", "rows_kept", "first_ts",
                 "last_ts", "out_file", "utc"]
_DAILY_KEY = re.compile(r"^data/futures/um/daily/(aggTrades|bookTicker)/([^/]+)/\2-\1-(\d{4}-\d{2}-\d{2})\.zip$")
_HEX64 = re.compile(r"[0-9a-f]{64}")


class HoldoutError(PermissionError):
    """Something would read data on or after HOLDOUT_START."""


# ---------------------------------------------------------------- keys and listing

def key_last_day(key: str) -> date:
    """Daily key '...-2023-05-16.zip' -> that day; monthly '...-2024-01.zip' -> 2024-01-31 (its last day)."""
    p = bv.key_period(key)
    if len(p) == 10:
        return date.fromisoformat(p)
    if len(p) == 7:
        y, m = int(p[:4]), int(p[5:])
        return date(y, m, calendar.monthrange(y, m)[1])
    raise ValueError(f"no period in key: {key}")


def is_dev_key(key: str) -> bool:
    try:
        return key_last_day(key) <= DEV_LAST_DAY
    except ValueError:
        return False


def require_dev_keys(keys: Iterable[str]) -> None:
    bad = [k for k in keys if not is_dev_key(k)]
    if bad:
        raise HoldoutError(f"{len(bad)} key(s) end after {DEV_LAST_DAY} or have no period: {bad}")


def daily_key(dataset: str, symbol: str, day) -> str:
    return DATASETS[dataset].format(sym=symbol) + f"{symbol}-{dataset}-{_day_str(day)}.zip"


def _parse_daily_key(key: str) -> tuple[str, str, str]:
    m = _DAILY_KEY.match(key)
    if not m:
        raise ValueError(f"not a daily aggTrades/bookTicker key: {key}")
    return m.group(1), m.group(2), date.fromisoformat(m.group(3)).isoformat()


def _list_sizes(prefix: str, client: httpx.Client | None = None) -> list[tuple[str, int]]:
    """(key, bytes) of every object under an S3 prefix, following pagination (bv._list with sizes)."""
    ns = bv._S3_NS
    out: list[tuple[str, int]] = []
    marker = ""
    own = client is None
    client = client or httpx.Client(timeout=60)
    try:
        while True:
            r = bv._get(client, bv.LIST_URL, {"delimiter": "/", "prefix": prefix, "marker": marker})
            root = ElementTree.fromstring(r.text)
            page = [(c.findtext(f"{ns}Key"), int(c.findtext(f"{ns}Size") or -1)) for c in root.iter(f"{ns}Contents")]
            pre = [p.findtext(f"{ns}Prefix") for p in root.iter(f"{ns}CommonPrefixes")]
            out += page
            if root.findtext(f"{ns}IsTruncated") != "true" or not (page or pre):
                break
            marker = root.findtext(f"{ns}NextMarker") or max([k for k, _ in page] + pre)
    finally:
        if own:
            client.close()
    return out


def list_dev_daily(dataset: str, symbol: str,
                   lister: Callable[[str], list[tuple[str, int]]] | None = None) -> list[tuple[str, int]]:
    """Daily archive keys (with sizes) of one dataset and symbol whose day is on or before DEV_LAST_DAY."""
    rows = (lister or _list_sizes)(DATASETS[dataset].format(sym=symbol))
    out = []
    for key, size in rows:
        try:
            ds, sym, _ = _parse_daily_key(key)
        except ValueError:
            continue  # .CHECKSUM files, monthly keys, anything else
        if ds == dataset and sym == symbol and is_dev_key(key):
            out.append((key, int(size)))
    return sorted(out)


def dev_days(symbols: Iterable[str], datasets: Iterable[str] = tuple(DATASETS),
             lister: Callable[[str], list[tuple[str, int]]] | None = None) -> dict[str, list[str]]:
    """{symbol: sorted dev days that have a daily archive in every one of datasets}: the `avail` of
    select_b1_days and leg_windows. build() fails on a job whose archive does not exist."""
    out = {}
    for sym in symbols:
        sets = [{_parse_daily_key(k)[2] for k, _ in list_dev_daily(ds, sym, lister)} for ds in datasets]
        out[sym] = sorted(set.intersection(*sets)) if sets else []
    return out


# ---------------------------------------------------------------- days and windows

def _day_str(day) -> str:
    if isinstance(day, date):  # also datetime and pd.Timestamp
        return day.strftime("%Y-%m-%d")
    return date.fromisoformat(str(day)).isoformat()


def _day_ms(day) -> int:
    return (date.fromisoformat(_day_str(day)) - date(1970, 1, 1)).days * DAY_MS


def _iso_day(day_index: int) -> str:
    return (date(1970, 1, 1) + timedelta(days=int(day_index))).isoformat()


def select_b1_days(avail: dict[str, list[str]], seed: int = 1101, per_coin: int = 15, lo: str = "2023-05-16",
                   hi: str = "2023-12-30") -> list[tuple[str, str]]:
    """per_coin days per symbol drawn without replacement from the available days in [lo, hi]. One generator,
    symbols in sorted order, so the draw does not depend on how the dict was built."""
    lo, hi = _day_str(lo), _day_str(hi)
    if date.fromisoformat(hi) > DEV_LAST_DAY:
        raise HoldoutError(f"B1 days must end on or before {DEV_LAST_DAY}, got hi={hi}")
    rng = np.random.default_rng(seed)
    out = []
    for sym in sorted(avail):
        days = sorted({d for d in map(_day_str, avail[sym]) if lo <= d <= hi})
        if len(days) < per_coin:
            raise ValueError(f"{sym}: only {len(days)} days available in [{lo}, {hi}], need {per_coin}")
        pick = rng.choice(len(days), size=per_coin, replace=False)
        out += [(sym, days[i]) for i in sorted(pick)]
    return out


def _live_ms(col: pd.Series) -> np.ndarray:
    """t_live_ms as int64 epoch ms; NaN, fractions, datetimes and µs/ns values raise instead of being cast."""
    if not len(col):
        return np.empty(0, np.int64)
    if not pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
        raise ValueError(f"t_live_ms must be numeric epoch ms, got dtype {col.dtype}")
    v = col.to_numpy(np.float64, na_value=np.nan)
    bad = ~np.isfinite(v) | (v != np.floor(v)) | (v < 0) | (v >= 1e14)
    if bad.any():
        raise ValueError(f"t_live_ms is not a whole number of epoch ms for legs {col.index[bad].tolist()}")
    return v.astype(np.int64)


def leg_windows(legs: pd.DataFrame, pre_ms: int = 600_000, post_ms: int = 7_200_000,
                avail: dict[str, Iterable[str]] | None = None) -> pd.DataFrame:
    """Window [t_live_ms - pre_ms, t_live_ms + post_ms) per leg, split at UTC midnights (one daily archive per
    piece); `leg` is the input row label. A window with end_ms >= HOLDOUT_START is dropped whole and reported
    in out.attrs["dropped"]. With avail ({symbol: days}, e.g. dev_days()), a leg with a piece on a day not in
    avail[symbol] is dropped whole and reported in out.attrs["missing"]. attrs hold plain dicts, so they
    survive pd.concat."""
    if pre_ms < 0 or post_ms <= 0:
        raise ValueError("pre_ms must be >= 0 and post_ms > 0")
    t = _live_ms(legs["t_live_ms"])
    s, e = t - pre_ms, t + post_ms
    drop = e >= HOLDOUT_MS
    syms = legs["symbol"].to_numpy()
    ok_days = None if avail is None else {sym: set(map(_day_str, d)) for sym, d in avail.items()}
    rows, missing = [], []
    for j in np.flatnonzero(~drop):
        a, b = int(s[j]), int(e[j])
        pieces = [{"symbol": syms[j], "day": _iso_day(d), "start_ms": max(a, d * DAY_MS),
                   "end_ms": min(b, (d + 1) * DAY_MS), "leg": legs.index[j]}
                  for d in range(a // DAY_MS, (b - 1) // DAY_MS + 1)]
        gone = [] if ok_days is None else [p["day"] for p in pieces if p["day"] not in ok_days.get(syms[j], ())]
        if gone:
            missing.append({"leg": legs.index[j], "symbol": syms[j], "t_live_ms": int(t[j]), "days": gone})
        else:
            rows += pieces
    out = pd.DataFrame(rows, columns=["symbol", "day", "start_ms", "end_ms", "leg"])
    out = out.astype({"start_ms": "int64", "end_ms": "int64"})
    out.attrs["dropped"] = [{"leg": legs.index[j], "symbol": syms[j], "t_live_ms": int(t[j]), "end_ms": int(e[j])}
                            for j in np.flatnonzero(drop)]
    out.attrs["missing"] = missing
    return out


def full_day_windows(pairs: Iterable[tuple[str, str]]) -> pd.DataFrame:
    """Whole-UTC-day windows for (symbol, day) pairs, e.g. select_b1_days' output."""
    rows = [{"symbol": s, "day": _day_str(d), "start_ms": _day_ms(d), "end_ms": _day_ms(d) + DAY_MS}
            for s, d in pairs]
    return pd.DataFrame(rows, columns=["symbol", "day", "start_ms", "end_ms"])


def jobs_from_windows(windows: pd.DataFrame, datasets: Iterable[str] = tuple(DATASETS)) -> pd.DataFrame:
    """build() jobs: one row per (dataset, symbol, day) with that day's list of (start_ms, end_ms)."""
    rows = []
    for (sym, day), g in windows.groupby(["symbol", "day"], sort=True):
        w = sorted(zip(g["start_ms"].astype("int64").tolist(), g["end_ms"].astype("int64").tolist()))
        rows += [{"dataset": ds, "symbol": sym, "day": _day_str(day), "windows": w} for ds in datasets]
    return pd.DataFrame(rows, columns=["dataset", "symbol", "day", "windows"])


def _windows(windows_ms, day=None) -> np.ndarray:
    """Validated, sorted, merged (k, 2) int64 windows. day: every window must lie inside that UTC day."""
    w = np.asarray([(int(a), int(b)) for a, b in windows_ms], dtype=np.int64).reshape(-1, 2)
    if not len(w):
        raise ValueError("no windows")
    if (w[:, 0] >= w[:, 1]).any():
        raise ValueError("a window has start_ms >= end_ms")
    if (w[:, 1] > HOLDOUT_MS).any():
        raise HoldoutError("a window reaches past HOLDOUT_START")
    if day is not None:
        d0 = _day_ms(day)
        if ((w[:, 0] < d0) | (w[:, 1] > d0 + DAY_MS)).any():
            raise ValueError(f"a window lies outside the archive's day {_day_str(day)}")
    w = w[np.argsort(w[:, 0], kind="stable")]
    merged = [[int(w[0, 0]), int(w[0, 1])]]
    for a, b in w[1:].tolist():
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return np.asarray(merged, dtype=np.int64)


def _inside(ts: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Mask of ts inside any of the sorted, disjoint windows w."""
    i = np.searchsorted(w[:, 0], ts, side="right") - 1
    ok = i >= 0
    ok[ok] = ts[ok] < w[i[ok], 1]
    return ok


def out_file_for(dataset: str, symbol: str, day, windows_ms, out_dir: Path | None = None) -> Path:
    """Output parquet path. The name carries a hash of the merged windows, so two sets cutting different
    windows from the same archive never share a file."""
    tag = hashlib.sha256(json.dumps(_windows(windows_ms).tolist()).encode()).hexdigest()[:10]
    day = _day_str(day)
    return Path(out_dir or TICK_DIR) / dataset / symbol / f"{symbol}-{dataset}-{day}-w{tag}.parquet"


# ---------------------------------------------------------------- read and cut

def read_zip_batches(path, dataset: str, block_size: int = 8 << 20) -> Iterator[pa.RecordBatch]:
    """Typed record batches of the single CSV in an archive zip, streamed. Files with and without a header
    row both occur; a header must name COLS[dataset]. Leading blank lines are skipped."""
    cols = COLS[dataset]
    schema = SCHEMAS[dataset]
    with zipfile.ZipFile(path) as zf:
        members = [n for n in zf.namelist() if not n.endswith("/")]
        if len(members) != 1:
            raise ValueError(f"{path}: expected one CSV member, found {members}")
        skip = 0
        with zf.open(members[0]) as f:
            first = f.readline()
            while first and not first.strip():
                skip += 1
                first = f.readline()
        if not first:
            return
        header = not first[:1].isdigit()
        if header:
            names = [x.strip().lower() for x in first.decode("utf-8-sig", "replace").strip().split(",")]
            if names != [c.lower() for c in cols]:
                raise ValueError(f"{path}: unexpected {dataset} header {names}")
            if not first.endswith(b"\n"):
                return  # header only, without a newline (pyarrow cannot skip it)
        # pyarrow's skip_rows counts the blank lines too
        with zf.open(members[0]) as f:
            reader = pacsv.open_csv(
                f, read_options=pacsv.ReadOptions(column_names=cols, skip_rows=skip + header, block_size=block_size),
                convert_options=pacsv.ConvertOptions(column_types=dict(zip(schema.names, schema.types))))
            yield from reader


def _ms(col: pa.Array) -> np.ndarray:
    if col.null_count:
        raise ValueError("null timestamps")
    a = col.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    return np.where(a >= 10**14, a // 1000, a)  # microsecond values -> ms


def extract(zip_path, dataset: str, windows_ms, out_path, meta: dict | None = None,
            block_size: int = 8 << 20) -> dict:
    """Keep the rows whose timestamp lies in any window and write them to parquet in file order, with the time
    columns in ms. A parsed row with any time column on or after HOLDOUT_START stops the read (a dev archive must
    not hold one, kept or not). Before the file is moved into place every kept time is re-read and must be before
    HOLDOUT_START, and every kept timestamp inside a window. Either failure is a HoldoutError (a PermissionError)
    and leaves no output file."""
    w = _windows(windows_ms)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    part = out_path.with_name(out_path.name + ".part")
    tcol = TS_COL[dataset]
    md = {f"x1_{k}": str(v) for k, v in (meta or {}).items()}
    md["x1_windows"] = json.dumps(w.tolist())
    schema = SCHEMAS[dataset].with_metadata(md)
    total = kept = us = decreases = 0
    lo = hi = prev = None
    batches = read_zip_batches(zip_path, dataset, block_size)
    try:
        with pq.ParquetWriter(part, schema) as writer:
            for b in batches:
                total += b.num_rows
                times = {n: _ms(b.column(n)) for n in TIME_COLS[dataset]}
                late = np.logical_or.reduce([a >= HOLDOUT_MS for a in times.values()])
                if late.any():
                    raise HoldoutError(f"{Path(zip_path).name}: {int(late.sum())} row(s) of this batch are stamped "
                                       "on or after HOLDOUT_START")
                us += int((b.column(tcol).to_numpy() >= 10**14).sum())
                ts_all = times[tcol]
                m = _inside(ts_all, w)
                if not m.any():
                    continue
                ts = ts_all[m]
                b = b.filter(pa.array(m))
                arrays = [pa.array(times[n][m]) if n in times else b.column(n) for n in schema.names]
                writer.write_batch(pa.RecordBatch.from_arrays(arrays, schema=schema))
                kept += len(ts)
                decreases += int((np.diff(ts) < 0).sum()) + int(prev is not None and ts[0] < prev)
                prev = int(ts[-1])
                lo = int(ts.min()) if lo is None else min(lo, int(ts.min()))
                hi = int(ts.max()) if hi is None else max(hi, int(ts.max()))
        _verify(part, dataset, w, kept)
        os.replace(part, out_path)
    finally:
        batches.close()
        _unlink(part)
    return {"dataset": dataset, "rows_total": total, "rows_kept": kept, "first_ts": lo, "last_ts": hi,
            "us_rows": us, "ts_decreases": decreases, "windows": len(w), "out_file": str(out_path)}


def _verify(path: Path, dataset: str, w: np.ndarray, kept: int) -> None:
    """Independent of the filter: re-read every written time column and check it against the holdout, and the
    timestamps against the windows (broadcast compare, not _inside)."""
    tcols = list(TIME_COLS[dataset])
    n = 0
    for b in pq.ParquetFile(path).iter_batches(batch_size=1 << 20, columns=tcols):
        for c in tcols:
            if (b.column(c).to_numpy() >= HOLDOUT_MS).any():
                raise HoldoutError(f"{path.name}: a kept {c} is on or after HOLDOUT_START")
        ts = b.column(TS_COL[dataset]).to_numpy()
        n += len(ts)
        if not ((ts[:, None] >= w[None, :, 0]) & (ts[:, None] < w[None, :, 1])).any(axis=1).all():
            raise HoldoutError(f"{path.name}: a kept timestamp lies outside every window")
    if n != kept:
        raise ValueError(f"{path.name}: wrote {kept} rows, read back {n}")


# ---------------------------------------------------------------- download

def _http_get(url: str, dest: Path, client: httpx.Client | None = None, tries: int = 5) -> None:
    """Stream url to dest, retrying network errors, 429 and 5xx with exponential backoff."""
    own = client is None
    client = client or httpx.Client(timeout=120, follow_redirects=True)
    try:
        for attempt in range(tries):
            try:
                with client.stream("GET", url) as r:
                    if r.status_code != 429 and r.status_code < 500:
                        r.raise_for_status()
                        with open(dest, "wb") as f:
                            for chunk in r.iter_bytes(1 << 20):
                                f.write(chunk)
                        return
                    if attempt == tries - 1:
                        r.raise_for_status()
            except httpx.TransportError:
                if attempt == tries - 1:
                    raise
            time.sleep(2 ** attempt)
    finally:
        if own:
            client.close()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _checksum_value(path: Path) -> str:
    # bytes, not text: CHECKSUM files name the archive (bv._verified)
    parts = path.read_bytes().split()
    v = parts[0].decode("ascii", "replace").strip().lower() if parts else ""
    if not _HEX64.fullmatch(v):
        raise ValueError(f"bad CHECKSUM file {path.name}")
    return v


def _unlink(path: Path, tries: int = 5) -> None:
    """Windows can briefly hold a handle (antivirus, indexer): retry before giving up."""
    for attempt in range(tries):
        try:
            Path(path).unlink(missing_ok=True)
            return
        except OSError:
            if attempt == tries - 1:
                raise
            time.sleep(0.5 * (attempt + 1))


def fetch_extract(key: str, windows_ms, out_path, getter: Callable[[str, Path], None] | None = None) -> dict:
    """Download one dev daily archive to SCRATCH, verify its SHA256 against the .CHECKSUM, extract the windows
    and delete the zip (also on failure). getter(url, dest) writes the body of url to dest."""
    require_dev_keys([key])
    dataset, symbol, day = _parse_daily_key(key)
    w = _windows(windows_ms, day)
    get = getter or _http_get
    SCRATCH.mkdir(parents=True, exist_ok=True)
    zpath = SCRATCH / key.rsplit("/", 1)[1]
    cpath = zpath.with_name(zpath.name + ".CHECKSUM")
    try:
        get(bv.BASE_URL + key, zpath)
        get(bv.BASE_URL + key + ".CHECKSUM", cpath)
        sha = _sha256_file(zpath)
        if sha != _checksum_value(cpath):
            raise ValueError(f"checksum mismatch: {key}")
        size = zpath.stat().st_size
        st = extract(zpath, dataset, w.tolist(), out_path, meta={"source_key": key, "source_sha256": sha})
        return {"key": key, "symbol": symbol, "day": day, "bytes": size, "sha256": sha, **st}
    finally:
        _unlink(zpath)
        _unlink(cpath)


# ---------------------------------------------------------------- build and manifest

def _rel(p) -> str:
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def _resolve(s: str) -> Path:
    p = Path(s)
    return (p if p.is_absolute() else ROOT / p).resolve()


def read_manifest(path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=MANIFEST_COLS)
    ints = {c: "Int64" for c in ("bytes", "rows_total", "rows_kept", "first_ts", "last_ts")}
    strs = {c: str for c in ("dataset", "symbol", "day", "key", "sha256", "out_file", "utc")}
    return pd.read_csv(p, dtype={**ints, **strs}, keep_default_na=False, na_values={c: [""] for c in ints})


def _write_manifest(df: pd.DataFrame, path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    df.sort_values(["dataset", "symbol", "day"]).to_csv(tmp, index=False, lineterminator="\n")
    os.replace(tmp, p)


def _plan(jobs: pd.DataFrame, out_dir: Path) -> list[dict]:
    """Validate every job before anything is downloaded: dev keys first, then the windows."""
    missing = {"dataset", "symbol", "day", "windows"} - set(jobs.columns)
    if missing:
        raise ValueError(f"jobs lack columns {sorted(missing)}")
    bad = sorted(set(jobs["dataset"]) - set(DATASETS))
    if bad:
        raise ValueError(f"unknown datasets {bad}")
    keys = [daily_key(ds, sym, day) for ds, sym, day in zip(jobs["dataset"], jobs["symbol"], jobs["day"])]
    require_dev_keys(keys)
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate (dataset, symbol, day) jobs")
    plan = []
    for key, (ds, sym, day, wins) in zip(keys, jobs[["dataset", "symbol", "day", "windows"]].itertuples(index=False)):
        w = _windows(wins, day)
        plan.append({"key": key, "dataset": ds, "symbol": sym, "day": _day_str(day), "windows": w.tolist(),
                     "out_file": out_file_for(ds, sym, day, w.tolist(), out_dir)})
    return plan


def _done(job: dict, row) -> bool:
    """Skip a job when its out_file exists and was cut from the archive whose SHA256 the manifest records,
    with the same windows (both stored in the parquet's metadata)."""
    if row is None or not isinstance(row["sha256"], str) or not _HEX64.fullmatch(row["sha256"]):
        return False
    out = _resolve(row["out_file"])
    if out != Path(job["out_file"]).resolve() or not out.exists():
        return False
    try:
        md = pq.read_schema(out).metadata or {}
    except Exception:  # noqa: BLE001 - an unreadable file is redone
        return False
    return (md.get(b"x1_source_sha256") == row["sha256"].encode() and md.get(b"x1_source_key") == job["key"].encode()
            and md.get(b"x1_windows") == json.dumps(job["windows"]).encode())


@contextmanager
def _scratch_lock() -> Iterator[None]:
    """One build at a time per SCRATCH, so a start-up cleanup never deletes another build's zips. An OS file
    lock (next to SCRATCH, kept): the OS releases it if the process dies, so it never goes stale."""
    path = SCRATCH.with_name(SCRATCH.name + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT)
    try:
        try:
            if os.name == "nt":
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError(f"another build is using {SCRATCH} ({path.name} is locked)") from None
        try:
            yield
        finally:
            if os.name == "nt":
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    finally:
        os.close(fd)


def build(jobs: pd.DataFrame, manifest_path, getter: Callable[[str, Path], None] | None = None, workers: int = 2,
          out_dir: Path | None = None) -> pd.DataFrame:
    """Fetch and cut every job (columns dataset, symbol, day, windows), at most `workers` archives at a time.
    Every job's archive must exist (filter days with dev_days / leg_windows(avail=...) first). Resumable:
    finished jobs are skipped, and the manifest (MANIFEST_COLS, one row per key) is rewritten after each
    completed job. A holdout guard aborts everything; other failures are collected and raised at the end after
    the successful rows are saved. A second build while one runs raises RuntimeError. Returns the manifest."""
    if workers < 1:
        raise ValueError("workers must be >= 1")
    plan = _plan(pd.DataFrame(jobs), Path(out_dir or TICK_DIR))
    with _scratch_lock():
        man = read_manifest(manifest_path)
        rows = {r["key"]: r for r in man.to_dict("records")}
        todo = [j for j in plan if not _done(j, rows.get(j["key"]))]
        SCRATCH.mkdir(parents=True, exist_ok=True)
        for p in SCRATCH.iterdir():  # leftovers of an interrupted run
            if p.is_file() and p.name.endswith((".zip", ".zip.CHECKSUM")):
                _unlink(p)
        errors: list[str] = []
        ex = ThreadPoolExecutor(max_workers=workers)
        try:
            futs = {ex.submit(fetch_extract, j["key"], j["windows"], j["out_file"], getter): j for j in todo}
            for fut in as_completed(futs):
                j = futs[fut]
                try:
                    st = fut.result()
                except HoldoutError:
                    raise
                except Exception as e:  # noqa: BLE001 - one bad archive must not abort the batch
                    errors.append(f"{type(e).__name__} on {j['key']}: {e}")
                    continue
                rows[j["key"]] = {"dataset": j["dataset"], "symbol": j["symbol"], "day": j["day"], "key": j["key"],
                                  "bytes": st["bytes"], "sha256": st["sha256"], "rows_total": st["rows_total"],
                                  "rows_kept": st["rows_kept"], "first_ts": st["first_ts"], "last_ts": st["last_ts"],
                                  "out_file": _rel(j["out_file"]),
                                  "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
                man = pd.DataFrame(list(rows.values()), columns=MANIFEST_COLS)
                _write_manifest(man, manifest_path)
        finally:
            ex.shutdown(wait=True, cancel_futures=True)
    if errors:
        raise RuntimeError(f"{len(errors)} of {len(todo)} jobs failed, first: {errors[0]}")
    return read_manifest(manifest_path) if Path(manifest_path).exists() else man
