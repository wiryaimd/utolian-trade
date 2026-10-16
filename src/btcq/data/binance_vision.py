"""Bulk downloader for data.binance.vision archives with SHA256 verification."""
from __future__ import annotations

import asyncio
import hashlib
import io
import re
import time
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import httpx
import pandas as pd
from tqdm import tqdm

from btcq.config import RAW_DIR, SYMBOL

BASE_URL = "https://data.binance.vision/"
LIST_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
_S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]

# Archive prefixes; {sym} is filled with the symbol at download time.
DATASETS = {
    "perp_1m_monthly": "data/futures/um/monthly/klines/{sym}/1m/",
    "perp_1m_daily": "data/futures/um/daily/klines/{sym}/1m/",
    "perp_15m_monthly": "data/futures/um/monthly/klines/{sym}/15m/",
    "perp_4h_monthly": "data/futures/um/monthly/klines/{sym}/4h/",
    "perp_1d_monthly": "data/futures/um/monthly/klines/{sym}/1d/",
    "perp_1d_daily": "data/futures/um/daily/klines/{sym}/1d/",
    "spot_1m_monthly": "data/spot/monthly/klines/{sym}/1m/",
    "spot_1d_monthly": "data/spot/monthly/klines/{sym}/1d/",
    "spot_1d_daily": "data/spot/daily/klines/{sym}/1d/",
    "funding_monthly": "data/futures/um/monthly/fundingRate/{sym}/",
    "metrics_daily": "data/futures/um/daily/metrics/{sym}/",
    "premium_1h_monthly": "data/futures/um/monthly/premiumIndexKlines/{sym}/1h/",
    "premium_1h_daily": "data/futures/um/daily/premiumIndexKlines/{sym}/1h/",
}


def _get(client: httpx.Client, url: str, params: dict | None = None, tries: int = 6) -> httpx.Response:
    """GET with exponential backoff on network errors, 429 and 5xx."""
    for attempt in range(tries):
        try:
            r = client.get(url, params=params)
            if r.status_code != 429 and r.status_code < 500:
                return r.raise_for_status()
        except httpx.TransportError:
            if attempt == tries - 1:
                raise
        if attempt == tries - 1:
            r.raise_for_status()
        time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def _list(prefix: str, client: httpx.Client | None = None) -> tuple[list[str], list[str]]:
    """All (Keys, CommonPrefixes) under an S3 prefix, following pagination."""
    keys: list[str] = []
    prefixes: list[str] = []
    marker = ""
    own = client is None
    client = client or httpx.Client(timeout=60)
    try:
        while True:
            r = _get(client, LIST_URL, {"delimiter": "/", "prefix": prefix, "marker": marker})
            root = ElementTree.fromstring(r.text)
            page = [k.text for k in root.iter(f"{_S3_NS}Key")]
            pre = [p.findtext(f"{_S3_NS}Prefix") for p in root.iter(f"{_S3_NS}CommonPrefixes")]
            keys += page
            prefixes += pre
            if root.findtext(f"{_S3_NS}IsTruncated") != "true" or not (page or pre):
                break
            marker = root.findtext(f"{_S3_NS}NextMarker") or max(page + pre)
    finally:
        if own:
            client.close()
    return keys, prefixes


def list_keys(prefix: str, client: httpx.Client | None = None) -> list[str]:
    """List every .zip key under an S3 prefix (handles pagination)."""
    return sorted(k for k in _list(prefix, client)[0] if k.endswith(".zip"))


def list_prefixes(prefix: str, client: httpx.Client | None = None) -> list[str]:
    """Child "directory" names under an S3 prefix, e.g. the symbols of a dataset."""
    return sorted({p[len(prefix):].strip("/") for p in _list(prefix, client)[1]})


def list_um_symbols() -> list[str]:
    """Every USD-M futures symbol with kline archives (listed, delisted and dated contracts)."""
    return sorted(set(list_prefixes("data/futures/um/monthly/klines/"))
                  | set(list_prefixes("data/futures/um/daily/klines/")))


def _local_path(key: str) -> Path:
    return RAW_DIR / key.removeprefix("data/")


def _verified(path: Path) -> bool:
    ck = path.with_name(path.name + ".CHECKSUM")
    if not (path.exists() and ck.exists()):
        return False
    # bytes, not text: CHECKSUM files name the archive, and some symbols are not ASCII
    expected = ck.read_bytes().split()[0].decode("ascii").strip().lower()
    return hashlib.sha256(path.read_bytes()).hexdigest() == expected


async def _fetch(client: httpx.AsyncClient, key: str, sem: asyncio.Semaphore) -> str | None:
    try:
        path = _local_path(key)
        if _verified(path):
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        async with sem:
            for attempt in range(5):
                try:
                    body = (await client.get(BASE_URL + key)).raise_for_status().content
                    ck = (await client.get(BASE_URL + key + ".CHECKSUM")).raise_for_status().content
                    break
                except httpx.HTTPError:
                    if attempt == 4:
                        return f"download failed: {key}"
                    await asyncio.sleep(2 ** attempt)
        expected = ck.split()[0].decode("ascii").strip().lower()
        if hashlib.sha256(body).hexdigest() != expected:
            return f"checksum mismatch: {key}"
        path.write_bytes(body)
        path.with_name(path.name + ".CHECKSUM").write_bytes(ck)
        return None
    except Exception as e:  # noqa: BLE001 - one bad file must not abort the whole batch
        return f"{type(e).__name__} on {key}: {e}"


async def _download_all(keys: list[str], concurrency: int) -> list[str]:
    sem = asyncio.Semaphore(concurrency)
    errors: list[str] = []
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        tasks = [asyncio.create_task(_fetch(client, k, sem)) for k in keys]
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), leave=False):
            err = await fut
            if err:
                errors.append(err)
    return errors


def download(dataset: str, keep=lambda key: True, concurrency: int = 8,
             symbol: str = SYMBOL) -> list[Path]:
    """Download (and checksum-verify) every archive in a dataset. Returns local paths."""
    keys = [k for k in list_keys(DATASETS[dataset].format(sym=symbol)) if keep(k)]
    errors = asyncio.run(_download_all(keys, concurrency))
    if errors:
        raise RuntimeError(f"{len(errors)} files failed, first: {errors[0]}")
    return [_local_path(k) for k in keys]


def download_keys(keys: list[str], concurrency: int = 16) -> list[Path]:
    """Download (and checksum-verify) an explicit list of archive keys, e.g. across symbols."""
    errors = asyncio.run(_download_all(keys, concurrency))
    if errors:
        raise RuntimeError(f"{len(errors)} files failed, first: {errors[0]}")
    return [_local_path(k) for k in keys]


def key_period(key: str) -> str:
    """'...-1m-2024-01.zip' -> '2024-01'; '...-2024-01-05.zip' -> '2024-01-05'."""
    m = re.search(r"(\d{4}-\d{2}(?:-\d{2})?)\.zip$", key)
    return m.group(1) if m else ""


def _read_csv_from_zip(path: Path, names: list[str]) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        raw = zf.read(zf.namelist()[0])
    has_header = not raw[:1].isdigit()
    return pd.read_csv(io.BytesIO(raw), header=0 if has_header else None,
                       names=None if has_header else names)


def _to_ms(ts: pd.Series) -> pd.Series:
    ts = ts.astype("int64")
    # Spot archives switched to microseconds in 2025.
    return ts.where(ts < 10**14, ts // 1000)


def parse_klines(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = _read_csv_from_zip(p, KLINE_COLS)
        df.columns = KLINE_COLS[: len(df.columns)]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame({"ts": _to_ms(df["open_time"])})
    for col in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume"]:
        out[col] = df[col].astype("float64")
    out["count"] = df["count"].astype("int64")
    return out.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def parse_funding(paths: list[Path]) -> pd.DataFrame:
    df = pd.concat([_read_csv_from_zip(p, ["calc_time", "funding_interval_hours", "last_funding_rate"])
                    for p in paths], ignore_index=True)
    out = pd.DataFrame({"ts": _to_ms(df["calc_time"]), "rate": df["last_funding_rate"].astype("float64")})
    return out.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def parse_metrics(paths: list[Path]) -> pd.DataFrame:
    frames = [_read_csv_from_zip(p, []) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    # pandas >= 3 parses these strings at us/s resolution, so convert via an explicit unit.
    ts = pd.to_datetime(df["create_time"], utc=True).dt.as_unit("ms").astype("int64")
    out = pd.DataFrame({"ts": ts})
    for col in ["sum_open_interest", "sum_open_interest_value", "count_toptrader_long_short_ratio",
                "sum_toptrader_long_short_ratio", "count_long_short_ratio",
                "sum_taker_long_short_vol_ratio"]:
        out[col] = pd.to_numeric(df[col], errors="coerce")
    return out.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
