"""Bulk downloader for data.binance.vision archives with SHA256 verification."""
from __future__ import annotations

import asyncio
import hashlib
import io
import re
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
    "spot_1m_monthly": "data/spot/monthly/klines/{sym}/1m/",
    "funding_monthly": "data/futures/um/monthly/fundingRate/{sym}/",
    "metrics_daily": "data/futures/um/daily/metrics/{sym}/",
    "premium_1h_monthly": "data/futures/um/monthly/premiumIndexKlines/{sym}/1h/",
    "premium_1h_daily": "data/futures/um/daily/premiumIndexKlines/{sym}/1h/",
}


def list_keys(prefix: str) -> list[str]:
    """List every .zip key under an S3 prefix (handles pagination)."""
    keys: list[str] = []
    marker = ""
    with httpx.Client(timeout=60) as client:
        while True:
            r = client.get(LIST_URL, params={"delimiter": "/", "prefix": prefix, "marker": marker})
            r.raise_for_status()
            root = ElementTree.fromstring(r.text)
            page = [k.text for k in root.iter(f"{_S3_NS}Key")]
            keys += [k for k in page if k.endswith(".zip")]
            truncated = root.findtext(f"{_S3_NS}IsTruncated") == "true"
            if not truncated or not page:
                break
            marker = root.findtext(f"{_S3_NS}NextMarker") or page[-1]
    return sorted(keys)


def _local_path(key: str) -> Path:
    return RAW_DIR / key.removeprefix("data/")


def _verified(path: Path) -> bool:
    ck = path.with_name(path.name + ".CHECKSUM")
    if not (path.exists() and ck.exists()):
        return False
    expected = ck.read_text().split()[0].strip().lower()
    return hashlib.sha256(path.read_bytes()).hexdigest() == expected


async def _fetch(client: httpx.AsyncClient, key: str, sem: asyncio.Semaphore) -> str | None:
    path = _local_path(key)
    if _verified(path):
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    async with sem:
        for attempt in range(5):
            try:
                body = (await client.get(BASE_URL + key)).raise_for_status().content
                ck = (await client.get(BASE_URL + key + ".CHECKSUM")).raise_for_status().text
                break
            except httpx.HTTPError:
                if attempt == 4:
                    return f"download failed: {key}"
                await asyncio.sleep(2 ** attempt)
    expected = ck.split()[0].strip().lower()
    if hashlib.sha256(body).hexdigest() != expected:
        return f"checksum mismatch: {key}"
    path.write_bytes(body)
    path.with_name(path.name + ".CHECKSUM").write_text(ck)
    return None


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
