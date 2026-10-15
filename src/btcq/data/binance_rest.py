"""Binance USDS-M public REST endpoints used to fill gaps the archive doesn't cover."""
from __future__ import annotations

import time

import httpx
import pandas as pd

from btcq.config import SYMBOL

FAPI = "https://fapi.binance.com"
TESTNET_FAPI = "https://testnet.binancefuture.com"


def funding_history(start_ms: int, end_ms: int | None = None, symbol: str = SYMBOL,
                    base: str = FAPI) -> pd.DataFrame:
    """All settled funding rates in [start_ms, end_ms)."""
    rows: list[dict] = []
    cursor = start_ms
    end_ms = end_ms or int(time.time() * 1000)
    with httpx.Client(timeout=30) as client:
        while cursor < end_ms:
            r = client.get(f"{base}/fapi/v1/fundingRate",
                           params={"symbol": symbol, "startTime": cursor, "endTime": end_ms - 1,
                                   "limit": 1000})
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            rows += batch
            cursor = batch[-1]["fundingTime"] + 1
            if len(batch) < 1000:
                break
    df = pd.DataFrame({"ts": [int(x["fundingTime"]) for x in rows],
                       "rate": [float(x["fundingRate"]) for x in rows]})
    return df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def klines(interval: str, start_ms: int, end_ms: int | None = None,
           symbol: str = SYMBOL, base: str = FAPI) -> pd.DataFrame:
    """Closed perp klines in [start_ms, end_ms), same columns as the archive parser."""
    rows: list[list] = []
    cursor = start_ms
    now = int(time.time() * 1000)
    end_ms = min(end_ms or now, now)
    with httpx.Client(timeout=30) as client:
        while cursor < end_ms:
            r = client.get(f"{base}/fapi/v1/klines",
                           params={"symbol": symbol, "interval": interval, "startTime": cursor,
                                   "endTime": end_ms - 1, "limit": 1500})
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            rows += batch
            cursor = batch[-1][0] + 1
    # Drop the still-forming bar.
    rows = [x for x in rows if x[6] < now]
    df = pd.DataFrame({
        "ts": [int(x[0]) for x in rows],
        "open": [float(x[1]) for x in rows], "high": [float(x[2]) for x in rows],
        "low": [float(x[3]) for x in rows], "close": [float(x[4]) for x in rows],
        "volume": [float(x[5]) for x in rows], "quote_volume": [float(x[7]) for x in rows],
        "taker_buy_volume": [float(x[9]) for x in rows], "count": [int(x[8]) for x in rows],
    })
    return df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
