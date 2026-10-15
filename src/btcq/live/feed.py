"""Market data feeds. ReplayFeed streams the local archive; BinanceFeed is the live source.

Both expose the same async interface used by the app:
  history(symbol, cutoff_ms, tf, n)        -> closed bars with ts < cutoff_ms (last n)
  minutes(symbol, start_ms, end_ms)        -> closed 1m bars in [start_ms, end_ms)
  funding_map(symbol, start_ms, end_ms)    -> {minute_ts: settled rate} for funding in that window
"""
from __future__ import annotations

import asyncio
import heapq
import json
import logging
import time

import numpy as np
import pandas as pd

from btcq.config import HOLDOUT_START, MINUTE_MS, TF_MS
from btcq.data import binance_rest, store
from btcq.live.bars import Minute

log = logging.getLogger("btcq.feed")


def funding_minutes(minute_ts: np.ndarray, f: pd.DataFrame) -> dict[int, float]:
    """Map settlements to the first 1m bar at/after each (minute-rounded) funding time,
    exactly as `backtest.engine.Market` does."""
    if not len(f):
        return {}
    times = (f["ts"].to_numpy(np.int64) + MINUTE_MS // 2) // MINUTE_MS * MINUTE_MS
    idx = np.searchsorted(minute_ts, times)
    out: dict[int, float] = {}
    for i, r in zip(idx, f["rate"].to_numpy(np.float64)):
        if i < len(minute_ts):
            k = int(minute_ts[i])
            out[k] = out.get(k, 0.0) + float(r)
    return out


def _to_minutes(df: pd.DataFrame) -> list[Minute]:
    cols = ["ts", "open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume"]
    return [Minute(int(r[0]), *map(float, r[1:])) for r in df[cols].itertuples(index=False)]


class ReplayFeed:
    """Replays archived 1m bars (dev period by default; holdout only if unlocked)."""

    def __init__(self, symbols: list[str], period: str = "dev"):
        self.m1, self.fund = {}, {}
        for s in symbols:
            self.m1[s] = store.load_1m(period, s)
            self.fund[s] = funding_minutes(self.m1[s]["ts"].to_numpy(np.int64), store.load_funding(period, s))

    async def history(self, symbol: str, cutoff_ms: int, tf: str, n: int) -> pd.DataFrame:
        m = self.m1[symbol]
        bars = store.resample(m[m["ts"] < cutoff_ms], tf)
        return bars[bars["ts"] + TF_MS[tf] <= cutoff_ms].tail(n).reset_index(drop=True)

    async def minutes(self, symbol: str, start_ms: int, end_ms: int) -> list[Minute]:
        m = self.m1[symbol]
        return _to_minutes(m[(m["ts"] >= start_ms) & (m["ts"] < end_ms)])

    async def funding_map(self, symbol: str, start_ms: int, end_ms: int) -> dict[int, float]:
        return {k: v for k, v in self.fund[symbol].items() if start_ms <= k < end_ms}

    def merged(self, start: dict[str, int], end_ms: int):
        """All symbols' minutes in global time order: yields (symbol, Minute, funding_rate)."""
        def keyed(sym: str, ts: np.ndarray):
            for i, t in enumerate(ts):
                yield int(t), sym, i

        rows, iters = {}, []
        for s, m in self.m1.items():
            part = m[(m["ts"] >= start[s]) & (m["ts"] < end_ms)]
            ts = part["ts"].to_numpy(np.int64)
            rows[s] = part[["open", "high", "low", "close", "volume", "quote_volume",
                            "taker_buy_volume"]].to_numpy()
            iters.append(keyed(s, ts))
        for t, s, i in heapq.merge(*iters):
            yield s, Minute(t, *map(float, rows[s][i])), self.fund[s].get(t, 0.0)


class BinanceFeed:
    """Live Binance USDS-M data: REST for history/backfill, websocket for closed 1m klines and
    mark-price updates (whose funding estimate becomes the settled rate at funding time)."""

    def __init__(self, symbols: list[str], env: str = "mainnet"):
        from btcq.live.binance_api import ENVS
        self.symbols = symbols
        self.base, self.ws_url = ENVS[env]["rest"], ENVS[env]["ws"]
        self.next_rate: dict[str, dict[int, float]] = {s: {} for s in symbols}
        self.silence_s = 60.0

    async def history(self, symbol: str, cutoff_ms: int, tf: str, n: int) -> pd.DataFrame:
        start = cutoff_ms - n * TF_MS[tf]
        df = await asyncio.to_thread(binance_rest.klines, tf, start, cutoff_ms, symbol, self.base)
        return df[df["ts"] + TF_MS[tf] <= cutoff_ms].tail(n).reset_index(drop=True)

    async def minutes(self, symbol: str, start_ms: int, end_ms: int) -> list[Minute]:
        if end_ms <= start_ms:
            return []
        df = await asyncio.to_thread(binance_rest.klines, "1m", start_ms, end_ms, symbol, self.base)
        return _to_minutes(df[df["ts"] < end_ms])

    async def funding_map(self, symbol: str, start_ms: int, end_ms: int) -> dict[int, float]:
        if end_ms <= start_ms:
            return {}
        f = await asyncio.to_thread(binance_rest.funding_history, start_ms - MINUTE_MS, end_ms, symbol,
                                    self.base)
        minute_ts = np.arange(start_ms, end_ms, MINUTE_MS, dtype=np.int64)
        return funding_minutes(minute_ts, f)

    def live_funding(self, symbol: str, minute_ts: int) -> float:
        """Settled rate for a funding time = the last estimate streamed for that funding time."""
        return self.next_rate[symbol].pop(minute_ts, 0.0)

    async def stream(self):
        """Yields ("kline", symbol, Minute) for closed 1m klines and ("reconnected", None, None)
        after every reconnect (the caller backfills gaps via REST). Reconnects forever."""
        import websockets  # optional dependency, only needed live

        streams = "/".join(f"{s.lower()}@kline_1m/{s.lower()}@markPrice@1s" for s in self.symbols)
        backoff = 1
        first = True
        while True:
            try:
                async with websockets.connect(self.ws_url + streams, ping_interval=20, ping_timeout=20,
                                              max_size=2**22) as ws:
                    if not first:
                        yield "reconnected", None, None
                    first = False
                    log.info("websocket connected (%d symbols)", len(self.symbols))
                    while True:
                        # Mark price arrives every second, so silence means a dead or mis-routed
                        # connection that is still "open". Drop it; the caller backfills over REST.
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=self.silence_s)
                        except asyncio.TimeoutError:
                            raise ConnectionError(f"no messages for {self.silence_s:g}s") from None
                        backoff = 1
                        msg = json.loads(raw).get("data", {})
                        if msg.get("e") == "kline":
                            k = msg["k"]
                            if k.get("x"):
                                yield "kline", msg["s"], Minute(int(k["t"]), float(k["o"]), float(k["h"]),
                                                               float(k["l"]), float(k["c"]), float(k["v"]),
                                                               float(k["q"]), float(k["V"]))
                        elif msg.get("e") == "markPriceUpdate":
                            nxt = int(msg["T"]) // MINUTE_MS * MINUTE_MS
                            self.next_rate[msg["s"]][nxt] = float(msg["r"])
            except Exception as e:  # noqa: BLE001
                log.warning("websocket error: %s; reconnecting in %ss", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
                first = False  # the next successful connect triggers a REST catch-up


def now_ms() -> int:
    return int(time.time() * 1000)


def holdout_start_ms() -> int:
    return int(HOLDOUT_START.value // 10**6)
