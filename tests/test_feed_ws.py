"""Live websocket feed: silent connections are dropped and followed by a REST catch-up signal."""
import asyncio
import json

import websockets

from btcq.live.binance_api import MAINNET
from btcq.live.feed import BinanceFeed

KLINE = {"stream": "btcusdt@kline_1m", "data": {"e": "kline", "s": "BTCUSDT", "k": {
    "t": 1_700_000_000_000, "o": "100", "h": "101", "l": "99", "c": "100.5", "v": "2", "q": "201", "V": "1",
    "x": True}}}


def test_mainnet_uses_market_stream_path():
    assert MAINNET["ws"].startswith("wss://fstream.binance.com/market/stream")


def test_silent_connection_is_dropped_then_reconnect_signals_catch_up():
    async def go():
        conns = []

        async def handler(ws):
            conns.append(1)
            if len(conns) > 1:                       # second connection works
                await ws.send(json.dumps(KLINE))
            await ws.wait_closed()                   # first connection: open but silent

        async with websockets.serve(handler, "127.0.0.1", 0) as srv:
            port = srv.sockets[0].getsockname()[1]
            feed = BinanceFeed(["BTCUSDT"], "testnet")
            feed.ws_url = f"ws://127.0.0.1:{port}/stream?streams="
            feed.silence_s = 0.5
            events = []
            gen = feed.stream()
            async for kind, sym, m in gen:
                events.append(kind)
                if kind == "kline":
                    bar = (sym, m.ts, m.close)
                    break
            await gen.aclose()
        return events, len(conns), bar

    events, n, bar = asyncio.run(asyncio.wait_for(go(), timeout=20))
    assert events == ["reconnected", "kline"] and n == 2
    assert bar == ("BTCUSDT", 1_700_000_000_000, 100.5)
