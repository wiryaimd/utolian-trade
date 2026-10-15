"""Signed Binance USDS-M Futures REST client (async, httpx).

Only the endpoints the bot needs. Conditional orders (stop / take-profit) use the Algo Order
API (POST/GET/DELETE /fapi/v1/algoOrder), which Binance requires for STOP_MARKET and
TAKE_PROFIT_MARKET orders.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import time
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from urllib.parse import urlencode

import httpx

log = logging.getLogger("btcq.binance")

MAINNET = {"rest": "https://fapi.binance.com", "ws": "wss://fstream.binance.com/stream?streams="}
TESTNET = {"rest": "https://testnet.binancefuture.com", "ws": "wss://stream.binancefuture.com/stream?streams="}
ENVS = {"mainnet": MAINNET, "testnet": TESTNET}


class BinanceError(Exception):
    def __init__(self, status: int, code: int, msg: str):
        super().__init__(f"HTTP {status} code {code}: {msg}")
        self.status, self.code, self.msg = status, code, msg


class SymbolRules:
    """Exchange filters for one symbol, with exact decimal rounding."""

    def __init__(self, info: dict):
        f = {x["filterType"]: x for x in info["filters"]}
        self.symbol = info["symbol"]
        self.tick = Decimal(f["PRICE_FILTER"]["tickSize"])
        mkt = f.get("MARKET_LOT_SIZE", f["LOT_SIZE"])
        self.step = Decimal(mkt["stepSize"])
        self.min_qty = Decimal(mkt["minQty"])
        self.max_qty = Decimal(mkt["maxQty"])
        self.min_notional = Decimal(f.get("MIN_NOTIONAL", {}).get("notional", "0"))

    def qty(self, q: float) -> Decimal:
        """Round a quantity DOWN to the lot step (never size up)."""
        return (Decimal(repr(q)) / self.step).to_integral_value(ROUND_DOWN) * self.step

    def price(self, p: float) -> Decimal:
        return (Decimal(repr(p)) / self.tick).to_integral_value(ROUND_HALF_UP) * self.tick

    @staticmethod
    def fmt(d: Decimal) -> str:
        s = format(d.normalize(), "f")
        return s


class BinanceFutures:
    def __init__(self, api_key: str, api_secret: str, env: str = "mainnet",
                 client: httpx.AsyncClient | None = None, recv_window: int = 10_000):
        self.key, self.secret = api_key, api_secret.encode()
        self.base = ENVS[env]["rest"]
        self.client = client or httpx.AsyncClient(timeout=20)
        self.recv_window = recv_window
        self.offset_ms = 0

    # ------------------------------------------------------------ transport

    async def sync_time(self) -> int:
        local = int(time.time() * 1000)
        server = (await self._request("GET", "/fapi/v1/time"))["serverTime"]
        self.offset_ms = int(server) - local
        return self.offset_ms

    def _sign(self, params: dict) -> str:
        params = {k: v for k, v in params.items() if v is not None}
        params["timestamp"] = int(time.time() * 1000) + self.offset_ms
        params["recvWindow"] = self.recv_window
        q = urlencode(params)
        sig = hmac.new(self.secret, q.encode(), hashlib.sha256).hexdigest()
        return f"{q}&signature={sig}"

    async def _request(self, method: str, path: str, params: dict | None = None, signed: bool = False,
                       retries: int = 2):
        """GETs are retried on network errors / 5xx; order-placing calls are not retried here
        (callers check order status by client id instead, to stay idempotent)."""
        attempt = 0
        while True:
            q = self._sign(dict(params or {})) if signed else urlencode(params or {})
            url = f"{self.base}{path}" + (f"?{q}" if q else "")
            try:
                r = await self.client.request(method, url, headers={"X-MBX-APIKEY": self.key} if signed else None)
            except httpx.HTTPError as e:
                if method == "GET" and attempt < retries:
                    attempt += 1
                    await asyncio.sleep(0.5 * attempt)
                    continue
                raise BinanceError(0, -1, f"network error: {e}") from e
            if r.status_code >= 500 and method == "GET" and attempt < retries:
                attempt += 1
                await asyncio.sleep(0.5 * attempt)
                continue
            try:
                body = r.json()
            except ValueError:
                body = {"code": -1, "msg": r.text[:200]}
            if r.status_code >= 400 or (isinstance(body, dict) and int(body.get("code", 0) or 0) < 0):
                code = int(body.get("code", -1)) if isinstance(body, dict) else -1
                if code == -1021 and attempt < retries:   # timestamp outside recvWindow: resync once
                    attempt += 1
                    await self.sync_time()
                    continue
                raise BinanceError(r.status_code, code, body.get("msg", "") if isinstance(body, dict) else str(body))
            return body

    # ------------------------------------------------------------ market / account

    async def exchange_info(self) -> dict:
        return await self._request("GET", "/fapi/v1/exchangeInfo")

    async def rules(self, symbols: list[str]) -> dict[str, SymbolRules]:
        info = await self.exchange_info()
        out = {s["symbol"]: SymbolRules(s) for s in info["symbols"] if s["symbol"] in symbols}
        missing = set(symbols) - set(out)
        if missing:
            raise BinanceError(0, -1, f"symbols not listed on this exchange: {sorted(missing)}")
        return out

    async def account(self) -> dict:
        return await self._request("GET", "/fapi/v3/account", signed=True)

    async def positions(self) -> dict[str, float]:
        """One-way mode position amount per symbol (only non-zero ones are returned)."""
        rows = await self._request("GET", "/fapi/v3/positionRisk", signed=True)
        return {r["symbol"]: float(r["positionAmt"]) for r in rows if float(r["positionAmt"]) != 0.0}

    async def hedge_mode(self) -> bool:
        return bool((await self._request("GET", "/fapi/v1/positionSide/dual", signed=True))["dualSidePosition"])

    async def set_leverage(self, symbol: str, leverage: int) -> dict:
        return await self._request("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage}, signed=True)

    async def set_cross_margin(self, symbol: str) -> None:
        try:
            await self._request("POST", "/fapi/v1/marginType", {"symbol": symbol, "marginType": "CROSSED"},
                                signed=True)
        except BinanceError as e:
            if e.code != -4046:  # "No need to change margin type."
                raise

    async def user_trades(self, symbol: str, start_ms: int) -> list[dict]:
        return await self._request("GET", "/fapi/v1/userTrades", {"symbol": symbol, "startTime": start_ms,
                                                                  "limit": 100}, signed=True)

    # ------------------------------------------------------------ orders

    async def market_order(self, symbol: str, side: str, qty: Decimal, client_id: str,
                           reduce_only: bool = False) -> dict:
        params = {"symbol": symbol, "side": side, "type": "MARKET", "quantity": SymbolRules.fmt(qty),
                  "newClientOrderId": client_id, "newOrderRespType": "RESULT"}
        if reduce_only:
            params["reduceOnly"] = "true"
        return await self._request("POST", "/fapi/v1/order", params, signed=True)

    async def get_order(self, symbol: str, client_id: str) -> dict:
        return await self._request("GET", "/fapi/v1/order", {"symbol": symbol, "origClientOrderId": client_id},
                                   signed=True)

    async def close_all_conditional(self, symbol: str, side: str, order_type: str, trigger: Decimal,
                                    client_id: str, working_type: str = "CONTRACT_PRICE") -> dict:
        """STOP_MARKET / TAKE_PROFIT_MARKET with closePosition=true via the Algo Order API."""
        return await self._request("POST", "/fapi/v1/algoOrder", {
            "algoType": "CONDITIONAL", "symbol": symbol, "side": side, "type": order_type,
            "triggerPrice": SymbolRules.fmt(trigger), "closePosition": "true", "workingType": working_type,
            "clientAlgoId": client_id}, signed=True)

    async def get_algo(self, client_id: str) -> dict:
        return await self._request("GET", "/fapi/v1/algoOrder", {"clientAlgoId": client_id}, signed=True)

    async def cancel_algo(self, client_id: str) -> dict:
        return await self._request("DELETE", "/fapi/v1/algoOrder", {"clientAlgoId": client_id}, signed=True)

    async def open_algo_orders(self) -> list[dict]:
        r = await self._request("GET", "/fapi/v1/openAlgoOrders", signed=True)
        return r if isinstance(r, list) else r.get("orders", [])
