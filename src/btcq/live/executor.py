"""Live order execution on Binance USDS-M Futures for one strategy book.

Mirrors the paper book's decision order (signal exit / time stop / reverse, then trailing stop,
then entry) but acts the moment a strategy bar closes:

  entry  -> MARKET order (sized from real account equity) -> STOP_MARKET closePosition stop
            (+ TAKE_PROFIT_MARKET if the strategy has a target)
  exit   -> cancel stop/target -> reduce-only MARKET close
  stop   -> fires on the exchange by itself; `reconcile()` (every minute) records it

Safety rails: one-way mode required; cross margin; stop placement failure => immediate close and
pause; entries older than max_entry_delay are skipped (exits are always executed); sizes below
exchange minimums are skipped; a notional sanity cap; unknown or mismatched exchange positions
pause new entries and alert; equity drawdown kill switch.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Callable

from btcq.live.binance_api import BinanceError, BinanceFutures, SymbolRules

log = logging.getLogger("btcq.executor")
SIDE = {1: "BUY", -1: "SELL"}
CLOSE_SIDE = {1: "SELL", -1: "BUY"}
UNKNOWN_ORDER = {-2011, -2013, -20001}


class LiveSetupError(RuntimeError):
    pass


@dataclass
class LivePos:
    dir: int = 0
    qty: str = "0"
    entry_px: float = 0.0
    model_px: float = 0.0
    stop: float = math.nan
    stop_id: str = ""
    tp: float = math.nan
    tp_id: str = ""
    entry_id: str = ""
    entry_time: int = 0
    held: int = 0
    max_bars: int = 0
    risk_usd: float = 0.0
    ext: float = 0.0


@dataclass
class ExecConfig:
    risk_per_trade: float = 0.01
    max_leverage: float = 3.0
    leverage_setting: int = 5
    taker_fee: float = 0.0005
    stop_working_type: str = "CONTRACT_PRICE"
    max_entry_delay_min: int = 10
    max_drawdown_pause: float = 0.35
    reconcile_every_s: int = 55
    order_check_every_s: int = 300


def now_ms() -> int:
    return int(time.time() * 1000)


class LiveExecutor:
    def __init__(self, name: str, api: BinanceFutures, symbols: list[str], cfg: ExecConfig,
                 on_event: Callable[[dict], None] | None = None, clock: Callable[[], int] = now_ms):
        self.name, self.api, self.symbols, self.cfg = name, api, list(symbols), cfg
        self.on_event = on_event or (lambda e: None)
        self.clock = clock
        self.tag = name.lower().replace(" ", "")[:4]
        self.pos = {s: LivePos() for s in symbols}
        self.rules: dict[str, SymbolRules] = {}
        self.paused, self.paused_reason = False, ""
        self.equity = math.nan
        self.peak = math.nan
        self.last_reconcile = 0
        self.last_order_check = 0
        self.alerted: set[str] = set()

    # ------------------------------------------------------------ helpers

    def _emit(self, kind: str, **data) -> None:
        self.on_event({"book": self.name, "kind": kind, "ts": self.clock(), **data})

    def _cid(self, symbol: str, ts_ms: int, kind: str) -> str:
        return f"{self.tag}-{symbol}-{ts_ms // 1000}-{kind}"[:36]

    def pause(self, reason: str) -> None:
        if not self.paused:
            self.paused, self.paused_reason = True, reason
            self._emit("live_paused", reason=reason)

    def resume(self) -> None:
        self.paused, self.paused_reason = False, ""
        if not math.isnan(self.equity):
            self.peak = self.equity
        self.alerted.clear()
        self._emit("live_resumed", equity=self.equity)

    async def _refresh_equity(self) -> float:
        acct = await self.api.account()
        self.equity = float(acct["totalMarginBalance"])
        self.peak = self.equity if math.isnan(self.peak) else max(self.peak, self.equity)
        return self.equity

    # ------------------------------------------------------------ startup

    async def startup(self) -> None:
        await self.api.sync_time()
        self.rules = await self.api.rules(self.symbols)
        if await self.api.hedge_mode():
            raise LiveSetupError("Binance Futures is in Hedge Mode; switch the account to One-way Mode first.")
        for s in self.symbols:
            await self.api.set_cross_margin(s)
            await self.api.set_leverage(s, self.cfg.leverage_setting)
        await self._refresh_equity()
        await self.reconcile(force=True)
        self._emit("live_ready", equity=self.equity,
                   positions={s: p.dir for s, p in self.pos.items() if p.dir})

    # ------------------------------------------------------------ decisions

    def on_minute(self, symbol: str, high: float, low: float) -> None:
        p = self.pos[symbol]
        if p.dir == 1:
            p.ext = max(p.ext, high)
        elif p.dir == -1:
            p.ext = min(p.ext, low)

    async def on_decision(self, symbol: str, bar_end_ms: int, d: dict, ref_price: float) -> None:
        p = self.pos[symbol]
        if p.dir:
            p.held += 1
            reason = None
            if (p.dir == 1 and d["exit_long"]) or (p.dir == -1 and d["exit_short"]):
                reason = "exit_signal"
            elif p.max_bars > 0 and p.held >= p.max_bars:
                reason = "time"
            elif d["entry"] == -p.dir:
                reason = "reverse"
            if reason:
                await self.close(symbol, reason, bar_end_ms)
            elif not math.isnan(d["trail"]):
                new = p.ext - d["trail"] if p.dir == 1 else p.ext + d["trail"]
                if (new - p.stop) * p.dir > 0:
                    await self._move_stop(symbol, new, bar_end_ms)
        p = self.pos[symbol]
        if p.dir == 0 and d["entry"]:
            if self.paused:
                self._emit("live_skipped", symbol=symbol, dir=d["entry"], reason=f"paused: {self.paused_reason}")
                return
            age_min = (self.clock() - bar_end_ms) / 60_000
            if age_min > self.cfg.max_entry_delay_min:
                self._emit("live_skipped", symbol=symbol, dir=d["entry"],
                           reason=f"signal is {age_min:.0f} min old (limit {self.cfg.max_entry_delay_min})")
                return
            await self.open(symbol, d, ref_price, bar_end_ms)

    # ------------------------------------------------------------ entry / exit

    async def open(self, symbol: str, d: dict, ref: float, ts: int) -> None:
        side, stop = d["entry"], d["stop"]
        rules = self.rules[symbol]
        if math.isnan(stop) or (stop - ref) * side >= 0:
            self._emit("live_skipped", symbol=symbol, dir=side, reason="stop is on the wrong side of price")
            return
        try:
            eq = await self._refresh_equity()
        except BinanceError as e:
            self._emit("live_error", symbol=symbol, msg=f"entry skipped, account unavailable: {e}")
            return
        cap = self.cfg.max_leverage * eq / len(self.symbols)
        dist = abs(ref - stop)
        want = min(self.cfg.risk_per_trade * eq / dist, cap / ref) if math.isnan(d["size"]) \
            else min(d["size"] * eq / ref, cap / ref)
        qty = rules.qty(want)
        if qty < rules.min_qty or qty * Decimal(repr(ref)) < rules.min_notional:
            self._emit("live_skipped", symbol=symbol, dir=side,
                       reason=f"size {qty} below exchange minimum (min qty {rules.min_qty}, "
                              f"min notional {rules.min_notional})")
            return
        if float(qty) * ref > cap * 1.02:
            self.pause(f"sanity check failed: {symbol} notional {float(qty) * ref:,.0f} > cap {cap:,.0f}")
            return
        cid = self._cid(symbol, ts, "e")
        r = await self._market(symbol, SIDE[side], qty, cid, reduce_only=False)
        if r is None:
            return
        filled, avg = Decimal(r["executedQty"]), float(r["avgPrice"])
        if filled <= 0:
            self._emit("live_error", symbol=symbol, msg=f"entry order {cid} not filled: {r.get('status')}")
            return
        p = LivePos(dir=side, qty=str(filled), entry_px=avg, model_px=ref, entry_id=cid, entry_time=self.clock(),
                    max_bars=d["max_bars"], risk_usd=float(filled) * abs(avg - stop), ext=avg)
        self.pos[symbol] = p
        if not await self._place_stop(symbol, stop, ts):
            return
        if not math.isnan(d["target"]):
            await self._place_target(symbol, d["target"], ts)
        self._emit("live_entry", symbol=symbol, dir=side, price=avg, model_price=ref, qty=float(filled),
                   stop=p.stop, notional=float(filled) * avg, risk=p.risk_usd, equity=eq,
                   slippage_bps=(avg / ref - 1) * 1e4 * side)

    async def _market(self, symbol: str, side: str, qty: Decimal, cid: str, reduce_only: bool) -> dict | None:
        try:
            return await self.api.market_order(symbol, side, qty, cid, reduce_only=reduce_only)
        except BinanceError as e:
            if e.code == -1:  # network: the order may still have gone through
                try:
                    o = await self.api.get_order(symbol, cid)
                    if o.get("status") in ("FILLED", "PARTIALLY_FILLED"):
                        return o
                except BinanceError:
                    pass
            self._emit("live_error", symbol=symbol, msg=f"{side} {qty} {symbol} failed: {e}")
            return None

    async def _place_stop(self, symbol: str, stop: float, ts: int, suffix: str = "s") -> bool:
        p = self.pos[symbol]
        trig = self.rules[symbol].price(stop)
        for attempt in range(2):
            cid = self._cid(symbol, ts, suffix + ("" if attempt == 0 else "2"))
            try:
                await self.api.close_all_conditional(symbol, CLOSE_SIDE[p.dir], "STOP_MARKET", trig, cid,
                                                     self.cfg.stop_working_type)
                p.stop, p.stop_id = float(trig), cid
                return True
            except BinanceError as e:
                if e.code == -2021:  # would immediately trigger: price is already through the stop
                    await self.close(symbol, "stop", ts, cancel=False)
                    return False
                last = e
        self._emit("live_error", symbol=symbol, msg=f"could not place stop ({last}); closing position for safety")
        await self.close(symbol, "no_stop", ts, cancel=False)
        self.pause(f"stop placement failed on {symbol}")
        return False

    async def _place_target(self, symbol: str, target: float, ts: int) -> None:
        p = self.pos[symbol]
        cid = self._cid(symbol, ts, "t")
        try:
            await self.api.close_all_conditional(symbol, CLOSE_SIDE[p.dir], "TAKE_PROFIT_MARKET",
                                                 self.rules[symbol].price(target), cid, self.cfg.stop_working_type)
            p.tp, p.tp_id = target, cid
        except BinanceError as e:
            self._emit("live_error", symbol=symbol, msg=f"take-profit not placed: {e} (stop is active)")

    async def _move_stop(self, symbol: str, new_stop: float, ts: int) -> None:
        p = self.pos[symbol]
        old = p.stop_id
        if await self._place_stop(symbol, new_stop, ts, suffix="m"):
            await self._cancel(symbol, old)
            self._emit("live_stop_moved", symbol=symbol, stop=self.pos[symbol].stop)

    async def _cancel(self, symbol: str, cid: str) -> None:
        if not cid:
            return
        try:
            await self.api.cancel_algo(cid)
        except BinanceError as e:
            if e.code not in UNKNOWN_ORDER:
                self._emit("live_error", symbol=symbol, msg=f"cancel {cid} failed: {e}")

    async def close(self, symbol: str, reason: str, ts: int, cancel: bool = True) -> None:
        p = self.pos[symbol]
        if not p.dir:
            return
        if cancel:
            await self._cancel(symbol, p.stop_id)
            await self._cancel(symbol, p.tp_id)
        try:
            amt = (await self.api.positions()).get(symbol, 0.0)
        except BinanceError as e:
            self._emit("live_error", symbol=symbol, msg=f"close {reason}: positions unavailable: {e}")
            return
        if amt == 0.0:
            await self._resolve_external(symbol)
            return
        qty = self.rules[symbol].qty(abs(amt))
        r = await self._market(symbol, CLOSE_SIDE[p.dir], qty, self._cid(symbol, ts, "x"), reduce_only=True)
        if r is None:
            self.pause(f"could not close {symbol}; check the exchange manually")
            return
        self._record_exit(symbol, float(r["avgPrice"]), reason)

    def _record_exit(self, symbol: str, price: float, reason: str) -> None:
        p = self.pos[symbol]
        q = float(p.qty)
        gross = q * (price - p.entry_px) * p.dir
        fees = q * (p.entry_px + price) * self.cfg.taker_fee
        net = gross - fees
        self.pos[symbol] = LivePos()
        self._emit("live_exit", symbol=symbol, dir=p.dir, entry_px=p.entry_px, exit_px=price, qty=q,
                   entry_time=p.entry_time, exit_time=self.clock(), gross=gross, fees_est=fees, net=net,
                   R=net / p.risk_usd if p.risk_usd else 0.0, reason=reason, equity=self.equity)

    async def _resolve_external(self, symbol: str) -> None:
        """The exchange position is gone while we still hold it: a stop/target fired (or a manual close)."""
        p = self.pos[symbol]
        price, reason = math.nan, "external"
        for cid, why in ((p.stop_id, "stop"), (p.tp_id, "target")):
            if not cid:
                continue
            try:
                o = await self.api.get_algo(cid)
            except BinanceError:
                continue
            if float(o.get("actualPrice") or 0) > 0:
                price, reason = float(o["actualPrice"]), why
                break
        if math.isnan(price):
            try:
                trades = await self.api.user_trades(symbol, p.entry_time)
                closing = [t for t in trades if t["side"] == CLOSE_SIDE[p.dir]]
                if closing:
                    price = float(closing[-1]["price"])
            except BinanceError:
                pass
        for cid in (p.stop_id, p.tp_id):
            await self._cancel(symbol, cid)
        self._record_exit(symbol, price if not math.isnan(price) else p.entry_px, reason)

    # ------------------------------------------------------------ reconciliation

    async def maybe_reconcile(self) -> None:
        if self.clock() - self.last_reconcile >= self.cfg.reconcile_every_s * 1000:
            await self.reconcile()

    async def reconcile(self, force: bool = False) -> None:
        self.last_reconcile = self.clock()
        try:
            exch = await self.api.positions()
        except BinanceError as e:
            self._emit("live_warning", msg=f"reconcile skipped: {e}")
            return
        for s in self.symbols:
            p, amt = self.pos[s], exch.get(s, 0.0)
            if p.dir and amt == 0.0:
                await self._resolve_external(s)
            elif not p.dir and amt != 0.0:
                if s not in self.alerted:
                    self.alerted.add(s)
                    self._emit("live_error", symbol=s, msg=f"exchange has an unknown {s} position ({amt}); "
                                                           "not touching it. Close it manually or /resume.")
                self.pause(f"unknown position on {s}")
            elif p.dir and (amt > 0) != (p.dir > 0):
                self.pause(f"{s} position direction mismatch (bot {p.dir}, exchange {amt})")
            elif p.dir:
                step = self.rules[s].step if s in self.rules else Decimal("0")
                if abs(Decimal(repr(abs(amt))) - Decimal(p.qty)) > step / 2:
                    self._emit("live_warning", msg=f"{s} size differs: bot {p.qty}, exchange {abs(amt)}; adopting")
                    p.qty = str(Decimal(repr(abs(amt))))
        if force or self.clock() - self.last_order_check >= self.cfg.order_check_every_s * 1000:
            self.last_order_check = self.clock()
            try:
                open_ids = {o.get("clientAlgoId") for o in await self.api.open_algo_orders()}
            except BinanceError as e:
                self._emit("live_warning", msg=f"open-order check skipped: {e}")
                open_ids = None
            if open_ids is not None:
                for s, p in self.pos.items():
                    if p.dir and p.stop_id not in open_ids and exch.get(s, 0.0) != 0.0:
                        self._emit("live_warning", msg=f"{s} stop {p.stop_id or '(none)'} missing; re-placing")
                        await self._place_stop(s, p.stop, self.clock(), suffix="r")
        try:
            await self._refresh_equity()
        except BinanceError:
            return
        if self.peak > 0 and not self.paused and 1 - self.equity / self.peak >= self.cfg.max_drawdown_pause:
            self.pause(f"account drawdown {1 - self.equity / self.peak:.1%} >= {self.cfg.max_drawdown_pause:.0%}")

    # ------------------------------------------------------------ persistence

    def state(self) -> dict:
        return {"pos": {s: asdict(p) for s, p in self.pos.items()}, "paused": self.paused,
                "paused_reason": self.paused_reason, "peak": self.peak}

    def restore(self, st: dict) -> None:
        self.pos = {s: LivePos(**st["pos"][s]) if s in st.get("pos", {}) else LivePos() for s in self.symbols}
        self.paused, self.paused_reason = st.get("paused", False), st.get("paused_reason", "")
        self.peak = st.get("peak", math.nan) if st.get("peak") is not None else math.nan
