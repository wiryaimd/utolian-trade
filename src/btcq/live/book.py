"""Paper trading book: an incremental, multi-symbol replica of `backtest.engine._simulate`.

Per closed 1m bar and symbol, in this order (same as the backtest kernel):
  1. funding settles on a position held before this minute;
  2. decisions from strategy bars whose execution time has arrived run at this minute's open
     (signal exits, time stops, reversals, trailing-stop updates, then new entries);
  3. stop / target are checked on this minute's range, stop first on ties, gaps fill at the open;
  4. the favourable extreme since entry is updated.

Sizing uses one shared account: risk_per_trade x equity at the stop, with notional capped at
max_leverage x equity / n_symbols per symbol (total notional <= max_leverage x equity).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Callable

REASONS = {0: "stop", 1: "target", 2: "exit_signal", 3: "time", 4: "reverse", 6: "manual"}


@dataclass
class Position:
    dir: int = 0
    qty: float = 0.0
    epx: float = 0.0
    sl: float = 0.0
    tp: float = math.nan
    ext: float = 0.0
    held: int = 0
    max_bars: int = 0
    e_time: int = 0
    e_fee: float = 0.0
    e_fund: float = 0.0
    e_eq: float = 0.0
    e_risk: float = 0.0
    init_sl: float = 0.0


@dataclass
class BookConfig:
    risk_per_trade: float = 0.01
    max_leverage: float = 3.0
    initial_equity: float = 10_000.0
    taker_fee: float = 0.0005
    slippage: dict = field(default_factory=dict)  # per symbol
    latency_min: int = 1
    max_drawdown_pause: float = 1.0               # 1.0 = never pause


class PaperBook:
    def __init__(self, name: str, symbols: list[str], cfg: BookConfig,
                 on_event: Callable[[dict], None] | None = None):
        self.name = name
        self.symbols = list(symbols)
        self.cfg = cfg
        self.on_event = on_event or (lambda e: None)
        self.cash = cfg.initial_equity
        self.pos = {s: Position() for s in symbols}
        self.queue: dict[str, list] = {s: [] for s in symbols}   # [exec_after_ms, decision]
        self.last_close = {s: math.nan for s in symbols}
        self.peak = cfg.initial_equity
        self.paused = False
        self.paused_reason = ""
        self.trades: list[dict] = []

    # ------------------------------------------------------------ accounting

    def equity(self) -> float:
        eq = self.cash
        for s, p in self.pos.items():
            if p.dir and not math.isnan(self.last_close[s]):
                eq += p.qty * (self.last_close[s] - p.epx) * p.dir
        return eq

    def drawdown(self) -> float:
        return 1.0 - self.equity() / self.peak if self.peak > 0 else 0.0

    def _emit(self, kind: str, **data) -> None:
        self.on_event({"book": self.name, "kind": kind, **data})

    # ------------------------------------------------------------ inputs

    def schedule(self, symbol: str, bar_end_ms: int, decision: dict) -> None:
        """A strategy bar closed at bar_end_ms; its decision executes after the latency."""
        self.queue[symbol].append([bar_end_ms + self.cfg.latency_min * 60_000, decision])

    def on_minute(self, symbol: str, ts: int, o: float, h: float, l: float, c: float,
                  funding_rate: float = 0.0) -> None:
        p = self.pos[symbol]
        slip = self.cfg.slippage.get(symbol, 0.0)
        if p.dir and funding_rate:
            f = p.qty * o * funding_rate * p.dir
            self.cash -= f
            p.e_fund += f
        q = self.queue[symbol]
        while q and ts >= q[0][0]:
            _, d = q.pop(0)
            self._decide(symbol, ts, o, d, slip)
        p = self.pos[symbol]
        if p.dir:
            px, reason = math.nan, -1
            if p.dir == 1:
                if l <= p.sl:
                    px, reason = min(o, p.sl), 0
                elif not math.isnan(p.tp) and h >= p.tp:
                    px, reason = max(o, p.tp), 1
            else:
                if h >= p.sl:
                    px, reason = max(o, p.sl), 0
                elif not math.isnan(p.tp) and l <= p.tp:
                    px, reason = min(o, p.tp), 1
            if reason >= 0:
                self._close(symbol, ts, px * (1.0 - slip * p.dir), reason)
            elif p.dir == 1:
                p.ext = max(p.ext, h)
            else:
                p.ext = min(p.ext, l)
        self.last_close[symbol] = c
        self._check_drawdown()

    # ------------------------------------------------------------ mechanics

    def _decide(self, symbol: str, ts: int, o: float, d: dict, slip: float) -> None:
        p = self.pos[symbol]
        if p.dir:
            p.held += 1
            reason = -1
            if (p.dir == 1 and d["exit_long"]) or (p.dir == -1 and d["exit_short"]):
                reason = 2
            elif p.max_bars > 0 and p.held >= p.max_bars:
                reason = 3
            elif d["entry"] == -p.dir:
                reason = 4
            if reason >= 0:
                self._close(symbol, ts, o * (1.0 - slip * p.dir), reason)
            elif not math.isnan(d["trail"]):
                if p.dir == 1:
                    p.sl = max(p.sl, p.ext - d["trail"])
                else:
                    p.sl = min(p.sl, p.ext + d["trail"])
        p = self.pos[symbol]
        if p.dir == 0 and d["entry"] != 0:
            if self.paused:
                self._emit("skipped", symbol=symbol, ts=ts, dir=d["entry"], reason=self.paused_reason)
                return
            self._open(symbol, ts, o, d, slip)

    def _open(self, symbol: str, ts: int, o: float, d: dict, slip: float) -> None:
        side = d["entry"]
        px = o * (1.0 + slip * side)
        s = d["stop"]
        if math.isnan(s) or (s - px) * side >= 0:
            return
        eq = self.equity()
        dist = abs(px - s)
        cap = self.cfg.max_leverage * eq / len(self.symbols) / px
        qty = min(self.cfg.risk_per_trade * eq / dist, cap) if math.isnan(d["size"]) \
            else min(d["size"] * eq / px, cap)
        if qty <= 0:
            return
        fee = qty * px * self.cfg.taker_fee
        self.cash -= fee
        self.pos[symbol] = Position(dir=side, qty=qty, epx=px, sl=s, tp=d["target"], ext=o, held=0,
                                    max_bars=d["max_bars"], e_time=ts, e_fee=fee, e_eq=eq,
                                    e_risk=qty * dist, init_sl=s)
        self._emit("entry", symbol=symbol, ts=ts, dir=side, price=px, stop=s, target=d["target"], qty=qty,
                   notional=qty * px, risk=qty * dist, equity=eq)

    def _close(self, symbol: str, ts: int, px: float, reason: int) -> dict:
        p = self.pos[symbol]
        gross = p.qty * (px - p.epx) * p.dir
        fee = p.qty * px * self.cfg.taker_fee
        self.cash += gross - fee
        net = gross - p.e_fee - fee - p.e_fund
        t = {"book": self.name, "symbol": symbol, "dir": p.dir, "entry_time": p.e_time, "exit_time": ts,
             "entry_px": p.epx, "exit_px": px, "qty": p.qty, "init_risk": p.e_risk, "gross": gross,
             "fees": p.e_fee + fee, "funding": p.e_fund, "net": net, "R": net / p.e_risk if p.e_risk else 0.0,
             "reason": REASONS[reason], "eq_before": p.e_eq}
        self.trades.append(t)
        self.pos[symbol] = Position()
        self._emit("exit", **t, equity=self.equity())
        return t

    def close_manual(self, symbol: str, ts: int, price: float) -> dict | None:
        """Paper-close an open position at `price` (Telegram /close)."""
        p = self.pos[symbol]
        if not p.dir:
            return None
        return self._close(symbol, ts, price * (1.0 - self.cfg.slippage.get(symbol, 0.0) * p.dir), 6)

    def _check_drawdown(self) -> None:
        eq = self.equity()
        if eq > self.peak:
            self.peak = eq
        elif not self.paused and self.drawdown() >= self.cfg.max_drawdown_pause:
            self.pause(f"drawdown {self.drawdown():.1%} >= {self.cfg.max_drawdown_pause:.0%} limit")

    def pause(self, reason: str) -> None:
        self.paused, self.paused_reason = True, reason
        self._emit("paused", reason=reason, equity=self.equity())

    def resume(self) -> None:
        """Resume entries. The drawdown baseline restarts at current equity, so the
        kill switch only fires again on a fresh drawdown of the same size."""
        self.paused, self.paused_reason = False, ""
        self.peak = self.equity()
        self._emit("resumed", equity=self.equity())

    # ------------------------------------------------------------ persistence

    def state(self) -> dict:
        return {"cash": self.cash, "peak": self.peak, "paused": self.paused, "paused_reason": self.paused_reason,
                "pos": {s: asdict(p) for s, p in self.pos.items()}, "queue": self.queue,
                "last_close": self.last_close}

    def restore(self, st: dict) -> None:
        self.cash, self.peak = st["cash"], st["peak"]
        self.paused, self.paused_reason = st["paused"], st["paused_reason"]
        self.pos = {s: Position(**st["pos"][s]) if s in st["pos"] else Position() for s in self.symbols}
        self.queue = {s: [list(x) for x in st["queue"].get(s, [])] for s in self.symbols}
        self.last_close = {s: float(st["last_close"].get(s, math.nan)) for s in self.symbols}
