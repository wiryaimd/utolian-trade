"""Wires bars, signals and paper books together. Feed-agnostic: replay and live share it."""
from __future__ import annotations

import logging
import math
from typing import Callable

import pandas as pd

from btcq.config import TF_MS
from btcq.live.bars import BarBuilder, Minute, SignalRunner
from btcq.live.book import BookConfig, PaperBook
from btcq.live.config import BotConfig
from btcq.live.d1 import D1Book
from btcq.strategies.r3_trend import S1Frozen, S1FrozenLong

F1_PARAMS = {"n": 40, "k": 3.5, "exit": "channel"}
log = logging.getLogger("btcq.trader")


def default_books(cfg: BotConfig) -> list[tuple[str, object, dict]]:
    books = [("F1", S1Frozen(), F1_PARAMS)]
    if cfg.shadow_long_only:
        books.append(("F1-long", S1FrozenLong(), F1_PARAMS))
    return books


class Trader:
    def __init__(self, cfg: BotConfig, on_event: Callable[[dict], None] | None = None,
                 books: list[tuple[str, object, dict]] | None = None):
        self.cfg = cfg
        self.on_event = on_event or (lambda e: None)
        spec = books if books is not None else default_books(cfg)
        tfs = {s.tf for _, s, _ in spec}
        if len(tfs) != 1:
            raise ValueError("all books must use one strategy timeframe")
        self.tf = tfs.pop()
        bc = BookConfig(risk_per_trade=cfg.risk_per_trade, max_leverage=cfg.max_leverage,
                        initial_equity=cfg.initial_equity, taker_fee=cfg.taker_fee,
                        slippage={s: cfg.slippage(s) for s in cfg.symbols}, latency_min=cfg.latency_min,
                        max_drawdown_pause=cfg.max_drawdown_pause)
        self.books = {name: PaperBook(name, cfg.symbols, bc, self._event) for name, _, _ in spec}
        # Round 5 D1 daily ensemble, paper shadow: its own daily decisions, its registered coins only
        if cfg.d1_shadow and set(cfg.d1_symbols) <= set(cfg.symbols):
            self.books["D1"] = D1Book("D1", list(cfg.d1_symbols), bc, self._event)
        elif cfg.d1_shadow:
            log.warning("D1 shadow book left off: it needs %s in SYMBOLS", ", ".join(cfg.d1_symbols))
        self.runners = {name: SignalRunner(strat, params) for name, strat, params in spec}
        self.builders: dict[str, BarBuilder] = {}
        # Called as hook(book_name, symbol, bar_end_ms, decision, bar_close) when a bar closes;
        # the live executor uses it to act immediately instead of a minute later.
        self.decision_hook: Callable[[str, str, int, dict, float], None] | None = None
        self.last_minute: dict[str, int] = {}
        self.last_bar_end: dict[str, int] = {}     # per symbol: end of the last strategy bar evaluated
        self.day: int | None = None

    def _event(self, e: dict) -> None:
        self.on_event(e)

    # ------------------------------------------------------------ setup

    def bootstrap(self, symbol: str, history: pd.DataFrame) -> None:
        """Load closed strategy-timeframe bars that precede the first minute to be processed."""
        self.builders[symbol] = BarBuilder(self.tf, history, self.cfg.history_bars)

    def bootstrap_daily(self, symbol: str, daily: pd.DataFrame | None) -> None:
        """Closed daily bars before the D1 book's first minute (it keeps each coin's full daily history)."""
        if "D1" in self.books:
            self.books["D1"].bootstrap(symbol, daily)

    # ------------------------------------------------------------ the pipeline

    def process_minute(self, symbol: str, m: Minute, funding_rate: float = 0.0) -> None:
        if m.ts <= self.last_minute.get(symbol, -1):
            return  # duplicate (e.g. websocket replay after reconnect)
        day = m.ts // TF_MS["1d"]
        if self.day is not None and day > self.day:
            self.on_event({"kind": "day_end", "day_ms": self.day * TF_MS["1d"], "ts": m.ts})
        self.day = day if self.day is None else max(self.day, day)
        b = self.builders[symbol]
        for bar in b.close_if_new_bucket(m.ts):
            self._on_bar_close(symbol, bar)
        for book in self.books.values():
            book.on_minute(symbol, m.ts, m.open, m.high, m.low, m.close, funding_rate)
        for bar in b.add(m):
            self._on_bar_close(symbol, bar)
        self.last_minute[symbol] = m.ts

    def _on_bar_close(self, symbol: str, bar: list) -> None:
        frame = self.builders[symbol].frame()
        end = int(bar[0]) + TF_MS[self.tf]
        self.last_bar_end[symbol] = end
        for name, runner in self.runners.items():
            d = runner.evaluate(frame)
            self.books[name].schedule(symbol, end, d)
            if self.decision_hook:
                self.decision_hook(name, symbol, end, d, float(bar[4]))
            if d["entry"] or d["exit_long"] or d["exit_short"]:
                self.on_event({"kind": "signal", "book": name, "symbol": symbol, "ts": end, **d})

    # ------------------------------------------------------------ commands / views

    def snapshot(self) -> dict:
        out = {}
        for name, bk in self.books.items():
            positions = []
            for s, p in bk.pos.items():
                if p.dir:
                    last = bk.last_close[s]
                    upnl = p.qty * (last - p.epx) * p.dir if not math.isnan(last) else 0.0
                    positions.append({"symbol": s, "dir": p.dir, "entry_px": p.epx, "stop": p.sl, "last": last,
                                      "qty": p.qty, "upnl": upnl, "R_open": upnl / p.e_risk if p.e_risk else 0.0,
                                      "since": p.e_time})
            out[name] = {"equity": bk.equity(), "cash": bk.cash if not isinstance(bk.cash, dict) else sum(bk.cash.values()),
                         "peak": bk.peak, "drawdown": bk.drawdown(), "paused": bk.paused,
                         "paused_reason": bk.paused_reason, "positions": positions, "n_trades": len(bk.trades)}
            if hasattr(bk, "sub_snapshot"):
                out[name]["subs"] = bk.sub_snapshot()
                out[name]["slots"] = bk.slots
        return out

    def state(self) -> dict:
        return {"books": {n: b.state() for n, b in self.books.items()},
                "builders": {s: b.state() for s, b in self.builders.items()},
                "last_minute": self.last_minute, "last_bar_end": self.last_bar_end, "day": self.day}

    def restore(self, st: dict) -> None:
        for n, bst in st.get("books", {}).items():
            if n in self.books:
                try:
                    self.books[n].restore(bst)
                except ValueError as e:          # a D1 state for other coins: D1 starts a new record (app warns)
                    log.warning("%s state not restored: %s", n, e)
        for s, bst in st.get("builders", {}).items():
            if s in self.builders:
                self.builders[s].restore(bst)
        self.last_minute = {s: int(v) for s, v in st.get("last_minute", {}).items()}
        self.last_bar_end = {s: int(v) for s, v in st.get("last_bar_end", {}).items()}
        self.day = st.get("day")
