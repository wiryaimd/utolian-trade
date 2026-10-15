"""Bot application: routes trader events to SQLite + Telegram, answers commands, runs live/replay."""
from __future__ import annotations

import asyncio
import html
import logging
import math
from datetime import datetime, timezone

from btcq.config import MINUTE_MS, TF_MS
from btcq.live.bars import Minute
from btcq.live.config import BotConfig
from btcq.live.db import StateDB
from btcq.live.feed import now_ms
from btcq.live.trader import Trader

log = logging.getLogger("btcq.app")
SIDE = {1: "LONG", -1: "SHORT"}
SNAP_MS = 15 * 60_000
HELP = ("<b>Commands</b>\n/status: equity, drawdown, open positions\n/positions: open positions in detail\n"
        "/trades [n]: last n closed trades (default 10)\n/pause: stop opening new positions\n"
        "/resume: allow new positions again\n/close SYMBOL: paper-close a position now\n"
        "/compare: F1 vs long-only shadow\n/help: this list")


def _t(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _px(x: float) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:,.6g}"


def _dur(ms: int) -> str:
    h = ms / 3_600_000
    return f"{h / 24:.1f}d" if h >= 48 else f"{h:.0f}h"


class App:
    def __init__(self, cfg: BotConfig, feed, notifier, db: StateDB, notify_books=("F1",),
                 books=None, live: bool = False, executor=None):
        self.cfg, self.feed, self.notifier, self.db = cfg, feed, notifier, db
        self.live = live
        self.executor = executor
        # With real orders, the paper F1 book becomes the quiet "model"; live fills are what you hear about.
        self.notify_books = set() if executor else set(notify_books)
        self.trader = Trader(cfg, on_event=self._on_event, books=books)
        self.outbox: list[str] = []
        self.day_start_eq: dict[str, float] = {}
        self.stale_alerted = False
        self.pending_decisions: list[tuple] = []
        self.snap_slot: int | None = None
        if executor:
            executor.on_event = self._on_event
            self.trader.decision_hook = self._on_decision

    def _on_decision(self, book: str, symbol: str, bar_end: int, d: dict, close: float) -> None:
        if book == self.executor.name:
            self.pending_decisions.append((symbol, bar_end, d, close))

    # ------------------------------------------------------------ startup / persistence

    def _cutoff(self, symbol: str, st: dict | None, start_ms: int) -> int:
        tf_ms = TF_MS[self.trader.tf]
        if st:
            partial = st.get("builders", {}).get(symbol, {}).get("partial")
            if partial:
                return int(partial[0])
            last = st.get("last_minute", {}).get(symbol)
            if last is not None:
                return (int(last) + MINUTE_MS) // tf_ms * tf_ms
        return start_ms // tf_ms * tf_ms

    async def start(self, start_ms: int) -> None:
        st = self.db.load_state()
        for s in self.cfg.symbols:
            hist = await self.feed.history(s, self._cutoff(s, st, start_ms), self.trader.tf,
                                           self.cfg.history_bars)
            self.trader.bootstrap(s, hist)
        if st:
            self.trader.restore(st)
            if self.executor and st.get("live"):
                self.executor.restore(st["live"])
            log.info("restored state; last minutes %s", {s: _t(v) for s, v in self.trader.last_minute.items()})
        for name, b in self.trader.books.items():
            self.day_start_eq.setdefault(name, b.equity())

    def _meta_config(self) -> dict:
        c = self.cfg
        return {"mode": c.mode, "env": c.binance_env, "symbols": c.symbols, "initial_equity": c.initial_equity,
                "risk_per_trade": c.risk_per_trade, "max_leverage": c.max_leverage,
                "max_drawdown_pause": c.max_drawdown_pause, "books": list(self.trader.books),
                "live_book": self.executor.name if self.executor else None}

    def first_minute(self, symbol: str, start_ms: int) -> int:
        last = self.trader.last_minute.get(symbol)
        if last is not None:
            return last + MINUTE_MS
        return start_ms // TF_MS[self.trader.tf] * TF_MS[self.trader.tf]

    def save(self) -> None:
        st = self.trader.state()
        if self.executor:
            st["live"] = self.executor.state()
        self.db.save_state(st)

    # ------------------------------------------------------------ processing

    async def process(self, symbol: str, m: Minute, funding_rate: float) -> None:
        self.trader.process_minute(symbol, m, funding_rate)
        if self.executor:
            self.executor.on_minute(symbol, m.high, m.low)
            while self.pending_decisions:
                sym, end, d, close = self.pending_decisions.pop(0)
                await self.executor.on_decision(sym, end, d, close)
            await self.executor.maybe_reconcile()
        self._snapshot(m.ts)
        await self.flush()

    def _snapshot(self, ts: int) -> None:
        """Equity snapshot every 15 minutes of market time (for the dashboard)."""
        slot = ts // SNAP_MS
        if slot == self.snap_slot:
            return
        self.snap_slot = slot
        t = slot * SNAP_MS
        for name, b in self.trader.books.items():
            self.db.add_snapshot(name, t, b.equity(), b.drawdown(), sum(1 for p in b.pos.values() if p.dir))
        ex = self.executor
        if ex and not math.isnan(ex.equity):
            dd = 1 - ex.equity / ex.peak if ex.peak and not math.isnan(ex.peak) else 0.0
            self.db.add_snapshot(f"{ex.name}-live", t, ex.equity, dd, sum(1 for p in ex.pos.values() if p.dir))

    async def flush(self) -> None:
        while self.outbox:
            await self.notifier.send(self.outbox.pop(0))

    async def catch_up(self, until_ms: int, start_ms: int | None = None) -> int:
        """Process every closed minute up to until_ms (exclusive) for all symbols, in time order."""
        batches = []
        for s in self.cfg.symbols:
            a = self.first_minute(s, start_ms if start_ms is not None else until_ms)
            mins = await self.feed.minutes(s, a, until_ms)
            fund = await self.feed.funding_map(s, a, until_ms)
            batches += [(m.ts, s, m, fund.get(m.ts, 0.0)) for m in mins]
        batches.sort(key=lambda x: (x[0], x[1]))
        for _, s, m, r in batches:
            await self.process(s, m, r)
        return len(batches)

    # ------------------------------------------------------------ events -> db / messages

    def _on_event(self, e: dict) -> None:
        kind, book = e["kind"], e.get("book", "")
        if kind == "exit":
            self.db.add_trade(e)
        if kind == "live_exit":
            self.db.add_trade({**e, "book": f"{book}-live", "fees": e["fees_est"], "funding": 0.0,
                               "init_risk": abs(e["net"] / e["R"]) if e["R"] else 0.0, "eq_before": e["equity"]})
        if kind in ("entry", "exit", "paused", "resumed", "skipped") or kind.startswith("live_"):
            self.db.add_event(int(e.get("ts", 0) or 0), book, kind,
                              {k: v for k, v in e.items() if k not in ("kind", "book")})
        if kind == "day_end":
            self._daily(e)
            return
        if book not in self.notify_books and kind in ("entry", "exit", "skipped"):
            return
        msg = self._format(e)
        if msg:
            self.outbox.append(msg)

    def _format(self, e: dict) -> str | None:
        kind = e["kind"]
        tag = f"[{html.escape(e.get('book', ''))}{' paper' if self.cfg.mode == 'paper' else ''}]"
        if kind == "entry":
            stop_pct = abs(e["price"] - e["stop"]) / e["price"]
            return (f"{tag} <b>{SIDE[e['dir']]} {e['symbol']}</b> @ {_px(e['price'])}\n"
                    f"stop {_px(e['stop'])} ({stop_pct:.1%} away) · risk ${e['risk']:,.0f} "
                    f"({e['risk'] / e['equity']:.2%} of equity)\n"
                    f"size {e['qty']:.6g} · notional ${e['notional']:,.0f} · {_t(e['ts'])} UTC")
        if kind == "exit":
            mark = "✅" if e["net"] > 0 else "❌"
            return (f"{tag} {mark} closed <b>{SIDE[e['dir']]} {e['symbol']}</b> @ {_px(e['exit_px'])}\n"
                    f"<b>{e['R']:+.2f} R</b> (${e['net']:+,.0f}) · {e['reason'].replace('_', ' ')} · held "
                    f"{_dur(e['exit_time'] - e['entry_time'])}\nequity ${e['equity']:,.0f}")
        if kind == "paused":
            return f"{tag} ⏸ <b>new entries paused</b>: {html.escape(e['reason'])}. Open stops stay active. /resume to continue."
        if kind == "resumed":
            return f"{tag} ▶️ entries resumed, equity ${e['equity']:,.0f}"
        if kind == "skipped":
            return f"{tag} skipped {SIDE[e['dir']]} {e['symbol']} (paused: {html.escape(e['reason'])})"
        if kind.startswith("live_"):
            return self._format_live(e)
        return None

    def _format_live(self, e: dict) -> str | None:
        kind = e["kind"]
        env = "LIVE" if self.cfg.binance_env == "mainnet" else "TESTNET"
        tag = f"[{html.escape(e.get('book', ''))} {env}]"
        if kind == "live_entry":
            return (f"🟢 {tag} <b>{SIDE[e['dir']]} {e['symbol']}</b> filled @ {_px(e['price'])} "
                    f"(model {_px(e['model_price'])}, slippage {e['slippage_bps']:+.1f} bp)\n"
                    f"stop {_px(e['stop'])} · risk ${e['risk']:,.0f} ({e['risk'] / e['equity']:.2%} of equity) · "
                    f"notional ${e['notional']:,.0f}")
        if kind == "live_exit":
            mark = "✅" if e["net"] > 0 else "❌"
            return (f"{mark} {tag} closed <b>{SIDE[e['dir']]} {e['symbol']}</b> @ {_px(e['exit_px'])}: "
                    f"<b>{e['R']:+.2f} R</b> (${e['net']:+,.0f} est.) · {e['reason'].replace('_', ' ')} · held "
                    f"{_dur(e['exit_time'] - e['entry_time'])}")
        if kind == "live_skipped":
            return f"{tag} skipped {SIDE[e['dir']]} {e['symbol']}: {html.escape(e['reason'])}"
        if kind == "live_error":
            return f"🚨 {tag} {html.escape(e.get('symbol', ''))} {html.escape(e['msg'])}"
        if kind == "live_warning":
            return f"⚠️ {tag} {html.escape(e['msg'])}"
        if kind == "live_paused":
            return f"⏸ {tag} <b>new entries paused</b>: {html.escape(e['reason'])}. Stops stay active. /resume to continue."
        if kind == "live_resumed":
            return f"▶️ {tag} entries resumed"
        if kind == "live_ready":
            pos = ", ".join(f"{SIDE[d]} {s}" for s, d in e["positions"].items()) or "none"
            return f"🔌 {tag} connected to Binance: equity ${e['equity']:,.2f}, open positions: {pos}"
        if kind == "live_stop_moved":
            return f"{tag} {e['symbol']} stop moved to {_px(e['stop'])}"
        return None

    def _daily(self, e: dict) -> None:
        day = _t(e["day_ms"])[:10]
        snap = self.trader.snapshot()
        lines = [f"📊 <b>Daily summary {day}</b> (UTC)"]
        if self.executor:
            ex = self.executor
            pos = ", ".join(f"{SIDE[p.dir]} {s}" for s, p in ex.pos.items() if p.dir) or "none"
            dd = 1 - ex.equity / ex.peak if ex.peak and not math.isnan(ex.peak) else 0.0
            lines.append(f"<b>{html.escape(ex.name)} account ({self.cfg.binance_env})</b>: ${ex.equity:,.2f}, "
                         f"drawdown {dd:.1%}{' · PAUSED' if ex.paused else ''}\n  open: {pos}")
        for name, s in snap.items():
            start = self.day_start_eq.get(name, s["equity"])
            self.db.set_equity(name, e["day_ms"], s["equity"])
            total = s["equity"] / self.cfg.initial_equity - 1
            day_ret = s["equity"] / start - 1 if start else 0.0
            n_today = sum(1 for t in self.trader.books[name].trades if t["exit_time"] // TF_MS["1d"] * TF_MS["1d"]
                          == e["day_ms"])
            pos = ", ".join(f"{SIDE[p['dir']]} {p['symbol']} {p['R_open']:+.2f}R" for p in s["positions"]) or "none"
            label = name + (" (paper model)" if self.executor and name == self.executor.name
                            else " (shadow)" if name not in self.notify_books else "")
            lines.append(f"<b>{html.escape(label)}</b>: ${s['equity']:,.0f} ({day_ret:+.2%} day, {total:+.1%} total), "
                         f"drawdown {s['drawdown']:.1%}{' · PAUSED' if s['paused'] else ''}\n"
                         f"  open: {pos} · closed today: {n_today}")
            self.day_start_eq[name] = s["equity"]
        self.outbox.append("\n".join(lines))

    # ------------------------------------------------------------ commands

    async def command(self, text: str) -> str | None:
        parts = text.split()
        cmd = parts[0].split("@")[0].lower()
        snap = self.trader.snapshot()
        main = next(iter(self.trader.books))
        if cmd in ("/start", "/help"):
            return HELP
        if cmd == "/status":
            last = max(self.trader.last_minute.values()) if self.trader.last_minute else 0
            out = [f"<b>Status</b> · data up to {_t(last + MINUTE_MS) if last else '–'} UTC · mode {self.cfg.mode}"
                   f" · {self.cfg.binance_env}"]
            if self.executor:
                ex = self.executor
                out.append(f"<b>{html.escape(ex.name)} live account</b>: ${ex.equity:,.2f}, open "
                           f"{sum(1 for p in ex.pos.values() if p.dir)}"
                           f"{' · PAUSED: ' + html.escape(ex.paused_reason) if ex.paused else ''}")
            for name, s in snap.items():
                out.append(f"<b>{html.escape(name)}</b>: ${s['equity']:,.0f} ({s['equity'] / self.cfg.initial_equity - 1:+.1%}), "
                           f"drawdown {s['drawdown']:.1%}, open {len(s['positions'])}, closed trades {s['n_trades']}"
                           f"{' · PAUSED: ' + html.escape(s['paused_reason']) if s['paused'] else ''}")
            return "\n".join(out)
        if cmd == "/positions" and self.executor:
            ps = [(s, p) for s, p in self.executor.pos.items() if p.dir]
            if not ps:
                return "live: no open positions"
            return "\n".join(f"<b>{SIDE[p.dir]} {s}</b> since {_t(p.entry_time)}: qty {p.qty}, entry "
                             f"{_px(p.entry_px)}, stop {_px(p.stop)}" for s, p in ps)
        if cmd == "/positions":
            ps = snap[main]["positions"]
            if not ps:
                return f"{main}: no open positions"
            return "\n".join(f"<b>{SIDE[p['dir']]} {p['symbol']}</b> since {_t(p['since'])}: entry {_px(p['entry_px'])}, "
                             f"stop {_px(p['stop'])}, last {_px(p['last'])}, {p['R_open']:+.2f} R (${p['upnl']:+,.0f})"
                             for p in ps)
        if cmd == "/trades":
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 10
            ts = self.db.trades(f"{main}-live" if self.executor else main, n)
            if not ts:
                return "no closed trades yet"
            return "\n".join(f"{_t(t['exit_time'])} {SIDE[t['dir']]} {t['symbol']} {t['R']:+.2f}R "
                             f"(${t['net']:+,.0f}) {t['reason']}" for t in ts)
        if cmd == "/pause":
            for b in self.trader.books.values():
                if not b.paused:
                    b.pause("paused by /pause")
            if self.executor:
                self.executor.pause("paused by /pause")
            self.save()
            await self.flush()
            return None
        if cmd == "/resume":
            for b in self.trader.books.values():
                b.resume()
            if self.executor:
                self.executor.resume()
            self.save()
            await self.flush()
            return None
        if cmd == "/close":
            if len(parts) < 2:
                return "usage: /close SYMBOL"
            sym = parts[1].upper()
            if self.executor:
                if sym not in self.executor.pos or not self.executor.pos[sym].dir:
                    return f"no open live position in {sym}"
                await self.executor.close(sym, "manual", now_ms())
                self.save()
                await self.flush()
                return None
            b = self.trader.books[main]
            if sym not in b.pos or not b.pos[sym].dir:
                return f"no open {main} position in {sym}"
            ts = self.trader.last_minute.get(sym, 0) + MINUTE_MS
            b.close_manual(sym, ts, b.last_close[sym])
            self.save()
            await self.flush()
            return None
        if cmd == "/compare":
            out = ["<b>Book comparison</b>"]
            for name, b in self.trader.books.items():
                rs = [t["R"] for t in b.trades]
                avg = sum(rs) / len(rs) if rs else 0.0
                out.append(f"{html.escape(name)}: {len(rs)} trades, avg {avg:+.3f} R, equity ${b.equity():,.0f}")
            return "\n".join(out)
        return "unknown command, try /help"

    # ------------------------------------------------------------ loops

    async def watchdog(self) -> None:
        while True:
            await asyncio.sleep(60)
            if not self.trader.last_minute:
                continue
            lag_min = (now_ms() - (max(self.trader.last_minute.values()) + 2 * MINUTE_MS)) / MINUTE_MS
            if lag_min > self.cfg.stale_after_min and not self.stale_alerted:
                self.stale_alerted = True
                await self.notifier.send(f"⚠️ no market data for {lag_min:.0f} min; reconnecting/backfilling")
            elif lag_min <= self.cfg.stale_after_min and self.stale_alerted:
                self.stale_alerted = False
                await self.notifier.send("✅ market data flowing again")

    def heartbeat(self) -> None:
        (self.cfg.state_dir / "heartbeat").write_text(str(now_ms()))

    async def run_live(self) -> None:
        self.db.set_meta("source", "live")
        self.db.set_meta("config", self._meta_config())
        start = now_ms()
        await self.start(start)
        if self.executor:
            await self.executor.startup()
            await self.flush()
        closed_until = now_ms() // MINUTE_MS * MINUTE_MS
        n = await self.catch_up(closed_until, start_ms=start)
        self.save()
        snap = self.trader.snapshot()
        await self.notifier.send(
            f"🟢 <b>Bot started</b> ({self.cfg.mode}) · {len(self.cfg.symbols)} markets · caught up {n} minutes\n"
            + "\n".join(f"{html.escape(k)}: ${v['equity']:,.0f}, open {len(v['positions'])}" for k, v in snap.items()))
        tasks = [asyncio.create_task(self.notifier.poll_commands(self.command)),
                 asyncio.create_task(self.watchdog())]
        try:
            async for kind, s, m in self.feed.stream():
                if kind == "reconnected":
                    await self.catch_up(now_ms() // MINUTE_MS * MINUTE_MS)
                    self.save()
                    continue
                last = self.trader.last_minute.get(s)
                if last is not None and m.ts > last + MINUTE_MS:
                    await self.catch_up(m.ts)  # backfill the gap (all symbols) from REST first
                await self.process(s, m, self.feed.live_funding(s, m.ts))
                self.save()
                self.heartbeat()
        finally:
            for t in tasks:
                t.cancel()

    async def run_replay(self, start_ms: int, end_ms: int, save_every: int = 0) -> None:
        self.db.set_meta("source", "replay")
        self.db.set_meta("config", self._meta_config())
        await self.start(start_ms)
        starts = {s: self.first_minute(s, start_ms) for s in self.cfg.symbols}
        for i, (s, m, rate) in enumerate(self.feed.merged(starts, end_ms)):
            await self.process(s, m, rate)
            if save_every and i % save_every == 0:
                self.save()
        self.save()
