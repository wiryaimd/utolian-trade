"""D1 shadow paper book: the Round 5 daily long-only Donchian ensemble, run forward on live data.

Paper only, information only (D1 failed Round 5's dev gates; see research/PREREGISTRATION.md, "D1 shadow paper
book"). It replicates `research.round5.d1_run` minute by minute:
- 5 coins x 9 lookbacks = 45 sub-sleeves, each starting with 1/45 of the paper equity and its own cash;
- the daily close is the last 1m close of the UTC day; at that close every sub-sleeve's state is recomputed with
  `strategies.r5_d1` (the code the backtest ran) on the coin's full daily history;
- decisions execute as taker orders at the open of the first minute at or after 00:00 + latency (00:01);
- level signals: a flat sub-sleeve whose state is long enters, sized min(0.25 / sigma, 2) x its own cash; a long
  sub-sleeve whose state is flat exits; no stops. Real funding is charged on open positions;
- R uses the trailing level at entry, floored at 0.1% of the entry price (`r5_d1.apply_r`);
- a UTC day with no prices is filled with the last close, as the research's daily calendar does
  (`store.stitch_daily`): it counts in later days' lookbacks and volatility, but it is never traded on, because
  the research engine only acts on days that have prices.

D1 is never paused: the registered rule has no kill switch, so /pause, /resume and the drawdown pause act on
the F1 books only, and D1's peak (its drawdown baseline) is never reset.

It exposes the PaperBook interface the trader, app and dashboard use (equity, pos, trades, state, ...).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from btcq.config import MINUTE_MS, TF_MS
from btcq.live.book import BookConfig, Position
from btcq.strategies import r5_d1 as D

DAY_MS = TF_MS["1d"]
LOOKBACKS = tuple(int(L) for L in D.LOOKBACKS)
R_FLOOR = 1e-3


def key(symbol: str, L: int) -> str:
    return f"{symbol}|{L}"


@dataclass
class SubPos:
    qty: float = 0.0
    epx: float = 0.0
    e_time: int = 0
    e_fee: float = 0.0
    e_fund: float = 0.0
    e_eq: float = 0.0
    e_risk: float = 0.0


def decide(closes: np.ndarray) -> dict[str, dict]:
    """Every lookback's decision at the last close of a coin's daily history: state (1 long / 0 flat), size
    (NaN until sigma is ready) and the trailing level (for R)."""
    c = np.asarray(closes, np.float64)
    size = D.vol_size(c)
    ready = np.isfinite(size)
    out = {}
    for L in LOOKBACKS:
        state, _, trail, _ = D.donchian_state(c, L, ready)
        out[str(L)] = {"state": int(state[-1]) if len(c) else 0, "size": float(size[-1]) if len(c) else math.nan,
                       "trail": float(trail[-1]) if len(c) else math.nan}
    return out


def ew_index(start_close: dict, started: int | None, by_day: dict[int, dict], last_close: dict,
             now_ms: int | None) -> list[list]:
    """Equal-weight hold of the coins, rebalanced at every daily close, from D1's first minute (growth 1.0) to
    now: [[ts, growth], ...]. by_day maps a UTC day to its coins' daily closes; days that closed before D1
    started are ignored. This is the benchmark the pre-registered alpha test uses."""
    if started is None:
        return []
    ok = lambda v: isinstance(v, (int, float)) and math.isfinite(v) and v > 0  # noqa: E731
    prev = {s: float(v) for s, v in start_close.items() if ok(v)}
    out, idx = [[int(started), 1.0]], 1.0
    for d in sorted(k for k in by_day if k + DAY_MS > started):
        cur = {s: float(v) for s, v in by_day[d].items() if ok(v)}
        rs = [cur[s] / prev[s] - 1 for s in cur if s in prev]
        if rs:
            idx *= 1 + sum(rs) / len(rs)
        out.append([d + DAY_MS, idx])
        prev.update(cur)
    if now_ms is not None and now_ms > out[-1][0]:
        rs = [float(v) / prev[s] - 1 for s, v in last_close.items() if s in prev and ok(v)]
        out.append([int(now_ms), idx * (1 + sum(rs) / len(rs)) if rs else idx])
    return out


def order_count(trades: list[dict], subs: list[dict], since_ms: int) -> int:
    """Netted orders at or after since_ms: one per coin per day with any sub-sleeve entry or exit (as the
    research counts them)."""
    marks = ([(t["symbol"], t["entry_time"]) for t in trades] + [(t["symbol"], t["exit_time"]) for t in trades]
             + [(x["symbol"], x["since"]) for x in subs])
    return len({(s, ts // DAY_MS) for s, ts in marks if ts is not None and ts >= since_ms})


def worst_drawdown(equity: list[float], start: float) -> float:
    """Largest fall from a running peak (which starts at the starting equity)."""
    peak, worst = start, 0.0
    for v in equity:
        peak = max(peak, v)
        if peak > 0:
            worst = max(worst, 1.0 - v / peak)
    return worst


def restore_problem(st: dict, symbols: list[str], last_minute: dict) -> str | None:
    """Why a saved D1 state must not be continued (None if it can be): a different coin list, or the bot ran
    on without D1 (D1 was switched off), so D1's record has a hole it cannot fill."""
    if "symbols" in st and list(st["symbols"]) != list(symbols):
        return f"its saved coins ({', '.join(st['symbols'])}) differ from the registered ones"
    seen = st.get("last_ts")
    if seen is not None and any(v is not None for v in seen.values()):      # (a D1 that never ran has no record)
        for s in symbols:
            if last_minute.get(s) is not None and seen.get(s) != last_minute.get(s):
                return "the bot kept running while D1 was switched off, so its record has a gap"
    return None


class D1Book:
    """45 independent long-only sub-sleeves on daily decisions (see module docstring)."""

    kind = "d1"
    pausable = False

    def __init__(self, name: str, symbols: list[str], cfg: BookConfig,
                 on_event: Callable[[dict], None] | None = None):
        self.name = name
        self.symbols = list(symbols)
        self.cfg = cfg
        self.on_event = on_event or (lambda e: None)
        self.slots = len(self.symbols) * len(LOOKBACKS)
        self.cash = {key(s, L): cfg.initial_equity / self.slots for s in self.symbols for L in LOOKBACKS}
        self.subs = {k: SubPos() for k in self.cash}
        self.queue: dict[str, list] = {s: [] for s in self.symbols}      # [exec_after_ms, decisions]
        self.last_close = {s: math.nan for s in self.symbols}
        self.closes: dict[str, list[list]] = {s: [] for s in self.symbols}  # [day_ms, close], closed days
        self.partial: dict[str, list | None] = {s: None for s in self.symbols}  # [day_ms, close] of today
        self.start_close = {s: math.nan for s in self.symbols}
        self.last_ts: dict[str, int | None] = {s: None for s in self.symbols}
        self.started: int | None = None                  # D1's first minute: the start of its record
        self.peak = cfg.initial_equity
        self.worst_dd = 0.0                              # deepest drawdown seen on any minute (never goes down)
        self.paused = False                              # never set: D1 has no kill switch (see docstring)
        self.paused_reason = ""
        self.trades: list[dict] = []

    # ------------------------------------------------------------ accounting

    def sub_equity(self, k: str) -> float:
        sp, s = self.subs[k], k.split("|")[0]
        eq = self.cash[k]
        if sp.qty and not math.isnan(self.last_close[s]):
            eq += sp.qty * (self.last_close[s] - sp.epx)
        return eq

    def equity(self) -> float:
        return sum(self.sub_equity(k) for k in self.cash)

    def drawdown(self) -> float:
        return 1.0 - self.equity() / self.peak if self.peak > 0 else 0.0

    @property
    def pos(self) -> dict[str, Position]:
        """Per-coin aggregate of the open sub-sleeves, shaped like PaperBook.pos (for snapshots and counts)."""
        out = {}
        for s in self.symbols:
            held = [self.subs[key(s, L)] for L in LOOKBACKS if self.subs[key(s, L)].qty > 0]
            if not held:
                out[s] = Position()
                continue
            qty = sum(p.qty for p in held)
            out[s] = Position(dir=1, qty=qty, epx=sum(p.qty * p.epx for p in held) / qty, sl=math.nan,
                              e_time=min(p.e_time for p in held), e_risk=sum(p.e_risk for p in held))
        return out

    def sub_snapshot(self) -> list[dict]:
        out = []
        for s in self.symbols:
            for L in LOOKBACKS:
                sp = self.subs[key(s, L)]
                if sp.qty > 0:
                    last = self.last_close[s]
                    upnl = sp.qty * (last - sp.epx) if not math.isnan(last) else 0.0
                    out.append({"symbol": s, "lookback": L, "entry_px": sp.epx, "qty": sp.qty, "last": last,
                                "notional": sp.qty * last if not math.isnan(last) else None, "upnl": upnl,
                                "R_open": upnl / sp.e_risk if sp.e_risk else 0.0, "since": sp.e_time,
                                "sub_equity": self.sub_equity(key(s, L))})
        return out

    def _emit(self, kind: str, **data) -> None:
        self.on_event({"book": self.name, "kind": kind, **data})

    # ------------------------------------------------------------ inputs

    def bootstrap(self, symbol: str, daily) -> None:
        """Closed daily bars (ts = day open, close) that precede the first minute to be processed. Missing days
        are forward-filled, as the research calendar does."""
        if symbol not in self.closes or daily is None or not len(daily):
            return
        hist: list[list] = []
        for t, c in zip(daily["ts"], daily["close"]):
            t = int(t)
            while hist and t - hist[-1][0] > DAY_MS:
                hist.append([hist[-1][0] + DAY_MS, hist[-1][1]])
            if not hist or t > hist[-1][0]:
                hist.append([t, float(c)])
        self.closes[symbol] = hist

    def schedule(self, symbol: str, bar_end_ms: int, decision: dict) -> None:   # 4h runners never target D1
        raise TypeError("D1 makes its own daily decisions")

    def on_minute(self, symbol: str, ts: int, o: float, h: float, l: float, c: float,
                  funding_rate: float = 0.0) -> None:
        if symbol not in self.queue:               # not one of D1's coins
            return
        day = ts // DAY_MS * DAY_MS
        p = self.partial[symbol]
        if p is not None and p[0] != day:          # the day's last minute never came: close it now
            self._close_day(symbol)
        hist = self.closes[symbol]
        if hist and day - hist[-1][0] > DAY_MS:    # whole days without prices: fill, never trade (see docstring)
            n = 0
            while day - hist[-1][0] > DAY_MS:
                bar = [hist[-1][0] + DAY_MS, hist[-1][1]]
                hist.append(bar)
                self._emit("daily_close", symbol=symbol, day_ms=bar[0], close=bar[1], ts=bar[0] + DAY_MS)
                n += 1
            self._emit("warning", symbol=symbol, ts=ts,
                       msg=f"D1: {symbol} had no prices for {n} day(s); filled with the last close and not traded, "
                           f"as the research calendar does" + (" (the research allowed at most 2)" if n > 2 else ""))
        if funding_rate:
            for L in LOOKBACKS:
                k = key(symbol, L)
                sp = self.subs[k]
                if sp.qty > 0:
                    f = sp.qty * o * funding_rate
                    self.cash[k] -= f
                    sp.e_fund += f
        q = self.queue[symbol]
        while q and ts >= q[0][0]:
            _, d = q.pop(0)
            self._execute(symbol, ts, o, d)
        self.partial[symbol] = [day, c]
        if ts + MINUTE_MS == day + DAY_MS:
            self._close_day(symbol)
        if math.isnan(self.start_close[symbol]):
            self.start_close[symbol] = c
        if self.started is None:
            self.started = ts
        self.last_close[symbol] = c
        self.last_ts[symbol] = ts
        self._check_drawdown()

    # ------------------------------------------------------------ mechanics

    def _close_day(self, symbol: str) -> None:
        day, close = self.partial[symbol]
        self.partial[symbol] = None
        hist = self.closes[symbol]
        if hist and hist[-1][0] >= day:
            return
        hist.append([day, close])
        self._emit("daily_close", symbol=symbol, day_ms=day, close=close, ts=day + DAY_MS)
        d = decide(np.array([x[1] for x in hist]))
        self.queue[symbol].append([day + DAY_MS + self.cfg.latency_min * MINUTE_MS, d])

    def _execute(self, symbol: str, ts: int, o: float, d: dict) -> None:
        slip = self.cfg.slippage.get(symbol, 0.0)
        for L in LOOKBACKS:
            k = key(symbol, L)
            x = d[str(L)]
            if self.subs[k].qty > 0 and x["state"] != 1:
                self._exit(symbol, L, ts, o * (1.0 - slip), "exit_signal")
            if self.subs[k].qty == 0 and x["state"] == 1 and math.isfinite(x["size"]):
                self._enter(symbol, L, ts, o * (1.0 + slip), x)

    def _enter(self, symbol: str, L: int, ts: int, px: float, x: dict) -> None:
        k = key(symbol, L)
        cash = self.cash[k]
        if cash <= 0:
            return
        qty = min(x["size"] * cash / px, D.LEV_CAP * cash / px)
        if qty <= 0:
            return
        fee = qty * px * self.cfg.taker_fee
        dist = px - x["trail"]
        if not math.isfinite(dist) or dist < R_FLOOR * px:
            dist = R_FLOOR * px
        self.cash[k] = cash - fee
        self.subs[k] = SubPos(qty=qty, epx=px, e_time=ts, e_fee=fee, e_eq=cash, e_risk=qty * dist)
        self._emit("entry", symbol=symbol, lookback=L, ts=ts, dir=1, price=px, stop=math.nan, target=math.nan,
                   qty=qty, notional=qty * px, risk=qty * dist, equity=self.equity(), size=x["size"],
                   exposure=qty * px / cash)

    def _exit(self, symbol: str, L: int, ts: int, px: float, reason: str) -> dict:
        k = key(symbol, L)
        sp = self.subs[k]
        gross = sp.qty * (px - sp.epx)
        fee = sp.qty * px * self.cfg.taker_fee
        self.cash[k] += gross - fee
        net = gross - sp.e_fee - fee - sp.e_fund
        t = {"book": self.name, "symbol": symbol, "lookback": L, "dir": 1, "entry_time": sp.e_time, "exit_time": ts,
             "entry_px": sp.epx, "exit_px": px, "qty": sp.qty, "init_risk": sp.e_risk, "gross": gross,
             "fees": sp.e_fee + fee, "funding": sp.e_fund, "net": net, "R": net / sp.e_risk if sp.e_risk else 0.0,
             "reason": reason, "eq_before": sp.e_eq}
        self.trades.append(t)
        self.subs[k] = SubPos()
        self._emit("exit", **t, equity=self.equity())
        return t

    def close_manual(self, symbol: str, ts: int, price: float) -> dict | None:
        return None                                  # /close acts on the main book only

    def _check_drawdown(self) -> None:
        self.peak = max(self.peak, self.equity())      # tracked for the drawdown figures; never pauses D1
        self.worst_dd = max(self.worst_dd, self.drawdown())

    # ------------------------------------------------------------ persistence

    def state(self) -> dict:
        return {"kind": self.kind, "symbols": self.symbols, "cash": self.cash,
                "subs": {k: asdict(v) for k, v in self.subs.items()}, "queue": self.queue,
                "last_close": self.last_close, "partial": self.partial, "start_close": self.start_close,
                "last_ts": self.last_ts, "started": self.started, "peak": self.peak, "worst_dd": self.worst_dd}

    def restore(self, st: dict) -> None:
        if "symbols" in st and list(st["symbols"]) != self.symbols:
            raise ValueError(f"D1 state is for {st['symbols']}, this book trades {self.symbols}")
        self.cash.update({k: float(v) for k, v in st.get("cash", {}).items() if k in self.cash})
        self.subs.update({k: SubPos(**v) for k, v in st.get("subs", {}).items() if k in self.subs})
        self.queue = {s: [list(x) for x in st.get("queue", {}).get(s, [])] for s in self.symbols}
        self.last_close = {s: float(st.get("last_close", {}).get(s, math.nan)) for s in self.symbols}
        self.partial = {s: st.get("partial", {}).get(s) for s in self.symbols}
        self.start_close = {s: float(st.get("start_close", {}).get(s, math.nan)) for s in self.symbols}
        lt = st.get("last_ts") or {}
        self.last_ts = {s: (int(lt[s]) if lt.get(s) is not None else None) for s in self.symbols}
        self.started = int(st["started"]) if st.get("started") is not None else None
        self.peak = float(st.get("peak", self.peak))
        self.worst_dd = float(st.get("worst_dd", 0.0))
