"""System and API health for the dashboard's "System health" tab.

Two kinds of checks, each {"group", "name", "status": ok|watch|bad|wait|off, "value", "note", "kind"}:
- live checks ("kind": "live"), computed on every page refresh without any network call: the bot's saved state, the
  health record the bot writes with every save (websocket, REST, Telegram, Binance account API), the heartbeat file,
  and the server's disk, memory and load;
- full checks ("kind": "full"), run by the "Run full check now" button and automatically every 15 minutes: they call
  Binance's and Telegram's public endpoints from the server and compare the bot's prices, F1's current 4h bar and D1's
  daily closes and positions with Binance's own data.

The dashboard has no API keys and opens the bot's database read-only, so full checks use public endpoints only and
never change anything.
"""
from __future__ import annotations

import asyncio
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

MIN = 60_000
H4 = 4 * 3_600_000
DAY = 86_400_000
SERVER, MARKET, TELEGRAM, ACCOUNT = "server", "market", "telegram", "account"
GROUP_TITLES = {SERVER: "Bot and server", MARKET: "Binance market data", TELEGRAM: "Telegram",
                ACCOUNT: "Binance account (live trading)"}
RANK = {"bad": 3, "watch": 2, "ok": 1}
TELEGRAM_URL = "https://api.telegram.org/"
PROBE_TIMEOUT_S = 6.0
FULL_DEADLINE_S = 20.0           # the dashboard cuts every connection after 30 s


# ------------------------------------------------------------------ formatting

def utc(ms: int | None) -> str:
    if ms is None:
        return "–"
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%d %b %H:%M UTC")


def span(ms: float) -> str:
    s = max(int(ms // 1000), 0)
    if s < 60:
        return f"{s} s"
    m = s // 60
    if m < 60:
        return f"{m} min"
    h = m // 60
    if h < 48:
        return f"{h} h {m % 60} min"
    return f"{h // 24} days"


def ago(ms: float) -> str:
    return "just now" if ms < 1000 else span(ms) + " ago"


def worst(checks: list[dict]) -> str:
    ranked = [c["status"] for c in checks if c["status"] in RANK]
    return max(ranked, key=RANK.get) if ranked else "ok"


def _check(group: str, name: str, status: str, value: str, note: str, kind: str = "live") -> dict:
    return {"group": group, "name": name, "status": status, "value": value, "note": note, "kind": kind}


def _band(x: float, ok_below: float, watch_below: float) -> str:
    return "ok" if x < ok_below else "watch" if x < watch_below else "bad"


# ------------------------------------------------------------------ server facts

def read_heartbeat(state_dir: Path) -> int | None:
    try:
        return int((state_dir / "heartbeat").read_text().strip())
    except (OSError, ValueError):
        return None


def memory_mb() -> tuple[float, float] | None:
    """(available, total) MB from /proc/meminfo; inside Docker this is the whole server."""
    try:
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            info[k] = float(v.split()[0]) / 1024
        return info["MemAvailable"], info["MemTotal"]
    except (OSError, KeyError, ValueError):
        return None


def load_avg() -> tuple[float, float, float] | None:
    try:
        a, b, c = Path("/proc/loadavg").read_text().split()[:3]
        return float(a), float(b), float(c)
    except (OSError, ValueError):
        return None


def disk_free(path: Path) -> tuple[float, float] | None:
    """(free, total) bytes of the file system holding path."""
    try:
        st = os.statvfs(path)
        return st.f_bavail * st.f_frsize, st.f_blocks * st.f_frsize
    except (OSError, AttributeError):          # no statvfs on Windows
        import shutil
        try:
            u = shutil.disk_usage(path)
            return float(u.free), float(u.total)
        except OSError:
            return None


# ------------------------------------------------------------------ live checks

def live_checks(st: dict | None, health: dict | None, meta: dict, source: str, now: int, state_dir: Path,
                db_path: Path, daily_closes: list[tuple], dashboard_code: str) -> list[dict]:
    out: list[dict] = []
    add = lambda *a: out.append(_check(*a))  # noqa: E731
    replay = source == "replay"
    st = st or {}
    lm = {s: int(v) for s, v in st.get("last_minute", {}).items()
          if not meta.get("symbols") or s in meta["symbols"]}    # a market removed from SYMBOLS stays in the state
    last = max(lm.values()) if lm else None
    data_now = last + MIN if last is not None else None
    h = health or {}
    written = h.get("written")
    stale = written is None or now - written > 3 * MIN         # the bot saves after every market's minute

    # -- bot and server
    hb = read_heartbeat(state_dir)
    if replay:
        add(SERVER, "Bot process", "off", "replay database", "This database comes from an archive replay; no bot is "
            "running, so the live checks below are mostly switched off.")
    elif hb is None:
        add(SERVER, "Bot process", "bad", "no heartbeat file", "The bot writes state/heartbeat after every processed "
            "minute. There is none: the bot has not run yet, or the dashboard cannot see its folder.")
    else:
        s = _band(now - hb, 3 * MIN, 10 * MIN)
        add(SERVER, "Bot process", s, f"last heartbeat {ago(now - hb)}",
            "The bot finished a minute recently." if s == "ok" else
            "The bot is late; it may be backfilling after a reconnect. Watch the next minutes." if s == "watch" else
            "The bot looks stopped or stuck. On the VPS: sudo docker compose ps, then sudo docker compose logs bot.")
    if not h:
        add(SERVER, "Bot version", "wait", "no health record yet", "The bot writes a health record with every save; "
            "it appears once a bot with this version has processed a minute.")
    else:
        same = h.get("code") == dashboard_code
        note = ("Bot and dashboard run the same code." if same else
                "The bot and the dashboard run different code. Rebuild both: sudo docker compose up -d --build.")
        if h.get("started"):
            note += (f" Running since {utc(h['started'])} (up {span(now - h['started'])}) in {h.get('mode')} mode, "
                     f"market data from Binance {h.get('env')}.")
        add(SERVER, "Bot version", "ok" if same else "watch",
            f"code {h.get('code')}" + ("" if same else f", dashboard {dashboard_code}"), note)
        pa = h.get("processed_at")
        if pa and last is not None and not replay:
            lag = pa - (last + MIN)
            s = _band(lag, 30_000, 120_000)
            add(SERVER, "Processing speed", s, f"{max(lag, 0) / 1000:.0f} s after the minute closed",
                "Time from the end of a 1-minute bar to the bot finishing it. "
                + ("Normal." if s == "ok" else "Slow: the server may be overloaded."))
    size = sum(p.stat().st_size for p in (db_path, db_path.with_name(db_path.name + "-journal")) if p.exists())
    add(SERVER, "Database", "ok" if size < 1e9 else "watch", f"{size / 1e6:,.1f} MB",
        "The bot's SQLite file (state, trades, events, equity snapshots), opened read-only here."
        + ("" if size < 1e9 else " It is large; the dashboard may get slow."))
    df = disk_free(state_dir)
    if df:
        free, total = df
        share = free / total if total else 0.0
        s = "bad" if free < 0.5e9 or share < 0.03 else "watch" if free < 2e9 or share < 0.10 else "ok"
        add(SERVER, "Disk space", s, f"{free / 1e9:,.1f} GB free of {total / 1e9:,.0f} GB ({share:.0%})",
            "Free space where the bot keeps its database." + ("" if s == "ok" else " Free up space before it runs out."))
    mem = memory_mb()
    if mem:
        avail, total = mem
        s = "ok" if avail >= 200 else "watch" if avail >= 100 else "bad"
        add(SERVER, "Server memory", s, f"{avail:,.0f} MB free of {total:,.0f} MB",
            "Memory still available on the VPS." + ("" if s == "ok" else " Low: the bot could be killed."))
    la = load_avg()
    if la:
        cores = os.cpu_count() or 1
        s = _band(la[1], 0.8 * cores, 1.5 * cores)
        add(SERVER, "Server load", s, f"{la[0]:.2f} / {la[1]:.2f} / {la[2]:.2f} ({cores} core{'s' * (cores > 1)})",
            "Average number of busy processes over 1, 5 and 15 minutes. "
            + ("Fine." if s == "ok" else "The server is busy; trading could lag."))

    # -- Binance market data (as seen by the bot)
    if replay:
        add(MARKET, "Latest prices", "off", f"replay up to {utc(data_now)}", "Archive replay, not live data.")
    elif data_now is None:
        add(MARKET, "Latest prices", "bad", "no data yet", "The bot has not processed any market data yet.")
    else:
        s = _band(now - data_now, 3 * MIN, 10 * MIN)
        behind = [x for x, v in lm.items() if v < last - 2 * MIN]
        if behind and s == "ok":
            s = "watch"
        add(MARKET, "Latest prices", s, f"up to {utc(data_now)} ({ago(now - data_now)})",
            ("Fresh 1-minute bars for every market. " if s == "ok" else
             f"Behind: {', '.join(behind)}. " if behind else
             "Market data is late; the bot backfills gaps from Binance's REST API by itself. ")
            + f"Markets: {', '.join(sorted(lm))}.")
    feed = h.get("feed")
    if replay or not feed:
        add(MARKET, "Websocket stream", "off" if replay else "wait", "–" if replay else "no record yet",
            "Live 1-minute bars and funding rates arrive over Binance's websocket.")
    elif stale:
        add(MARKET, "Websocket stream", "wait", "health record is old",
            f"The bot last wrote its health record {ago(now - written) if written else 'never'}; see Bot process.")
    else:
        silent = written - (feed.get("ws_last_msg") or 0)
        if not feed.get("ws_connected") and feed.get("ws_since") is None and not feed.get("ws_last_error"):
            add(MARKET, "Websocket stream", "wait", "connecting",
                "The bot has just started and is opening its websocket connection.")
        elif feed.get("ws_connected") and silent < 75_000:
            add(MARKET, "Websocket stream", "ok", f"connected for {span(written - feed['ws_since'])}",
                "Binance pushes 1-minute bars and mark prices to the bot as they happen.")
        elif feed.get("ws_connected"):
            add(MARKET, "Websocket stream", "watch", f"connected, silent for {span(silent)}",
                "No message for over a minute; the bot drops a silent connection after 60 s and reconnects.")
        else:
            add(MARKET, "Websocket stream", "bad", f"disconnected since {utc(feed.get('ws_last_error_at'))}",
                f"Last error: {feed.get('ws_last_error') or '–'}. The bot keeps retrying and backfills over REST.")
        drops = [t for t in feed.get("ws_drops", []) if now - t < DAY]
        s = "ok" if len(drops) <= 6 else "watch" if len(drops) <= 24 else "bad"
        add(MARKET, "Websocket drops (24 h)", s, str(len(drops)),
            "Binance closes every connection after 24 h, so 1-2 a day is normal; missed minutes are backfilled."
            + (f" Last error {utc(feed.get('ws_last_error_at'))}: {feed.get('ws_last_error')}."
               if feed.get("ws_last_error") and drops else ""))
        err_at, ok_at = feed.get("rest_last_error_at"), feed.get("rest_last_ok")
        if err_at and now - err_at < 3_600_000 and (ok_at is None or err_at > ok_at):
            add(MARKET, "Binance REST (bot)", "watch", f"last call failed {ago(now - err_at)}",
                f"{feed.get('rest_last_error')}. The bot uses REST at start-up and to backfill gaps; it retries.")
        else:
            add(MARKET, "Binance REST (bot)", "ok", f"last OK {ago(now - ok_at)}" if ok_at else "not used yet",
                "The bot uses Binance's REST API at start-up and to backfill missed minutes.")
        cu = h.get("catch_up")
        add(MARKET, "Last backfill", "ok", f"{cu['minutes']} minute rows at {utc(cu['at'])}" if cu else "none needed",
            "Minutes fetched over REST after a start or a dropped connection (one row per market and minute).")

    # -- Telegram
    tg = h.get("telegram")
    if replay or not tg or not tg.get("configured"):
        add(TELEGRAM, "Telegram bot", "off", "not configured" if not replay else "–",
            "No Telegram token is set, so the bot only logs its messages." if not replay else
            "Archive replay: messages go to the log.")
    elif stale:
        add(TELEGRAM, "Telegram bot", "wait", "health record is old", "See Bot process.")
    else:
        poll = tg.get("last_poll_ok")
        quiet = written - poll if poll else None
        if quiet is None:
            s = "wait" if h.get("started") and written - h["started"] < 3 * MIN else "bad"
        else:
            s = _band(quiet, 3 * MIN, 15 * MIN)
        if s == "ok" and tg.get("poll_fail_streak", 0) >= 3:
            s = "watch"
        perr = tg.get("poll_error_at")
        add(TELEGRAM, "Telegram commands", s, f"last contact {ago(quiet)}" if quiet is not None else "no contact yet",
            "The bot asks Telegram for new commands about once a minute (/status, /d1, /pause ...)."
            + ("" if s in ("ok", "wait") else " It is not getting answers; commands may not arrive.")
            + (f" Last error {utc(perr)}: {tg.get('poll_error')}." if perr and now - perr < DAY else ""))
        sent, serr = tg.get("last_send_ok"), tg.get("send_error_at")
        failing = serr is not None and (sent is None or serr > sent)
        s = ("bad" if failing and (tg.get("send_fail_streak", 0) >= 3 or sent is None or now - sent > DAY)
             else "watch" if failing else "ok")
        add(TELEGRAM, "Telegram messages", s, f"last sent {ago(now - sent)}" if sent else "none sent yet",
            (f"The last message failed ({utc(serr)}): {tg.get('send_error')}." if failing else
             "Messages are going out.") + " Trade alerts and the daily summary go out as messages.")

    # -- Binance account API (live trading only)
    mode = h.get("mode") or meta.get("mode", "paper")
    if mode != "live":
        add(ACCOUNT, "Binance account API", "off", "not used (paper mode)",
            "Paper trading needs no API keys and sends nothing to Binance.")
    else:
        api = h.get("account_api") or {}
        ok_at, err_at = api.get("last_ok"), api.get("last_error_at")
        if stale:
            add(ACCOUNT, "Binance account API", "wait", "health record is old", "See Bot process.")
        else:
            s = "bad" if not ok_at else _band(written - ok_at, 3 * MIN, 10 * MIN)
            newer = err_at is not None and (ok_at is None or err_at > ok_at)
            if s == "ok" and newer and now - err_at < 3_600_000:
                s = "watch"
            add(ACCOUNT, "Binance account API", s, f"last OK {ago(written - ok_at)}" if ok_at else "no successful call",
                "The bot checks positions, stops and balance with your API key about once a minute."
                + (f" Last error {utc(err_at)}: {api.get('last_error')}." if newer else ""))
        lv = st.get("live")
        if lv:
            reason = lv.get("paused_reason") or ""
            if lv.get("paused"):
                add(ACCOUNT, "Live trading", "watch" if reason == "paused by /pause" else "bad", "paused",
                    f"{reason}. Stops stay on Binance; send /resume in Telegram once the cause is fixed.")
            else:
                add(ACCOUNT, "Live trading", "ok", "active", "The live executor may open new positions.")
            open_ = {x: p for x, p in lv.get("pos", {}).items() if p.get("dir")}
            nostop = [x for x, p in open_.items() if not p.get("stop_id")
                      or not (isinstance(p.get("stop"), (int, float)) and math.isfinite(p["stop"]))]
            add(ACCOUNT, "Live positions and stops", "bad" if nostop else "ok", f"{len(open_)} open",
                ("No stop order recorded for " + ", ".join(nostop) + ". The bot re-places missing stops at its next "
                 "check; if this stays, protect or close the position on Binance yourself.") if nostop else
                "Every open live position has a stop order on Binance.")

    # -- strategies
    for name, b in st.get("books", {}).items():
        if b.get("kind") == "d1":
            out += _d1_checks(name, b, lm, data_now, daily_closes)
        else:
            out += _book_checks(name, b, st, lm, data_now, h, now)
    return out


def _book_checks(name: str, b: dict, st: dict, lm: dict, data_now: int | None, h: dict, now: int) -> list[dict]:
    out: list[dict] = []
    add = lambda *a: out.append(_check(name, *a))  # noqa: E731
    lbe = {s: int(v) for s, v in st.get("last_bar_end", {}).items()}
    if data_now is None:
        add("4h decisions on schedule", "wait", "no data yet", "Decisions are made at every 4h bar close.")
    else:
        due = data_now // H4 * H4
        # each market against its own last minute: the bot saves after every market's minute, so at a 4h boundary
        # one market can already be past the close while the next is not yet
        missed = [s for s, t in lm.items() if s in lbe and lbe[s] < (t + MIN) // H4 * H4]
        unknown = [s for s in lm if s not in lbe]
        if missed:
            add("4h decisions on schedule", "bad", f"missed for {', '.join(missed)}",
                f"The 4h bar that closed at {utc(due)} was not evaluated for every market.")
        elif unknown:
            add("4h decisions on schedule", "wait", f"next at {utc(due + H4)}",
                "Recorded from the first 4h bar close after the bot started with this version"
                + (f" ({', '.join(unknown)})." if lbe else "."))
        else:
            add("4h decisions on schedule", "ok", f"last {utc(due)}, next {utc(due + H4)}",
                "Signals are checked at every 4h close (00, 04, 08, 12, 16, 20 UTC) for every market.")
    if b.get("paused"):
        add("New entries", "watch", "paused", f"{b.get('paused_reason')}. Open positions keep their stops; "
            "send /resume in Telegram to allow new entries.")
    else:
        add("New entries", "ok", "allowed", "The book may open new positions.")
    bad, n = [], 0
    for s, p in b.get("pos", {}).items():
        if not p.get("dir"):
            continue
        n += 1
        last, sl = b.get("last_close", {}).get(s), p.get("sl")
        if not (isinstance(sl, (int, float)) and math.isfinite(sl)):
            bad.append(f"{s} has no stop")
        elif isinstance(last, (int, float)) and math.isfinite(last) and (last - sl) * p["dir"] <= 0:
            bad.append(f"{s} last price {last:g} is past its stop {sl:g}")
    add("Open positions and stops", "bad" if bad else "ok", f"{n} open",
        ("; ".join(bad) + ". A processed minute through the stop should have closed it.") if bad else
        "Every open position has a stop on the right side of the last price.")
    out.append(_queue_check(name, b, lm, "Orders waiting"))
    return out


def _queue_check(group: str, b: dict, lm: dict, title: str) -> dict:
    stuck, waiting = [], []
    for s, q in b.get("queue", {}).items():
        for item in q:
            (stuck if s in lm and int(item[0]) <= lm[s] else waiting).append((s, int(item[0])))
    if stuck:
        return _check(group, title, "bad", f"{len(stuck)} overdue",
                      "Orders that should already have filled: " + ", ".join(f"{s} at {utc(t)}" for s, t in stuck))
    if waiting:
        return _check(group, title, "ok", f"{len(waiting)} waiting until {utc(min(t for _, t in waiting))}",
                      "Decisions fill one minute after the bar close, as in the backtest.")
    return _check(group, title, "ok", "none waiting", "Decisions fill one minute after the bar close.")


def _d1_checks(name: str, b: dict, lm: dict, data_now: int | None, daily_closes: list[tuple]) -> list[dict]:
    out: list[dict] = []
    add = lambda *a: out.append(_check(name, *a))  # noqa: E731
    started = b.get("started")
    symbols = b.get("symbols") or sorted({k.split("|")[0] for k in b.get("cash", {})})
    if started is None or data_now is None:
        add("Daily decisions on schedule", "wait", "not started", "D1 starts with the first minute it processes.")
    else:
        due = data_now // DAY * DAY - DAY            # the last UTC day that has closed
        if started >= due + DAY:
            add("Daily decisions on schedule", "wait", f"first at {utc(due + 2 * DAY)}",
                "D1 decides at each 00:00 UTC daily close and fills at 00:01; it started after the last one.")
        else:
            last = {}
            for s, d, _ in daily_closes:
                if int(d) + DAY > started:
                    last[s] = max(last.get(s, 0), int(d))
            own = {s: (lm[s] + MIN) // DAY * DAY - DAY for s in symbols if s in lm}
            missed = [s for s, d in own.items() if started < d + DAY and last.get(s) != d]
            add("Daily decisions on schedule", "bad" if missed else "ok",
                f"missed for {', '.join(missed)}" if missed else
                f"last close {utc(due + DAY)}, next {utc(due + 2 * DAY)}",
                "D1 records each coin's daily close at 00:00 UTC and decides on it." if not missed else
                "A daily close was not recorded for every coin.")
    out.append(_queue_check(name, b, lm, "Orders filled on time"))
    add("Never paused", "bad" if b.get("paused") else "ok", "paused" if b.get("paused") else "running",
        "D1's pre-registered forward test has no pause; /pause and the kill switch act on F1 only.")
    subs = b.get("subs", {})
    n = sum(1 for v in subs.values() if v.get("qty", 0) > 0)
    neg = [k for k, v in b.get("cash", {}).items() if v < 0]
    add("Sub-positions", "bad" if neg else "ok", f"{n} of {len(subs) or 45} open",
        f"Negative cash in {', '.join(neg)}." if neg else "Each of the 45 sub-positions has its own cash, none negative.")
    return out


# ------------------------------------------------------------------ full checks (network)

def http_get_default(url: str, params: dict | None = None, timeout: float = PROBE_TIMEOUT_S):
    """(status, body, milliseconds): body is parsed JSON when possible."""
    import httpx
    for attempt in (1, 2):                   # one retry on fast failures (connection errors, 5xx); never on timeouts
        t0 = time.perf_counter()
        try:
            with httpx.Client(timeout=timeout, follow_redirects=False) as c:
                r = c.get(url, params=params)
        except (httpx.ConnectError, httpx.RemoteProtocolError):
            if attempt == 2:
                raise
            time.sleep(0.5)
            continue
        if r.status_code >= 500 and attempt == 1:
            time.sleep(0.5)
            continue
        ms = (time.perf_counter() - t0) * 1000
        try:
            body = r.json()
        except ValueError:
            body = r.text[:300]
        return r.status_code, body, ms


def ws_probe_default(url: str, timeout: float = PROBE_TIMEOUT_S) -> float:
    """Milliseconds until the first websocket message."""
    async def go() -> float:
        import websockets
        t0 = time.perf_counter()
        async with websockets.connect(url, open_timeout=timeout, close_timeout=2, ping_interval=None,
                                      max_size=2**20) as ws:
            await asyncio.wait_for(ws.recv(), timeout)
        return (time.perf_counter() - t0) * 1000
    return asyncio.run(go())


def _err(e: BaseException) -> str:
    from btcq.live.feed import short_error
    return short_error(e)


class ProbeError(RuntimeError):
    """The service answered with an error. transient: worth a second try (5xx, rate limit)."""

    def __init__(self, msg: str, transient: bool):
        super().__init__(msg)
        self.transient = transient


def _expect_ok(status: int, body) -> None:
    if status != 200:
        msg = body.get("msg") if isinstance(body, dict) else str(body)[:120]
        raise ProbeError(f"HTTP {status}" + (f": {msg}" if msg else "")
                         + (" (Binance blocks this server's location)" if status == 451 else ""),
                         transient=status >= 500 or status in (418, 429))


def warm_up() -> None:
    """Compile D1's numba code once, so the first button press doesn't wait for it."""
    import numpy as np
    from btcq.live.d1 import decide
    decide(np.linspace(1.0, 2.0, 400))


def full_check_names(books: dict) -> list[tuple[str, str]]:
    names = [(MARKET, "Binance REST API"), (MARKET, "Clock vs Binance"), (MARKET, "Markets trading"),
             (MARKET, "Websocket reachable"), (MARKET, "Bot prices match Binance"), (TELEGRAM, "Telegram reachable")]
    for name, b in books.items():
        if b.get("kind") == "d1":
            names += [(name, "Daily closes match Binance"), (name, "Positions follow the rule")]
        else:
            names.append((name, "Current 4h bar matches Binance"))
    return names


def placeholders(books: dict) -> list[dict]:
    return [_check(g, n, "wait", "not checked yet", "Press 'Run full check now', or wait for the automatic run.",
                   "full") for g, n in full_check_names(books)]


def _klines(http_get, rest: str, symbol: str, interval: str, **params) -> list:
    status, body, _ = http_get(f"{rest}/fapi/v1/klines", {"symbol": symbol, "interval": interval, **params})
    _expect_ok(status, body)
    return body


def full_checks(st: dict | None, meta: dict, source: str, daily_closes: list[tuple], rest: str, ws: str,
                http_get: Callable = http_get_default, ws_probe: Callable = ws_probe_default,
                clock: Callable[[], float] = time.time, deadline_s: float = FULL_DEADLINE_S) -> list[dict]:
    st = st or {}
    books = st.get("books", {})
    lm = {s: int(v) for s, v in st.get("last_minute", {}).items()}
    symbols = meta.get("symbols") or sorted(lm)
    replay = source == "replay"
    paper = next((b for b in books.values() if b.get("kind") != "d1"), None) or next(iter(books.values()), {})

    def rest_api():
        status, body, ms = http_get(f"{rest}/fapi/v1/ping")
        _expect_ok(status, body)
        return [_check(MARKET, "Binance REST API", "ok" if ms < 1500 else "watch", f"reachable, {ms:.0f} ms",
                       "Binance's REST API answers from this server" + ("." if ms < 1500 else ", but slowly."), "full")]

    def clock_check():
        t0 = clock() * 1000
        status, body, ms = http_get(f"{rest}/fapi/v1/time")
        _expect_ok(status, body)
        skew = int(body["serverTime"]) - (t0 + ms / 2)
        s = _band(abs(skew), 1000, 5000)
        return [_check(MARKET, "Clock vs Binance", s, f"server clock {-skew:+.0f} ms vs Binance",
                       "Small differences are harmless. Live orders need it under a few seconds.", "full")]

    def markets():
        status, body, _ = http_get(f"{rest}/fapi/v1/exchangeInfo")
        _expect_ok(status, body)
        info = {x["symbol"]: x.get("status") for x in body.get("symbols", [])}
        off = [f"{s} ({info.get(s, 'not listed')})" for s in symbols if info.get(s) != "TRADING"]
        return [_check(MARKET, "Markets trading", "bad" if off else "ok",
                       f"{len(symbols) - len(off)} of {len(symbols)} trading",
                       ("Not trading: " + ", ".join(off) + ".") if off else
                       "Binance lists every market the bot follows as TRADING.", "full")]

    def websocket():
        sym = (symbols or ["BTCUSDT"])[0].lower()
        ms = ws_probe(f"{ws}{sym}@markPrice@1s")
        return [_check(MARKET, "Websocket reachable", "ok", f"first message after {ms:.0f} ms",
                       "A fresh connection to Binance's market stream works from this server.", "full")]

    def prices():
        if replay:
            return [_check(MARKET, "Bot prices match Binance", "off", "replay", "Archive replay: not compared.", "full")]
        mism, same = [], 0
        for s in symbols:
            t, mine = lm.get(s), paper.get("last_close", {}).get(s)
            if t is None or mine is None:
                continue
            k = _klines(http_get, rest, s, "1m", startTime=t, limit=1)
            if not k or int(k[0][0]) != t:
                mism.append(f"{s}: Binance has no bar at {utc(t)}")
            elif abs(float(k[0][4]) - float(mine)) > 1e-9 * abs(float(mine)):
                mism.append(f"{s}: bot {float(mine):g} vs Binance {float(k[0][4]):g}")
            else:
                same += 1
        return [_check(MARKET, "Bot prices match Binance", "bad" if mism else "ok",
                       f"{same} of {len(symbols)} identical",
                       ("; ".join(mism) + ".") if mism else
                       f"The close of the last minute the bot processed ({utc(max(lm.values()) if lm else None)}) "
                       "equals Binance's for every market.", "full")]

    def telegram():
        status, body, ms = http_get(TELEGRAM_URL)
        if status >= 500:
            raise RuntimeError(f"HTTP {status}")
        return [_check(TELEGRAM, "Telegram reachable", "ok", f"reachable, {ms:.0f} ms",
                       "This server can reach Telegram (no token is used for this check).", "full")]

    def bars_4h(names: list[str]):
        if replay:
            return [_check(name, "Current 4h bar matches Binance", "off", "replay", "Archive replay.", "full")
                    for name in names]
        bad, n = [], 0
        for s, bs in st.get("builders", {}).items():
            p = bs.get("partial")
            if not p:
                continue
            k = _klines(http_get, rest, s, "4h", startTime=int(p[0]), limit=1)
            if not k or int(k[0][0]) != int(p[0]):
                bad.append(f"{s}: no Binance bar at {utc(int(p[0]))}")
                continue
            o, hi, lo = float(k[0][1]), float(k[0][2]), float(k[0][3])
            tol = 1e-9 * o
            if abs(o - p[1]) > tol or p[2] > hi + tol or p[3] < lo - tol:
                bad.append(f"{s}: bot open/high/low {p[1]:g}/{p[2]:g}/{p[3]:g} vs Binance {o:g}/{hi:g}/{lo:g}")
            else:
                n += 1
        return [_check(name, "Current 4h bar matches Binance", "bad" if bad else "ok", f"{n} markets checked",
                       ("; ".join(bad) + ".") if bad else
                       "The 4h bar the bot is building has Binance's open, and its high and low are inside Binance's."
                       + (" The 4h books share these bars." if len(names) > 1 else ""), "full") for name in names]

    def d1(name: str, b: dict):
        if replay:
            return [_check(name, n, "off", "replay", "Archive replay.", "full")
                    for n in ("Daily closes match Binance", "Positions follow the rule")]
        import numpy as np
        from btcq.live.d1 import LOOKBACKS, decide, key
        started = b.get("started")
        coins = b.get("symbols") or symbols
        today = int(clock() * 1000) // DAY * DAY
        hist: dict[str, list] = {}
        for s in coins:
            rows, start = [], 0
            while True:                     # every closed daily bar since listing (as the bot's start-up does)
                k = _klines(http_get, rest, s, "1d", startTime=start, endTime=today - 1, limit=1500)
                rows += [(int(x[0]), float(x[4])) for x in k if int(x[0]) + DAY <= today]
                if len(k) < 1500:
                    break
                start = int(k[-1][0]) + 1
            filled: list[tuple[int, float]] = []         # whole days without prices: as D1Book.bootstrap does
            for t, c in rows:
                while filled and t - filled[-1][0] > DAY:
                    filled.append((filled[-1][0] + DAY, filled[-1][1]))
                if not filled or t > filled[-1][0]:
                    filled.append((t, c))
            hist[s] = filled
        mine = {}
        for s, d, c in daily_closes:
            if started is not None and int(d) + DAY > started:
                mine.setdefault(s, {})[int(d)] = float(c)
        diff, compared = [], 0
        for s in coins:
            ref = dict(hist[s])
            for d, c in sorted(mine.get(s, {}).items())[-5:]:
                compared += 1
                if d not in ref or abs(ref[d] - c) > 1e-9 * abs(c):
                    diff.append(f"{s} {utc(d + DAY)[:6]}: bot {c:g} vs Binance {ref.get(d, float('nan')):g}")
        out = [_check(name, "Daily closes match Binance", "bad" if diff else "ok" if compared else "wait",
                      f"{compared} closes compared" if compared else "no daily close yet",
                      ("; ".join(diff) + ".") if diff else
                      "D1's recorded daily closes (last 5 per coin) equal Binance's daily bars." if compared else
                      "D1 records its first daily close at 00:00 UTC.", "full")]
        decided = {s: max(mine[s]) for s in coins if mine.get(s)}
        if len(decided) < len(coins):
            longs = 0
            for s in coins:
                closes = np.array([c for _, c in hist[s]])
                longs += sum(1 for L in LOOKBACKS if decide(closes)[str(L)]["state"] == 1) if len(closes) else 0
            out.append(_check(name, "Positions follow the rule", "wait", "first decision pending",
                              f"On Binance's data the rule says {longs} of {len(coins) * len(LOOKBACKS)} sub-positions "
                              "are in an uptrend now; D1 takes them at its first 00:00 UTC decision.", "full"))
            return out
        wrong, agree = [], 0
        for s in coins:
            closes = np.array([c for d, c in hist[s] if d <= decided[s]])
            dec = decide(closes)
            queued = b.get("queue", {}).get(s, [])
            for L in LOOKBACKS:
                want = dec[str(L)]["state"] == 1
                have = (queued[-1][1][str(L)]["state"] == 1) if queued else \
                    b.get("subs", {}).get(key(s, L), {}).get("qty", 0) > 0
                if want == have:
                    agree += 1
                else:
                    wrong.append(f"{s} {L}d: rule {'long' if want else 'flat'}, D1 {'long' if have else 'flat'}")
        return out + [_check(name, "Positions follow the rule", "bad" if wrong else "ok",
                             f"{agree} of {agree + len(wrong)} agree",
                             ("; ".join(wrong[:6]) + ("…" if len(wrong) > 6 else "") + ".") if wrong else
                             "Re-running D1's rule on Binance's own daily bars gives exactly D1's positions "
                             "(including orders waiting for 00:01).", "full")]

    jobs: list[tuple[list[tuple[str, str]], Callable]] = [
        ([(MARKET, "Binance REST API")], rest_api), ([(MARKET, "Clock vs Binance")], clock_check),
        ([(MARKET, "Markets trading")], markets), ([(MARKET, "Websocket reachable")], websocket),
        ([(MARKET, "Bot prices match Binance")], prices), ([(TELEGRAM, "Telegram reachable")], telegram)]
    four_h = [name for name, b in books.items() if b.get("kind") != "d1"]
    if four_h:
        jobs.append(([(name, "Current 4h bar matches Binance") for name in four_h], lambda: bars_4h(four_h)))
    for name, b in books.items():
        if b.get("kind") == "d1":
            jobs.append(([(name, "Daily closes match Binance"), (name, "Positions follow the rule")],
                         lambda name=name, b=b: d1(name, b)))
    out: list[dict] = []
    pool = ThreadPoolExecutor(max_workers=8)
    futs = {pool.submit(fn): names for names, fn in jobs}
    done, _ = wait(futs, timeout=deadline_s)
    for f, names in futs.items():
        if f in done and f.exception() is None:
            out += f.result()
        else:
            exc = f.exception() if f in done else None
            why = "timed out" if f not in done else _err(exc)
            # a network failure may be a blip: the runner shows a first one as Watch (see FullCheckRunner.run)
            transient = not isinstance(exc, ProbeError) or exc.transient
            out += [{**_check(g, n, "bad", "failed", f"The check could not finish: {why}", "full"),
                     "transient": transient} for g, n in names]
    pool.shutdown(wait=False, cancel_futures=True)
    return out


class FullCheckRunner:
    """Runs the full checks on demand (with a cooldown) and on a timer; keeps the last result in memory."""

    def __init__(self, load: Callable[[], dict], rest: str, ws: str, cooldown_s: float = 30, every_s: float = 900,
                 http_get: Callable = http_get_default, ws_probe: Callable = ws_probe_default,
                 clock: Callable[[], float] = time.time):
        self.load, self.rest, self.ws = load, rest, ws
        self.cooldown_s, self.every_s = cooldown_s, every_s
        self.http_get, self.ws_probe, self.clock = http_get, ws_probe, clock
        self.result: dict | None = None
        self.running = False
        self.last_start = -math.inf
        self._lock = threading.Lock()

    def status(self) -> dict:
        r = self.result
        left = max(0.0, self.cooldown_s - (self.clock() - self.last_start))
        return {"ran_at": r["ran_at"] if r else None, "took_ms": r["took_ms"] if r else None,
                "trigger": r["trigger"] if r else None, "running": self.running,
                "cooldown_s": math.ceil(left), "every_min": round(self.every_s / 60)}

    def checks(self, books: dict) -> list[dict]:
        r = self.result
        return r["checks"] if r else placeholders(books)

    def run(self, trigger: str = "button") -> str | None:
        """Runs one full check now. Returns a notice instead when one is running or ran within the cooldown."""
        with self._lock:
            if self.running:
                return "A full check is already running; its results appear when it finishes."
            left = self.cooldown_s - (self.clock() - self.last_start)
            if trigger == "button" and left > 0:
                return f"A full check ran {self.cooldown_s - left:.0f} s ago; the next one is possible in {math.ceil(left)} s."
            self.running, self.last_start = True, self.clock()
        t0 = self.clock()
        try:
            ctx = self.load()
            checks = full_checks(ctx.get("st"), ctx.get("meta", {}), ctx.get("source", "live"),
                                 ctx.get("daily_closes", []), ctx.get("rest", self.rest), ctx.get("ws", self.ws),
                                 self.http_get, self.ws_probe, self.clock)
        except Exception as e:  # noqa: BLE001 - a broken check must not break the dashboard
            checks = [_check(SERVER, "Full check", "bad", "failed", f"The full check could not run: {_err(e)}", "full")]
        finally:
            with self._lock:
                self.running = False
        before = {(c["group"], c["name"]) for c in (self.result or {}).get("checks", []) if c.get("transient")}
        for c in checks:
            if not c.pop("transient", False):
                continue
            c["transient"] = True              # remembered, so a repeat on the next run stays a problem
            if (c["group"], c["name"]) not in before:
                c["status"] = "watch"
                c["note"] += " First failure: it shows as a problem if the next full check fails too."
        self.result = {"ran_at": int(t0 * 1000), "took_ms": int((self.clock() - t0) * 1000), "trigger": trigger,
                       "checks": checks}
        return None

    def start(self, first_after_s: float = 60) -> threading.Thread:
        def loop():
            try:
                warm_up()
            except Exception:  # noqa: BLE001
                pass
            time.sleep(first_after_s)
            while True:
                try:
                    self.run("auto")
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(self.every_s)
        t = threading.Thread(target=loop, name="full-health-check", daemon=True)
        t.start()
        return t
