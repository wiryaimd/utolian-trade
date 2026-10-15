"""Read-only web dashboard for the bot.

    python -m btcq.live dashboard [--db state/bot.sqlite] [--host 127.0.0.1] [--port 8080]

Serves one page (dashboard.html) and a JSON API (/api/data) built from the bot's SQLite file,
opened read-only. HTTP Basic auth is required whenever it listens beyond localhost
(DASHBOARD_USER / DASHBOARD_PASSWORD).
"""
from __future__ import annotations

import base64
import hmac
import http.client
import json
import math
import socket
import sqlite3
import statistics
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from btcq.live.config import BotConfig
from btcq.live.db import StateDB

PAGE = Path(__file__).with_name("dashboard.html")
SIDE = {1: "LONG", -1: "SHORT"}

# What the research says F1 should look like (reports/results/F1.json, 5-market portfolio).
RESEARCH = {
    "trades_per_week": (2.5, 2.9),      # holdout 2024-26 alts / dev 2020-23
    "avg_R": (0.23, 0.33),
    "win_rate": (0.33, 0.38),
    "max_dd_typical": 0.28,             # dev and holdout both saw ~28%
    "max_dd_p95": 0.41,                 # 95th percentile of bootstrapped dev drawdowns
    # Longest losing streak per calendar year in the research ranged 11-23 trades (dev 2020-23 and
    # holdout 2024-26, 5 markets): the coins move together, so chop stops several out at once.
    "losing_streak_normal": 23,
    "losing_streak_bad": 30,
}


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _longest_losing_streak(rs: list[float]) -> tuple[int, int]:
    best = cur = 0
    for r in rs:
        cur = cur + 1 if r <= 0 else 0
        best = max(best, cur)
    return best, cur


def book_stats(trades: list[dict], weeks: float) -> dict:
    rs = [t["R"] for t in sorted(trades, key=lambda t: t["exit_time"])]
    n = len(rs)
    out = {"n": n, "per_week": n / weeks if weeks > 0 else None}
    if n:
        out["avg_R"] = sum(rs) / n
        out["win_rate"] = sum(r > 0 for r in rs) / n
        out["sum_R"] = sum(rs)
        out["se_R"] = statistics.stdev(rs) / math.sqrt(n) if n > 1 else None
        out["longest_losing"], out["current_losing"] = _longest_losing_streak(rs)
        gains, losses = sum(r for r in rs if r > 0), -sum(r for r in rs if r < 0)
        out["pf"] = gains / losses if losses > 0 else None
    return out


def health_checks(main: dict, stats: dict, weeks: float, data_age_min: float | None, is_replay: bool,
                  live: dict | None, kill: float) -> list[dict]:
    """Plain-language checks of the running system against the research. status: ok / watch / bad / wait."""
    R = RESEARCH
    out = []

    def add(name, status, value, expect, note):
        out.append({"name": name, "status": status, "value": value, "expect": expect, "note": note})

    if is_replay:
        add("Data feed", "ok", "replay", "live 1m bars", "This database comes from an archive replay, not live data.")
    elif data_age_min is None:
        add("Data feed", "bad", "no data yet", "< 3 min old", "The bot has not processed any market data yet.")
    else:
        st = "ok" if data_age_min < 3 else "watch" if data_age_min < 10 else "bad"
        add("Data feed", st, f"{data_age_min:.0f} min old", "< 3 min old",
            "Fresh 1-minute bars are arriving." if st == "ok" else
            "Market data is late; the bot backfills automatically, but check the VPS and logs if this persists.")

    paused = main.get("paused")
    add("Trading", "bad" if paused else "ok", "paused" if paused else "active", "active",
        f"New entries are paused: {main.get('paused_reason')}. Stops stay active; /resume in Telegram." if paused
        else "The bot is allowed to open new positions.")

    dd = main.get("drawdown") or 0.0
    st = "ok" if dd < 0.20 else "watch" if dd < 0.30 else "bad"
    add("Drawdown from peak", st, f"{dd:.1%}", f"research max ~{R['max_dd_typical']:.0%}; pause at {kill:.0%}",
        "Within the normal range." if st == "ok" else
        "Deep but still inside what the research saw." if st == "watch" else
        "Beyond the research's typical worst case; the kill switch pauses entries at the limit.")

    n = stats.get("n", 0)
    if weeks >= 4:
        pw = stats.get("per_week") or 0.0
        st = "ok" if 1.5 <= pw <= 4.5 else "watch"
        add("Trades per week", st, f"{pw:.1f}", f"{R['trades_per_week'][0]}–{R['trades_per_week'][1]}",
            "Trading at the expected pace." if st == "ok" else
            "Pace differs from the research; trend strategies are bursty, so judge over several months.")
    else:
        add("Trades per week", "wait", f"{n} trades so far", f"{R['trades_per_week'][0]}–{R['trades_per_week'][1]}",
            "Needs at least 4 weeks of running before the pace means anything.")

    if n >= 30:
        avg, se = stats["avg_R"], stats.get("se_R") or 0.0
        lo = R["avg_R"][0]
        if n >= 100 and avg < 0:
            st, note = "bad", "Average trade is negative after 100+ trades: the edge may be gone. Consider pausing."
        elif avg + 2 * se < lo:
            st, note = "watch", "Below the research even allowing for luck; keep watching before concluding anything."
        else:
            st, note = "ok", "Consistent with the research once normal luck is allowed for."
        add("Average R per trade", st, f"{avg:+.2f} ± {2 * se:.2f}", f"+{R['avg_R'][0]:.2f} to +{R['avg_R'][1]:.2f}",
            note)
        wr = stats["win_rate"]
        st = "ok" if 0.28 <= wr <= 0.45 else "watch"
        add("Win rate", st, f"{wr:.0%}", f"{R['win_rate'][0]:.0%}–{R['win_rate'][1]:.0%}",
            "Most trades lose small and a few winners pay for them; that is how this strategy works."
            if st == "ok" else "Win rate is outside the usual band.")
    else:
        add("Average R per trade", "wait", f"{n} trades", f"+{R['avg_R'][0]:.2f} to +{R['avg_R'][1]:.2f}",
            "Fewer than 30 closed trades: any average is mostly luck at this point.")

    if n and "longest_losing" in stats:
        ls, cur = stats["longest_losing"], stats["current_losing"]
        st = "ok" if ls <= R["losing_streak_normal"] else "watch" if ls <= R["losing_streak_bad"] else "bad"
        add("Losing streak", st, f"longest {ls}, current {cur}",
            f"up to ~{R['losing_streak_normal']} in a row happened every year",
            "Long losing runs are normal here: the 5 coins move together, so one choppy week stops out several "
            "positions at once. The research saw 16–23 in a row in most years." if st == "ok" else
            "Longer than anything the research produced; keep watching.")

    if live is not None:
        slip = live.get("avg_slippage_bps")
        if slip is None:
            add("Live fills vs model", "wait", "no live fills yet", "< 5 bp slippage",
                "Shows how far real entry prices land from the backtest's assumption.")
        else:
            st = "ok" if abs(slip) < 5 else "watch" if abs(slip) < 15 else "bad"
            add("Live fills vs model", st, f"{slip:+.1f} bp avg entry slippage", "< 5 bp",
                "Real fills are close to the model." if st == "ok" else
                "Fills are worse than modelled; costs will eat part of the edge.")
    return out


LOOPBACK = ("127.0.0.1", "localhost", "::1")


def access_check(access: dict | None) -> dict | None:
    """Health item about how the dashboard itself is exposed (shown only when it is public)."""
    if not access or not access.get("public"):
        return None
    n = access.get("password_len", 0)
    if n < 16:
        return {"name": "Dashboard access", "status": "bad", "value": f"public, plain HTTP, {n}-char password",
                "expect": "16+ random characters",
                "note": "Anyone on the internet can reach this login page and the password crosses the network "
                        "unencrypted. Set DASHBOARD_PASSWORD to 16+ random characters (openssl rand -base64 18), "
                        "or switch back to the SSH tunnel."}
    return {"name": "Dashboard access", "status": "watch", "value": "public, plain HTTP", "expect": "HTTPS or tunnel",
            "note": "The password crosses the network unencrypted: avoid logging in on public Wi-Fi, or use the "
                    "SSH tunnel / an HTTPS proxy. Repeated wrong logins are locked out automatically."}


class DashboardData:
    def __init__(self, db_path: Path | str, access: dict | None = None):
        self.db_path = Path(db_path)
        self.access = access
        self.lock = threading.Lock()

    def payload(self) -> dict:
        with self.lock:
            if not self.db_path.exists():
                return {"ready": False, "message": f"No bot database at {self.db_path} yet. Start the bot first."}
            db = StateDB(self.db_path, readonly=True)
            try:
                return _clean(self._build(db))
            finally:
                db.close()

    def _build(self, db: StateDB) -> dict:
        st = db.load_state()
        if not st:
            return {"ready": False, "message": "The bot has not saved any state yet."}
        meta = db.meta("config", {}) or {}
        source = db.meta("source", "live")
        books = meta.get("books") or list(st.get("books", {}))
        cfg = BotConfig(symbols=meta.get("symbols") or sorted({s for b in st["books"].values() for s in b["pos"]}),
                        initial_equity=meta.get("initial_equity", 10_000.0),
                        shadow_long_only="F1-long" in books)
        from btcq.live.trader import Trader
        trader = Trader(cfg)
        trader.restore(st)
        snap = trader.snapshot()
        last_minute = max(trader.last_minute.values()) if trader.last_minute else None
        now = int(time.time() * 1000)
        data_ts = (last_minute + 60_000) if last_minute is not None else None
        data_age = (now - data_ts) / 60_000 if data_ts else None

        all_snaps = db.snapshots()
        first_ts = all_snaps[0][1] if all_snaps else data_ts or now
        end_ts = data_ts if source == "replay" and data_ts else now
        weeks = max((end_ts - first_ts) / (7 * 86_400_000), 0.0)

        live_name = meta.get("live_book")
        trades = db.trades()
        by_book: dict[str, list] = {}
        for t in trades:
            by_book.setdefault(t["book"], []).append(t)

        out_books = {}
        for name, s in snap.items():
            out_books[name] = {**s, "start_equity": cfg.initial_equity, "role": "model" if live_name == name else
                               ("main" if name == "F1" else "shadow"), "stats": book_stats(by_book.get(name, []), weeks)}

        live = None
        if live_name and st.get("live"):
            lst = st["live"]
            last_close = st["books"].get(live_name, {}).get("last_close", {})
            pos = []
            for sym, p in lst["pos"].items():
                if p["dir"]:
                    last = last_close.get(sym)
                    upnl = float(p["qty"]) * (last - p["entry_px"]) * p["dir"] if last else None
                    pos.append({"symbol": sym, "dir": p["dir"], "entry_px": p["entry_px"], "stop": p["stop"],
                                "last": last, "qty": float(p["qty"]), "upnl": upnl,
                                "R_open": upnl / p["risk_usd"] if upnl is not None and p["risk_usd"] else None,
                                "since": p["entry_time"]})
            fills = [e for e in db.events(1000) if e["kind"] == "live_entry"]
            ltrades = by_book.get(f"{live_name}-live", [])
            eq = lst.get("equity")
            peak = lst.get("peak")
            live = {"name": live_name, "equity": eq, "peak": peak,
                    "drawdown": (1 - eq / peak) if eq and peak else None,
                    "paused": lst.get("paused"), "paused_reason": lst.get("paused_reason"), "positions": pos,
                    "stats": book_stats(ltrades, weeks),
                    "avg_slippage_bps": (sum(f["slippage_bps"] for f in fills) / len(fills)) if fills else None,
                    "n_fills": len(fills)}

        main_name = f"{live_name}-live" if live else ("F1" if "F1" in snap else next(iter(snap)))
        main = live if live else out_books[main_name]
        series = {}
        for book, ts, eq, dd, n_open in all_snaps:
            series.setdefault(book, []).append((ts, eq))
        for k, v in series.items():
            step = max(1, len(v) // 1500)
            series[k] = v[::step] + ([v[-1]] if (len(v) - 1) % step else [])

        per_symbol = {}
        for name, ts_ in by_book.items():
            agg = {}
            for t in ts_:
                a = agg.setdefault(t["symbol"], {"symbol": t["symbol"], "trades": 0, "sum_R": 0.0, "wins": 0})
                a["trades"] += 1
                a["sum_R"] += t["R"]
                a["wins"] += t["R"] > 0
            per_symbol[name] = sorted(({**a, "avg_R": a["sum_R"] / a["trades"], "win_rate": a["wins"] / a["trades"]}
                                       for a in agg.values()), key=lambda a: a["symbol"])

        return {
            "ready": True, "source": source, "mode": meta.get("mode", "paper"), "env": meta.get("env", "mainnet"),
            "generated": now, "data_ts": data_ts, "data_age_min": data_age, "weeks": weeks,
            "symbols": cfg.symbols, "risk_per_trade": meta.get("risk_per_trade"),
            "kill_switch": meta.get("max_drawdown_pause", 0.35), "main": main_name,
            "books": out_books, "live": live, "series": series,
            "trades": sorted(trades, key=lambda t: t["exit_time"], reverse=True)[:300],
            "events": db.events(150), "per_symbol": per_symbol, "research": RESEARCH,
            "health": [h for h in [access_check(self.access)] if h] + health_checks(
                main, main["stats"], weeks, data_age, source == "replay", live, meta.get("max_drawdown_pause", 0.35)),
        }


class LoginGuard:
    """Lockouts after repeated WRONG credentials (a browser's first, credential-less request is not a
    failure):
      - per IP: max_fails within window_s locks that IP for lock_s;
      - global: global_max wrong logins within global_window_s from ALL IPs together pause every login
        for global_lock_s. This caps distributed guessing (~2,400/day by default) at the cost of letting
        an attacker block logins temporarily, an acceptable trade for a read-only dashboard.
    Thread-safe; memory stays bounded under attacks from many addresses."""

    def __init__(self, max_fails: int = 10, window_s: int = 900, lock_s: int = 900, global_max: int = 100,
                 global_window_s: int = 3600, global_lock_s: int = 1800, clock=time.monotonic):
        self.max_fails, self.window_s, self.lock_s, self.clock = max_fails, window_s, lock_s, clock
        self.global_max, self.global_window_s, self.global_lock_s = global_max, global_window_s, global_lock_s
        self.fails: dict[str, list[float]] = {}
        self.locked: dict[str, float] = {}
        self.global_fails: list[float] = []
        self.global_until = 0.0
        self.last_try: dict[str, float] = {}
        self._last_sweep = -1e9
        self.lock = threading.Lock()

    def _sweep(self, now: float) -> None:
        if now - self._last_sweep < 60 and len(self.locked) < 50_000 and len(self.fails) < 50_000:
            return
        self._last_sweep = now
        self.fails = {k: v for k, v in self.fails.items() if v and now - v[-1] < self.window_s}
        self.locked = {k: u for k, u in self.locked.items() if u > now}
        self.last_try = {k: t for k, t in self.last_try.items() if now - t < 60}
        if len(self.locked) > 50_000:  # hard cap: keep the longest-running locks
            self.locked = dict(sorted(self.locked.items(), key=lambda kv: kv[1])[-50_000:])
        if len(self.fails) > 50_000:
            self.fails = dict(sorted(self.fails.items(), key=lambda kv: kv[1][-1])[-50_000:])

    def may_try(self, ip: str) -> bool:
        """While an IP (not the whole server) is locked, allow one credential attempt per minute, so the
        owner can get back in with the right password; guessing stays at 1/min for that IP."""
        now = self.clock()
        with self.lock:
            if self.global_until > now or now - self.last_try.get(ip, -1e9) < 60:
                return False
            self.last_try[ip] = now
            return True

    def retry_after(self, ip: str) -> int:
        with self.lock:
            now = self.clock()
            left = max(self.locked.get(ip, 0.0), self.global_until) - now
            if self.locked.get(ip, 0.0) <= now:
                self.locked.pop(ip, None)
            return int(left) + 1 if left > 0 else 0

    def failed(self, ip: str) -> str | None:
        """Record a wrong login. Returns "ip" or "global" when this failure triggers a lockout."""
        now = self.clock()
        with self.lock:
            self._sweep(now)
            self.global_fails = [t for t in self.global_fails if now - t < self.global_window_s] + [now]
            if len(self.global_fails) >= self.global_max and self.global_until <= now:
                self.global_until = now + self.global_lock_s
                self.global_fails = []
                return "global"
            recent = [t for t in self.fails.get(ip, []) if now - t < self.window_s] + [now]
            self.fails[ip] = recent
            if len(recent) >= self.max_fails:
                self.locked[ip] = now + self.lock_s
                self.fails.pop(ip, None)
                return "ip"
            return None

    def succeeded(self, ip: str) -> None:
        with self.lock:
            self.fails.pop(ip, None)
            self.locked.pop(ip, None)
            self.last_try.pop(ip, None)


def make_handler(data: DashboardData, user: str, password: str, guard: LoginGuard | None = None):
    token = base64.b64encode(f"{user}:{password}".encode()).decode() if password else None
    guard = guard or LoginGuard()

    class Handler(BaseHTTPRequestHandler):
        server_version = "httpd"
        sys_version = ""
        timeout = 20  # seconds per socket read/write: slow or stuck clients can't pin threads

        def log_message(self, fmt, *args):  # keep container logs quiet
            pass

        def _auth_ok(self) -> bool:
            if token is None:
                return True
            ip = self.client_address[0]
            got = self.headers.get("Authorization", "")
            if not got:
                return False  # first request from a browser: ask for credentials, not a failure
            if got.startswith("Basic ") and hmac.compare_digest(got[6:].encode("latin-1", "replace"),
                                                                token.encode("ascii")):
                guard.succeeded(ip)
                return True
            hit = guard.failed(ip)
            if hit == "ip":
                print(f"dashboard: {ip} locked out for {guard.lock_s // 60} min after "
                      f"{guard.max_fails} wrong logins", flush=True)
            elif hit == "global":
                print(f"dashboard: {guard.global_max} wrong logins from all addresses within "
                      f"{guard.global_window_s // 60} min; ALL logins paused for {guard.global_lock_s // 60} min",
                      flush=True)
            return False

        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/healthz":
                return self._send(200, b"ok", "text/plain")
            ip = self.client_address[0]
            wait = guard.retry_after(ip)
            if wait and not (self.headers.get("Authorization") and guard.may_try(ip)):
                return self._send(429, b"too many wrong logins, try again later", "text/plain",
                                  {"Retry-After": str(wait)})
            if not self._auth_ok():
                return self._send(401, b"authentication required", "text/plain",
                                  {"WWW-Authenticate": 'Basic realm="restricted", charset="UTF-8"'})
            if path in ("/", "/index.html"):
                return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if path == "/api/data":
                try:
                    body = json.dumps(data.payload(), allow_nan=False, default=str).encode()
                except sqlite3.Error:
                    return self._send(503, b'{"ready": false, "message": "bot database busy, retrying"}',
                                      "application/json", {"Retry-After": "5"})
                except Exception as e:  # noqa: BLE001 - never leak details to the client
                    print(f"dashboard: payload failed: {type(e).__name__}: {e}"[:240], flush=True)
                    return self._send(500, b'{"ready": false, "message": "dashboard error, see container log"}',
                                      "application/json")
                return self._send(200, body, "application/json")
            return self._send(404, b"not found", "text/plain")

    return Handler


def _hard_close(request) -> None:
    try:
        request.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass


class QuietServer(ThreadingHTTPServer):
    """Internet-facing hardening for the stdlib threaded server:
      - at most max_connections concurrent connections (extra ones are closed, no thread spawned);
      - each connection is cut after request_deadline_s in total (per-read timeouts alone let a
        client trickle bytes forever);
      - client disconnects are silent; other errors log at most one short line per minute."""
    daemon_threads = True
    request_queue_size = 128
    max_connections = 64
    request_deadline_s = 30

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._slots = threading.BoundedSemaphore(self.max_connections)

    def process_request(self, request, client_address):
        if not self._slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address):
        timer = threading.Timer(self.request_deadline_s, _hard_close, (request,))
        timer.daemon = True
        timer.start()
        try:
            super().process_request_thread(request, client_address)
        finally:
            timer.cancel()
            self._slots.release()
    _last_log = -1e9
    _suppressed = 0
    _log_lock = threading.Lock()

    def handle_error(self, request, client_address):
        import sys
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
            return  # resets, aborted or deadline-cut connections: expected on the internet
        with self._log_lock:
            now = time.monotonic()
            if now - self._last_log < 60:
                QuietServer._suppressed += 1
                return
            extra = f" ({QuietServer._suppressed} similar errors suppressed)" if QuietServer._suppressed else ""
            QuietServer._last_log, QuietServer._suppressed = now, 0
        print(f"dashboard: request from {client_address[0]} failed: {type(exc).__name__}: {exc}"[:240] + extra,
              flush=True)


def serve(db_path: Path | str, host: str = "127.0.0.1", port: int = 8080, user: str = "admin",
          password: str = "", allow_no_auth: bool = False) -> ThreadingHTTPServer:
    if not password and host not in LOOPBACK and not allow_no_auth:
        raise SystemExit("Set DASHBOARD_PASSWORD to serve the dashboard beyond localhost "
                         "(or bind --host 127.0.0.1 and use an SSH tunnel).")
    if password and len(password) < 16:
        print(f"WARNING: DASHBOARD_PASSWORD is only {len(password)} characters. If the dashboard is reachable "
              "from the internet, use 16+ random characters (e.g. `openssl rand -base64 18`).", flush=True)
    http.client._MAXLINE = 8192      # longest header line: oversize -> 431 instead of buffering 64 KB
    http.client._MAXHEADERS = 32     # header count
    # Inside Docker the server always listens on 0.0.0.0; DASHBOARD_BIND says what is actually published.
    import os
    public = os.environ.get("DASHBOARD_BIND", host) not in LOOPBACK
    access = {"public": public, "password_len": len(password)}
    return QuietServer((host, port), make_handler(DashboardData(db_path, access), user, password))
