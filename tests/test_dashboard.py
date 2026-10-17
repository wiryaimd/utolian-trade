"""Dashboard: payload from a replay database, live-mode payload, HTTP auth."""
import asyncio
import base64
import json
import threading
import urllib.error
import urllib.request

import pandas as pd
import pytest

from btcq.live.app import App
from btcq.live.config import BotConfig
from btcq.live.dashboard import DashboardData, health_checks, serve
from btcq.live.db import StateDB
from btcq.live.feed import ReplayFeed
from btcq.live.telegram import LogNotifier
from conftest import needs_data

ms = lambda s: int(pd.Timestamp(s, tz="UTC").value // 10**6)  # noqa: E731


@pytest.fixture(scope="module")
def replay_db(tmp_path_factory):
    d = tmp_path_factory.mktemp("dash")
    cfg = BotConfig(symbols=["BTCUSDT", "ETHUSDT"], shadow_long_only=True, state_dir=d)
    app = App(cfg, ReplayFeed(cfg.symbols), LogNotifier(), StateDB(d / "bot.sqlite"))
    asyncio.run(app.run_replay(ms("2023-07-01"), ms("2024-01-01")))
    return d / "bot.sqlite"


@needs_data
def test_payload_from_replay(replay_db):
    p = DashboardData(replay_db).payload()
    json.dumps(p, allow_nan=False)                       # strict JSON for the browser
    assert p["ready"] and p["source"] == "replay" and p["mode"] == "paper"
    assert set(p["books"]) == {"F1", "F1-long"} and p["main"] == "F1"
    assert len(p["series"]["F1"]) > 100 and p["weeks"] > 20
    assert p["books"]["F1"]["stats"]["n"] > 10
    names = [h["name"] for h in p["health"]]
    assert names[:3] == ["Data feed", "Trading", "Drawdown from peak"] and "Trades per week" in names
    assert all(h["status"] in ("ok", "watch", "bad", "wait") for h in p["health"])
    assert p["per_symbol"]["F1"] and p["events"] and p["trades"]


def test_health_checks_flag_problems_in_plain_language():
    stats = {"n": 120, "per_week": 0.5, "avg_R": -0.2, "se_R": 0.1, "win_rate": 0.2,
             "longest_losing": 35, "current_losing": 3}
    h = {x["name"]: x for x in health_checks({"drawdown": 0.33, "paused": True, "paused_reason": "drawdown"},
                                             stats, 30, 25.0, False, {"avg_slippage_bps": 20.0}, 0.35)}
    assert h["Data feed"]["status"] == "bad" and h["Trading"]["status"] == "bad"
    assert h["Drawdown from peak"]["status"] == "bad"
    assert h["Average R per trade"]["status"] == "bad" and "edge may be gone" in h["Average R per trade"]["note"]
    assert h["Losing streak"]["status"] == "bad" and h["Live fills vs model"]["status"] == "bad"
    early = {x["name"]: x for x in health_checks({"drawdown": 0.0}, {"n": 3}, 1, 1.0, False, None, 0.35)}
    assert early["Average R per trade"]["status"] == "wait" and early["Trades per week"]["status"] == "wait"


@needs_data
def test_http_requires_password_and_serves_page_and_api(replay_db):
    srv = serve(replay_db, "127.0.0.1", 0, "admin", "s3cret")
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(base + "/api/data")
        assert e.value.code == 401
        assert urllib.request.urlopen(base + "/healthz").read() == b"ok"
        auth = {"Authorization": "Basic " + base64.b64encode(b"admin:s3cret").decode()}
        page = urllib.request.urlopen(urllib.request.Request(base + "/", headers=auth)).read().decode()
        assert "Trading monitor" in page
        data = json.loads(urllib.request.urlopen(urllib.request.Request(base + "/api/data", headers=auth)).read())
        assert data["ready"] and data["books"]["F1"]["stats"]["n"] > 10
        bad = {"Authorization": "Basic " + base64.b64encode(b"admin:wrong").decode()}
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(urllib.request.Request(base + "/api/data", headers=bad))
    finally:
        srv.shutdown()
        srv.server_close()


def test_refuses_public_bind_without_password(tmp_path):
    with pytest.raises(SystemExit):
        serve(tmp_path / "x.sqlite", "0.0.0.0", 0, "admin", "")


def test_missing_database_is_reported_not_crashed(tmp_path):
    p = DashboardData(tmp_path / "nope.sqlite").payload()
    assert p["ready"] is False and "Start the bot" in p["message"]


@needs_data
def test_live_payload_shows_real_account(tmp_path):
    from test_executor import SECRET, Clock, FakeBinance, make
    fake = FakeBinance(prices={"BTCUSDT": 27_000.0, "ETHUSDT": 1_600.0})
    clock = Clock(0)
    _, ex, _ = make(fake, clock=clock, symbols=("BTCUSDT",))
    cfg = BotConfig(symbols=["BTCUSDT"], shadow_long_only=False, state_dir=tmp_path, mode="live",
                    binance_env="testnet", api_key="k", api_secret=SECRET)
    app = App(cfg, ReplayFeed(cfg.symbols), LogNotifier(), StateDB(tmp_path / "bot.sqlite"), executor=ex)
    orig = app.trader.process_minute

    def step(symbol, m, rate=0.0):
        clock.t = m.ts + 60_000
        fake.bar(symbol, m.open, m.high, m.low, m.close)
        orig(symbol, m, rate)
    app.trader.process_minute = step

    async def go():
        await app.start(ms("2023-10-01"))
        await ex.startup()
        await app.run_replay(ms("2023-10-01"), ms("2024-01-01"))
    asyncio.run(go())
    p = DashboardData(tmp_path / "bot.sqlite").payload()
    json.dumps(p, allow_nan=False)
    assert p["mode"] == "live" and p["main"] == "F1-live" and p["live"]["equity"] > 0
    assert p["live"]["stats"]["n"] >= 3 and p["live"]["n_fills"] >= 3
    assert "F1-live" in p["series"] and p["books"]["F1"]["role"] == "model"
    assert any(h["name"] == "Live fills vs model" for h in p["health"])


def test_cli_dashboard_without_db_flag(monkeypatch, tmp_path):
    """Docker runs `python -m btcq.live dashboard` with no --db; this once crashed on startup."""
    import sys

    import btcq.live.__main__ as cli
    import btcq.live.dashboard as dash
    seen = {}

    class FakeServer:
        def serve_forever(self):
            seen["served"] = True

    monkeypatch.setattr(dash, "serve", lambda db, host, port, user, pw: seen.update(db=db) or FakeServer())
    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    monkeypatch.setenv("DASHBOARD_PASSWORD", "x")
    monkeypatch.setattr(sys, "argv", ["btcq.live", "dashboard", "--host", "0.0.0.0", "--port", "8080"])
    cli.main()
    assert seen["served"] and seen["db"].endswith("bot.sqlite") and str(tmp_path) in seen["db"]


def test_login_guard_locks_out_and_expires():
    from btcq.live.dashboard import LoginGuard
    now = [0.0]
    g = LoginGuard(max_fails=3, window_s=60, lock_s=120, clock=lambda: now[0])
    assert not g.failed("1.2.3.4") and not g.failed("1.2.3.4")
    assert g.retry_after("1.2.3.4") == 0
    assert g.failed("1.2.3.4")                       # third wrong login locks
    assert 0 < g.retry_after("1.2.3.4") <= 121
    assert g.retry_after("5.6.7.8") == 0             # other IPs unaffected
    now[0] += 121
    assert g.retry_after("1.2.3.4") == 0             # lock expires
    g.failed("9.9.9.9"); now[0] += 61
    assert not g.failed("9.9.9.9")                   # failures outside the window are forgotten


@needs_data
def test_http_lockout_counts_only_wrong_credentials(replay_db):
    from http.server import ThreadingHTTPServer

    from btcq.live.dashboard import DashboardData, LoginGuard, make_handler
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(DashboardData(replay_db), "admin", "right-pass",
                                                               LoginGuard(max_fails=3)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}/api/data"

    def code(user_pass=None):
        h = {"Authorization": "Basic " + base64.b64encode(user_pass.encode()).decode()} if user_pass else {}
        try:
            return urllib.request.urlopen(urllib.request.Request(base, headers=h)).status
        except urllib.error.HTTPError as e:
            return e.code
    try:
        assert [code() for _ in range(5)] == [401] * 5          # browser prompts never count
        assert code("admin:right-pass") == 200
        assert [code("admin:guess") for _ in range(3)] == [401, 401, 401]   # IP now locked
        assert code("admin:guess4") == 401                      # locked IPs get one try per minute...
        assert code("admin:right-pass") == 429                  # ...so the next attempt is refused
    finally:
        srv.shutdown()
        srv.server_close()


def test_global_budget_stops_distributed_guessing():
    from btcq.live.dashboard import LoginGuard
    now = [0.0]
    g = LoginGuard(max_fails=10, global_max=20, global_window_s=3600, global_lock_s=1800, clock=lambda: now[0])
    hits = [g.failed(f"10.0.0.{i}") for i in range(20)]       # 20 different IPs, 1 guess each
    assert hits[:19] == [None] * 19 and hits[19] == "global"
    assert g.retry_after("203.0.113.9") > 0                   # every address paused, even new ones
    now[0] += 1801
    assert g.retry_after("203.0.113.9") == 0


def test_non_ascii_authorization_is_a_counted_failure_not_a_crash():
    from btcq.live.dashboard import LoginGuard, make_handler

    class Req:  # minimal stand-in for the handler's request state
        pass
    guard = LoginGuard(max_fails=2)
    H = make_handler(None, "admin", "pw", guard)
    h = H.__new__(H)
    h.client_address = ("198.51.100.7", 1234)
    h.headers = {"Authorization": "Basic \xe9\xe9"}
    assert h._auth_ok() is False and h._auth_ok() is False   # no TypeError
    assert guard.retry_after("198.51.100.7") > 0              # counted -> locked after 2


def test_access_check_warns_when_public():
    from btcq.live.dashboard import access_check
    assert access_check({"public": False, "password_len": 4}) is None
    weak = access_check({"public": True, "password_len": 8})
    assert weak["status"] == "bad" and "8-char" in weak["value"]
    assert access_check({"public": True, "password_len": 24})["status"] == "watch"


def _start(server):
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server.server_address[1]


def test_slow_trickle_client_is_cut_at_deadline_and_connections_are_capped(tmp_path):
    import socket
    import time as _t

    from btcq.live.dashboard import DashboardData, QuietServer, make_handler

    class Small(QuietServer):
        max_connections = 2
        request_deadline_s = 1.5
    srv = Small(("127.0.0.1", 0), make_handler(DashboardData(tmp_path / "x.sqlite"), "admin", "pw"))
    port = _start(srv)
    try:
        slow = socket.create_connection(("127.0.0.1", port))
        slow.sendall(b"GET / HTTP/1.1\r\n")
        t0, closed = _t.time(), False
        while _t.time() - t0 < 6:                     # trickle a header byte every 0.4 s
            try:
                slow.sendall(b"X")
                _t.sleep(0.4)
            except OSError:
                closed = True
                break
        assert closed and _t.time() - t0 < 4          # cut near the 1.5 s deadline, not kept forever
        a = socket.create_connection(("127.0.0.1", port))
        b = socket.create_connection(("127.0.0.1", port))
        _t.sleep(0.3)
        c = socket.create_connection(("127.0.0.1", port))
        c.settimeout(3)
        assert c.recv(10) == b""                      # third concurrent connection dropped immediately
        for s_ in (a, b, c, slow):
            s_.close()
    finally:
        srv.shutdown()
        srv.server_close()


@needs_data
def test_oversized_header_gets_431_and_locked_owner_can_still_log_in(replay_db):
    import socket

    from btcq.live.dashboard import DashboardData, LoginGuard, QuietServer, make_handler, serve
    srv = serve(replay_db, "127.0.0.1", 0, "admin", "right-pass")      # serve() installs header limits
    port = _start(srv)
    try:
        s_ = socket.create_connection(("127.0.0.1", port))
        s_.sendall(b"GET /healthz HTTP/1.1\r\nX-Big: " + b"a" * 20_000 + b"\r\n\r\n")
        s_.settimeout(5)
        assert b" 431 " in s_.recv(200)
        s_.close()
    finally:
        srv.shutdown()
        srv.server_close()

    guard = LoginGuard(max_fails=2)
    srv = QuietServer(("127.0.0.1", 0), make_handler(DashboardData(replay_db), "admin", "right-pass", guard))
    port = _start(srv)
    url = f"http://127.0.0.1:{port}/api/data"

    def code(cred):
        h = {"Authorization": "Basic " + base64.b64encode(cred.encode()).decode()}
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=h)).status
        except urllib.error.HTTPError as e:
            return e.code
    try:
        assert [code("admin:x"), code("admin:y")] == [401, 401]       # IP now locked
        assert code("admin:right-pass") == 200                        # one try/min while locked: owner gets in
        assert code("admin:z") == 401 and code("admin:z2") == 401      # relocks after 2 more wrong
        assert code("admin:w") == 401                                 # uses this minute's single try
        assert code("admin:right-pass") == 429                        # so even the right password waits
    finally:
        srv.shutdown()
        srv.server_close()


@needs_data
def test_locked_database_returns_clean_503(replay_db, tmp_path):
    import shutil
    import sqlite3 as sq

    from btcq.live.dashboard import DashboardData, QuietServer, make_handler
    db = tmp_path / "copy.sqlite"
    shutil.copy(replay_db, db)
    holder = sq.connect(db)
    holder.execute("BEGIN EXCLUSIVE")
    srv = QuietServer(("127.0.0.1", 0), make_handler(DashboardData(db), "admin", "pw"))
    port = _start(srv)
    h = {"Authorization": "Basic " + base64.b64encode(b"admin:pw").decode()}
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/api/data", headers=h),
                                   timeout=10)
        assert e.value.code == 503 and json.loads(e.value.read())["ready"] is False
    finally:
        holder.rollback()
        holder.close()
        srv.shutdown()
        srv.server_close()


def test_one_address_cannot_take_every_connection_slot(tmp_path):
    import socket
    import time as _t

    from btcq.live.dashboard import DashboardData, QuietServer, make_handler

    class Small(QuietServer):
        max_connections = 10
        per_ip_connections = 2
    srv = Small(("127.0.0.1", 0), make_handler(DashboardData(tmp_path / "x.sqlite"), "admin", "pw"))
    port = _start(srv)
    try:
        held = [socket.create_connection(("127.0.0.1", port)) for _ in range(2)]
        _t.sleep(0.3)
        extra = socket.create_connection(("127.0.0.1", port))
        extra.settimeout(3)
        assert extra.recv(10) == b""                  # 3rd from the same address: dropped
        for s_ in held:
            s_.close()
        _t.sleep(0.3)
        ok = urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=5).read()
        assert ok == b"ok"                            # slots are released again
        extra.close()
    finally:
        srv.shutdown()
        srv.server_close()
