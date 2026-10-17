"""System health: bot telemetry, token redaction, live checks, full checks against a fake Binance, and the HTTP button."""
import asyncio
import base64
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict

import httpx
import numpy as np
import pandas as pd
import pytest

from btcq.live import health as H
from btcq.live.app import App
from btcq.live.binance_api import BinanceError, BinanceFutures
from btcq.live.book import BookConfig, Position
from btcq.live.config import BotConfig
from btcq.live.d1 import LOOKBACKS, D1Book, SubPos, decide, key
from btcq.live.dashboard import DashboardData, QuietServer, make_handler
from btcq.live.db import StateDB
from btcq.live.feed import BinanceFeed
from btcq.live.telegram import LogNotifier, Telegram, redact
from btcq.live.trader import Trader
from btcq.live.version import code_id

MIN, H4, DAY = 60_000, 4 * 3_600_000, 86_400_000
NOW = int(pd.Timestamp("2026-10-18 09:00:30", tz="UTC").value // 10**6)
LAST = NOW // MIN * MIN - MIN                       # 08:59, the last closed minute
SYMS = ["BTCUSDT", "ETHUSDT"]
TOKEN = "123456789:AAFabcdefghijklmnopqrstuvwxyz0123456"
REST, WS = "https://binance.test", "wss://stream.test/stream?streams="


# ---------------------------------------------------------------- a healthy bot, as the dashboard reads it

def daily_history(n=700):
    rng = np.random.default_rng(7)
    end = NOW // DAY * DAY - DAY                     # yesterday, the last closed day
    out = {}
    for i, s in enumerate(SYMS):
        c = 100 * (i + 1) * np.exp(np.cumsum(rng.normal(0.003, 0.03, n)))
        out[s] = [(end - (n - 1 - j) * DAY, float(c[j])) for j in range(n)]
    return out


def healthy(hist=None):
    hist = hist or daily_history()
    f1 = {"cash": 10_000.0, "peak": 10_000.0, "paused": False, "paused_reason": "",
          "pos": {s: asdict(Position()) for s in SYMS}, "queue": {s: [] for s in SYMS},
          "last_close": {"BTCUSDT": 61_000.5, "ETHUSDT": 2_400.25}}
    f1["pos"]["BTCUSDT"] = asdict(Position(dir=1, qty=0.1, epx=60_000.0, sl=57_000.0))
    d1 = D1Book("D1", SYMS, BookConfig())
    d1.started = NOW - 2 * DAY
    for s in SYMS:
        dec = decide(np.array([c for _, c in hist[s]]))
        for L in LOOKBACKS:
            if dec[str(L)]["state"] == 1:
                d1.subs[key(s, L)] = SubPos(qty=0.001, epx=100.0)
        d1.last_close[s] = hist[s][-1][1]
    st = {"books": {"F1": f1, "D1": d1.state()}, "last_minute": {s: LAST for s in SYMS},
          "last_bar_end": {s: NOW // H4 * H4 for s in SYMS},
          "builders": {"BTCUSDT": {"partial": [NOW // H4 * H4, 60_000.0, 61_500.0, 59_800.0, 61_000.5, 1, 1, 1]},
                       "ETHUSDT": {"partial": [NOW // H4 * H4, 2_390.0, 2_410.0, 2_380.0, 2_400.25, 1, 1, 1]}}}
    health = {"written": NOW - 5_000, "started": NOW - 3 * 3_600_000, "code": code_id(), "mode": "paper",
              "env": "mainnet", "processed_at": LAST + MIN + 2_000, "catch_up": {"at": NOW - 3 * 3_600_000, "minutes": 10},
              "feed": {"ws_connected": True, "ws_since": NOW - 3 * 3_600_000, "ws_last_msg": NOW - 6_000,
                       "ws_drops": [NOW - 2 * 3_600_000], "ws_last_error": "ConnectionError: x",
                       "ws_last_error_at": NOW - 2 * 3_600_000, "rest_last_ok": NOW - 2 * 3_600_000,
                       "rest_last_error": None, "rest_last_error_at": None},
              "telegram": {"configured": True, "last_poll_ok": NOW - 20_000, "last_send_ok": NOW - 3_600_000,
                           "poll_error": None, "poll_error_at": None, "poll_fail_streak": 0,
                           "send_error": None, "send_error_at": None, "send_fail_streak": 0}, "account_api": None}
    closes = [(s, d, c) for s in SYMS for d, c in hist[s][-2:]]
    return st, health, closes


def by_name(checks):
    return {(c["group"], c["name"]): c for c in checks}


def run_live(tmp_path, st, health, closes, heartbeat=NOW - 20_000, now=NOW, mode="paper"):
    (tmp_path / "heartbeat").write_text(str(heartbeat))
    db = tmp_path / "bot.sqlite"
    db.write_bytes(b"")
    return by_name(H.live_checks(st, health, {"mode": mode, "symbols": SYMS}, "live", now, tmp_path, db, closes,
                                 code_id()))


def test_live_checks_of_a_healthy_bot(tmp_path):
    c = run_live(tmp_path, *healthy())
    want_ok = [("server", "Bot process"), ("server", "Bot version"), ("server", "Processing speed"),
               ("market", "Latest prices"), ("market", "Websocket stream"), ("market", "Websocket drops (24 h)"),
               ("market", "Binance REST (bot)"), ("telegram", "Telegram commands"), ("telegram", "Telegram messages"),
               ("F1", "4h decisions on schedule"), ("F1", "New entries"), ("F1", "Open positions and stops"),
               ("F1", "Orders waiting"), ("D1", "Daily decisions on schedule"), ("D1", "Orders filled on time"),
               ("D1", "Never paused"), ("D1", "Sub-positions")]
    assert {k: c[k]["status"] for k in want_ok} == {k: "ok" for k in want_ok}
    assert c[("account", "Binance account API")]["status"] == "off"
    assert all(x["kind"] == "live" for x in c.values())


def test_live_checks_flag_a_broken_bot(tmp_path):
    st, health, closes = healthy()
    old = LAST - 20 * MIN
    st["last_minute"] = {s: old for s in SYMS}
    st["last_bar_end"] = {s: old // H4 * H4 - H4 for s in SYMS}
    st["books"]["F1"]["last_close"]["BTCUSDT"] = 56_000.0                  # below the long's stop
    st["books"]["F1"]["queue"]["ETHUSDT"] = [[old - MIN, {}]]             # should have filled already
    st["books"]["D1"]["paused"] = True
    health["feed"]["ws_connected"] = False
    health["telegram"]["last_poll_ok"] = NOW - 30 * MIN
    health["code"] = "00000000"
    c = run_live(tmp_path, st, health, [], heartbeat=NOW - 20 * MIN)
    bad = [("server", "Bot process"), ("market", "Latest prices"), ("market", "Websocket stream"),
           ("telegram", "Telegram commands"), ("F1", "4h decisions on schedule"), ("F1", "Open positions and stops"),
           ("F1", "Orders waiting"), ("D1", "Daily decisions on schedule"), ("D1", "Never paused")]
    assert {k: c[k]["status"] for k in bad} == {k: "bad" for k in bad}
    assert c[("server", "Bot version")]["status"] == "watch" and "different code" in c[("server", "Bot version")]["note"]
    assert "past its stop" in c[("F1", "Open positions and stops")]["note"]


def test_live_checks_wait_for_a_fresh_d1_and_a_fresh_record(tmp_path):
    st, health, closes = healthy()
    st["books"]["D1"]["started"] = NOW - 2 * 3_600_000                     # started this morning
    st["last_bar_end"] = {}
    c = run_live(tmp_path, st, health, [])
    assert c[("D1", "Daily decisions on schedule")]["status"] == "wait"
    assert c[("F1", "4h decisions on schedule")]["status"] == "wait"
    c = run_live(tmp_path, st, None, [])
    assert c[("server", "Bot version")]["status"] == "wait" and c[("market", "Websocket stream")]["status"] == "wait"


def test_no_false_alarms_in_normal_operation(tmp_path):
    st, health, closes = healthy()
    # just restarted: the websocket is not open yet
    health["feed"].update(ws_connected=False, ws_since=None, ws_last_error=None)
    # a single getUpdates blip after the last message: commands recovered, messages unaffected
    health["telegram"].update(poll_error="HTTPStatusError: 502", poll_error_at=NOW - 60_000, poll_fail_streak=0)
    # a market removed from SYMBOLS is still in the saved state
    st["last_minute"]["DOGEUSDT"] = LAST - 3 * DAY
    c = run_live(tmp_path, st, health, closes)
    assert c[("market", "Websocket stream")]["status"] == "wait"
    assert c[("telegram", "Telegram messages")]["status"] == "ok"
    assert c[("telegram", "Telegram commands")]["status"] == "ok"
    assert c[("market", "Latest prices")]["status"] == "ok"
    # 00:00:30 UTC: BTC's 23:59 minute is saved, ETH's not yet; D1 is not late for ETH
    mid = NOW // DAY * DAY + 30_000
    st2, health2, _ = healthy()
    st2["last_minute"] = {"BTCUSDT": mid // DAY * DAY - MIN, "ETHUSDT": mid // DAY * DAY - 2 * MIN}
    day = mid // DAY * DAY
    closes2 = [("BTCUSDT", day - DAY, 1.0), ("ETHUSDT", day - 2 * DAY, 1.0)]
    health2["written"] = mid - 5_000
    c = run_live(tmp_path, st2, health2, closes2, heartbeat=mid - 5_000, now=mid)
    assert c[("D1", "Daily decisions on schedule")]["status"] == "ok"


def test_telegram_send_failures_and_live_trading_problems(tmp_path):
    st, health, closes = healthy()
    health["telegram"].update(send_error="HTTP 400: chat not found", send_error_at=NOW - 60_000, send_fail_streak=3)
    health.update(mode="live", account_api={"last_ok": NOW - 30_000, "last_error": "/fapi/v1/marginType: -4046",
                                            "last_error_at": NOW - 3_600_000})
    st["live"] = {"paused": True, "paused_reason": "stop placement failed on BTCUSDT",
                  "pos": {"BTCUSDT": {"dir": 1, "qty": "0.1", "stop": float("nan"), "stop_id": ""}}}
    c = run_live(tmp_path, st, health, closes, mode="live")
    assert c[("telegram", "Telegram messages")]["status"] == "bad"
    assert "chat not found" in c[("telegram", "Telegram messages")]["note"]
    api = c[("account", "Binance account API")]
    assert api["status"] == "ok" and "-4046" not in api["note"]        # an old, harmless error is not shown
    assert c[("account", "Live trading")]["status"] == "bad"
    assert c[("account", "Live positions and stops")]["status"] == "bad"


# ---------------------------------------------------------------- full checks against a fake Binance

class FakeBinance:
    def __init__(self, hist, close_1m=None, open_4h=None, ping_status=200, delay=0.0):
        self.hist, self.calls, self.delay, self.calls_4h = hist, 0, delay, 0
        self.close_1m = close_1m or {"BTCUSDT": "61000.5", "ETHUSDT": "2400.25"}
        self.open_4h = open_4h or {"BTCUSDT": "60000", "ETHUSDT": "2390"}
        self.ping_status = ping_status

    def __call__(self, url, params=None, timeout=6.0):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        p = params or {}
        if url == H.TELEGRAM_URL:
            return 302, "", 30.0
        path = url.split("/fapi/")[-1]
        if path == "v1/ping":
            return (200, {}, 50.0) if self.ping_status == 200 else \
                (self.ping_status, {"code": 0, "msg": "Service unavailable from a restricted location"}, 50.0)
        if path == "v1/time":
            return 200, {"serverTime": NOW + 150}, 40.0
        if path == "v1/exchangeInfo":
            return 200, {"symbols": [{"symbol": s, "status": "TRADING"} for s in SYMS]}, 80.0
        if path == "v1/klines":
            s, iv, start = p["symbol"], p["interval"], int(p.get("startTime", 0))
            if iv == "1m":
                return 200, [[start, "1", "1", "1", self.close_1m[s]]], 30.0
            if iv == "4h":
                self.calls_4h += 1
                return 200, [[start, self.open_4h[s], "61600" if s == "BTCUSDT" else "2420",
                              "59700" if s == "BTCUSDT" else "2370", "1"]], 30.0
            rows = [[d, "0", "0", "0", repr(c)] for d, c in self.hist[s] if start <= d <= p["endTime"]]
            return 200, rows[:p["limit"]], 30.0
        return 404, "not found", 10.0


def full(st, closes, fake, **kw):
    return by_name(H.full_checks(st, {"symbols": SYMS, "env": "mainnet"}, "live", closes, REST, WS, fake,
                                 lambda url, timeout=6.0: 120.0, clock=lambda: NOW / 1000, **kw))


def test_full_checks_pass_on_matching_binance_data():
    hist = daily_history()
    st, _, closes = healthy(hist)
    fake = FakeBinance(hist)
    c = full(st, closes, fake)
    names = [("market", "Binance REST API"), ("market", "Clock vs Binance"), ("market", "Markets trading"),
             ("market", "Websocket reachable"), ("market", "Bot prices match Binance"),
             ("telegram", "Telegram reachable"), ("F1", "Current 4h bar matches Binance"),
             ("D1", "Daily closes match Binance"), ("D1", "Positions follow the rule")]
    assert {k: c[k]["status"] for k in names} == {k: "ok" for k in names}, {k: c[k]["note"] for k in names}
    assert c[("D1", "Positions follow the rule")]["value"] == "18 of 18 agree"
    assert all(x["kind"] == "full" for x in c.values())


def test_full_checks_catch_wrong_prices_positions_and_a_blocked_api():
    hist = daily_history()
    st, _, closes = healthy(hist)
    longs = [L for L in LOOKBACKS if st["books"]["D1"]["subs"][key("BTCUSDT", L)]["qty"] > 0]
    st["books"]["D1"]["subs"][key("BTCUSDT", longs[0])]["qty"] = 0.0            # D1 flat where the rule says long
    fake = FakeBinance(hist, close_1m={"BTCUSDT": "61000.6", "ETHUSDT": "2400.25"},
                       open_4h={"BTCUSDT": "60001", "ETHUSDT": "2390"}, ping_status=451)
    c = full(st, closes, fake)
    assert c[("market", "Binance REST API")]["status"] == "bad"
    assert "blocks this server's location" in c[("market", "Binance REST API")]["note"]
    assert c[("market", "Bot prices match Binance")]["status"] == "bad"
    assert "61000.6" in c[("market", "Bot prices match Binance")]["note"]
    assert c[("F1", "Current 4h bar matches Binance")]["status"] == "bad"
    assert c[("D1", "Positions follow the rule")]["status"] == "bad"
    assert f"BTCUSDT {longs[0]}d: rule long, D1 flat" in c[("D1", "Positions follow the rule")]["note"]


def test_a_price_that_binance_has_not_finalised_yet_is_asked_again():
    hist = daily_history()
    st, _, closes = healthy(hist)
    fake = FakeBinance(hist, close_1m={"BTCUSDT": "61000.4", "ETHUSDT": "2400.25"})
    waits = []

    def sleep(sec):                                   # by the time the check asks again, Binance has the final close
        waits.append(sec)
        fake.close_1m["BTCUSDT"] = "61000.5"
    c = full(st, closes, fake, sleep=sleep)
    assert c[("market", "Bot prices match Binance")]["status"] == "ok" and waits == [3.0]
    c = full(st, closes, FakeBinance(hist, close_1m={"BTCUSDT": "61000.4", "ETHUSDT": "2400.25"}),
             sleep=lambda sec: None)
    assert c[("market", "Bot prices match Binance")]["status"] == "bad"


def test_full_checks_share_4h_bars_and_fill_missing_days_like_the_bot():
    hist = daily_history()
    gap = len(hist["BTCUSDT"]) - 40
    rest = {s: [x for j, x in enumerate(h) if j != gap] for s, h in hist.items()}   # Binance lacks one day
    bot = {s: h[:gap] + [(h[gap][0], h[gap - 1][1])] + h[gap + 1:] for s, h in hist.items()}  # the bot filled it
    st, _, closes = healthy(bot)
    st["books"]["F1-long"] = st["books"]["F1"]
    fake = FakeBinance(rest)
    c = full(st, closes, fake)
    assert c[("D1", "Positions follow the rule")]["status"] == "ok", c[("D1", "Positions follow the rule")]["note"]
    assert c[("F1", "Current 4h bar matches Binance")]["status"] == "ok"
    assert c[("F1-long", "Current 4h bar matches Binance")]["status"] == "ok" and fake.calls_4h == len(SYMS)


def test_a_first_network_failure_is_a_warning_and_a_repeat_a_problem():
    hist = daily_history()
    st, _, closes = healthy(hist)
    fake = FakeBinance(hist)

    def flaky(url, params=None, timeout=6.0):
        if url.endswith("/ping"):
            raise httpx.ConnectError("connection refused")
        return fake(url, params, timeout)
    r = H.FullCheckRunner(lambda: {"st": st, "meta": {"symbols": SYMS}, "source": "live", "daily_closes": closes},
                          REST, WS, http_get=flaky, ws_probe=lambda u, timeout=6.0: 90.0, clock=lambda: NOW / 1000)
    r.run("auto")
    first = by_name(r.checks(st["books"]))
    assert first[("market", "Binance REST API")]["status"] == "watch"
    assert first[("market", "Clock vs Binance")]["status"] == "ok"
    r.run("auto")
    assert by_name(r.checks(st["books"]))[("market", "Binance REST API")]["status"] == "bad"
    blocked = H.FullCheckRunner(lambda: {"st": st, "meta": {"symbols": SYMS}, "source": "live",
                                         "daily_closes": closes}, REST, WS, http_get=FakeBinance(hist, ping_status=451),
                                ws_probe=lambda u, timeout=6.0: 90.0, clock=lambda: NOW / 1000)
    blocked.run("auto")                                       # a definite answer is a problem at once
    assert by_name(blocked.checks(st["books"]))[("market", "Binance REST API")]["status"] == "bad"


def test_full_checks_before_d1_decides_and_on_timeouts():
    hist = daily_history()
    st, _, _ = healthy(hist)
    c = full(st, [], FakeBinance(hist))
    assert c[("D1", "Positions follow the rule")]["status"] == "wait"
    assert "sub-positions are in an uptrend now" in c[("D1", "Positions follow the rule")]["note"]
    c = full(st, [], FakeBinance(hist, delay=1.0), deadline_s=0.2)            # the fake websocket answers at once
    assert all(x["status"] == "bad" and "timed out" in x["note"] for k, x in c.items() if k[1] != "Websocket reachable")


# ---------------------------------------------------------------- the dashboard endpoint and button

def test_health_endpoint_and_the_run_now_button(tmp_path):
    hist = daily_history()
    st, health, closes = healthy(hist)
    db = StateDB(tmp_path / "bot.sqlite")
    db.set_meta("config", {"mode": "paper", "env": "mainnet", "symbols": SYMS, "books": ["F1", "D1"]})
    db.set_health(health)
    for s, d, c in closes:
        db.add_daily_close(s, d, c)
    db.save_state(st)
    db.close()
    (tmp_path / "heartbeat").write_text(str(int(time.time() * 1000)))
    data = DashboardData(tmp_path / "bot.sqlite")
    fake = FakeBinance(hist)
    runner = H.FullCheckRunner(data.health_context, REST, WS, http_get=fake, ws_probe=lambda u, timeout=6.0: 90.0,
                               clock=lambda: NOW / 1000)
    srv = QuietServer(("127.0.0.1", 0), make_handler(data, "admin", "s3cret", runner=runner))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    auth = "Basic " + base64.b64encode(b"admin:s3cret").decode()

    def post(headers):
        req = urllib.request.Request(base + "/api/health/check", data=b"", method="POST", headers=headers)
        return json.loads(urllib.request.urlopen(req).read())

    try:
        for headers, code in [({}, 401), ({"Authorization": auth}, 403),
                              ({"Authorization": auth, "X-Requested-With": "btcq-dashboard",
                                "Origin": "http://evil.example"}, 403),
                              ({"Authorization": auth, "X-Requested-With": "btcq-dashboard",
                                "Sec-Fetch-Site": "cross-site"}, 403)]:
            with pytest.raises(urllib.error.HTTPError) as e:
                post(headers)
            assert e.value.code == code
        assert fake.calls == 0
        ok = {"Authorization": auth, "X-Requested-With": "btcq-dashboard", "Origin": base}
        p = post(ok)
        assert p["ready"] and p["full"]["ran_at"] and p["notice"] is None and fake.calls > 0
        fulls = [c for g in p["groups"] for c in g["checks"] if c["kind"] == "full"]
        assert fulls and all(c["status"] in ("ok", "off") for c in fulls), [c for c in fulls if c["status"] != "ok"]
        import http.client
        conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=5)
        conn.putrequest("POST", "/api/health/check")
        for k, v in {**ok, "Content-Length": "-1"}.items():
            conn.putheader(k, v)
        conn.endheaders()
        assert conn.getresponse().status == 400                              # never reads an unbounded body
        conn.close()
        n = fake.calls
        again = post(ok)                                                     # within the 30 s cooldown
        assert "next one is possible" in again["notice"] and fake.calls == n
        got = json.loads(urllib.request.urlopen(urllib.request.Request(
            base + "/api/health", headers={"Authorization": auth})).read())
        assert got["ready"] and [g["key"] for g in got["groups"]][:4] == ["server", "market", "telegram", "account"]
        assert {"F1", "D1"} <= {g["key"] for g in got["groups"]}
    finally:
        srv.shutdown()
        srv.server_close()


# ---------------------------------------------------------------- telemetry and redaction

def test_telegram_errors_never_reveal_the_token(caplog):
    assert TOKEN not in redact(f"Server error for url 'https://api.telegram.org/bot{TOKEN}/getUpdates'")
    tg = Telegram(TOKEN, "1", client=httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(502, text="Bad Gateway", request=req))))
    caplog.set_level(logging.WARNING)

    async def go():
        await tg.send("hi")
        task = asyncio.create_task(tg.poll_commands(lambda t: None))
        await asyncio.sleep(0.05)
        task.cancel()
    asyncio.run(go())
    assert "telegram poll error" in caplog.text and TOKEN not in caplog.text and "<token>" in caplog.text
    assert TOKEN not in json.dumps(tg.health) and tg.health["poll_error"] and tg.health["send_error"]
    assert tg.health["poll_fail_streak"] >= 1 and tg.health["send_fail_streak"] == 1
    assert LogNotifier().health == {"configured": False}


def test_feed_and_account_api_record_their_last_error_without_query_strings():
    feed = BinanceFeed(["BTCUSDT"])

    def boom(*a):
        raise RuntimeError("Client error '451' for url 'https://fapi.binance.com/fapi/v1/klines?symbol=X&sig=abc'")
    with pytest.raises(RuntimeError):
        asyncio.run(feed._rest(boom))
    assert "sig=abc" not in feed.health["rest_last_error"] and feed.health["rest_last_error_at"]

    def handler(req):
        if req.url.path == "/fapi/v1/time":
            return httpx.Response(200, json={"serverTime": 1})
        return httpx.Response(401, json={"code": -2015, "msg": "Invalid API-key, IP, or permissions for action."})
    api = BinanceFutures("k", "s", "testnet", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    asyncio.run(api.sync_time())
    assert api.health["last_ok"] and api.health["last_error"] is None
    with pytest.raises(BinanceError):
        asyncio.run(api.account())
    assert "-2015" in api.health["last_error"] and "signature" not in api.health["last_error"]


def test_the_bot_saves_a_health_record_and_its_last_4h_decisions(tmp_path):
    cfg = BotConfig(symbols=SYMS, d1_shadow=False, state_dir=tmp_path)
    app = App(cfg, None, LogNotifier(), StateDB(tmp_path / "bot.sqlite"))
    app.trader.last_bar_end = {"BTCUSDT": 123}
    app.save()
    rec = StateDB(tmp_path / "bot.sqlite", readonly=True).health()
    assert rec["code"] == code_id() and rec["mode"] == "paper" and rec["telegram"] == {"configured": False}
    assert rec["feed"] is None and rec["written"] >= rec["started"]
    t = Trader(cfg)
    t.restore(app.trader.state())
    assert t.last_bar_end == {"BTCUSDT": 123}
