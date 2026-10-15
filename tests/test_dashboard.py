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
        assert "F1 trading monitor" in page
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
