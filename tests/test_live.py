"""Paper bot: parity with the backtest engine, restart, multi-market plumbing, kill switch, Telegram."""
import asyncio
import math

import httpx
import numpy as np
import pandas as pd
import pytest

from btcq.backtest import engine
from btcq.config import SLIPPAGE, Costs, Risk
from btcq.data import store
from btcq.live.app import App
from btcq.live.bars import BarBuilder, Minute
from btcq.live.book import BookConfig, PaperBook
from btcq.live.config import BotConfig
from btcq.live.db import StateDB
from btcq.live.feed import ReplayFeed
from btcq.live.telegram import LogNotifier, Telegram
from btcq.live.trader import F1_PARAMS
from btcq.strategies.r3_trend import S1Frozen
from conftest import needs_data

START = pd.Timestamp("2023-03-01", tz="UTC")
SPLIT = pd.Timestamp("2023-08-15 10:37", tz="UTC")
END = pd.Timestamp("2024-01-01", tz="UTC")
ms = lambda t: int(t.value // 10**6)  # noqa: E731


def _cfg(tmp_path, symbols, shadow=False):
    return BotConfig(symbols=symbols, shadow_long_only=shadow, max_drawdown_pause=1.0, state_dir=tmp_path)


def _replay(cfg, db_path, start, end, feed=None):
    app = App(cfg, feed or ReplayFeed(cfg.symbols), LogNotifier(), StateDB(db_path))
    asyncio.run(app.run_replay(ms(start), ms(end)))
    return app


def _engine_trades(symbol, start):
    mk = engine.Market.load("dev", symbol)
    res = engine.run(mk, S1Frozen().compute(mk, **F1_PARAMS), Costs(slippage=SLIPPAGE[symbol]),
                     Risk(0.01, 3.0, 10_000.0), start=start, latency_min=1)
    t = res.trades
    return t[t["reason"] != "end"].reset_index(drop=True)


@pytest.fixture(scope="module")
def btc_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("btc")
    return _replay(_cfg(d, ["BTCUSDT"]), d / "bot.sqlite", START, END)


@needs_data
def test_paper_book_reproduces_backtest_engine_exactly(btc_run):
    book = pd.DataFrame(btc_run.trader.books["F1"].trades)
    # The engine may act on the 4h bar that closed exactly at START; the bot only acts on bars
    # that close after it starts, so align the engine's start one hour later.
    ref = _engine_trades("BTCUSDT", START + pd.Timedelta(hours=1))
    assert len(book) == len(ref) > 10
    assert (pd.to_datetime(book["entry_time"], unit="ms", utc=True) == ref["entry_time"]).all()
    assert (pd.to_datetime(book["exit_time"], unit="ms", utc=True) == ref["exit_time"]).all()
    assert (book["dir"].to_numpy() == ref["dir"].to_numpy()).all()
    assert list(book["reason"]) == list(ref["reason"])
    for col in ("entry_px", "exit_px", "qty", "fees", "funding", "net", "R"):
        np.testing.assert_allclose(book[col].to_numpy(), ref[col].to_numpy(), rtol=1e-9, atol=1e-9,
                                   err_msg=col)


@needs_data
def test_restart_mid_bar_gives_identical_trades(btc_run, tmp_path):
    cfg = _cfg(tmp_path, ["BTCUSDT"])
    db = tmp_path / "bot.sqlite"
    feed = ReplayFeed(cfg.symbols)
    _replay(cfg, db, START, SPLIT, feed)                    # run, persist, "crash"
    app2 = _replay(cfg, db, SPLIT, END, feed)               # fresh process restores from SQLite
    got = sorted((t["entry_time"], t["exit_time"], round(t["R"], 9)) for t in app2.db.trades("F1"))
    want = sorted((t["entry_time"], t["exit_time"], round(t["R"], 9)) for t in btc_run.trader.books["F1"].trades)
    assert got == want


@needs_data
def test_five_markets_shared_account_and_long_only_shadow(tmp_path):
    syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT"]
    events = []
    cfg = _cfg(tmp_path, syms, shadow=True)
    app = App(cfg, ReplayFeed(syms), LogNotifier(), StateDB(tmp_path / "bot.sqlite"))
    orig = app._on_event
    app.trader.on_event = lambda e: (events.append(e), orig(e))
    for b in app.trader.books.values():
        b.on_event = app.trader._event
    s, e = pd.Timestamp("2023-09-01", tz="UTC"), END
    asyncio.run(app.run_replay(ms(s), ms(e)))
    f1 = pd.DataFrame(app.trader.books["F1"].trades)
    shadow = pd.DataFrame(app.trader.books["F1-long"].trades)
    assert len(f1) > 10 and len(shadow) > 0
    assert (shadow["dir"] == 1).all()
    entries = [x for x in events if x["kind"] == "entry" and x["book"] == "F1"]
    for x in entries:  # per-symbol notional cap = 3x equity / 5 markets
        assert x["notional"] <= 3 * x["equity"] / 5 * (1 + 1e-9)
        assert x["risk"] <= 0.01 * x["equity"] * (1 + 1e-9)
    # Sizing is shared, but timing per market must match the single-market engine exactly.
    ref = _engine_trades("ETHUSDT", s + pd.Timedelta(hours=1))
    ref = ref[ref["exit_time"] < e]
    eth = f1[f1["symbol"] == "ETHUSDT"]
    assert list(pd.to_datetime(eth["entry_time"], unit="ms", utc=True)) == list(ref["entry_time"])
    assert any(x["kind"] == "day_end" for x in events)


def test_bar_builder_matches_resample_with_gap():
    rng = np.random.default_rng(5)
    t0 = ms(pd.Timestamp("2023-01-02", tz="UTC"))
    ts = t0 + 60_000 * np.arange(60 * 9)
    ts = np.delete(ts, np.arange(200, 260))   # one hour missing inside the second 4h bar
    px = 100 + np.cumsum(rng.normal(0, 0.1, len(ts)))
    m1 = pd.DataFrame({"ts": ts, "open": px, "high": px + 0.3, "low": px - 0.3, "close": px + 0.1,
                       "volume": 1.0, "quote_volume": px, "taker_buy_volume": 0.5})
    b = BarBuilder("4h")
    closed = []
    for r in m1.itertuples(index=False):
        closed += b.close_if_new_bucket(r.ts)
        closed += b.add(Minute(*r))
    ours = pd.DataFrame(closed, columns=["ts", "open", "high", "low", "close", "volume", "quote_volume",
                                         "taker_buy_volume", "n_min"])
    ref = store.resample(m1, "4h")
    pd.testing.assert_frame_equal(ours[ref.columns.drop("n_min")].iloc[:2].reset_index(drop=True),
                                  ref[ref.columns.drop("n_min")].iloc[:2].reset_index(drop=True),
                                  check_dtype=False)


def _row(entry=0, stop=math.nan):
    return {"entry": entry, "stop": stop, "target": math.nan, "trail": math.nan, "exit_long": False,
            "exit_short": False, "size": math.nan, "max_bars": 0}


def test_kill_switch_pauses_new_entries_but_keeps_exits():
    events = []
    bk = PaperBook("F1", ["X"], BookConfig(risk_per_trade=0.04, max_drawdown_pause=0.06, taker_fee=0.0,
                                           latency_min=0), events.append)
    t = 0
    for _ in range(2):  # two stopped-out longs at -4% each
        bk.schedule("X", t, _row(1, 90.0))
        bk.on_minute("X", t, 100, 100, 100, 100)
        bk.on_minute("X", t + 60_000, 100, 100, 89, 89)
        t += 120_000
    assert bk.paused and any(e["kind"] == "paused" for e in events)
    bk.schedule("X", t, _row(1, 90.0))
    bk.on_minute("X", t, 100, 100, 100, 100)
    assert bk.pos["X"].dir == 0 and events[-1]["kind"] == "skipped"
    bk.resume()
    bk.schedule("X", t + 60_000, _row(1, 90.0))
    bk.on_minute("X", t + 60_000, 100, 100, 100, 100)
    assert bk.pos["X"].dir == 1


def test_telegram_only_answers_configured_chat():
    sent = []

    def handler(req: httpx.Request):
        sent.append(req.read().decode())
        return httpx.Response(200, json={"ok": True})

    tg = Telegram("T", "42", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    calls = []

    async def on_cmd(text):
        calls.append(text)
        return "pong"

    asyncio.run(tg.handle_updates([
        {"update_id": 1, "message": {"chat": {"id": 999}, "text": "/status"}},
        {"update_id": 2, "message": {"chat": {"id": 42}, "text": "/status"}}], on_cmd))
    assert calls == ["/status"] and len(sent) == 1 and '"chat_id":"42"' in sent[0].replace(" ", "")
    assert tg.offset == 3


@needs_data
def test_commands_pause_resume_status(tmp_path):
    cfg = _cfg(tmp_path, ["BTCUSDT"], shadow=True)
    app = App(cfg, ReplayFeed(cfg.symbols), LogNotifier(), StateDB(tmp_path / "bot.sqlite"))
    asyncio.run(app.run_replay(ms(pd.Timestamp("2023-11-01", tz="UTC")), ms(END)))
    asyncio.run(app.command("/pause"))
    assert all(b.paused for b in app.trader.books.values())
    assert "PAUSED" in asyncio.run(app.command("/status"))
    asyncio.run(app.command("/resume"))
    assert not any(b.paused for b in app.trader.books.values())
    assert "F1" in asyncio.run(app.command("/compare"))
