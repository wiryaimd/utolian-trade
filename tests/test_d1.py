"""D1 shadow paper book: parity with the Round 5 research backtest, restart, mechanics, persistence."""
import asyncio
import math

import numpy as np
import pandas as pd
import pytest

from btcq.config import PERP_START
from btcq.data import store
from btcq.live.app import App
from btcq.live.book import BookConfig
from btcq.live.config import D1_SYMBOLS, BotConfig
from btcq.live.d1 import LOOKBACKS, D1Book, decide, ew_index, key, order_count, restore_problem, worst_drawdown
from btcq.live.db import StateDB
from btcq.live.feed import ReplayFeed
from btcq.live.telegram import LogNotifier
from btcq.live.trader import Trader
from btcq.research import round5 as R5
from btcq.strategies import r5_d1 as D
from conftest import needs_data

SYMS = ["BTCUSDT", "ETHUSDT"]
START = pd.Timestamp("2022-03-01", tz="UTC")
SPLIT = pd.Timestamp("2022-11-09 13:37", tz="UTC")
END = pd.Timestamp("2023-06-01", tz="UTC")
DAY = 86_400_000
ms = lambda t: int(t.value // 10**6)  # noqa: E731


def _cfg(tmp, syms=SYMS):
    return BotConfig(symbols=syms, d1_symbols=syms, shadow_long_only=False, d1_shadow=True, max_drawdown_pause=1.0,
                     state_dir=tmp)


def _replay(cfg, db, start, end, feed):
    app = App(cfg, feed, LogNotifier(), StateDB(db))
    asyncio.run(app.run_replay(ms(start), ms(end)))
    return app


@pytest.fixture(scope="module")
def feed():
    return ReplayFeed(SYMS)


@pytest.fixture(scope="module")
def d1_run(tmp_path_factory, feed):
    d = tmp_path_factory.mktemp("d1")
    return _replay(_cfg(d), d / "bot.sqlite", START, END, feed)


def _reference():
    """research.round5.d1_run on the same data: daily closes resampled from the same 1m archive, the perp
    minutes for execution, real funding, and the engine started one day later (the bot only acts on daily
    bars that close after it starts)."""
    coins, closes = {}, {}
    for s in SYMS:
        m1 = store.load_1m("dev", s)
        bars = store.resample(m1[m1["ts"] < ms(END)], "1d")
        closes[s] = pd.Series(bars["close"].to_numpy(np.float64), index=bars["ts"].to_numpy(np.int64))
        mm = m1[(m1["ts"] >= int(PERP_START.value // 10**6)) & (m1["ts"] < ms(END))].reset_index(drop=True)
        coins[s] = R5.coin_from_m1(s, mm, store.load_funding("dev", s))
    subs = R5.d1_build(coins, closes)
    parts, _ = R5.d1_run(subs, coins, R5.Window("t", "dev", START + pd.Timedelta(days=1), END))
    return parts


@needs_data
def test_d1_book_reproduces_the_research_backtest(d1_run):
    book = pd.DataFrame(d1_run.trader.books["D1"].trades)
    n = 0
    for (s, L), (_, ref) in _reference().items():
        ref = ref[ref["reason"] != "end"].reset_index(drop=True)
        got = book[(book["symbol"] == s) & (book["lookback"] == L)].reset_index(drop=True)
        assert len(got) == len(ref), (s, L, len(got), len(ref))
        n += len(got)
        if not len(ref):
            continue
        assert (pd.to_datetime(got["entry_time"], unit="ms", utc=True) == ref["entry_time"]).all(), (s, L)
        assert (pd.to_datetime(got["exit_time"], unit="ms", utc=True) == ref["exit_time"]).all(), (s, L)
        for col in ("entry_px", "exit_px"):
            np.testing.assert_allclose(got[col].to_numpy(), ref[col].to_numpy(), rtol=1e-12, err_msg=f"{s} {L} {col}")
        np.testing.assert_allclose((got["net"] / got["eq_before"]).to_numpy(), (ref["net"] / ref["eq_before"]).to_numpy(),
                                   rtol=1e-9, atol=1e-12, err_msg=f"{s} {L} ret")
        np.testing.assert_allclose(got["R"].to_numpy(), ref["R"].to_numpy(), rtol=1e-9, atol=1e-9, err_msg=f"{s} {L} R")
        np.testing.assert_allclose((got["funding"] / got["eq_before"]).to_numpy(),
                                   (ref["funding"] / ref["eq_before"]).to_numpy(), rtol=1e-9, atol=1e-12)
    assert n > 60


@needs_data
def test_restart_mid_day_gives_identical_d1_trades(d1_run, tmp_path, feed):
    cfg = _cfg(tmp_path)
    db = tmp_path / "bot.sqlite"
    _replay(cfg, db, START, SPLIT, feed)                    # run, persist, "crash" in the middle of a day
    app2 = _replay(cfg, db, SPLIT, END, feed)               # a fresh process restores from SQLite
    got = sorted((t["symbol"], t["lookback"], t["entry_time"], t["exit_time"], round(t["R"], 9))
                 for t in app2.db.trades("D1"))
    want = sorted((t["symbol"], t["lookback"], t["entry_time"], t["exit_time"], round(t["R"], 9))
                  for t in d1_run.trader.books["D1"].trades)
    assert got == want and len(got) > 60
    assert len(app2.db.daily_closes()) >= 2 * 400
    # the restarted process reloads closed trades, so /d1, /compare and the daily summary count all of them
    assert len(app2.trader.books["D1"].trades) == len(want) and app2.trader.books["D1"].started == ms(START)
    exits = [e for e in app2.db.events(100_000) if e["kind"] == "exit"]
    assert exits and all(e["ts"] == e["exit_time"] for e in exits)


@needs_data
def test_d1_daily_summary_and_command(d1_run):
    msg = d1_run._d1_text()
    assert "D1 daily trend" in msg and "holding the 2 coins" in msg and "sub-positions" in msg
    snap = d1_run.trader.snapshot()["D1"]
    assert snap["slots"] == 18 and all(x["lookback"] in LOOKBACKS for x in snap["subs"])


# ---------------------------------------------------------------- mechanics on synthetic minutes

def _minutes(day0: int, closes: list[float], gap_last: bool = False):
    """One minute stream: each day is 3 minutes (00:00, 00:01, 23:59) at that day's close."""
    out = []
    for i, c in enumerate(closes):
        d = day0 + i * DAY
        mins = [d, d + 60_000] + ([] if gap_last and i == len(closes) - 2 else [d + DAY - 60_000])
        out += [(t, c) for t in mins]
    return out


def test_decide_matches_the_research_functions():
    c = 100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0.002, 0.03, 600)))
    got = decide(c)
    s = pd.Series(c, index=np.arange(600, dtype=np.int64) * DAY)
    for L in LOOKBACKS:
        sub = D.sub_signals(s, L)
        assert got[str(L)]["state"] == int(sub.state[-1])
        assert got[str(L)]["size"] == pytest.approx(float(sub.size[-1]), nan_ok=True)
        assert got[str(L)]["trail"] == pytest.approx(float(sub.trail[-1]), nan_ok=True)


def test_book_enters_at_the_next_minute_open_charges_funding_and_exits():
    cfg = BookConfig(initial_equity=4500.0, taker_fee=0.0005, slippage={"BTCUSDT": 0.0001}, latency_min=1)
    events = []
    b = D1Book("D1", ["BTCUSDT"], cfg, events.append)
    hist = 100 * np.exp(np.cumsum(np.full(400, 0.004) + np.random.default_rng(2).normal(0, 0.01, 400)))
    day0 = 1_000 * DAY
    b.bootstrap("BTCUSDT", pd.DataFrame({"ts": day0 + np.arange(400) * DAY, "close": hist}))
    up = [hist[-1] * 1.02]
    for t, c in _minutes(day0 + 400 * DAY, up):
        b.on_minute("BTCUSDT", t, c, c, c, c)
    assert [e["kind"] for e in events] == ["daily_close"] and len(b.queue["BTCUSDT"]) == 1
    d = b.queue["BTCUSDT"][0][1]
    longs = [L for L in LOOKBACKS if d[str(L)]["state"] == 1]
    assert longs and all(b.subs[key("BTCUSDT", L)].qty == 0 for L in LOOKBACKS)
    nxt = day0 + 401 * DAY
    b.on_minute("BTCUSDT", nxt, 999.0, 999.0, 999.0, 999.0)            # 00:00: not yet (latency)
    assert not b.trades and all(b.subs[key("BTCUSDT", L)].qty == 0 for L in LOOKBACKS)
    b.on_minute("BTCUSDT", nxt + 60_000, 200.0, 200.0, 200.0, 200.0)    # 00:01 open: fills
    sp = b.subs[key("BTCUSDT", longs[0])]
    cash0 = 4500.0 / 9
    assert sp.epx == pytest.approx(200.0 * 1.0001)
    assert sp.qty == pytest.approx(min(d[str(longs[0])]["size"], 2.0) * cash0 / sp.epx)
    assert b.cash[key("BTCUSDT", longs[0])] == pytest.approx(cash0 - sp.qty * sp.epx * 0.0005)
    b.on_minute("BTCUSDT", nxt + 120_000, 200.0, 200.0, 200.0, 200.0, funding_rate=0.001)
    assert sp.e_fund == pytest.approx(sp.qty * 200.0 * 0.001)
    b2 = D1Book("D1", ["BTCUSDT"], cfg)
    b2.restore(b.state())
    assert b2.equity() == pytest.approx(b.equity()) and b2.subs == b.subs and b2.queue == b.queue
    pos = b.pos["BTCUSDT"]
    assert pos.dir == 1 and pos.qty == pytest.approx(sum(b.subs[key("BTCUSDT", L)].qty for L in longs))
    c = 200.0 * 0.4                                                     # a deep fall flips the fast lookbacks flat
    b.on_minute("BTCUSDT", nxt + DAY - 60_000, c, c, c, c)
    b.on_minute("BTCUSDT", nxt + DAY + 60_000, 79.0, 79.0, 79.0, 79.0)
    assert b.trades and all(t["reason"] == "exit_signal" and t["exit_px"] == pytest.approx(79.0 * 0.9999)
                            for t in b.trades)
    assert all(b.subs[key("BTCUSDT", L)].qty == 0 for L in longs if L <= 30)


def test_a_missing_last_minute_still_closes_the_day():
    cfg = BookConfig(initial_equity=900.0, slippage={}, latency_min=1)
    events = []
    b = D1Book("D1", ["BTCUSDT"], cfg, events.append)
    day0 = 2_000 * DAY
    b.bootstrap("BTCUSDT", pd.DataFrame({"ts": day0 + np.arange(200) * DAY, "close": np.linspace(10, 20, 200)}))
    d1 = day0 + 200 * DAY
    b.on_minute("BTCUSDT", d1 + 600_000, 21.0, 21.0, 21.0, 21.0)       # the day's 23:59 minute never arrives
    b.on_minute("BTCUSDT", d1 + DAY + 300_000, 22.0, 22.0, 22.0, 22.0)  # next day: closes the old day first
    closes = [e for e in events if e["kind"] == "daily_close"]
    assert len(closes) == 1 and closes[0]["day_ms"] == d1 and closes[0]["close"] == 21.0
    assert b.trades == [] and any(b.subs[key("BTCUSDT", L)].qty > 0 for L in LOOKBACKS)   # executed at 00:05


def test_trader_runs_d1_only_when_enabled_and_snapshots_sub_positions(tmp_path):
    t = Trader(BotConfig(symbols=["BTCUSDT"], d1_symbols=["BTCUSDT"], d1_shadow=True, shadow_long_only=False,
                         state_dir=tmp_path))
    assert set(t.books) == {"F1", "D1"} and t.books["D1"].slots == 9
    assert "D1" not in Trader(BotConfig(symbols=["BTCUSDT"], state_dir=tmp_path)).books
    snap = t.snapshot()["D1"]
    assert snap["subs"] == [] and snap["slots"] == 9 and snap["equity"] == pytest.approx(10_000.0)
    assert math.isclose(snap["cash"], 10_000.0)


@needs_data
def test_dashboard_shows_the_d1_panel(d1_run):
    import json
    from btcq.live.dashboard import DashboardData
    p = DashboardData(d1_run.cfg.state_dir / "bot.sqlite").payload()
    json.dumps(p, allow_nan=False)
    assert "D1" in p["books"] and p["books"]["D1"]["role"] == "shadow" and p["main"] == "F1"
    d1 = p["d1"]
    assert d1["slots"] == 18 and [c["symbol"] for c in d1["coins"]] == SYMS and d1["days_running"] > 400
    assert len(d1["hold_series"]) > 400 and d1["alpha"] and 0.0 <= d1["alpha"]["psr_alpha"] <= 1.0
    assert [h["name"] for h in d1["health"]] == ["D1 sub-trades per week", "D1 days with an order",
                                                 "D1 worst drawdown", "D1 vs holding the coins"]
    assert all(h["status"] in ("ok", "watch", "bad", "wait") for h in d1["health"])
    assert any(t.get("lookback") for t in p["trades"] if t["book"] == "D1") and "D1" in p["series"]
    assert d1["sub_trades_per_week"] and 0 < d1["order_days_share"] <= 1
    assert d1["max_drawdown"] >= d1["drawdown"] >= 0 and d1["hold_series"][0][1] == pytest.approx(10_000.0)
    assert 0 < d1["max_drawdown_eod"] <= d1["max_drawdown"]
    assert d1["hold_return"] == pytest.approx(d1["hold_series"][-1][1] / 10_000.0 - 1)
    weeks = (p["data_ts"] - d1["started"]) / (7 * DAY)
    assert d1["sub_trades_per_week"] == pytest.approx(d1["closed_trades"] / weeks)
    assert p["books"]["D1"]["stats"]["per_week"] == pytest.approx(d1["closed_trades"] / weeks)
    n_f1 = len(d1_run.db.trades("F1"))
    assert sum(t["book"] == "F1" for t in p["trades"]) == min(n_f1, 300)       # per-book cap
    assert all(e["ts"] for e in p["events"])


# ---------------------------------------------------------------- isolation, universe, restart record

def test_pause_resume_and_the_kill_switch_never_touch_d1(tmp_path):
    cfg = BotConfig(symbols=list(D1_SYMBOLS), d1_shadow=True, state_dir=tmp_path)    # production defaults
    assert cfg.max_drawdown_pause == 0.35
    app = App(cfg, None, LogNotifier(), StateDB(tmp_path / "bot.sqlite"))
    d1 = app.trader.books["D1"]
    d1.peak = 20_000.0                       # D1 is 50% below its peak: still never paused
    d1.on_minute("BTCUSDT", 1_000 * DAY, 1.0, 1.0, 1.0, 1.0)
    assert not d1.paused and d1.peak == 20_000.0
    asyncio.run(app.command("/pause"))
    assert app.trader.books["F1"].paused and not d1.paused
    asyncio.run(app.command("/resume"))
    assert not app.trader.books["F1"].paused and d1.peak == 20_000.0
    assert not any(e["book"] == "D1" for e in app.db.events(100))


def test_d1_trades_only_its_registered_coins(tmp_path):
    t = Trader(BotConfig(symbols=list(D1_SYMBOLS) + ["ADAUSDT"], d1_shadow=True, state_dir=tmp_path))
    d1 = t.books["D1"]
    assert d1.symbols == list(D1_SYMBOLS) and d1.slots == 45
    d1.on_minute("ADAUSDT", 1_000 * DAY, 1.0, 1.0, 1.0, 1.0)            # not a D1 coin: ignored
    assert d1.started is None and d1.equity() == pytest.approx(10_000.0)
    assert "D1" not in Trader(BotConfig(symbols=["BTCUSDT", "ETHUSDT"], d1_shadow=True, state_dir=tmp_path)).books
    st = d1.state()
    with pytest.raises(ValueError):
        D1Book("D1", ["BTCUSDT"], d1.cfg).restore(st)
    assert restore_problem(st, ["BTCUSDT"], {}) and restore_problem(st, list(D1_SYMBOLS), {}) is None
    never_ran = {s: 1_000 * DAY for s in D1_SYMBOLS}                       # a saved D1 that never saw a minute
    assert restore_problem(st, list(D1_SYMBOLS), never_ran) is None


def test_a_d1_that_missed_minutes_starts_a_new_record(tmp_path):
    cfg = BotConfig(symbols=["BTCUSDT"], d1_symbols=["BTCUSDT"], d1_shadow=True, state_dir=tmp_path)
    t = Trader(cfg)
    b = t.books["D1"]
    b.on_minute("BTCUSDT", 1_000 * DAY, 1.0, 1.0, 1.0, 1.0)
    st = {"books": {"D1": b.state()}, "last_minute": {"BTCUSDT": 1_000 * DAY}}
    assert restore_problem(st["books"]["D1"], ["BTCUSDT"], st["last_minute"]) is None
    st["last_minute"]["BTCUSDT"] += 5 * DAY                               # the bot ran on without D1
    assert "switched off" in restore_problem(st["books"]["D1"], ["BTCUSDT"], st["last_minute"])
    app = App(cfg, None, LogNotifier(), StateDB(tmp_path / "bot.sqlite"))
    app._check_d1_state(st)
    assert "D1" not in st["books"] and "new record" in app.outbox[-1]
    app2 = App(cfg, None, LogNotifier(), StateDB(tmp_path / "bot2.sqlite"))
    app2._check_d1_state({"books": {}, "last_minute": {}})
    assert "started" in app2.outbox[-1]


def test_a_day_without_prices_is_filled_but_not_traded_like_the_research_calendar():
    cfg = BookConfig(initial_equity=900.0, slippage={}, latency_min=1)
    events = []
    b = D1Book("D1", ["BTCUSDT"], cfg, events.append)
    day0 = 3_000 * DAY
    hist = np.linspace(10, 20, 200)
    b.bootstrap("BTCUSDT", pd.DataFrame({"ts": np.r_[day0 + np.arange(150) * DAY, day0 + np.arange(151, 200) * DAY],
                                         "close": np.r_[hist[:150], hist[151:]]}))
    assert [x[0] for x in b.closes["BTCUSDT"]] == list(day0 + np.arange(200) * DAY)   # bootstrap gap filled
    assert b.closes["BTCUSDT"][150][1] == hist[149]
    d1 = day0 + 200 * DAY
    b.on_minute("BTCUSDT", d1 + DAY - 60_000, 21.0, 21.0, 21.0, 21.0)     # day d1 closes
    b.on_minute("BTCUSDT", d1 + 2 * DAY, 22.0, 22.0, 22.0, 22.0)          # day d1+1 had no minutes at all
    closes = [e["day_ms"] for e in events if e["kind"] == "daily_close"]
    assert closes == [d1, d1 + DAY] and b.closes["BTCUSDT"][-1] == [d1 + DAY, 21.0]
    assert any(e["kind"] == "warning" for e in events)
    # d1's decision executed at this minute; the filled day is never traded (the engine only acts on days with
    # prices), but it counts in the history the next real close decides on
    assert b.queue["BTCUSDT"] == [] and any(b.subs[key("BTCUSDT", L)].qty > 0 for L in LOOKBACKS)
    b.on_minute("BTCUSDT", d1 + 3 * DAY - 60_000, 23.0, 23.0, 23.0, 23.0)
    ref = decide(np.array([x[1] for x in b.closes["BTCUSDT"]]))
    assert b.closes["BTCUSDT"][-2:] == [[d1 + DAY, 21.0], [d1 + 2 * DAY, 23.0]]
    assert len(b.queue["BTCUSDT"]) == 1 and b.queue["BTCUSDT"][0][0] == d1 + 3 * DAY + 60_000
    assert repr(b.queue["BTCUSDT"][0][1]) == repr(ref)                     # (repr: NaN trails compare equal)


def test_benchmark_and_order_helpers():
    started = 10 * DAY + 12 * 3_600_000
    by_day = {9 * DAY: {"A": 1.0, "B": 1.0}, 10 * DAY: {"A": 110.0, "B": 90.0}, 11 * DAY: {"A": 121.0, "B": 90.0}}
    idx = ew_index({"A": 100.0, "B": 100.0}, started, by_day, {"A": 121.0, "B": 99.0}, 12 * DAY + 3_600_000)
    assert [t for t, _ in idx] == [started, 11 * DAY, 12 * DAY, 12 * DAY + 3_600_000]   # day 9 closed before start
    assert idx[1][1] == pytest.approx(1.0) and idx[2][1] == pytest.approx(1.05)
    assert idx[3][1] == pytest.approx(1.05 * 1.05)
    trades = [{"symbol": "A", "entry_time": 5 * DAY + 60_000, "exit_time": 8 * DAY + 60_000},
              {"symbol": "A", "entry_time": 8 * DAY + 60_000, "exit_time": 9 * DAY + 60_000}]
    subs = [{"symbol": "B", "since": 9 * DAY + 60_000}]
    assert order_count(trades, subs, 8 * DAY) == 3 and order_count(trades, subs, 0) == 4
    assert worst_drawdown([100.0, 120.0, 90.0, 130.0], 100.0) == pytest.approx(0.25)


def test_switching_d1_off_and_on_ends_its_record_and_says_so(tmp_path):
    on = BotConfig(symbols=["BTCUSDT"], d1_symbols=["BTCUSDT"], d1_shadow=True, state_dir=tmp_path)
    off = BotConfig(symbols=["BTCUSDT"], d1_symbols=["BTCUSDT"], d1_shadow=False, state_dir=tmp_path)
    path = tmp_path / "bot.sqlite"
    a1 = App(on, None, LogNotifier(), StateDB(path))
    a1._check_d1_state(None)
    assert "started" in a1.outbox[-1]
    a1.trader.books["D1"].on_minute("BTCUSDT", 1_000 * DAY, 1.0, 1.0, 1.0, 1.0)
    a1._snapshot(1_000 * DAY)
    a1.save()
    a2 = App(off, None, LogNotifier(), StateDB(path))
    a2._check_d1_state(a2.db.load_state())
    assert "record ends here" in a2.outbox[-1]
    a2.save()
    assert "D1" not in a2.db.load_state()["books"]
    a3 = App(on, None, LogNotifier(), StateDB(path))
    a3._check_d1_state(a3.db.load_state())
    assert "new record" in a3.outbox[-1] and "switched off" in a3.outbox[-1]


def test_paper_rows_are_committed_with_the_state_so_a_crash_cannot_duplicate_them(tmp_path):
    path = tmp_path / "bot.sqlite"
    db = StateDB(path)
    db.add_trade({"book": "F1-live", "symbol": "BTCUSDT", "exit_time": 2})   # a real fill: committed at once
    db.add_trade({"book": "D1", "symbol": "BTCUSDT", "exit_time": 1})
    db.add_event(1, "D1", "exit", {"exit_time": 1})
    db.set_equity("D1", 0, 1.0)
    db.conn.close()                                                         # "crash" before the state save
    db = StateDB(path)
    assert [t["book"] for t in db.trades()] == ["F1-live"] and db.events(5) == [] and db.equity_curve("D1") == []
    db.add_trade({"book": "D1", "symbol": "BTCUSDT", "exit_time": 3})
    db.save_state({"books": {}})
    db.conn.close()
    assert len(StateDB(path).trades("D1")) == 1


def test_drawdown_status_boundaries_follow_the_registered_rule():
    from btcq.live.dashboard import d1_health
    base = {"days_running": 0, "sub_trades_per_week": None, "closed_trades": 0, "order_days_share": None,
            "orders_last_7d": 0, "alpha": None, "drawdown": 0.0}
    status = lambda eod: d1_health({**base, "max_drawdown": eod, "max_drawdown_eod": eod})[2]["status"]  # noqa: E731
    assert [status(v) for v in (0.1999, 1 - 8000 / 10000, 1 - 7000 / 10000, 0.3001)] == ["ok", "watch", "watch", "bad"]


def test_old_exit_events_get_their_time_back(tmp_path):
    import json
    import sqlite3
    db = tmp_path / "bot.sqlite"
    StateDB(db).close()
    con = sqlite3.connect(db)
    con.execute("INSERT INTO events(ts, book, kind, data) VALUES(0, 'F1', 'exit', ?)", (json.dumps({"exit_time": 123}),))
    con.commit()
    con.close()
    assert StateDB(db).events(5)[0]["ts"] == 123
