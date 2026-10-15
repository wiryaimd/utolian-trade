"""Live executor against a simulated Binance USDS-M exchange (no network, no real orders)."""
import asyncio
import hashlib
import hmac
import math
import re
from urllib.parse import parse_qsl

import httpx
import pandas as pd
import pytest

from btcq.live.app import App
from btcq.live.binance_api import BinanceFutures
from btcq.live.config import LIVE_ACK, BotConfig
from btcq.live.db import StateDB
from btcq.live.executor import ExecConfig, LiveExecutor, LiveSetupError
from btcq.live.feed import ReplayFeed
from btcq.live.telegram import LogNotifier
from conftest import needs_data

SECRET = "test-secret"
CID = re.compile(r"^[\.A-Z\:/a-z0-9_-]{1,36}$")
RULES = {"BTCUSDT": ("0.10", "0.001", "100"), "ETHUSDT": ("0.01", "0.001", "20")}


class FakeBinance:
    """Minimal stateful USDS-M futures exchange: one-way mode, market orders, closePosition algos."""

    def __init__(self, wallet=10_000.0, prices=None):
        self.wallet = wallet
        self.price = dict(prices or {"BTCUSDT": 50_000.0, "ETHUSDT": 3_000.0})
        self.pos: dict[str, float] = {}
        self.entry: dict[str, float] = {}
        self.orders: dict[str, dict] = {}
        self.algos: dict[str, dict] = {}
        self.hedge = False
        self.fail: set[str] = set()
        self.calls: list[tuple[str, str, dict]] = []
        self.leverage: dict[str, int] = {}

    # -------- exchange mechanics
    def _fill(self, sym, side, qty, price):
        signed = qty if side == "BUY" else -qty
        old = self.pos.get(sym, 0.0)
        new = round(old + signed, 10)
        if old and (old > 0) != (signed > 0):  # reducing: realise PnL
            closed = min(abs(signed), abs(old))
            self.wallet += closed * (price - self.entry[sym]) * (1 if old > 0 else -1)
        if new == 0:
            self.pos.pop(sym, None)
            self.entry.pop(sym, None)
        else:
            if not old or (old > 0) == (signed > 0):
                self.entry[sym] = price
            self.pos[sym] = new

    def bar(self, sym, o, h, l, c):
        """Advance one 1m bar: trigger conditional orders on the bar's range, then set last price."""
        for a in list(self.algos.values()):
            if a["symbol"] != sym or a["algoStatus"] != "NEW":
                continue
            trig = float(a["triggerPrice"])
            stop = a["type"] == "STOP_MARKET"
            sell = a["side"] == "SELL"
            hit = (l <= trig) if (stop and sell) or (not stop and not sell) else (h >= trig)
            if hit and sym in self.pos:
                px = min(o, trig) if (stop and sell) else max(o, trig) if stop else trig
                self._fill(sym, a["side"], abs(self.pos[sym]), px)
                a.update(algoStatus="FINISHED", actualPrice=str(px), actualOrderId="1")
        self.price[sym] = c

    def unrealized(self):
        return sum(q * (self.price[s] - self.entry[s]) for s, q in self.pos.items())

    # -------- HTTP
    def handler(self, req: httpx.Request) -> httpx.Response:
        q = req.url.query.decode()
        params = dict(parse_qsl(q))
        if "signature" in params:
            body = q[: q.rindex("&signature=")]
            good = hmac.new(SECRET.encode(), body.encode(), hashlib.sha256).hexdigest()
            if params["signature"] != good:
                return httpx.Response(400, json={"code": -1022, "msg": "Signature for this request is not valid."})
            assert req.headers["X-MBX-APIKEY"] == "k"
        key = (req.method, req.url.path)
        self.calls.append((req.method, req.url.path, params))
        err = lambda code, msg: httpx.Response(400, json={"code": code, "msg": msg})  # noqa: E731
        if key == ("GET", "/fapi/v1/time"):
            return httpx.Response(200, json={"serverTime": 1})
        if key == ("GET", "/fapi/v1/exchangeInfo"):
            syms = [{"symbol": s, "filters": [{"filterType": "PRICE_FILTER", "tickSize": t},
                                              {"filterType": "LOT_SIZE", "stepSize": st, "minQty": st, "maxQty": "1000"},
                                              {"filterType": "MARKET_LOT_SIZE", "stepSize": st, "minQty": st,
                                               "maxQty": "100"},
                                              {"filterType": "MIN_NOTIONAL", "notional": mn}]}
                    for s, (t, st, mn) in RULES.items()]
            return httpx.Response(200, json={"symbols": syms})
        if key == ("GET", "/fapi/v1/positionSide/dual"):
            return httpx.Response(200, json={"dualSidePosition": self.hedge})
        if key == ("POST", "/fapi/v1/marginType"):
            return err(-4046, "No need to change margin type.")
        if key == ("POST", "/fapi/v1/leverage"):
            self.leverage[params["symbol"]] = int(params["leverage"])
            return httpx.Response(200, json={"leverage": int(params["leverage"])})
        if key == ("GET", "/fapi/v3/account"):
            eq = self.wallet + self.unrealized()
            return httpx.Response(200, json={"totalMarginBalance": str(eq), "availableBalance": str(eq)})
        if key == ("GET", "/fapi/v3/positionRisk"):
            return httpx.Response(200, json=[{"symbol": s, "positionAmt": str(a)} for s, a in self.pos.items()])
        if key == ("POST", "/fapi/v1/order"):
            if "order" in self.fail:
                return err(-2019, "Margin is insufficient.")
            sym, side, qty = params["symbol"], params["side"], float(params["quantity"])
            assert CID.match(params["newClientOrderId"])
            if params.get("reduceOnly") == "true":
                qty = min(qty, abs(self.pos.get(sym, 0.0)))
            px = self.price[sym]
            self._fill(sym, side, qty, px)
            o = {"clientOrderId": params["newClientOrderId"], "status": "FILLED", "avgPrice": str(px),
                 "executedQty": str(qty)}
            self.orders[params["newClientOrderId"]] = o
            return httpx.Response(200, json=o)
        if key == ("GET", "/fapi/v1/order"):
            o = self.orders.get(params["origClientOrderId"])
            return httpx.Response(200, json=o) if o else err(-2013, "Order does not exist.")
        if key == ("POST", "/fapi/v1/algoOrder"):
            if "algo" in self.fail:
                return err(-1000, "simulated failure")
            assert params["algoType"] == "CONDITIONAL" and params["closePosition"] == "true"
            assert "quantity" not in params and "reduceOnly" not in params
            assert CID.match(params["clientAlgoId"])
            trig, px, sell = float(params["triggerPrice"]), self.price[params["symbol"]], params["side"] == "SELL"
            immediate = (trig >= px if sell else trig <= px) if params["type"] == "STOP_MARKET" else \
                (trig <= px if sell else trig >= px)
            if immediate:
                return err(-2021, "Order would immediately trigger.")
            a = {**params, "algoStatus": "NEW", "actualPrice": "0"}
            self.algos[params["clientAlgoId"]] = a
            return httpx.Response(200, json=a)
        if key == ("GET", "/fapi/v1/algoOrder"):
            a = self.algos.get(params["clientAlgoId"])
            return httpx.Response(200, json=a) if a else err(-2013, "Order does not exist.")
        if key == ("DELETE", "/fapi/v1/algoOrder"):
            a = self.algos.get(params["clientAlgoId"])
            if not a or a["algoStatus"] != "NEW":
                return err(-2011, "Unknown order sent.")
            a["algoStatus"] = "CANCELED"
            return httpx.Response(200, json={"code": "200", "msg": "success"})
        if key == ("GET", "/fapi/v1/openAlgoOrders"):
            return httpx.Response(200, json=[a for a in self.algos.values() if a["algoStatus"] == "NEW"])
        if key == ("GET", "/fapi/v1/userTrades"):
            return httpx.Response(200, json=[])
        return err(-1, f"unhandled {key}")


class Clock:
    def __init__(self, t=1_700_000_000_000):
        self.t = t

    def __call__(self):
        return self.t


def make(fake=None, clock=None, symbols=("BTCUSDT", "ETHUSDT"), **cfg):
    fake = fake or FakeBinance()
    api = BinanceFutures("k", SECRET, "testnet", client=httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)))
    events = []
    ex = LiveExecutor("F1", api, list(symbols), ExecConfig(**cfg), events.append, clock or Clock())
    return fake, ex, events


def row(entry=0, stop=math.nan, exit_long=False, exit_short=False, target=math.nan):
    return {"entry": entry, "stop": stop, "target": target, "trail": math.nan, "exit_long": exit_long,
            "exit_short": exit_short, "size": math.nan, "max_bars": 0}


def run(coro):
    return asyncio.run(coro)


def test_startup_configures_account_and_refuses_hedge_mode():
    fake, ex, ev = make()
    run(ex.startup())
    assert fake.leverage == {"BTCUSDT": 5, "ETHUSDT": 5}
    assert ev[-1]["kind"] == "live_ready" and ev[-1]["equity"] == 10_000
    fake2, ex2, _ = make(FakeBinance())
    fake2.hedge = True
    with pytest.raises(LiveSetupError):
        run(ex2.startup())


def test_entry_sends_rounded_market_order_and_close_all_stop():
    clock = Clock()
    fake, ex, ev = make(clock=clock)
    run(ex.startup())
    run(ex.on_decision("BTCUSDT", clock.t, row(1, 48_765.4321), 50_000.0))
    order = [c for c in fake.calls if c[1] == "/fapi/v1/order"][0][2]
    algo = [c for c in fake.calls if c[1] == "/fapi/v1/algoOrder" and c[0] == "POST"][0][2]
    # 1% of 10,000 at a 1,234.57 stop distance = 0.08100.. BTC, floored to the 0.001 step
    assert order["type"] == "MARKET" and order["side"] == "BUY" and order["quantity"] == "0.081"
    assert algo["type"] == "STOP_MARKET" and algo["side"] == "SELL" and algo["triggerPrice"] == "48765.4"
    assert algo["workingType"] == "CONTRACT_PRICE"
    assert fake.pos["BTCUSDT"] == pytest.approx(0.081)
    e = ev[-1]
    assert e["kind"] == "live_entry" and e["qty"] == pytest.approx(0.081) and e["slippage_bps"] == pytest.approx(0)


def test_exit_signal_cancels_stop_and_closes_reduce_only():
    clock = Clock()
    fake, ex, ev = make(clock=clock)
    run(ex.startup())
    run(ex.on_decision("BTCUSDT", clock.t, row(1, 49_000.0), 50_000.0))
    fake.bar("BTCUSDT", 50_000, 52_100, 50_000, 52_000)
    run(ex.on_decision("BTCUSDT", clock.t, row(exit_long=True), 52_000.0))
    assert "BTCUSDT" not in fake.pos
    assert all(a["algoStatus"] == "CANCELED" for a in fake.algos.values())
    close = [c[2] for c in fake.calls if c[1] == "/fapi/v1/order"][-1]
    assert close["reduceOnly"] == "true" and close["side"] == "SELL"
    x = [e for e in ev if e["kind"] == "live_exit"][0]
    assert x["reason"] == "exit_signal" and x["exit_px"] == 52_000
    assert x["R"] == pytest.approx((2_000 * 0.01 - 0.01 * (50_000 + 52_000) * 0.0005) / (0.01 * 1_000))


def test_stop_fill_on_exchange_is_recorded_by_reconcile():
    clock = Clock()
    fake, ex, ev = make(clock=clock)
    run(ex.startup())
    run(ex.on_decision("BTCUSDT", clock.t, row(1, 49_000.0), 50_000.0))
    fake.bar("BTCUSDT", 49_500, 49_600, 48_800, 48_900)   # stop triggers on the exchange
    run(ex.reconcile(force=True))
    x = [e for e in ev if e["kind"] == "live_exit"][0]
    assert x["reason"] == "stop" and x["exit_px"] == 49_000 and x["R"] < -1
    assert ex.pos["BTCUSDT"].dir == 0


def test_stop_placement_failure_flattens_and_pauses():
    clock = Clock()
    fake, ex, ev = make(clock=clock)
    run(ex.startup())
    fake.fail.add("algo")
    run(ex.on_decision("BTCUSDT", clock.t, row(1, 49_000.0), 50_000.0))
    assert "BTCUSDT" not in fake.pos and ex.paused
    kinds = [e["kind"] for e in ev]
    assert "live_error" in kinds and "live_paused" in kinds and "live_exit" in kinds


def test_price_already_through_stop_closes_immediately():
    clock = Clock()
    fake, ex, ev = make(clock=clock)
    run(ex.startup())
    fake.price["BTCUSDT"] = 48_000.0   # market gapped below the stop between signal and fill
    run(ex.on_decision("BTCUSDT", clock.t, row(1, 49_000.0), 50_000.0))
    assert "BTCUSDT" not in fake.pos
    assert [e for e in ev if e["kind"] == "live_exit"][0]["reason"] == "stop"


def test_stale_entries_skipped_but_exits_always_run():
    clock = Clock()
    fake, ex, ev = make(clock=clock)
    run(ex.startup())
    bar_end = clock.t
    clock.t += 30 * 60_000
    run(ex.on_decision("BTCUSDT", bar_end, row(1, 49_000.0), 50_000.0))
    assert "BTCUSDT" not in fake.pos and ev[-1]["kind"] == "live_skipped"
    clock.t = bar_end
    run(ex.on_decision("BTCUSDT", bar_end, row(1, 49_000.0), 50_000.0))
    clock.t += 30 * 60_000
    run(ex.on_decision("BTCUSDT", bar_end, row(exit_long=True), 50_000.0))
    assert "BTCUSDT" not in fake.pos


def test_size_below_exchange_minimum_is_skipped():
    clock = Clock()
    fake, ex, ev = make(FakeBinance(wallet=50.0), clock=clock)
    run(ex.startup())
    run(ex.on_decision("BTCUSDT", clock.t, row(1, 49_000.0), 50_000.0))
    assert not fake.pos and "below exchange minimum" in ev[-1]["reason"]


def test_unknown_exchange_position_pauses_without_touching_it():
    fake = FakeBinance()
    fake.pos["ETHUSDT"], fake.entry["ETHUSDT"] = 1.0, 3_000.0
    _, ex, ev = make(fake)
    run(ex.startup())
    assert ex.paused and fake.pos["ETHUSDT"] == 1.0
    assert not [c for c in fake.calls if c[1] == "/fapi/v1/order"]


def test_restart_restores_position_and_replaces_missing_stop():
    clock = Clock()
    fake, ex, _ = make(clock=clock)
    run(ex.startup())
    run(ex.on_decision("BTCUSDT", clock.t, row(1, 49_000.0), 50_000.0))
    state = ex.state()
    for a in fake.algos.values():          # stop vanished while the bot was down
        a["algoStatus"] = "CANCELED"
    _, ex2, ev2 = make(fake, clock=clock)
    ex2.restore(state)
    run(ex2.startup())
    assert ex2.pos["BTCUSDT"].dir == 1
    assert any("missing" in e.get("msg", "") for e in ev2)
    assert sum(a["algoStatus"] == "NEW" for a in fake.algos.values()) == 1


def test_live_mainnet_needs_explicit_ack():
    c = BotConfig(mode="live", binance_env="mainnet", api_key="k", api_secret="s")
    with pytest.raises(SystemExit):
        c.validate()
    c.live_ack = LIVE_ACK
    c.validate()
    BotConfig(mode="live", binance_env="testnet", api_key="k", api_secret="s").validate()
    with pytest.raises(SystemExit):
        BotConfig(mode="live", binance_env="testnet").validate()


@needs_data
def test_bot_in_live_mode_mirrors_paper_decisions_on_simulated_exchange(tmp_path):
    """End to end: replay 2023-Q4 BTC through the App with the executor on the fake exchange.
    Every live round trip must match the paper model's entries/exits (fills differ by < 1 minute)."""
    fake = FakeBinance(prices={"BTCUSDT": 27_000.0, "ETHUSDT": 1_600.0})
    clock = Clock(0)
    _, ex, _ = make(fake, clock=clock, symbols=("BTCUSDT",))
    cfg = BotConfig(symbols=["BTCUSDT"], shadow_long_only=False, max_drawdown_pause=1.0, state_dir=tmp_path,
                    mode="live", binance_env="testnet", api_key="k", api_secret=SECRET)
    app = App(cfg, ReplayFeed(cfg.symbols), LogNotifier(), StateDB(tmp_path / "bot.sqlite"), executor=ex)
    orig = app.trader.process_minute

    def step(symbol, m, rate=0.0):
        clock.t = m.ts + 60_000                       # "now" = this minute just closed
        fake.bar(symbol, m.open, m.high, m.low, m.close)
        orig(symbol, m, rate)
    app.trader.process_minute = step

    async def go():
        await app.start(int(pd.Timestamp("2023-09-01", tz="UTC").value // 10**6))
        await ex.startup()
        await app.run_replay(int(pd.Timestamp("2023-09-01", tz="UTC").value // 10**6),
                             int(pd.Timestamp("2024-01-01", tz="UTC").value // 10**6))
    run(go())
    paper = app.trader.books["F1"].trades
    live = app.db.trades("F1-live")
    assert len(paper) >= 5
    # live executes at the 4h close, paper one minute later: same trades, entries 1-2 minutes apart
    lv = sorted(live, key=lambda t: t["entry_time"])
    assert len(lv) == len(paper)
    for p_, l_ in zip(paper, lv):
        assert p_["dir"] == l_["dir"]
        assert abs(p_["entry_time"] - l_["entry_time"]) <= 2 * 60_000
        assert l_["reason"] in (p_["reason"], "stop")
