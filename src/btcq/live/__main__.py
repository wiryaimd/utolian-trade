"""Paper-trading bot CLI.

    python -m btcq.live run                       # live paper trading (Binance data, Telegram)
    python -m btcq.live check                     # connectivity: Binance REST/WS + Telegram
    python -m btcq.live status                    # print saved state
    python -m btcq.live chat-id                   # find your Telegram chat id
    python -m btcq.live dashboard                 # web dashboard on http://127.0.0.1:8080
    python -m btcq.live replay --start 2023-06-01 --end 2023-12-31   # archive replay (dev data)
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

import pandas as pd

from btcq.live.app import App
from btcq.live.config import BotConfig
from btcq.live.db import StateDB
from btcq.live.telegram import LogNotifier, Telegram


def _logging(cfg: BotConfig) -> None:
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # emoji-safe on any console
    for h in (logging.StreamHandler(sys.stdout),
              RotatingFileHandler(cfg.state_dir / "bot.log", maxBytes=5_000_000, backupCount=5,
                                  encoding="utf-8")):
        h.setFormatter(fmt)
        root.addHandler(h)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _notifier(cfg: BotConfig):
    if cfg.telegram_token and cfg.telegram_chat_id:
        return Telegram(cfg.telegram_token, cfg.telegram_chat_id)
    logging.getLogger("btcq").warning("TELEGRAM_BOT_TOKEN/CHAT_ID not set: notifications go to the log only")
    return LogNotifier()


def _ms(day: str) -> int:
    return int(pd.Timestamp(day, tz="UTC").value // 10**6)


async def _check(cfg: BotConfig) -> int:
    from btcq.live.feed import BinanceFeed, now_ms
    ok = True
    print(f"Binance environment: {cfg.binance_env} · trading mode: {cfg.mode}")
    feed = BinanceFeed(cfg.symbols, cfg.binance_env)
    for s in cfg.symbols:
        try:
            m = await feed.minutes(s, now_ms() - 5 * 60_000, now_ms())
            print(f"REST  {s}: OK, last closed 1m close {m[-1].close if m else '–'}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"REST  {s}: FAILED {e}")
    try:
        async def first_kline():
            async for kind, s, m in feed.stream():
                if kind == "kline":
                    return s, m
        s, m = await asyncio.wait_for(first_kline(), timeout=90)
        print(f"WS    first closed kline: {s} {pd.Timestamp(m.ts, unit='ms')} close {m.close}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"WS    FAILED {e!r}")
    if cfg.api_key and cfg.api_secret:
        from btcq.live.binance_api import BinanceFutures
        api = BinanceFutures(cfg.api_key, cfg.api_secret, cfg.binance_env)
        try:  # read-only calls: no orders are sent
            off = await api.sync_time()
            acct = await api.account()
            hedge = await api.hedge_mode()
            pos = await api.positions()
            rules = await api.rules(cfg.symbols)
            print(f"KEYS  OK: clock offset {off} ms, margin balance {float(acct['totalMarginBalance']):,.2f} USDT, "
                  f"available {float(acct['availableBalance']):,.2f}")
            print(f"KEYS  position mode: {'HEDGE (must switch to One-way)' if hedge else 'One-way OK'}; "
                  f"open positions: {pos or 'none'}")
            print("KEYS  symbol rules: " + ", ".join(f"{s} step {r.step} min notional {r.min_notional}"
                                                    for s, r in rules.items()))
            ok = ok and not hedge
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"KEYS  FAILED {e}")
    else:
        print("KEYS  not configured (only needed for TRADING_MODE=live)")
    if cfg.telegram_token and cfg.telegram_chat_id:
        tg = Telegram(cfg.telegram_token, cfg.telegram_chat_id)
        try:
            me = await tg.get_me()
            print(f"TG    bot @{me['result']['username']} OK; sending a test message")
            await tg.send("✅ btcq bot connectivity check passed for Telegram.")
        except Exception as e:  # noqa: BLE001
            from btcq.live.telegram import redact
            ok = False
            print(f"TG    FAILED {redact(e)}")
    else:
        print("TG    not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID)")
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m btcq.live")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    sub.add_parser("check")
    sub.add_parser("status")
    sub.add_parser("chat-id", help="print chat ids that recently messaged the bot (Telegram setup)")
    dp = sub.add_parser("dashboard", help="read-only web dashboard")
    dp.add_argument("--db", default=None, help="defaults to <STATE_DIR>/bot.sqlite")
    dp.add_argument("--host", default=os.environ.get("DASHBOARD_HOST", "127.0.0.1"))
    dp.add_argument("--port", type=int, default=int(os.environ.get("DASHBOARD_PORT", "8080")))
    rp = sub.add_parser("replay")
    rp.add_argument("--start", required=True)
    rp.add_argument("--end", required=True)
    rp.add_argument("--state-dir", default=None, help="defaults to <STATE_DIR>/replay")
    rp.add_argument("--telegram", action="store_true", help="send replay notifications to Telegram")
    rp.add_argument("--unlock-holdout", action="store_true",
                    help="allow replaying 2024+ archive data (logged in research/holdout_log.jsonl)")
    a = ap.parse_args()
    if a.cmd == "dashboard":  # needs no trading config
        from btcq.live.dashboard import serve
        db = a.db or str(Path(os.environ.get("STATE_DIR", "state")) / "bot.sqlite")
        srv = serve(db, a.host, a.port, os.environ.get("DASHBOARD_USER", "admin"),
                    os.environ.get("DASHBOARD_PASSWORD", ""))
        auth = "password protected" if os.environ.get("DASHBOARD_PASSWORD") else "no password (localhost only)"
        print(f"dashboard on http://{a.host}:{a.port} reading {db} ({auth})", flush=True)
        srv.serve_forever()
        return
    cfg = BotConfig.from_env()

    if a.cmd == "run":
        from btcq.live.feed import BinanceFeed
        _logging(cfg)
        executor = None
        if cfg.mode == "live":
            from btcq.live.binance_api import BinanceFutures
            from btcq.live.executor import ExecConfig, LiveExecutor
            executor = LiveExecutor(
                "F1", BinanceFutures(cfg.api_key, cfg.api_secret, cfg.binance_env), cfg.symbols,
                ExecConfig(risk_per_trade=cfg.risk_per_trade, max_leverage=cfg.max_leverage,
                           leverage_setting=cfg.leverage_setting, taker_fee=cfg.taker_fee,
                           stop_working_type=cfg.stop_working_type, max_entry_delay_min=cfg.max_entry_delay_min,
                           max_drawdown_pause=cfg.max_drawdown_pause))
            logging.getLogger("btcq").warning("LIVE ORDER MODE on Binance %s", cfg.binance_env.upper())
        app = App(cfg, BinanceFeed(cfg.symbols, cfg.binance_env), _notifier(cfg),
                  StateDB(cfg.state_dir / "bot.sqlite"), live=True, executor=executor)
        asyncio.run(app.run_live())
    elif a.cmd == "chat-id":
        import httpx
        if not cfg.telegram_token:
            raise SystemExit("set TELEGRAM_BOT_TOKEN first, then send any message to your bot")
        r = httpx.get(f"https://api.telegram.org/bot{cfg.telegram_token}/getUpdates", timeout=30).json()
        chats = {(m["chat"]["id"], m["chat"].get("username") or m["chat"].get("title") or "")
                 for u in r.get("result", []) for m in [u.get("message") or {}] if m}
        lines = [f"chat id {c}  ({n})" for c, n in chats]
        print("\n".join(lines) if lines else "no messages yet: message your bot first")
    elif a.cmd == "check":
        sys.exit(asyncio.run(_check(cfg)))
    elif a.cmd == "status":
        db = StateDB(cfg.state_dir / "bot.sqlite")
        app = App(cfg, None, LogNotifier(), db)
        st = db.load_state()
        if not st:
            print("no saved state yet")
            return
        for s in cfg.symbols:
            app.trader.bootstrap(s, None)
        app.trader.restore(st)
        for name, s in app.trader.snapshot().items():
            print(f"{name}: equity {s['equity']:,.2f} drawdown {s['drawdown']:.1%} paused={s['paused']} "
                  f"open={[(p['symbol'], p['dir'], round(p['R_open'], 2)) for p in s['positions']]} "
                  f"trades={len(db.trades(name))}")
    elif a.cmd == "replay":
        from btcq.data import store
        from btcq.live.feed import ReplayFeed, holdout_start_ms
        start, end = _ms(a.start), _ms(a.end)
        period = "dev"
        if end > holdout_start_ms():
            if not a.unlock_holdout:
                raise SystemExit("--end is in the research holdout (2024+); add --unlock-holdout to allow it")
            store.unlock_holdout(f"paper-bot replay {a.start}..{a.end} (engineering check, not research)")
            period = "full"
        if a.state_dir:
            cfg.state_dir = Path(a.state_dir)
        else:
            cfg.state_dir = cfg.state_dir / "replay"
        _logging(cfg)
        db_path = cfg.state_dir / "bot.sqlite"
        if db_path.exists():
            db_path.unlink()
        app = App(cfg, ReplayFeed(cfg.symbols, period), _notifier(cfg) if a.telegram else LogNotifier(),
                  StateDB(db_path))
        asyncio.run(app.run_replay(start, end))
        for name, s in app.trader.snapshot().items():
            ts = app.db.trades(name)
            rs = [t["R"] for t in ts]
            peak, mdd = cfg.initial_equity, 0.0
            for _, eq in app.db.equity_curve(name):
                peak = max(peak, eq)
                mdd = max(mdd, 1 - eq / peak)
            print(f"{name}: {len(ts)} closed trades, avg {sum(rs) / max(len(rs), 1):+.3f} R, "
                  f"equity {s['equity']:,.2f}, max daily drawdown {mdd:.1%}, open {len(s['positions'])}")


if __name__ == "__main__":
    main()
