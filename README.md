# btcq: crypto perp strategy research + F1 paper-trading bot

## What's here
- `src/btcq/research`, `reports/`, `research/`: three pre-registered research rounds. The results are in
  `reports/summary.html`, and the rules and every amendment or deviation are in `research/PREREGISTRATION.md`.
- `src/btcq/live`: a paper-trading bot running **F1**.
  - The rule: on 4h bars, enter on a break of the 40-bar high (long) or low (short), with a stop 3.5 ATR away, and
    exit on the opposite 20-bar channel.
  - Markets: BTC, ETH, SOL, XRP and BNB USDⓈ-M perps.
  - Sizing: 1% of equity risked per trade, total notional ≤ 3×.
  - A long-only copy runs as a quiet "shadow" book for comparison.

The bot calls the same strategy code as the backtests. A parity test replays 2023 through the bot and must reproduce
the backtest engine's trades exactly.

## Run the bot on a VPS (Docker)
1. **Telegram:**
   - Create a bot with @BotFather and copy the token.
   - Send your bot any message.
   - Put the token in `.env`, then run `docker compose run --rm bot python -m btcq.live chat-id` to get your chat id.
2. **Configure:** `cp .env.example .env` and fill in `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`. The other defaults
   match the research.
3. **Check connectivity:** run `docker compose run --rm bot python -m btcq.live check`. It needs Binance REST,
   the Binance websocket and Telegram, and sends you a test message.
4. **Start:** `docker compose up -d --build`, then follow the output with `docker compose logs -f`.
   - State, trades and logs live in `./state/bot.sqlite` and `./state/bot.log`.
   - Restarts resume from SQLite and backfill missed minutes over REST.
5. Keep the VPS clock synced with NTP (`timedatectl`), because data freshness checks use wall time.

**Telegram commands:** `/status`, `/positions`, `/trades [n]`, `/pause`, `/resume`, `/close SYMBOL`, `/compare`, `/help`.

## Real orders (TRADING_MODE=live)
The bot can trade for real.
- **Orders:** at each 4h close it sends a market entry, sized from your real account equity. It then places an
  exchange-side `STOP_MARKET` with `closePosition=true` through Binance's Algo Order API.
- **Exits:** a signal exit cancels the stop and closes reduce-only.
- **Reconciliation:** every minute the bot checks Binance's positions, so a stop that fired on the exchange gets
  recorded.
- **The paper book keeps running** as a "model", so you can compare model fills with real fills.

Safety rails:
- One-way mode is required, and margin is set to cross.
- If the stop can't be placed, the position is closed immediately and trading pauses.
- Entries whose signal is older than `MAX_ENTRY_DELAY_MIN` are skipped; exits always run.
- Orders below exchange minimums are skipped.
- A notional sanity cap catches sizing bugs.
- Unknown or mismatched positions pause trading and alert you.
- The drawdown kill switch works on real account equity.
- Mainnet trading only starts with `LIVE_TRADING_ACK=I_ACCEPT_REAL_ORDERS`.

Suggested rollout:
1. **Paper on mainnet data** (`TRADING_MODE=paper`) for at least ~3 months.
2. **Testnet rehearsal:** set `TRADING_MODE=live` and `BINANCE_ENV=testnet`, using keys from
   https://testnet.binancefuture.com. This rehearses real order plumbing with fake money.
3. **Mainnet, small:** set `TRADING_MODE=live`, `BINANCE_ENV=mainnet`, `LIVE_TRADING_ACK=...` and
   `RISK_PER_TRADE=0.0025`. Raise risk only if live fills track the paper model.

API keys:
- Enable Futures only, with no withdrawals and IP-restricted to the VPS.
- Use a dedicated sub-account, because the bot assumes it owns every position on these symbols.
- `python -m btcq.live check` verifies the keys with read-only calls and never places orders.

**Location matters:** Binance refuses service from restricted locations (HTTP 451, "b. Eligibility" in its Terms). The
VPS must be somewhere Binance serves, and you must be eligible to use it. `check` shows a 451 if the location is
blocked.

**Messages:** every F1 entry and exit, a daily summary at 00:00 UTC, data-gap alerts, and the drawdown kill switch,
which pauses new entries at a 35% drawdown.

## Without Docker
```
pip install -e ".[live]"
python -m btcq.live check
python -m btcq.live run
python -m btcq.live status
```

## Replay (engineering check on archived data, no network)
```
python -m btcq.live replay --start 2023-01-01 --end 2024-01-01
```
This needs the local archive (`python -m btcq.research.run download`). Replays into 2024+ need `--unlock-holdout`,
which is logged.

## Tests
`pytest`
