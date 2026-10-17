"""Bot settings from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from btcq.config import SLIPPAGE, SYMBOLS, Costs


LIVE_ACK = "I_ACCEPT_REAL_ORDERS"
# D1's registered universe (Round 5): pinned, so editing SYMBOLS for F1 never resizes or drops D1 sub-sleeves.
D1_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


@dataclass
class BotConfig:
    mode: str = "paper"                      # "paper" (simulated fills) or "live" (real orders)
    binance_env: str = "mainnet"             # "mainnet" or "testnet" (data + orders)
    symbols: list[str] = field(default_factory=lambda: list(SYMBOLS))
    initial_equity: float = 10_000.0         # paper account, USDT
    risk_per_trade: float = 0.01             # fraction of account equity risked at the stop
    max_leverage: float = 3.0                # total notional cap, as a multiple of equity
    max_drawdown_pause: float = 0.35         # pause new entries beyond this drawdown from peak
    latency_min: int = 1                     # market orders fill one minute after the bar close
    taker_fee: float = Costs.taker
    shadow_long_only: bool = True            # also paper-trade F1 long-only for comparison
    d1_shadow: bool = False                  # also paper-trade Round 5's D1 daily ensemble (env D1_SHADOW, on)
    d1_symbols: list[str] = field(default_factory=lambda: list(D1_SYMBOLS))   # not an env setting
    history_bars: int = 1500                 # 4h bars kept for signal computation
    state_dir: Path = Path("state")
    telegram_token: str = ""
    telegram_chat_id: str = ""
    stale_after_min: int = 5                 # alert if no closed 1m bar for this long
    api_key: str = ""
    api_secret: str = ""
    live_ack: str = ""
    leverage_setting: int = 5                # exchange leverage setting (margin only; sizing caps notional)
    stop_working_type: str = "CONTRACT_PRICE"
    max_entry_delay_min: int = 10            # skip entries whose signal is older than this

    def slippage(self, symbol: str) -> float:
        return SLIPPAGE.get(symbol, 0.0002)

    @classmethod
    def from_env(cls) -> "BotConfig":
        c = cls()
        c.mode = _env("TRADING_MODE", c.mode).lower()
        c.binance_env = _env("BINANCE_ENV", c.binance_env).lower()
        syms = _env("SYMBOLS", "")
        if syms:
            c.symbols = [s.strip().upper() for s in syms.split(",") if s.strip()]
        c.initial_equity = float(_env("PAPER_EQUITY", str(c.initial_equity)))
        c.risk_per_trade = float(_env("RISK_PER_TRADE", str(c.risk_per_trade)))
        c.max_leverage = float(_env("MAX_LEVERAGE", str(c.max_leverage)))
        c.max_drawdown_pause = float(_env("MAX_DRAWDOWN_PAUSE", str(c.max_drawdown_pause)))
        c.shadow_long_only = _env("SHADOW_LONG_ONLY", "true").lower() in ("1", "true", "yes")
        c.d1_shadow = _env("D1_SHADOW", "true").lower() in ("1", "true", "yes")
        c.state_dir = Path(_env("STATE_DIR", str(c.state_dir)))
        c.telegram_token = _env("TELEGRAM_BOT_TOKEN", "")
        c.telegram_chat_id = _env("TELEGRAM_CHAT_ID", "")
        c.api_key = _env("BINANCE_API_KEY", "")
        c.api_secret = _env("BINANCE_API_SECRET", "")
        c.live_ack = _env("LIVE_TRADING_ACK", "")
        c.leverage_setting = int(_env("LEVERAGE_SETTING", str(c.leverage_setting)))
        c.stop_working_type = _env("STOP_WORKING_TYPE", c.stop_working_type).upper()
        c.max_entry_delay_min = int(_env("MAX_ENTRY_DELAY_MIN", str(c.max_entry_delay_min)))
        c.validate()
        return c

    def validate(self) -> None:
        if self.mode not in ("paper", "live"):
            raise SystemExit(f"TRADING_MODE must be 'paper' or 'live', got {self.mode!r}")
        if self.binance_env not in ("mainnet", "testnet"):
            raise SystemExit(f"BINANCE_ENV must be 'mainnet' or 'testnet', got {self.binance_env!r}")
        if self.stop_working_type not in ("CONTRACT_PRICE", "MARK_PRICE"):
            raise SystemExit("STOP_WORKING_TYPE must be CONTRACT_PRICE or MARK_PRICE")
        if self.mode == "live":
            if not (self.api_key and self.api_secret):
                raise SystemExit("TRADING_MODE=live needs BINANCE_API_KEY and BINANCE_API_SECRET")
            if self.binance_env == "mainnet" and self.live_ack != LIVE_ACK:
                raise SystemExit(f"Real-money trading is locked. Set LIVE_TRADING_ACK={LIVE_ACK} to enable it.")
            if self.leverage_setting * 1.0 < self.max_leverage:
                raise SystemExit("LEVERAGE_SETTING must be >= MAX_LEVERAGE so the account can hold full size")
