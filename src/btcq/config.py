"""Project-wide paths, periods, costs and risk settings."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PARQUET_DIR = DATA_DIR / "parquet"
REPORTS_DIR = ROOT / "reports"
RESEARCH_DIR = ROOT / "research"

SYMBOL = "BTCUSDT"
# Round 3 universe and per-side slippage assumptions (pre-registered).
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT"]
SLIPPAGE = {"BTCUSDT": 0.0001, "ETHUSDT": 0.0001, "SOLUSDT": 0.0002, "XRPUSDT": 0.0002,
            "BNBUSDT": 0.0002}
OI_LAG_MS = 5 * 60_000  # an OI snapshot stamped t is treated as known from t + 5 minutes

# Spot 1m is used as price history before the perp archive begins.
SPOT_START = pd.Timestamp("2017-08-17", tz="UTC")
PERP_START = pd.Timestamp("2020-01-01", tz="UTC")
# Holdout is locked: dev loaders never return data on or after this instant.
HOLDOUT_START = pd.Timestamp("2024-01-01", tz="UTC")

MINUTE_MS = 60_000
TF_MS = {"1m": 60_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}


@dataclass(frozen=True)
class Costs:
    taker: float = 0.0005  # Binance USDS-M VIP0 taker
    maker: float = 0.0002  # Binance USDS-M VIP0 maker
    slippage: float = 0.0001  # per side, on market/stop fills
    synthetic_funding: float = 0.0001  # per 8h, used before real funding history exists

    def scaled(self, k: float) -> "Costs":
        """Stress version: fees, slippage and synthetic funding multiplied by k."""
        return replace(self, taker=self.taker * k, maker=self.maker * k,
                       slippage=self.slippage * k)


@dataclass(frozen=True)
class Risk:
    risk_per_trade: float = 0.01
    max_leverage: float = 3.0
    initial_equity: float = 10_000.0
