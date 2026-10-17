import os

import numpy as np
import pandas as pd
import pytest

from btcq.config import PARQUET_DIR

# The dashboard's automatic full health check calls Binance and Telegram; tests never touch the network.
os.environ["DASHBOARD_FULL_CHECK_MIN"] = "0"

T0 = int(pd.Timestamp("2022-01-03", tz="UTC").value // 10**6)  # a Monday 00:00 UTC

needs_data = pytest.mark.skipif(not (PARQUET_DIR / "BTCUSDT" / "perp_1m.parquet").exists(),
                                reason="parquet store not built")


def make_m1(paths: list[tuple[float, float, float, float]], t0: int = T0) -> pd.DataFrame:
    """1m frame from (open, high, low, close) tuples."""
    a = np.asarray(paths, dtype=np.float64)
    n = len(a)
    return pd.DataFrame({"ts": t0 + 60_000 * np.arange(n, dtype=np.int64), "open": a[:, 0],
                         "high": a[:, 1], "low": a[:, 2], "close": a[:, 3],
                         "volume": np.ones(n), "quote_volume": a[:, 3],
                         "taker_buy_volume": np.full(n, 0.5), "is_perp": True})


def flat(px: float, n: int) -> list[tuple]:
    return [(px, px, px, px)] * n
