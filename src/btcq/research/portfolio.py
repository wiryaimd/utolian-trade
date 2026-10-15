"""Equal-capital multi-symbol sleeves combined into one account (round 3).

Each sleeve holds 1/N of the capital, risks 1% x N of its own capital at the stop and
is capped at 3x its own capital, which is 1% of the account per trade with total
notional <= 3x. The portfolio return is the daily mean of the available sleeves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from btcq.backtest import engine
from btcq.config import Risk


def sleeve_risk(n_symbols: int, base: Risk = Risk()) -> Risk:
    return Risk(risk_per_trade=base.risk_per_trade * n_symbols, max_leverage=base.max_leverage,
                initial_equity=base.initial_equity)


def assemble(parts: list[tuple[pd.Series, pd.DataFrame]],
             initial_equity: float = Risk().initial_equity) -> tuple[engine.Result, pd.Series]:
    """Combine per-sleeve (daily returns, trades) into one portfolio Result."""
    dailies = [d for d, _ in parts if len(d)]
    daily = pd.concat(dailies, axis=1).mean(axis=1).sort_index() if dailies else pd.Series(dtype=float)
    trades = pd.concat([t for _, t in parts], ignore_index=True)
    if len(trades):
        trades = trades.sort_values("entry_time").reset_index(drop=True)
        # A sleeve's per-trade equity return is diluted by its share of the account.
        trades["ret"] = trades["ret"] / len(parts)
    equity = initial_equity * (1.0 + daily).cumprod()
    start = daily.index[0] if len(daily) else pd.Timestamp("1970-01-01", tz="UTC")
    return engine.Result(trades, equity, initial_equity, start), daily


def combine_null(dists: list[np.ndarray], counts: list[int]) -> np.ndarray:
    """Trade-weighted portfolio null from per-symbol null distributions of mean trade return."""
    w = np.asarray(counts, dtype=np.float64)
    keep = w > 0
    if not keep.any():
        return np.zeros(1)
    stack = np.vstack([d for d, k in zip(dists, keep) if k])
    return (stack * w[keep][:, None]).sum(axis=0) / w[keep].sum()
