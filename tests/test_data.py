"""Sanity checks on the parquet store (skipped for symbols not built)."""
import numpy as np
import pandas as pd
import pytest

from btcq.config import OI_LAG_MS, SYMBOLS
from btcq.data import store

LO = int(pd.Timestamp("2020-01-01", tz="UTC").value // 10**6)
HI = int(pd.Timestamp("2027-01-01", tz="UTC").value // 10**6)


def _built(sym, name):
    return (store.symbol_dir(sym) / f"{name}.parquet").exists()


@pytest.mark.parametrize("sym", SYMBOLS)
def test_metrics_timestamps_are_real_5min_snapshots(sym):
    if not _built(sym, "metrics"):
        pytest.skip("metrics not built")
    raw = pd.read_parquet(store.symbol_dir(sym) / "metrics.parquet")
    ts = raw["ts"].to_numpy()
    assert ts.min() >= LO and ts.max() < HI, "metrics timestamps outside 2020-2026 (unit bug?)"
    assert np.median(np.diff(ts)) == 300_000
    # Loader exposes availability time = snapshot + lag.
    assert store.load_metrics("dev", sym)["ts"].iloc[0] == ts[0] + OI_LAG_MS


@pytest.mark.parametrize("sym", SYMBOLS)
def test_premium_is_hourly_and_in_range(sym):
    if not _built(sym, "premium_1h"):
        pytest.skip("premium not built")
    p = pd.read_parquet(store.symbol_dir(sym) / "premium_1h.parquet")
    assert p["ts"].min() >= LO and p["ts"].max() < HI
    assert np.median(np.diff(p["ts"].to_numpy())) == 3_600_000
    assert p["close"].abs().median() < 0.01  # premium is a small fraction of price


@pytest.mark.parametrize("sym", SYMBOLS)
def test_resampled_4h_matches_binance(sym):
    if not (_built(sym, "perp_1m") and _built(sym, "perp_4h_binance")):
        pytest.skip("not built")
    m1 = store.load_1m("dev", sym)
    ours = store.resample(m1, "4h")
    ours = ours[ours["is_perp"]]
    m = ours.merge(store.load_binance_klines("4h", sym), on="ts", suffixes=("", "_b"))
    assert len(m) > 3000
    bad = ~np.all([np.isclose(m[c], m[c + "_b"]) for c in ("open", "high", "low", "close")], axis=0)
    assert bad.mean() < 0.001, pd.to_datetime(m.loc[bad, "ts"], unit="ms").tolist()[:10]
