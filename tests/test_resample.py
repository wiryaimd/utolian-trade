import numpy as np
import pandas as pd

from btcq.data import store
from btcq.features.mtf import align_htf
from conftest import T0, make_m1, needs_data


def test_resample_ohlcv_and_drops_incomplete_tail():
    rng = np.random.default_rng(3)
    px = 100 + np.cumsum(rng.normal(0, 0.1, 47))
    paths = [(p, p + 0.2, p - 0.2, p + 0.05) for p in px]
    m1 = make_m1(paths)
    b = store.resample(m1, "15m")
    assert len(b) == 3  # 47 minutes -> 3 complete 15m bars, trailing 2 minutes dropped
    first = m1.iloc[:15]
    assert b.loc[0, "open"] == first["open"].iloc[0]
    assert b.loc[0, "high"] == first["high"].max()
    assert b.loc[0, "low"] == first["low"].min()
    assert b.loc[0, "close"] == first["close"].iloc[-1]
    assert b.loc[0, "volume"] == 15


def test_htf_values_appear_only_after_htf_close():
    ltf_ts = T0 + 900_000 * np.arange(32)  # 8h of 15m bars
    htf_ts = T0 + 14_400_000 * np.arange(2)
    v = align_htf(ltf_ts, "15m", htf_ts, "4h", np.array([1.0, 2.0]))
    assert np.isnan(v[:15]).all()        # 00:00-03:45 bars: first 4h bar not closed yet
    assert v[15] == 1.0                  # 03:45-04:00 bar closes exactly when the 4h bar closes
    assert (v[16:31] == 1.0).all()
    assert v[31] == 2.0


@needs_data
def test_resampled_prices_match_binance_klines():
    m1 = store.load_1m("dev")
    for tf in ("15m", "4h"):
        ours = store.resample(m1, tf)
        ours = ours[ours["is_perp"]]
        m = ours.merge(store.load_binance_klines(tf), on="ts", suffixes=("", "_b"))
        assert len(m) > 8000
        bad = ~np.all([np.isclose(m[c], m[c + "_b"]) for c in ("open", "high", "low", "close")], axis=0)
        # Only the known 2023-11-10 16:30-16:50 archive outage may differ.
        bad_times = pd.to_datetime(m.loc[bad, "ts"], unit="ms", utc=True)
        assert (bad_times.dt.date.astype(str) == "2023-11-10").all(), bad_times.tolist()
