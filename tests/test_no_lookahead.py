"""Truncation test: signals for any closed bar must not change when future data is removed."""
import numpy as np
import pandas as pd
import pytest

from btcq.data import store
from btcq.data.bardata import BarData
from btcq.strategies import ALL, ROUND3, ROUND4
from conftest import needs_data

CUTS = ["2022-11-09 13:37", "2023-01-02 00:00", "2023-03-10 04:00", "2023-03-12 21:15",
        "2023-06-30 23:59", "2023-08-17 08:05"]


@pytest.fixture(scope="module")
def data():
    m1 = store.load_1m("dev")
    m1 = m1[m1["ts"] >= int(pd.Timestamp("2022-01-01", tz="UTC").value // 10**6)]
    return BarData(m1.reset_index(drop=True), store.load_funding("dev"), store.load_metrics("dev"),
                   store.load_premium("dev"))


def _arrays(sig):
    return dict(zip(["entry", "stop", "target", "trail", "exit_long", "exit_short", "limit", "size"],
                    sig.arrays(len(sig.entry))))


@needs_data
@pytest.mark.parametrize("strat", ALL + ROUND3 + ROUND4, ids=lambda s: s.code)
def test_signals_unchanged_by_future_data(strat, data):
    rng = np.random.default_rng(0)
    configs = strat.configs()
    picks = [configs[0], configs[-1]] + [configs[i] for i in rng.integers(0, len(configs), 2)]
    total_entries = 0
    for params in picks:
        full = _arrays(strat.compute(data, **params))
        total_entries += int(np.abs(full["entry"]).sum())
        for cut in CUTS:
            cut_ms = int(pd.Timestamp(cut, tz="UTC").value // 10**6)
            part = _arrays(strat.compute(data.truncated(cut_ms), **params))
            n = len(part["entry"])
            assert n > 0
            for k, v in part.items():
                assert np.array_equal(v, full[k][:n], equal_nan=True), \
                    f"{strat.code} {params} '{k}' differs when data is cut at {cut}"
    assert total_entries > 0, f"{strat.code}: no entries in any tested config, so the test proves nothing"
