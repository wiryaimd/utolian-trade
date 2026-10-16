"""Round 4 engine extensions: per-minute slippage, worst-of-minute stops, forced exits.
All are opt-in; the defaults must reproduce the original engine bit for bit."""
import numpy as np
import pandas as pd
import pytest

from btcq.backtest import engine
from btcq.config import REPORTS_DIR, Costs, Risk
from conftest import flat, make_m1, needs_data

C = Costs(taker=0.0005, maker=0.0002, slippage=0.0001)
R = Risk(risk_per_trade=0.01, max_leverage=3.0, initial_equity=10_000.0)


def market(paths):
    return engine.Market(make_m1(paths), None, synthetic_funding=0.0)


def one_entry(n_bars, d, stop):
    e = np.zeros(n_bars, np.int8)
    e[0] = d
    s = np.full(n_bars, np.nan)
    s[0] = stop
    return engine.Signals("15m", e, s, np.full(n_bars, np.nan))


def multi_trade_case():
    """Target, stop, signal exit, reverse and end trades on one path (15m bars)."""
    up = [(100 + i, 101 + i, 99.5 + i, 101 + i) for i in range(15)]
    paths = (flat(100, 15) + up + flat(115, 15) + [(115, 115, 80, 85)] + flat(85, 14)
             + flat(85, 15) + flat(86, 15) + flat(87, 15) + flat(88, 15) + flat(89, 15))
    n = len(paths) // 15
    e = np.zeros(n, np.int8)
    s = np.full(n, np.nan)
    t = np.full(n, np.nan)
    xl = np.zeros(n, np.bool_)
    e[0], s[0], t[0] = 1, 95.0, 110.0          # long -> target
    e[2], s[2] = 1, 90.0                       # long -> stop on the crash bar
    e[4], s[4] = 1, 80.0                       # long -> exit signal
    xl[5] = True
    e[6], s[6] = 1, 80.0                       # long -> reversed by a short
    e[7], s[7] = -1, 95.0                      # short -> end of data
    return market(paths), engine.Signals("15m", e, s, t, exit_long=xl)


def test_constant_slippage_array_is_bit_identical_to_scalar():
    mk, sg = multi_trade_case()
    a = engine.run(mk, sg, C, R)
    b = engine.run(mk, sg, C, R, slippage=np.full(len(mk.ts), C.slippage))
    assert list(a.trades["reason"]) == ["target", "stop", "exit_signal", "reverse", "end"]
    pd.testing.assert_frame_equal(a.trades, b.trades, check_exact=True)
    pd.testing.assert_series_equal(a.equity, b.equity, check_exact=True)


def test_slippage_is_taken_at_the_fill_minute():
    paths = flat(100, 15) + [(100, 100.5, 99.5, 100)] + flat(100, 13) + [(100, 100, 94, 94)]
    mk = market(paths)
    slip = np.full(len(mk.ts), 0.0001)
    slip[15] = 0.001    # entry minute
    slip[29] = 0.002    # stop minute
    t = engine.run(mk, one_entry(2, 1, 95.0), C, R, slippage=slip).trades.iloc[0]
    assert t["entry_i"] == 15 and t["exit_i"] == 29
    assert t["entry_px"] == pytest.approx(100 * 1.001)
    assert t["exit_px"] == pytest.approx(95 * (1 - 0.002))


def test_worst_of_minute_stop_fills_at_low_and_high():
    paths = flat(100, 15) + [(100, 100.5, 99.5, 100)] + [(99, 99, 93, 96)] + flat(96, 13)
    t0 = engine.run(market(paths), one_entry(2, 1, 95.0), C, R).trades.iloc[0]
    t1 = engine.run(market(paths), one_entry(2, 1, 95.0), C, R, stop_mode=1).trades.iloc[0]
    assert t0["exit_px"] == pytest.approx(95 * (1 - 0.0001))
    assert t1["exit_px"] == pytest.approx(93 * (1 - 0.0001))
    assert t1["exit_i"] == t0["exit_i"] and t1["reason"] == "stop"
    paths_s = flat(100, 15) + [(100, 100.5, 99.5, 100)] + [(101, 107, 101, 104)] + flat(104, 13)
    s1 = engine.run(market(paths_s), one_entry(2, -1, 105.0), C, R, stop_mode=1).trades.iloc[0]
    assert s1["exit_px"] == pytest.approx(107 * (1 + 0.0001))
    # gap through the stop: worst-of-minute is never better than the gap fill at the open
    paths_g = flat(100, 15) + [(100, 100, 99, 99)] + [(90, 91, 88, 90)] + flat(90, 13)
    g0 = engine.run(market(paths_g), one_entry(2, 1, 95.0), C, R).trades.iloc[0]
    g1 = engine.run(market(paths_g), one_entry(2, 1, 95.0), C, R, stop_mode=1).trades.iloc[0]
    assert g0["exit_px"] == pytest.approx(90 * (1 - 0.0001))
    assert g1["exit_px"] == pytest.approx(88 * (1 - 0.0001))


def test_forced_exit_closes_with_penalty_and_no_end_trade():
    paths = flat(100, 15) + [(100, 100.5, 99.5, 100)] + flat(100, 10) + [(100, 101, 99, 98)] + flat(98, 3)
    mk = market(paths)
    fx = np.zeros(len(mk.ts), np.bool_)
    fx[26] = True
    res = engine.run(mk, one_entry(2, 1, 90.0), C, R, force_exit=fx, force_penalty=0.01)
    assert list(res.trades["reason"]) == ["delist"]
    t = res.trades.iloc[0]
    assert t["exit_i"] == 26
    assert t["exit_px"] == pytest.approx(98 * (1 - (0.0001 + 0.01)))
    assert res.equity.iloc[-1] == pytest.approx(10_000 + t["net"])


def test_stop_in_the_forced_exit_minute_wins():
    paths = flat(100, 15) + [(100, 100.5, 99.5, 100)] + flat(100, 10) + [(100, 100, 85, 88)] + flat(88, 3)
    mk = market(paths)
    fx = np.zeros(len(mk.ts), np.bool_)
    fx[26] = True
    t = engine.run(mk, one_entry(2, 1, 90.0), C, R, force_exit=fx, force_penalty=0.01).trades.iloc[0]
    assert t["reason"] == "stop" and t["exit_px"] == pytest.approx(90 * (1 - 0.0001))


def test_forced_exit_while_flat_does_nothing():
    mk, sg = multi_trade_case()
    a = engine.run(mk, sg, C, R)
    fx = np.zeros(len(mk.ts), np.bool_)
    fx[:15] = True                       # before the first entry executes
    b = engine.run(mk, sg, C, R, force_exit=fx, force_penalty=0.01)
    pd.testing.assert_frame_equal(a.trades, b.trades, check_exact=True)


def test_reason_codes():
    assert engine.REASONS[5] == "end" and engine.REASONS[6] == "manual" and engine.REASONS[7] == "delist"


def test_bad_array_lengths_are_rejected():
    mk, sg = multi_trade_case()
    with pytest.raises(ValueError):
        engine.run(mk, sg, C, R, slippage=np.zeros(3))


@needs_data
def test_f1_dev_trades_reproduce_exactly():
    """Rerun F1 (round 3) on the 5 markets and compare with the committed trade file."""
    from btcq.config import SLIPPAGE, SYMBOLS
    from btcq.research.portfolio import sleeve_risk
    from btcq.research.round3 import LATENCY_MIN
    from btcq.strategies.r3_trend import S1Frozen
    ref = pd.read_parquet(REPORTS_DIR / "results" / "F1_dev_trades.parquet")
    strat = S1Frozen()
    cols = ["entry_i", "exit_i", "dir", "entry_px", "exit_px", "qty", "fees", "funding", "net", "R"]
    for sym in SYMBOLS:
        mk = engine.Market.load("dev", sym, with_metrics=False)
        res = engine.run(mk, strat.compute(mk, n=40, k=3.5, exit="channel"), Costs(slippage=SLIPPAGE[sym]),
                         sleeve_risk(len(SYMBOLS)), start=strat.data_start, latency_min=LATENCY_MIN)
        got = res.trades[cols].reset_index(drop=True)
        want = ref[ref["symbol"] == sym][cols].reset_index(drop=True)
        pd.testing.assert_frame_equal(got, want, check_exact=False, rtol=1e-12, check_dtype=False)
