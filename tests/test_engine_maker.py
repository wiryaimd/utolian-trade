"""Maker-first execution (X1): T = 0 parity with engine.run, then hand-checked fills on synthetic 1m paths."""
import numpy as np
import pandas as pd
import pytest

from btcq.backtest import engine
from btcq.backtest import maker_engine as me
from btcq.config import HOLDOUT_START, PERP_START, Costs, Risk
from conftest import T0, flat, make_m1, needs_data

C = Costs(taker=0.0005, maker=0.0002, slippage=0.0001)
R = Risk(risk_per_trade=0.01, max_leverage=3.0, initial_equity=10_000.0)
TICK = 0.1
BASE = list(engine.TRADE_COLS) + ["entry_time", "exit_time", "R", "ret"]
TERMINAL = ["entry_maker_fills", "exit_maker_fills", "entry_fallbacks", "exit_fallbacks", "entries_missed",
            "exits_cancelled", "exits_preempted_by_stop"]


def market(paths, funding=None):
    return engine.Market(make_m1(paths), funding, synthetic_funding=0.0)


def signals(n_bars, entries=(), tf="15m", **kw):
    """entries: (bar, dir, stop[, target]) tuples."""
    e = np.zeros(n_bars, np.int8)
    s = np.full(n_bars, np.nan)
    t = np.full(n_bars, np.nan)
    for x in entries:
        e[x[0]], s[x[0]] = x[1], x[2]
        if len(x) > 3:
            t[x[0]] = x[3]
    return engine.Signals(tf, e, s, t, **kw)


def bools(n, idx):
    a = np.zeros(n, np.bool_)
    a[list(idx)] = True
    return a


def mrun(mk, sg, T, model="strict", fallback=True, **kw):
    return me.run(mk, sg, C, R, policy=me.MakerPolicy(T, model, fallback), tick=np.full(len(mk.ts), TICK), **kw)


def qty(cash, px, stop, size=np.nan):
    return min(R.risk_per_trade * cash / abs(px - stop), R.max_leverage * cash / px) if np.isnan(size) \
        else min(size * cash / px, R.max_leverage * cash / px)


def assert_parity(mk, sg, **kw):
    a = engine.run(mk, sg, C, R, **{k: v for k, v in kw.items() if k not in ("policy", "tick")})
    b = me.run(mk, sg, C, R, **kw)
    pd.testing.assert_frame_equal(b.trades[BASE], a.trades, check_exact=True)
    pd.testing.assert_series_equal(b.equity, a.equity, check_exact=True)
    assert b.start == a.start
    return a, b


def assert_accounting(res):
    """Every posted order ends exactly once; with T > 0 every position comes from an entry order."""
    info = res.info
    assert sum(info[k] for k in TERMINAL) == info["orders_posted"]
    assert len(info["missed_entry_tf"]) == info["entries_missed"]
    assert len(res.trades) == info["entry_maker_fills"] + info["entry_fallbacks"]


# ---------------------------------------------------------------- T = 0 parity

def case_multi():
    """Target, stop, signal exit, reverse and end trades on one path (as test_engine_r4)."""
    up = [(100 + i, 101 + i, 99.5 + i, 101 + i) for i in range(15)]
    paths = (flat(100, 15) + up + flat(115, 15) + [(115, 115, 80, 85)] + flat(85, 14)
             + flat(85, 15) + flat(86, 15) + flat(87, 15) + flat(88, 15) + flat(89, 15))
    n = len(paths) // 15
    sg = signals(n, [(0, 1, 95.0, 110.0), (2, 1, 90.0), (4, 1, 80.0), (6, 1, 80.0), (7, -1, 95.0)],
                 exit_long=bools(n, [5]))
    return market(paths), sg, {}


def case_short_capped_stop():
    paths = flat(100, 15) + [(100, 100.2, 99.0, 99.0)] + flat(99.0, 14)
    return market(paths), signals(2, [(0, -1, 100.1)]), {}


def case_gap_through_stop():
    paths = flat(100, 15) + [(100, 100, 99, 99)] + [(90, 91, 89, 90)] + flat(90, 13)
    return market(paths), signals(2, [(0, 1, 95.0)]), {}


def case_exit_and_entry_same_bar():
    wig = [(100 + np.sin(k / 3), 100.6 + np.sin(k / 3), 99.4 + np.sin(k / 3), 100 + np.sin((k + 1) / 3))
           for k in range(75)]
    sg = signals(5, [(0, 1, 95.0), (2, -1, 104.0)], exit_long=bools(5, [2]))
    return market(wig), sg, {}


def case_trailing():
    up = [(100 + i * 10 / 15, 100 + (i + 1) * 10 / 15, 100 + i * 10 / 15, 100 + (i + 1) * 10 / 15)
          for i in range(15)]
    paths = flat(100, 15) + up + [(110, 110, 106, 106)] + flat(106, 14)
    return market(paths), signals(3, [(0, 1, 90.0)], trail=np.full(3, 3.0)), {}


def case_time_stop():
    return market(flat(100, 90)), signals(6, [(0, 1, 90.0), (3, -1, 110.0)], max_bars=2), {}


def case_funding():
    n = 9 * 60
    fund = pd.DataFrame({"ts": [T0 + 8 * 3_600_000], "rate": [0.001]})
    nb = n // 15
    return market(flat(100, n), fund), signals(nb, [(0, 1, 90.0), (nb - 2, -1, 110.0)], max_bars=nb - 4), {}


def case_force_exit():
    paths = flat(100, 15) + [(100, 100.5, 99.5, 100)] + flat(100, 10) + [(100, 101, 99, 98)] + flat(98, 3)
    fx = bools(len(paths), [26])
    return market(paths), signals(2, [(0, 1, 90.0)]), {"force_exit": fx, "force_penalty": 0.01}


def case_size_multiple():
    paths = flat(100, 15) + [(100, 100.5, 99.5, 100)] + flat(100, 14)
    return market(paths), signals(2, [(0, 1, 90.0)], size=np.array([0.5, np.nan])), {}


HAND_CASES = [case_multi, case_short_capped_stop, case_gap_through_stop, case_exit_and_entry_same_bar,
              case_trailing, case_time_stop, case_funding, case_force_exit, case_size_multiple]


@pytest.mark.parametrize("case", HAND_CASES, ids=lambda f: f.__name__)
def test_t0_is_bit_identical_on_hand_cases(case):
    mk, sg, kw = case()
    a, _ = assert_parity(mk, sg, **kw)
    assert len(a.trades) > 0
    assert_parity(mk, sg, latency_min=1, stop_mode=1, **kw)
    # a T = 0 policy ignores the model, the fallback flag and the tick
    assert_parity(mk, sg, policy=me.MakerPolicy(0, "optimistic", False), tick=np.full(len(mk.ts), TICK), **kw)


def test_hand_cases_cover_every_engine_reason():
    reasons = set()
    for case in HAND_CASES:
        mk, sg, kw = case()
        reasons |= set(engine.run(mk, sg, C, R, **kw).trades["reason"])
    assert reasons == {"stop", "target", "exit_signal", "time", "reverse", "end", "delist"}


def test_t0_extra_columns_describe_taker_legs():
    mk, sg, _ = case_multi()
    res = me.run(mk, sg, C, R)
    t = res.trades
    decision = t["reason"].isin(["exit_signal", "time", "reverse"])
    assert list(t["reason"]) == ["target", "stop", "exit_signal", "reverse", "end"]
    assert (t["entry_post_i"] == t["entry_i"]).all() and (t["entry_ref_px"] == t["entry_px"]).all()
    assert (t.loc[decision, "exit_post_i"] == t.loc[decision, "exit_i"]).all()
    assert (t.loc[decision, "exit_ref_px"] == t.loc[decision, "exit_px"]).all()
    assert (t.loc[~decision, "exit_post_i"] == -1).all() and t.loc[~decision, "exit_ref_px"].isna().all()
    assert (t["exit_maker"] == 0).all()
    assert list(t.columns[:len(me.MF_COLS)]) == me.MF_COLS
    assert all(res.info[k] == 0 for k in me.INFO_KEYS) and len(res.info["missed_entry_tf"]) == 0


def random_case(seed, n_bars=240, tf="15m"):
    """A random 1m walk with occasional gaps, real-style 8h funding and random signals of every kind
    (including entries whose stop is on the wrong side, which the engine skips)."""
    rng = np.random.default_rng(seed)
    n = n_bars * 15
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    o = np.r_[100.0, c[:-1]] * (1 + rng.normal(0, 0.003, n) * (rng.random(n) < 0.03))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.001, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.001, n)))
    fts = T0 + 8 * 3_600_000 * np.arange(1, n // 480 + 1)
    mk = market(list(zip(o, h, l, c)), pd.DataFrame({"ts": fts, "rate": rng.normal(0, 0.0005, len(fts))}))
    bc = mk.bars(tf)["close"].to_numpy()
    nb = len(bc)
    u = rng.random(nb)
    entry = np.where(u < 0.12, 1, np.where(u > 0.88, -1, 0)).astype(np.int8)
    dist = bc * rng.uniform(0.003, 0.03, nb)
    stop = np.where(rng.random(nb) < 0.1, bc + entry * dist, bc - entry * dist)
    stop[entry == 0] = np.nan
    target = np.where(rng.random(nb) < 0.4, bc + entry * 2 * dist, np.nan)
    trail = np.where(rng.random(nb) < 0.3, dist, np.nan)
    size = np.where(rng.random(nb) < 0.2, rng.uniform(0.3, 5.0, nb), np.nan)
    sg = engine.Signals(tf, entry, stop, target, trail=trail, exit_long=rng.random(nb) < 0.08,
                        exit_short=rng.random(nb) < 0.08, max_bars=int(rng.integers(0, 12)), size=size)
    fx = rng.random(n) < 0.002
    slip = rng.uniform(0.0, 0.0004, n)
    return mk, sg, fx, slip


@pytest.mark.parametrize("seed", range(6))
def test_t0_is_bit_identical_on_random_paths(seed):
    mk, sg, fx, slip = random_case(seed)
    a, _ = assert_parity(mk, sg)
    assert len(a.trades) > 15 and a.trades["funding"].abs().sum() > 0
    assert_parity(mk, sg, latency_min=1, slippage=slip)
    assert_parity(mk, sg, stop_mode=1, force_exit=fx, force_penalty=0.003, slippage=slip)
    assert_parity(mk, sg, start=pd.Timestamp(int(mk.ts[700]), unit="ms", tz="UTC"), latency_min=2)


# ---------------------------------------------------------------- fill models

# bar 1 opens at 100; P = 99.9 is touched in minute 16 and traded through in minute 17
ENTRY_PATH = flat(100, 15) + [(100, 100.2, 99.95, 100), (100, 100, 99.9, 99.95),
                              (99.95, 99.95, 99.8, 99.85)] + flat(99.85, 12)


@pytest.mark.parametrize("model,fill_i,px", [("strict", 17, 100 * (1 + 0.0001)), ("central", 17, 100.0),
                                             ("optimistic", 16, 99.9)])
def test_entry_fill_models(model, fill_i, px):
    mk = market(ENTRY_PATH)
    res = mrun(mk, signals(2, [(0, 1, 95.0)]), 5, model)
    t = res.trades.iloc[0]
    ref = 100 * (1 + 0.0001)
    q = qty(10_000, ref, 95.0)                      # sized at the post with the taker price
    xpx = 99.85 * (1 - 0.0001)
    assert t["entry_i"] == fill_i and t["maker"] == 1 and t["entry_post_i"] == 15
    assert t["entry_px"] == pytest.approx(px, rel=1e-15) and t["entry_ref_px"] == pytest.approx(ref, rel=1e-15)
    assert t["qty"] == pytest.approx(q, rel=1e-14)
    assert t["init_risk"] == pytest.approx(q * (px - 95.0), rel=1e-14)
    assert t["fees"] == pytest.approx(q * px * 0.0002 + q * xpx * 0.0005, rel=1e-14)
    assert t["gross"] == pytest.approx(q * (xpx - px), rel=1e-12)
    assert t["reason"] == "end" and t["exit_post_i"] == -1 and np.isnan(t["exit_ref_px"])
    assert res.equity.iloc[-1] == pytest.approx(10_000 + t["net"], rel=1e-14)
    assert res.info["orders_posted"] == 1 and res.info["entry_maker_fills"] == 1
    assert_accounting(res)


EXIT_PATH = (flat(100, 15) + [(100, 100, 99.5, 100)] + flat(100, 14)
             + [(101, 101.1, 100.9, 101), (101, 101.2, 101, 101.1)] + flat(101.1, 13))


@pytest.mark.parametrize("model,exit_i,xpx,epx", [("strict", 31, 101 * (1 - 0.0001), 100 * (1 + 0.0001)),
                                                  ("central", 31, 101.0, 100.0),
                                                  ("optimistic", 30, 101.1, 99.9)])
def test_exit_fill_models(model, exit_i, xpx, epx):
    # the entry fills in minute 15 (low 99.5 < 99.9); the exit signal of bar 1 posts a sell at 101 + 0.1
    mk = market(EXIT_PATH)
    res = mrun(mk, signals(3, [(0, 1, 95.0)], exit_long=bools(3, [1])), 5, model)
    assert len(res.trades) == 1
    t = res.trades.iloc[0]
    q = qty(10_000, 100 * (1 + 0.0001), 95.0)
    assert t["entry_i"] == 15 and t["entry_px"] == pytest.approx(epx, rel=1e-15)
    assert t["reason"] == "exit_signal" and t["exit_i"] == exit_i and t["exit_maker"] == 1
    assert t["exit_post_i"] == 30 and t["exit_ref_px"] == pytest.approx(101 * (1 - 0.0001), rel=1e-15)
    assert t["exit_px"] == pytest.approx(xpx, rel=1e-15)
    assert t["fees"] == pytest.approx(q * epx * 0.0002 + q * xpx * 0.0002, rel=1e-14)
    assert res.info["exit_maker_fills"] == 1 and res.info["entry_maker_fills"] == 1
    assert_accounting(res)


def test_short_entry_mirrors_the_fill_condition():
    # sell at P = 100.1: a high of exactly 100.1 fills only the optimistic model
    paths = flat(100, 15) + [(100, 100.1, 99.9, 100), (100, 100.15, 100, 100.1)] + flat(100.1, 13)
    sg = signals(2, [(0, -1, 105.0)])
    s = mrun(market(paths), sg, 5, "strict").trades.iloc[0]
    op = mrun(market(paths), sg, 5, "optimistic").trades.iloc[0]
    assert s["dir"] == -1 and s["entry_i"] == 16 and s["entry_px"] == pytest.approx(100 * (1 - 0.0001))
    assert op["entry_i"] == 15 and op["entry_px"] == pytest.approx(100.1)


@pytest.mark.parametrize("o,touch,side,tick", [(100.2, 100.1, 1, 0.1),     # 100.2 - 0.1 = 100.10000000000001
                                               (100.1, 100.2, -1, 0.1),    # 100.1 + 0.1 = 100.19999999999999
                                               (100.02, 100.01, 1, 0.01)])  # 100.02 - 0.01 = 100.00999999999999
def test_an_exact_one_tick_touch_is_a_touch_despite_float_noise(o, touch, side, tick):
    assert o - side * tick != touch
    bar = (o, touch, o, o) if side == -1 else (o, o, touch, o)
    mk = market(flat(o, 15) + [bar] + flat(o, 14))
    sg = signals(2, [(0, side, o * (1 - 0.05 * side))])
    for model, entry_i, maker in (("strict", 20, 0), ("central", 20, 0), ("optimistic", 15, 1)):
        res = me.run(mk, sg, C, R, policy=me.MakerPolicy(5, model), tick=np.full(len(mk.ts), tick))
        t = res.trades.iloc[0]
        assert (t["entry_i"], t["maker"]) == (entry_i, maker), model


def grid_legs(tick, side, offset, model, n=4000, seed=5):
    """One leg per 2-row block: posted at a grid open o = k * tick; row 0 reaches o - side * offset ticks."""
    from decimal import Decimal
    k = np.random.default_rng(seed).integers(100, 2_000_000, n)
    q = Decimal(tick)
    o = np.array([float(Decimal(int(x)) * q) for x in k])
    reach = np.array([float(Decimal(int(x) - side * offset) * q) for x in k])
    rows = np.empty((2 * n, 4))
    rows[0::2] = np.c_[o, np.maximum(o, reach), np.minimum(o, reach), o]
    rows[1::2] = np.c_[o, o, o, o]
    ts = T0 + 60_000 * np.arange(2 * n, dtype=np.int64)
    return me.simulate_legs(rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3], ts, np.arange(0, 2 * n, 2),
                            np.full(n, side), float(tick), 0.0001, 1, model, True, markouts=())


@pytest.mark.parametrize("tick", ["0.1", "0.01", "0.001", "0.0001"])
@pytest.mark.parametrize("side", [1, -1])
def test_touch_and_trade_through_are_exact_on_decimal_price_grids(tick, side):
    assert not grid_legs(tick, side, 1, "strict")["filled"].any()           # a touch never fills strict
    assert not grid_legs(tick, side, 1, "central")["filled"].any()
    assert grid_legs(tick, side, 1, "optimistic")["filled"].all()           # and always fills optimistic
    assert grid_legs(tick, side, 2, "strict")["filled"].all()               # one tick through fills strict
    assert not grid_legs(tick, side, 0, "optimistic")["filled"].any()       # the open itself is no fill


@pytest.mark.parametrize("model,fill_px,stop", [("optimistic", 99.9, 99.9), ("optimistic", 99.9, 99.95),
                                                ("central", 100.0, 100.005)])
def test_a_maker_fill_at_or_beyond_its_own_stop_opens_nothing(model, fill_px, stop):
    # the stop is valid against ref = 100.01 but not against the model's fill price: as the engine's limit
    # fills, no position opens (it would have R = -inf or be stopped out above its entry) and the entry is missed
    mk = market(flat(100, 15) + [(100, 100.5, 99.5, 100)] + flat(100, 14))
    sg = signals(2, [(0, 1, stop)])
    res = mrun(mk, sg, 5, model)
    assert len(res.trades) == 0 and (res.equity == 10_000).all()
    assert res.info["entries_missed"] == 1 and list(res.info["missed_entry_tf"]) == [0]
    assert_accounting(res)
    assert len(engine.run(mk, signals(2, [(0, 1, stop)], limit=np.array([fill_px, np.nan])), C, R).trades) == 0
    s = mrun(mk, sg, 5, "strict").trades           # strict fills at ref, where the stop is valid
    assert len(s) == 1 and s.iloc[0]["init_risk"] > 0 and np.isfinite(s.iloc[0]["R"])


# ---------------------------------------------------------------- quantity, fallback, cancellation

RUNAWAY_PATH = flat(100, 15) + [(100 + 0.5 * k, 100.5 + 0.5 * k, 100 + 0.5 * k, 100.5 + 0.5 * k)
                                for k in range(15)]


@pytest.mark.parametrize("model", list(me.MODELS))
def test_run_away_entry_falls_back_with_the_quantity_fixed_at_post(model):
    mk = market(RUNAWAY_PATH)
    res = mrun(mk, signals(2, [(0, 1, 95.0)]), 5, model)
    t = res.trades.iloc[0]
    ref = 100 * (1 + 0.0001)
    fpx = 102.5 * (1 + 0.0001)                      # o[20] with the fallback minute's slippage
    q = qty(10_000, ref, 95.0)
    assert t["entry_i"] == 20 and t["entry_post_i"] == 15 and t["maker"] == 0
    assert t["entry_px"] == pytest.approx(fpx, rel=1e-15) and t["entry_ref_px"] == pytest.approx(ref, rel=1e-15)
    assert t["qty"] == pytest.approx(q, rel=1e-14)
    assert abs(t["qty"] / qty(10_000, fpx, 95.0) - 1) > 0.3   # a re-size at the fill would be ~33% smaller
    assert t["init_risk"] == pytest.approx(q * (fpx - 95.0), rel=1e-14)
    xpx = 107.5 * (1 - 0.0001)
    assert t["fees"] == pytest.approx(q * fpx * 0.0005 + q * xpx * 0.0005, rel=1e-14)
    assert res.info["entry_fallbacks"] == 1 and res.info["entry_maker_fills"] == 0
    assert_accounting(res)


def test_a_fallback_entry_can_hit_its_target_in_its_fill_minute():
    # a taker fallback is a market entry: the engine checks the target in that minute (o[20] = 102.5, high 103)
    res = mrun(market(RUNAWAY_PATH), signals(2, [(0, 1, 95.0, 103.0)]), 5)
    t = res.trades.iloc[0]
    assert t["entry_i"] == 20 and t["maker"] == 0 and t["reason"] == "target" and t["exit_i"] == 20
    assert t["exit_px"] == pytest.approx(103 * (1 - 0.0001), rel=1e-15)


def test_a_maker_fill_skips_the_target_in_its_fill_minute():
    # as the engine's limit fills: the fill minute's range is not used for the target
    paths = flat(100, 15) + [(100, 105, 99.5, 104), (104, 104.6, 104, 104.5)] + flat(104.5, 13)
    t = mrun(market(paths), signals(2, [(0, 1, 95.0, 104.5)]), 5).trades.iloc[0]
    assert t["entry_i"] == 15 and t["maker"] == 1 and t["reason"] == "target" and t["exit_i"] == 16
    assert t["exit_px"] == pytest.approx(104.5 * (1 - 0.0001), rel=1e-15)


def test_post_and_cancel_records_the_missed_entry():
    mk = market(RUNAWAY_PATH)
    res = mrun(mk, signals(2, [(0, 1, 95.0)]), 5, fallback=False)
    assert len(res.trades) == 0
    assert res.info["entries_missed"] == 1 and list(res.info["missed_entry_tf"]) == [0]
    assert (res.equity == 10_000).all()
    assert_accounting(res)


# bar 2 falls steadily (highs at the open, never above the resting sell); bar 3 bounces through it
RETRY_PATH = (flat(100, 15) + [(100, 100, 99.5, 100)] + flat(100, 14)
              + [(100 - 0.2 * k, 100 - 0.2 * k, 99.8 - 0.2 * k, 99.8 - 0.2 * k) for k in range(15)]
              + [(97, 97.5, 97, 97.2)] + flat(97.2, 14))


def test_cancelled_exit_is_retried_at_the_next_decision():
    mk = market(RETRY_PATH)
    sg = signals(4, [(0, 1, 90.0)], exit_long=bools(4, [1, 2]))
    res = mrun(mk, sg, 5, fallback=False)
    assert len(res.trades) == 1
    t = res.trades.iloc[0]
    assert t["reason"] == "exit_signal" and t["exit_i"] == 45 and t["exit_post_i"] == 45
    assert t["exit_maker"] == 1 and t["exit_px"] == pytest.approx(97 * (1 - 0.0001), rel=1e-15)
    assert res.info["exits_cancelled"] == 1 and res.info["exit_maker_fills"] == 1
    assert res.info["orders_posted"] == 3
    assert_accounting(res)
    # with the fallback the first exit crosses at m0 + 5 instead
    t2 = mrun(mk, sg, 5).trades.iloc[0]
    assert t2["exit_i"] == 35 and t2["exit_post_i"] == 30 and t2["exit_maker"] == 0
    assert t2["exit_px"] == pytest.approx(99.0 * (1 - 0.0001), rel=1e-15)
    assert t2["exit_ref_px"] == pytest.approx(100 * (1 - 0.0001), rel=1e-15)


def test_the_trailing_extreme_includes_minutes_while_an_exit_rests():
    # minute 30 makes a new high (100.1) that does not fill the resting sell at 100 + 0.1; the exit is
    # cancelled, so bar 2's trail of 3 ratchets the stop to 100.1 - 3 = 97.1, hit by minute 46's low
    paths = (flat(100, 15) + [(100, 100, 99.5, 100)] + flat(100, 14) + [(100, 100.1, 100, 100)] + flat(100, 14)
             + [(100, 100, 100, 100), (99, 99, 97.05, 98)] + flat(98, 13))
    sg = signals(4, [(0, 1, 90.0)], trail=np.full(4, 3.0), exit_long=bools(4, [1]))
    res = mrun(market(paths), sg, 5, fallback=False)
    t = res.trades.iloc[0]
    assert res.info["exits_cancelled"] == 1
    assert t["reason"] == "stop" and t["exit_i"] == 46
    assert t["exit_px"] == pytest.approx((100.1 - 3.0) * (1 - 0.0001), rel=1e-15)


@pytest.mark.parametrize("model", list(me.MODELS))
def test_a_maker_entry_trails_from_its_market_price_not_the_strict_ref(model):
    # strict fills at ref = 100.01, but its extreme starts at o = 100 as the taker's does: bar 1's trail of 3
    # sets the stop at 97 (not 97.01), which minute 30's low 97.005 does not reach
    paths = flat(100, 15) + [(100, 100, 99.5, 100)] + flat(100, 14) + [(100, 100, 97.005, 100)] + flat(100, 14)
    sg = signals(3, [(0, 1, 90.0)], trail=np.full(3, 3.0))
    a = engine.run(market(paths), sg, C, R).trades.iloc[0]
    t = mrun(market(paths), sg, 5, model).trades.iloc[0]
    assert (t["entry_i"], t["exit_i"], t["reason"]) == (a["entry_i"], a["exit_i"], a["reason"]) == (15, 44, "end")
    if model == "strict":                       # the same trade with the maker fee on the entry
        for col in ("entry_px", "exit_px", "qty", "init_risk", "gross"):
            assert t[col] == a[col], col
        assert t["fees"] == pytest.approx(a["fees"] - a["qty"] * a["entry_px"] * 0.0003, rel=1e-14)


def test_a_later_decision_forces_a_resting_order():
    # T = 30 on 15m bars: the next bar's decision (minute 30) cuts the order short
    mk = market(flat(100, 45))
    res = mrun(mk, signals(3, [(0, 1, 95.0)]), 30)
    t = res.trades.iloc[0]
    assert t["entry_i"] == 30 and t["entry_post_i"] == 15 and t["maker"] == 0
    assert t["entry_px"] == pytest.approx(100 * (1 + 0.0001), rel=1e-15)
    assert res.info["forced_fallbacks"] == 1 and res.info["entry_fallbacks"] == 1
    assert_accounting(res)
    res2 = mrun(mk, signals(3, [(0, 1, 95.0)]), 30, fallback=False)
    assert len(res2.trades) == 0 and res2.info["forced_fallbacks"] == 1 and res2.info["entries_missed"] == 1
    assert_accounting(res2)


# ---------------------------------------------------------------- stops while an exit rests

@pytest.mark.parametrize("model", list(me.MODELS))
def test_stop_preempts_a_resting_exit(model):
    # minute 31 touches both the stop (low 94) and the resting sell (high 100.5 > 100.1): the stop wins
    paths = (flat(100, 15) + [(100, 100, 99.5, 100)] + flat(100, 14)
             + [(100, 100, 99.9, 100), (100, 100.5, 94, 95)] + flat(95, 13))
    res = mrun(market(paths), signals(3, [(0, 1, 95.0)], exit_long=bools(3, [1])), 5, model)
    assert len(res.trades) == 1
    t = res.trades.iloc[0]
    assert t["reason"] == "stop" and t["exit_i"] == 31 and t["exit_maker"] == 0
    assert t["exit_px"] == pytest.approx(95 * (1 - 0.0001), rel=1e-15)
    assert t["exit_post_i"] == -1 and np.isnan(t["exit_ref_px"])
    assert res.info["exits_preempted_by_stop"] == 1 and res.info["exit_maker_fills"] == 0
    assert_accounting(res)


def reversal_sizes(cash_after_entry, epx, q1, stop2):
    """Quantity of the reversal's entry: sized on the cash a taker exit at minute 30 would leave."""
    xr = 100 * (1 - 0.0001)
    cp = cash_after_entry + (q1 * (xr - epx) * 1 - q1 * xr * 0.0005)
    return qty(cp, xr, stop2), xr


def test_merged_reversal_fills_exit_and_entry_together():
    paths = (flat(100, 15) + [(100, 100, 99.5, 100)] + flat(100, 14)
             + [(100, 100.3, 99.9, 100.2)] + flat(100.2, 14))
    res = mrun(market(paths), signals(3, [(0, 1, 95.0), (1, -1, 105.0)]), 5)
    assert list(res.trades["reason"]) == ["reverse", "end"]
    long, short = res.trades.iloc[0], res.trades.iloc[1]
    epx = 100 * (1 + 0.0001)
    q1 = qty(10_000, epx, 95.0)
    c1 = 10_000 - q1 * epx * 0.0002
    q2, xr = reversal_sizes(c1, epx, q1, 105.0)
    assert long["exit_i"] == 30 and long["exit_maker"] == 1 and long["exit_px"] == pytest.approx(xr, rel=1e-15)
    assert long["fees"] == pytest.approx(q1 * epx * 0.0002 + q1 * xr * 0.0002, rel=1e-14)
    c2 = c1 + (q1 * (xr - epx) - q1 * xr * 0.0002)
    assert short["dir"] == -1 and short["entry_i"] == 30 and short["maker"] == 1 and short["entry_post_i"] == 30
    assert short["qty"] == pytest.approx(q2, rel=1e-13)
    assert abs(short["qty"] / qty(c2, xr, 105.0) - 1) > 1e-5   # not re-sized on the realised cash
    assert short["eq_before"] == pytest.approx(c2, rel=1e-14)
    assert res.info["orders_posted"] == 3 and res.info["exit_maker_fills"] == 1
    assert res.info["entry_maker_fills"] == 2
    assert_accounting(res)


@pytest.mark.parametrize("minute31,entry_i,entry_px,maker", [
    ((99, 99, 94, 95), 35, 95 * (1 - 0.0001), 0),        # the entry never fills and falls back at m0 + 5
    ((100, 100.5, 94, 95), 31, 100 * (1 - 0.0001), 1),   # the entry fills in the stop minute
])
def test_stop_preempts_the_exit_of_a_reversal_and_the_entry_still_executes(minute31, entry_i, entry_px, maker):
    paths = (flat(100, 15) + [(100, 100, 99.5, 100)] + flat(100, 14)
             + [(100, 100, 99.9, 99.9), minute31] + flat(95, 13))
    res = mrun(market(paths), signals(3, [(0, 1, 95.0), (1, -1, 110.0)]), 5)
    assert list(res.trades["reason"]) == ["stop", "end"]
    long, short = res.trades.iloc[0], res.trades.iloc[1]
    epx = 100 * (1 + 0.0001)
    q1 = qty(10_000, epx, 95.0)
    c1 = 10_000 - q1 * epx * 0.0002
    q2, xr = reversal_sizes(c1, epx, q1, 110.0)
    assert long["exit_i"] == 31 and long["exit_px"] == pytest.approx(95 * (1 - 0.0001), rel=1e-15)
    assert long["exit_post_i"] == -1
    assert short["entry_i"] == entry_i and short["maker"] == maker and short["entry_post_i"] == 30
    assert short["entry_px"] == pytest.approx(entry_px, rel=1e-15)
    assert short["entry_ref_px"] == pytest.approx(xr, rel=1e-15)
    assert short["qty"] == pytest.approx(q2, rel=1e-13)
    assert short["eq_before"] == pytest.approx(c1 + long["gross"] - q1 * long["exit_px"] * 0.0005, rel=1e-13)
    assert res.info["exits_preempted_by_stop"] == 1
    assert res.info["entry_fallbacks"] == 1 - maker
    assert_accounting(res)


# ---------------------------------------------------------------- funding inside a resting window

FUND = pd.DataFrame({"ts": [T0 + 8 * 3_600_000], "rate": [0.001]})   # settles at minute 480


@pytest.mark.parametrize("T,counted", [(60, 1), (59, 1), (58, 0)])
def test_funding_inside_a_resting_entry_window_is_counted(T, counted):
    # 1h bars, latency 1: the bar-6 entry is posted at 07:01 (minute 421); flat prices never fill it
    mk = market(flat(100, 9 * 60), FUND)
    res = mrun(mk, signals(9, [(6, 1, 90.0)], tf="1h"), T, latency_min=1)
    t = res.trades.iloc[0]
    assert res.info["funding_in_window"] == counted
    assert t["entry_i"] == 421 + T
    # the position pays the 08:00 funding only when it opened before that minute
    assert t["funding"] == pytest.approx(0.0 if counted else t["qty"] * 100 * 0.001)


def test_funding_inside_a_resting_exit_window_is_counted_and_paid():
    mk = market(flat(100, 9 * 60), FUND)
    sg = signals(9, [(2, 1, 90.0)], tf="1h", exit_long=bools(9, [6]))
    res = mrun(mk, sg, 60, latency_min=1)
    t = res.trades.iloc[0]
    assert t["entry_i"] == 241 and t["exit_post_i"] == 421 and t["exit_i"] == 481
    assert res.info["funding_in_window"] == 1
    assert t["funding"] == pytest.approx(t["qty"] * 100 * 0.001)    # still open at 08:00, so it pays


def test_an_order_posted_at_the_funding_minute_is_not_counted():
    mk = market(flat(100, 10 * 60), FUND)
    res = mrun(mk, signals(10, [(7, 1, 90.0)], tf="1h"), 60)    # posted at 08:00, after the settlement
    assert res.info["funding_in_window"] == 0 and res.trades.iloc[0]["entry_i"] == 540


# ---------------------------------------------------------------- window end

def test_entry_resting_at_the_window_end_crosses_at_the_last_close():
    mk = market(flat(100, 18))       # one full bar; posted at 15, expiry at 20 lies beyond the data
    res = mrun(mk, signals(1, [(0, 1, 95.0)]), 5)
    t = res.trades.iloc[0]
    q = qty(10_000, 100 * (1 + 0.0001), 95.0)
    assert t["reason"] == "end" and t["entry_i"] == 17 and t["exit_i"] == 17 and t["maker"] == 0
    assert t["entry_post_i"] == 15 and t["exit_post_i"] == -1
    assert t["entry_px"] == pytest.approx(100 * (1 + 0.0001)) and t["exit_px"] == pytest.approx(100 * (1 - 0.0001))
    assert t["qty"] == pytest.approx(q, rel=1e-14)
    assert t["gross"] == pytest.approx(q * (t["exit_px"] - t["entry_px"]), rel=1e-12)
    assert t["fees"] == pytest.approx(q * (t["entry_px"] + t["exit_px"]) * 0.0005, rel=1e-14)
    assert res.equity.iloc[-1] == pytest.approx(10_000 + t["net"], rel=1e-14)
    assert res.info["entry_fallbacks"] == 1
    assert_accounting(res)
    res2 = mrun(mk, signals(1, [(0, 1, 95.0)]), 5, fallback=False)
    assert len(res2.trades) == 0 and list(res2.info["missed_entry_tf"]) == [0]
    assert res2.equity.iloc[-1] == 10_000
    assert_accounting(res2)


END_PATH = flat(100, 15) + [(100, 100, 99.5, 100)] + flat(100, 14) + flat(100, 3)    # 2 full bars


def test_exit_resting_at_the_window_end_is_cancelled_by_the_end_close():
    res = mrun(market(END_PATH), signals(2, [(0, 1, 95.0)], exit_long=bools(2, [1])), 5)
    t = res.trades.iloc[0]
    assert t["reason"] == "end" and t["exit_i"] == 32 and t["exit_maker"] == 0 and t["exit_post_i"] == -1
    assert t["exit_px"] == pytest.approx(100 * (1 - 0.0001))
    assert res.info["exits_cancelled"] == 1
    assert_accounting(res)


def test_reversal_resting_at_the_window_end():
    res = mrun(market(END_PATH), signals(2, [(0, 1, 95.0), (1, -1, 105.0)]), 5)
    assert list(res.trades["reason"]) == ["end", "end"]
    long, short = res.trades.iloc[0], res.trades.iloc[1]
    epx = 100 * (1 + 0.0001)
    q1 = qty(10_000, epx, 95.0)
    q2, _ = reversal_sizes(10_000 - q1 * epx * 0.0002, epx, q1, 105.0)
    assert long["exit_i"] == 32 and short["entry_i"] == 32 and short["exit_i"] == 32
    assert short["dir"] == -1 and short["qty"] == pytest.approx(q2, rel=1e-13)
    assert short["entry_px"] == pytest.approx(100 * (1 - 0.0001)) and short["exit_px"] == pytest.approx(100.01)
    assert short["entry_post_i"] == 30 and short["maker"] == 0
    assert res.info["exits_cancelled"] == 1 and res.info["entry_fallbacks"] == 1
    assert res.equity.iloc[-1] == pytest.approx(10_000 + res.trades["net"].sum(), rel=1e-14)
    assert_accounting(res)


# ---------------------------------------------------------------- forced exits and a bankrupt account

# bar 1 opens at 100 and rises from minute 16; a forced exit (delist) fires at minute 16
DELIST_PATH = (flat(100, 15) + [(100, 100.5, 100, 100.5)] + [(100.5 + 0.2 * k,) * 4 for k in range(14)]
               + flat(103, 15))


def test_a_forced_exit_takes_a_resting_entry_with_it():
    mk = market(DELIST_PATH)
    fx = bools(len(DELIST_PATH), [16])
    sg = signals(3, [(0, 1, 95.0)])
    a = engine.run(mk, sg, C, R, force_exit=fx, force_penalty=0.01).trades
    assert list(zip(a["entry_i"], a["exit_i"], a["reason"])) == [(15, 16, "delist")]
    res = mrun(mk, sg, 5, force_exit=fx, force_penalty=0.01)
    assert len(res.trades) == 1
    t = res.trades.iloc[0]
    c16 = 100.5
    q = qty(10_000, 100 * (1 + 0.0001), 95.0)
    assert (t["entry_i"], t["exit_i"], t["reason"], t["maker"], t["entry_post_i"]) == (16, 16, "delist", 0, 15)
    assert t["entry_px"] == pytest.approx(c16 * (1 + 0.0001), rel=1e-15)
    assert t["exit_px"] == pytest.approx(c16 * (1 - 0.0101), rel=1e-15)
    assert t["qty"] == pytest.approx(q, rel=1e-14) and t["funding"] == 0.0
    assert t["fees"] == pytest.approx(q * (t["entry_px"] + t["exit_px"]) * 0.0005, rel=1e-14)
    assert res.equity.iloc[-1] == pytest.approx(10_000 + t["net"], rel=1e-14)
    assert res.info["entry_fallbacks"] == 1
    assert_accounting(res)
    nf = mrun(mk, sg, 5, fallback=False, force_exit=fx, force_penalty=0.01)
    assert len(nf.trades) == 0 and list(nf.info["missed_entry_tf"]) == [0]
    assert_accounting(nf)


def test_a_forced_exit_inside_a_resting_reversal_closes_both_legs():
    # the long fills in minute 15; bar 1 reverses (sell X and N rest at 100.1); the delist fires at minute 31
    paths = flat(100, 15) + [(100, 100, 99.5, 100)] + flat(100, 14) + flat(100, 2) + [(99, 99, 98.9, 99)] * 13
    fx = bools(len(paths), [31])
    sg = signals(3, [(0, 1, 95.0), (1, -1, 105.0)])
    a = engine.run(market(paths), sg, C, R, force_exit=fx).trades
    assert list(a["reason"]) == ["reverse", "delist"] and list(a["exit_i"]) == [30, 31]
    res = mrun(market(paths), sg, 5, force_exit=fx)
    t = res.trades
    assert list(t["reason"]) == ["delist", "delist"] and list(t["exit_i"]) == [31, 31]
    assert list(t["dir"]) == [1, -1] and t.iloc[1]["entry_i"] == 31 and t.iloc[1]["entry_post_i"] == 30
    assert res.info["exits_preempted_by_stop"] == 1 and res.info["entry_fallbacks"] == 1
    assert_accounting(res)


@pytest.mark.parametrize("stop_mode,minute31", [(0, (40, 40, 39, 40)),       # the entry rests to the end
                                                (1, (100, 100.5, 39, 40))])  # the entry is touched in minute 31
def test_no_entry_opens_on_a_bankrupt_account(stop_mode, minute31):
    # a 3x long is stopped far below its stop in minute 31 while the reversal's sell rests: cash goes negative,
    # the run stops as the engine's does, and the resting entry is cancelled instead of opening
    paths = (flat(100, 15) + [(100, 100, 99.5, 100)] + flat(100, 14) + [(100, 100, 99.95, 100), minute31]
             + flat(40, 13))
    sg = signals(3, [(0, 1, 95.0), (1, -1, 105.0)], size=np.array([3.0, 0.5, np.nan]))
    res = mrun(market(paths), sg, 5, stop_mode=stop_mode)
    assert len(res.trades) == 1
    t = res.trades.iloc[0]
    assert t["reason"] == "stop" and t["exit_i"] == 31 and 10_000 + t["net"] < 0
    assert res.info["entries_missed"] == 1 and list(res.info["missed_entry_tf"]) == [1]
    assert_accounting(res)


# ---------------------------------------------------------------- accounting on random paths

@pytest.mark.parametrize("T", [1, 5, 14, 30])
@pytest.mark.parametrize("fallback", [True, False])
@pytest.mark.parametrize("model", list(me.MODELS))
def test_order_accounting_and_prices_on_random_paths(model, fallback, T):
    mk, sg, fx, slip = random_case(T + 7 * fallback)
    res = me.run(mk, sg, C, R, latency_min=1, policy=me.MakerPolicy(T, model, fallback),
                 tick=np.full(len(mk.ts), TICK), slippage=slip, force_exit=fx, force_penalty=0.003)
    t, info, n = res.trades, res.info, len(mk.ts)
    assert info["orders_posted"] > 20
    assert_accounting(res)
    assert len(t) == info["entry_maker_fills"] + info["entry_fallbacks"]
    if fallback:
        assert info["entries_missed"] == 0 and info["exits_cancelled"] <= 1
    if T == 30:
        assert info["forced_fallbacks"] > 0
    d = t["dir"].to_numpy()
    post, ei = t["entry_post_i"].to_numpy(), t["entry_i"].to_numpy()
    ref = mk.o[post] * (1.0 + slip[post] * d)
    assert (t["entry_ref_px"].to_numpy() == ref).all()
    mkr = t["maker"].to_numpy() == 1
    want = {"strict": ref, "central": mk.o[post], "optimistic": mk.o[post] - d * TICK}[model]
    assert (t["entry_px"].to_numpy()[mkr] == want[mkr]).all()
    assert ((ei >= post) & (mk.ts[ei] < mk.ts[post] + T * 60_000))[mkr].all()
    # taker entries fall back at an open, or cross at the close of the last minute or of a forced exit
    forced = fx[ei] & (t["reason"] == "delist").to_numpy() & (t["exit_i"].to_numpy() == ei)
    at_open = t["entry_px"].to_numpy() == mk.o[ei] * (1.0 + slip[ei] * d)
    at_close = t["entry_px"].to_numpy() == mk.c[ei] * (1.0 + slip[ei] * d)
    assert at_open[~mkr & (ei < n - 1) & ~forced].all()
    assert (at_open | at_close)[~mkr & forced].all()
    assert info["entry_maker_fills"] == mkr.sum() and info["exit_maker_fills"] == t["exit_maker"].sum()
    decision = t["reason"].isin(["exit_signal", "time", "reverse"]).to_numpy()
    xpost, xi = t["exit_post_i"].to_numpy(), t["exit_i"].to_numpy()
    assert (xpost[~decision] == -1).all() and t["exit_ref_px"][~decision].isna().all()
    xm = t["exit_maker"].to_numpy() == 1
    assert not (xm & ~decision).any()
    xref = mk.o[xpost] * (1.0 - slip[xpost] * d)
    assert (t["exit_ref_px"].to_numpy()[decision] == xref[decision]).all()
    xwant = {"strict": xref, "central": mk.o[xpost], "optimistic": mk.o[xpost] + d * TICK}[model]
    assert (t["exit_px"].to_numpy()[xm] == xwant[xm]).all()
    xtk = decision & ~xm
    assert (t["exit_px"].to_numpy()[xtk] == (mk.o[xi] * (1.0 - slip[xi] * d))[xtk]).all()
    assert info["exit_fallbacks"] == xtk.sum()
    fee = np.where(mkr, C.maker * t["qty"] * t["entry_px"], C.taker * t["qty"] * t["entry_px"]) + \
        np.where(xm, C.maker, C.taker) * t["qty"] * t["exit_px"]
    np.testing.assert_allclose(t["fees"], fee, rtol=1e-12)
    assert res.equity.iloc[-1] == pytest.approx(10_000 + t["net"].sum(), rel=1e-12)


# ---------------------------------------------------------------- inputs

def test_bad_inputs_are_rejected():
    mk, sg, _ = case_multi()
    tick = np.full(len(mk.ts), TICK)
    with pytest.raises(ValueError):
        me.run(mk, signals(9, [(0, 1, 95.0)], limit=np.full(9, np.nan)), C, R)
    with pytest.raises(ValueError):
        me.run(mk, sg, C, R, policy=me.MakerPolicy(5, "pessimistic"), tick=tick)
    with pytest.raises(ValueError):
        me.run(mk, sg, C, R, policy=me.MakerPolicy(-1), tick=tick)
    with pytest.raises(ValueError):
        me.run(mk, sg, C, R, policy=me.MakerPolicy(5))          # no tick
    with pytest.raises(ValueError):
        me.run(mk, sg, C, R, policy=me.MakerPolicy(5), tick=np.full(3, TICK))
    bad = tick.copy()
    bad[15] = np.nan                                             # the first post minute
    with pytest.raises(ValueError):
        me.run(mk, sg, C, R, policy=me.MakerPolicy(5), tick=bad)
    # the tick is only needed where an order is posted
    only = np.full(len(mk.ts), np.nan)
    only[[15, 45, 75, 90, 105, 120]] = TICK
    assert len(me.run(mk, sg, C, R, policy=me.MakerPolicy(5), tick=only).trades) == 5


def test_tick_array_maps_months():
    t0 = int(pd.Timestamp("2022-01-31 23:58", tz="UTC").value // 10**6)
    mk = engine.Market(make_m1(flat(100, 4), t0=t0), None, synthetic_funding=0.0)
    np.testing.assert_array_equal(me.tick_array(mk, {"2022-01": 0.1, "2022-02": 0.01}), [0.1, 0.1, 0.01, 0.01])
    np.testing.assert_array_equal(me.tick_array(mk, {"2022-01": 0.1}), [0.1, 0.1, np.nan, np.nan])


# ---------------------------------------------------------------- single legs

LEG_ROWS = [(100, 100, 100, 100),
            (100, 100.2, 100, 100.1),      # leg A (buy, m0 = 1): P = 99.9 not reached
            (100.1, 100.1, 99.9, 100),     # touched: the optimistic fill
            (100, 100, 99.85, 99.9),       # traded through: strict / central fill
            (99.9, 100.3, 99.9, 100.2),
            (100, 100, 99.5, 99.5),        # leg B (sell, m0 = 5): P = 100.1 never reached
            (99.5, 99.5, 99, 99),
            (99, 99, 98.5, 98.5),          # leg D (buy, m0 = 7) fills here, but its markout window is cut
            (98.5, 98.6, 98, 98.2),        # leg B's expiry: falls back at 98.5
            (98.2, 98.3, 98.2, 98.2),
            (98.2, 98.2, 98.2, 98.2),      # leg C (buy, m0 = 10): expiry beyond the data
            (98.2, 98.2, 98.2, 98.2)]


def legs(model, fallback=True, T=3, m0=(1, 5, 10, 7), side=(1, -1, 1, 1)):
    a = np.asarray(LEG_ROWS, np.float64)
    ts = T0 + 60_000 * np.arange(len(a), dtype=np.int64)
    return me.simulate_legs(a[:, 0], a[:, 1], a[:, 2], a[:, 3], ts, np.array(m0), np.array(side),
                            np.full(len(a), TICK), np.full(len(a), 0.0001), T, model, fallback, markouts=(1, 2))


@pytest.mark.parametrize("model,fill_i,px", [("strict", 3, 100 * (1 + 0.0001)), ("central", 3, 100.0),
                                             ("optimistic", 2, 99.9)])
def test_simulate_legs_maker_fill(model, fill_i, px):
    out = legs(model)
    ref = 100 * (1 + 0.0001)
    c = np.asarray(LEG_ROWS)[:, 3]
    assert out["filled"][0] and out["maker"][0] and not out["fell_back"][0] and out["fill_i"][0] == fill_i
    assert out["exp_i"][0] == 4 and out["wait_min"][0] == fill_i - 1
    assert out["ref_px"][0] == pytest.approx(ref, rel=1e-15) and out["px"][0] == pytest.approx(px, rel=1e-15)
    assert out["saving_bp"][0] == pytest.approx(((ref - px) / ref + 0.0003) * 1e4, rel=1e-12)
    assert out["markout_bp_1"][0] == pytest.approx((c[fill_i + 1] - px) / px * 1e4, rel=1e-12)
    assert out["markout_bp_2"][0] == pytest.approx((c[fill_i + 2] - px) / px * 1e4, rel=1e-12)
    assert out["complete"][0]
    if model == "strict":
        assert out["saving_bp"][0] == pytest.approx(3.0, abs=1e-9)


def test_simulate_legs_fallback_cancel_and_incomplete():
    out = legs("strict")
    ref_b, fpx = 100 * (1 - 0.0001), 98.5 * (1 - 0.0001)
    assert not out["filled"][1] and out["fell_back"][1] and out["fill_i"][1] == -1 and out["exp_i"][1] == 8
    assert out["px"][1] == pytest.approx(fpx, rel=1e-15) and out["wait_min"][1] == 3
    assert out["saving_bp"][1] == pytest.approx(-(ref_b - fpx) / ref_b * 1e4, rel=1e-12)   # paid the drift
    assert np.isnan(out["markout_bp_1"][1]) and out["complete"][1]
    # expiry beyond the data: no fallback price, incomplete
    assert out["exp_i"][2] == 12 and not out["filled"][2] and not out["fell_back"][2]
    assert np.isnan(out["px"][2]) and np.isnan(out["saving_bp"][2]) and np.isnan(out["wait_min"][2])
    assert not out["complete"][2]
    # filled, but exp + max(markouts) = 12 lies beyond the last row
    assert out["filled"][3] and out["fill_i"][3] == 7 and out["exp_i"][3] == 10 and not out["complete"][3]
    # post-and-cancel
    nf = legs("strict", fallback=False)
    assert not nf["fell_back"][1] and np.isnan(nf["px"][1]) and np.isnan(nf["saving_bp"][1])
    assert nf["wait_min"][1] == 3 and nf["filled"][0]


def test_simulate_legs_t0_is_the_taker_leg():
    out = legs("optimistic", T=0)
    assert not out["filled"][:2].any() and out["fell_back"][:2].all()
    np.testing.assert_array_equal(out["px"][:2], out["ref_px"][:2])
    np.testing.assert_array_equal(out["saving_bp"][:2], [0.0, 0.0])


def test_simulate_legs_expiry_is_by_time_across_a_gap():
    ts = T0 + 60_000 * np.array([0, 1, 2, 10, 11, 12], np.int64)
    p = np.full(6, 100.0)
    out = me.simulate_legs(p, p, p, p, ts, np.array([1]), np.array([1]), 0.1, 0.0001, 3, "strict", True,
                           markouts=(1,))
    assert out["exp_i"][0] == 3 and out["wait_min"][0] == 9 and out["fell_back"][0]
    assert not out["complete"][0]                       # the order rested across the gap


def test_simulate_legs_markouts_are_by_time_and_complete_excludes_gaps():
    # rows at minutes 0, 1, 2, 3, 10, 11, 12, 13; a buy posted at minute 1 is traded through in minute 2
    mins = np.array([0, 1, 2, 3, 10, 11, 12, 13], np.int64)
    ts = T0 + 60_000 * mins
    o = np.full(8, 100.0)
    lo = np.where(mins == 2, 99.8, 100.0)
    c = 100.0 + mins / 100
    out = me.simulate_legs(o, o, lo, c, ts, np.array([1]), np.array([1]), 0.1, 0.0, 2, "central", True,
                           markouts=(1, 2, 8))
    assert out["filled"][0] and out["fill_i"][0] == 2
    assert out["markout_bp_1"][0] == pytest.approx((c[3] - 100) / 100 * 1e4, rel=1e-12)   # minute 3
    assert np.isnan(out["markout_bp_2"][0])                                               # minute 4 is missing
    assert out["markout_bp_8"][0] == pytest.approx((c[4] - 100) / 100 * 1e4, rel=1e-12)   # minute 10
    assert not out["complete"][0]                       # minutes 1 .. 1 + 2 + 8 cross the gap
    # a leg posted after the gap at minute 10 (row 4), with markouts up to 2: minutes 10 .. 13 are all there
    short = me.simulate_legs(o, o, lo, c, ts, np.array([4]), np.array([1]), 0.1, 0.0, 1, "central", True,
                             markouts=(1, 2))
    assert short["complete"][0] and short["fell_back"][0]
    assert not me.simulate_legs(o, o, lo, c, ts, np.array([4]), np.array([1]), 0.1, 0.0, 1, "central", True,
                                markouts=(1, 3))["complete"][0]           # minute 14 is past the data


def test_simulate_legs_rejects_bad_legs():
    for kw in ({"m0": (12,), "side": (1,)}, {"m0": (1,), "side": (0,)}):
        with pytest.raises(ValueError):
            legs("strict", **kw)
    with pytest.raises(ValueError):
        legs("best")
    a = np.asarray(LEG_ROWS, np.float64)
    ts = T0 + 60_000 * np.arange(len(a), dtype=np.int64)
    with pytest.raises(ValueError):
        me.simulate_legs(a[:, 0], a[:, 1], a[:, 2], a[:, 3], ts, np.array([1]), np.array([1]), 0.0, 0.0001, 3,
                         "strict", True)


# ---------------------------------------------------------------- real dev data (F1)

@needs_data
def test_f1_eth_dev_parity_and_a_strict_maker_run():
    from btcq.research.portfolio import sleeve_risk
    from btcq.research.round3 import LATENCY_MIN, _costs
    from btcq.strategies.r3_trend import S1Frozen
    mk = engine.Market.load("dev", "ETHUSDT", with_metrics=False)
    assert mk.ts[-1] < HOLDOUT_START.value // 10**6
    sg = S1Frozen().compute(mk, n=40, k=3.5, exit="channel")
    kw = {"start": PERP_START, "latency_min": LATENCY_MIN}
    costs, risk = _costs("ETHUSDT"), sleeve_risk(5)
    a = engine.run(mk, sg, costs, risk, **kw)
    b = me.run(mk, sg, costs, risk, **kw)
    pd.testing.assert_frame_equal(b.trades[BASE], a.trades, check_exact=True)
    pd.testing.assert_series_equal(b.equity, a.equity, check_exact=True)
    assert len(a.trades) > 50
    s = me.run(mk, sg, costs, risk, policy=me.MakerPolicy(15), tick=np.full(len(mk.ts), 0.01), **kw)
    t = s.trades
    assert s.info["funding_in_window"] == 0 and s.info["entries_missed"] == 0
    assert_accounting(s)
    mkr = t["maker"] == 1
    assert mkr.any() and (t.loc[mkr, "entry_px"] == t.loc[mkr, "entry_ref_px"]).all()   # strict: fee only
    xm = t["exit_maker"] == 1
    assert xm.any() and (t.loc[xm, "exit_px"] == t.loc[xm, "exit_ref_px"]).all()
    assert (t["exit_time"] < HOLDOUT_START).all()
