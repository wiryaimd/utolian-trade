"""X1 tick queue simulator: hand-built bookTicker/aggTrades sequences, the sell mirror, validation, and a
random cross-check against a literal pure-Python transcription of the rules."""
import numpy as np
import pandas as pd
import pytest

from btcq.backtest import queue_sim as QS

T0 = int(pd.Timestamp("2023-06-01", tz="UTC").value // 10**6)
FT, FM = 0.0005, 0.0002


def book(rows) -> QS.Book:
    """rows: (dt_ms, bid, bid_qty, ask, ask_qty)."""
    a = np.asarray(rows, np.float64).reshape(-1, 5)
    return QS.Book(T0 + a[:, 0].astype(np.int64), a[:, 1], a[:, 2], a[:, 3], a[:, 4])


def tape(rows) -> QS.Tape:
    """rows: (dt_ms, px, qty, buyer_maker); buyer_maker True = seller-initiated."""
    if not rows:
        return QS.Tape(np.zeros(0, np.int64), np.zeros(0), np.zeros(0), np.zeros(0, np.bool_))
    return QS.Tape(T0 + np.array([r[0] for r in rows], np.int64), np.array([r[1] for r in rows], np.float64),
                   np.array([r[2] for r in rows], np.float64), np.array([r[3] for r in rows], np.bool_))


def leg(b, tp, dt_live, side, qty, T, queue="conservative", **kw) -> pd.Series:
    df = QS.simulate_legs(b, tp, T0 + dt_live, side, qty, T, queue=queue, **kw)
    assert len(df) == 1
    return df.iloc[0]


def saving(side, taker, avg, share):
    return (side * (taker - avg) / taker + share * (FT - FM)) * 1e4


# ---------------------------------------------------------------- hand-built sequences

def test_join_and_depletion_to_a_fill():
    b = book([(0, 100.0, 5, 100.1, 3)])
    tp = tape([(20, 100.0, 2, True),        # queue ahead 5 -> 3
               (25, 100.1, 4, False),       # buyer-initiated: never touches a resting bid
               (28, 100.1, 1, True),        # seller-initiated above P: ignored
               (30, 100.0, 2, True),        # 3 -> 1
               (40, 100.0, 3, True),        # 1 ahead, the excess 2 fills our 1
               (50, 99.0, 9, True)])        # after the fill: not processed
    r = leg(b, tp, 10, 1, 1.0, 1000)
    assert r["status"] == "ok"
    assert (r["post_px"], r["taker_px"], r["q_ahead0"]) == (100.0, 100.1, 5.0)
    assert r["filled_qty"] == 1.0 and r["maker_share"] == 1.0
    assert r["fill_ts"] == T0 + 40 and r["first_fill_ts"] == T0 + 40
    assert r["avg_maker_px"] == 100.0 and np.isnan(r["fallback_px"]) and r["avg_px"] == 100.0
    assert r["saving_bp"] == pytest.approx(saving(1, 100.1, 100.0, 1.0))
    assert r["saving_bp"] == pytest.approx(0.1 / 100.1 * 1e4 + 3.0)
    assert not r["trade_through"] and not r["level_vanished"]
    assert r["wait_ms"] == 30 and r["book_age_ms"] == 10
    # one unit short of the queue: no fill, everything crosses at the ask
    r2 = leg(b, tape([(20, 100.0, 4.9, True)]), 10, 1, 1.0, 1000)
    assert r2["filled_qty"] == 0.0 and np.isnan(r2["first_fill_ts"]) and np.isnan(r2["avg_maker_px"])
    assert r2["fallback_px"] == 100.1 and r2["avg_px"] == 100.1 and r2["saving_bp"] == 0.0
    assert r2["wait_ms"] == 1000


def test_trade_through_fills_the_whole_remainder():
    b = book([(0, 100.0, 5, 100.1, 3)])
    tp = tape([(15, 100.0, 1, True), (20, 99.9, 0.1, True)])   # a tiny print below P still fills all 3
    r = leg(b, tp, 10, 1, 3.0, 1000)
    assert r["trade_through"]
    assert r["filled_qty"] == 3.0 and r["maker_share"] == 1.0
    assert r["fill_ts"] == T0 + 20 and r["first_fill_ts"] == T0 + 20
    assert r["avg_px"] == 100.0 and np.isnan(r["fallback_px"])


def test_partial_fill_then_fallback_at_the_ask_at_t_exp():
    b = book([(0, 100.0, 1, 100.1, 3),
              (500, 100.0, 2, 100.2, 3),          # queue ahead stays 0 (min(0, 2))
              (1010, 100.0, 2, 100.3, 3),         # exactly t_exp: not an event, but the fallback's ask
              (1011, 100.0, 2, 100.5, 3)])        # after t_exp: unused
    tp = tape([(100, 100.0, 1.5, True),           # 1 ahead, 0.5 fills us
               (1010, 99.0, 5, True)])            # a trade-through at t_exp is too late
    r = leg(b, tp, 10, 1, 2.0, 1000, markouts_ms=(50,))
    assert r["filled_qty"] == 0.5 and r["maker_share"] == 0.25
    assert np.isnan(r["fill_ts"]) and r["first_fill_ts"] == T0 + 100
    assert r["avg_maker_px"] == 100.0 and r["fallback_px"] == 100.3
    assert r["avg_px"] == pytest.approx(0.25 * 100.0 + 0.75 * 100.3)
    assert r["saving_bp"] == pytest.approx(((100.1 - 100.225) / 100.1 + 0.25 * (FT - FM)) * 1e4)
    assert not r["trade_through"]
    assert r["wait_ms"] == 1000
    # the markout of a partial fill runs from the first fill: mid at 150 ms is row 0's
    assert r["markout_bp_50"] == pytest.approx((100.05 - 100.0) / 100.0 * 1e4)
    assert not r["complete"]                       # t_exp + 50 ms = 1060 is past the book's last row (1011)
    assert leg(b, tp, 10, 1, 2.0, 1000, markouts_ms=(1,))["complete"]


def test_level_vanishes_then_a_seller_fills_us():
    b = book([(0, 100.0, 5, 100.1, 3),
              (20, 99.9, 4, 100.1, 3),            # bid below P: we are the unseen best bid
              (35, 100.0, 7, 100.1, 3)])          # back at P: the 7 joined behind us
    tp = tape([(30, 100.0, 0.4, True), (45, 100.0, 0.6, True)])
    for q in QS.QUEUE:
        r = leg(b, tp, 10, 1, 1.0, 1000, queue=q)
        assert r["level_vanished"] and not r["trade_through"]
        assert r["first_fill_ts"] == T0 + 30 and r["fill_ts"] == T0 + 45
        assert r["filled_qty"] == 1.0


def test_bid_above_P_leaves_the_level_unseen_and_skips_prorata():
    b = book([(0, 100.0, 10, 100.2, 3),
              (15, 100.0, 20, 100.2, 3),          # 10 joined behind us
              (20, 100.1, 2, 100.2, 3),           # someone improves: our level is unseen
              (30, 100.0, 16, 100.2, 3)])         # back at P with 16: the drop was not observed, no pro-rata credit
    tp = tape([(40, 100.0, 9, True)])
    for q in QS.QUEUE:
        r = leg(b, tp, 10, 1, 1.0, 1000, queue=q)
        assert not r["level_vanished"]
        assert r["filled_qty"] == 0.0                     # 10 ahead either way (prorata would have said 8)


def test_same_ms_tape_before_book():
    b = book([(0, 100.0, 5, 100.1, 3), (20, 100.0, 2, 100.1, 3)])
    tp = tape([(20, 100.0, 3, True), (30, 100.0, 2.5, True)])
    # trade first: 5 -> 2, the book row shows 2 (explained, prorata takes nothing), then 2.5 fills 0.5.
    # Book first would cap the queue at 2 and the 3 at 20 ms would fill the whole order at 20 ms.
    for q in QS.QUEUE:
        r = leg(b, tp, 10, 1, 1.0, 1000, queue=q)
        assert r["filled_qty"] == 0.5 and r["first_fill_ts"] == T0 + 30


def test_file_order_within_a_stream():
    # two rows at the join ms: the later one is the snapshot
    b = book([(0, 100.0, 5, 100.1, 3), (10, 100.0, 8, 100.1, 3), (10, 100.0, 6, 100.1, 3)])
    assert leg(b, tape([]), 10, 1, 1.0, 100)["q_ahead0"] == 6.0
    # two trades in one ms: the at-P print completes the order before the trade-through is read
    tp = tape([(20, 100.0, 7, True), (20, 99.9, 1, True)])
    r = leg(b, tp, 10, 1, 1.0, 100)
    assert r["fill_ts"] == T0 + 20 and not r["trade_through"]
    tp2 = tape([(20, 99.9, 1, True), (20, 100.0, 7, True)])
    assert leg(b, tp2, 10, 1, 1.0, 100)["trade_through"]


def test_events_strictly_inside_the_window():
    b = book([(0, 100.0, 5, 100.1, 3)])
    tp = tape([(10, 99.0, 1, True)])                      # a trade-through at t_live is before the order is live
    r = leg(b, tp, 10, 1, 1.0, 1000)
    assert r["filled_qty"] == 0.0 and not r["trade_through"]
    r0 = leg(b, tape([(10, 99.0, 1, True), (11, 99.0, 1, True)]), 10, 1, 1.0, 0)   # T = 0: pure taker
    assert r0["filled_qty"] == 0.0 and r0["fallback_px"] == r0["taker_px"] == 100.1 and r0["saving_bp"] == 0.0


def test_prorata_vs_conservative_on_an_unexplained_drop():
    b = book([(0, 100.0, 10, 100.1, 3),
              (20, 100.0, 20, 100.1, 3),          # 10 joined behind us
              (30, 100.0, 16, 100.1, 3)])         # 4 cancelled, no trades: prorata puts 10/20 of them ahead
    tp = tape([(40, 100.0, 9, True)])
    c = leg(b, tp, 10, 1, 1.0, 1000, queue="conservative")
    p = leg(b, tp, 10, 1, 1.0, 1000, queue="prorata")
    assert c["filled_qty"] == 0.0                         # 10 ahead, 9 traded
    assert p["filled_qty"] == 1.0 and p["fill_ts"] == T0 + 40     # 10 - 4 * 10 / 20 = 8 ahead
    assert p["saving_bp"] > c["saving_bp"]


def test_size_drop_explained_by_trades_is_not_a_cancellation():
    b = book([(0, 100.0, 10, 100.1, 3), (20, 100.0, 20, 100.1, 3), (40, 100.0, 16, 100.1, 3)])
    tp = tape([(30, 100.0, 4, True), (50, 100.0, 6.5, True)])
    c = QS.simulate_legs(b, tp, T0 + 10, 1, 1.0, 1000, queue="conservative")
    p = QS.simulate_legs(b, tp, T0 + 10, 1, 1.0, 1000, queue="prorata")
    pd.testing.assert_frame_equal(c, p)
    assert c["filled_qty"].iloc[0] == 0.5                 # 10 -> 6 ahead, the 6.5 print fills 0.5
    # half explained: 2 traded, 2 cancelled -> prorata moves us up by 2 * 8 / 18
    tp2 = tape([(30, 100.0, 2, True), (50, 100.0, 7.5, True)])
    c2 = leg(b, tp2, 10, 1, 1.0, 1000, queue="conservative")
    p2 = leg(b, tp2, 10, 1, 1.0, 1000, queue="prorata")
    assert c2["filled_qty"] == 0.0
    assert p2["filled_qty"] == pytest.approx(7.5 - (8.0 - 2.0 * 8.0 / 18.0))


def test_sell_side_hand_checked():
    b = book([(0, 100.0, 3, 100.1, 5), (1010, 99.8, 1, 99.9, 1)])
    tp = tape([(20, 100.1, 2, True),               # seller-initiated: never touches a resting ask
               (30, 100.1, 4, False),              # buyers lift the ask: 5 -> 1
               (40, 100.1, 1.5, False)])           # 1 ahead, 0.5 fills us
    r = leg(b, tp, 10, -1, 2.0, 1000)
    assert (r["post_px"], r["taker_px"], r["q_ahead0"]) == (100.1, 100.0, 5.0)
    assert r["filled_qty"] == 0.5 and r["first_fill_ts"] == T0 + 40
    assert r["fallback_px"] == 99.8                # the bid at t_exp
    avg = 0.25 * 100.1 + 0.75 * 99.8
    assert r["avg_px"] == pytest.approx(avg)
    assert r["saving_bp"] == pytest.approx(saving(-1, 100.0, avg, 0.25))
    full = leg(b, tape([(30, 100.2, 0.1, False)]), 10, -1, 2.0, 1000)    # a buyer prints above P
    assert full["trade_through"] and full["filled_qty"] == 2.0
    assert full["saving_bp"] == pytest.approx((0.1 / 100.0 + FT - FM) * 1e4)
    vanish = leg(book([(0, 100.0, 3, 100.1, 5), (20, 100.0, 3, 100.2, 2)]), tape([(30, 100.1, 2, False)]),
                 10, -1, 2.0, 1000)
    assert vanish["level_vanished"] and vanish["filled_qty"] == 2.0


def mirror_book(b: QS.Book, c: float = 200.0) -> QS.Book:
    return QS.Book(b.ts, c - b.ask_px, b.ask_qty, c - b.bid_px, b.bid_qty)


def mirror_tape(t: QS.Tape, c: float = 200.0) -> QS.Tape:
    return QS.Tape(t.ts, c - t.px, t.qty, ~t.buyer_maker)


SAME = ["status", "q_ahead0", "filled_qty", "maker_share", "fill_ts", "first_fill_ts", "trade_through",
        "level_vanished", "book_crossed", "wait_ms", "book_age_ms", "complete"]


@pytest.mark.parametrize("queue", list(QS.QUEUE))
def test_sell_mirrors_buy(queue):
    rng = np.random.default_rng(7)
    b, tp = random_streams(rng)
    t_live = T0 + rng.integers(-50, 2000, 300)
    qty = 0.5 * rng.integers(1, 30, 300)
    T = rng.integers(0, 600, 300)
    buy = QS.simulate_legs(b, tp, t_live, 1, qty, T, queue=queue, markouts_ms=(25, 400))
    sell = QS.simulate_legs(mirror_book(b), mirror_tape(tp), t_live, -1, qty, T, queue=queue, markouts_ms=(25, 400))
    pd.testing.assert_frame_equal(buy[SAME], sell[SAME])
    for col in ("post_px", "taker_px", "avg_maker_px", "fallback_px", "avg_px"):
        np.testing.assert_allclose(200.0 - buy[col], sell[col], rtol=1e-12, equal_nan=True, err_msg=col)
    assert buy["filled_qty"].gt(0).any() and (buy["filled_qty"] < qty).any()


def test_prorata_never_fills_less_than_conservative():
    rng = np.random.default_rng(21)
    b, tp = random_streams(rng)
    t_live = T0 + rng.integers(0, 2000, 400)
    side = rng.choice([-1, 1], 400)
    qty = 0.5 * rng.integers(1, 30, 400)
    T = rng.integers(0, 300, 400)
    c = QS.simulate_legs(b, tp, t_live, side, qty, T, queue="conservative")
    p = QS.simulate_legs(b, tp, t_live, side, qty, T, queue="prorata")
    assert (p["filled_qty"] >= c["filled_qty"] - 1e-12).all()
    assert (p["filled_qty"] > c["filled_qty"]).any()


def test_no_book():
    b = book([(100, 100.0, 5, 100.1, 3)])
    df = QS.simulate_legs(b, tape([(150, 99.0, 1, True)]), [T0 + 50, T0 + 120], [1, 1], 1.0, 1000)
    nb, ok = df.iloc[0], df.iloc[1]
    assert nb["status"] == "no_book" and ok["status"] == "ok"
    flags = ("trade_through", "level_vanished", "book_crossed", "complete")
    num = [c for c in df.columns if c not in ("status",) + flags]
    assert nb[num].isna().all()
    assert not nb[list(flags)].any()
    assert ok["filled_qty"] == 1.0 and ok["trade_through"]
    empty = QS.simulate_legs(book(np.zeros((0, 5))), tape([]), [T0, T0 + 5], [1, -1], 1.0, 10)
    assert (empty["status"] == "no_book").all() and empty["saving_bp"].isna().all()


def test_markouts():
    b = book([(0, 100.0, 1, 100.1, 3), (150, 100.2, 1, 100.4, 3), (2000, 99.0, 1, 99.2, 3)])
    tp = tape([(20, 99.9, 1, True)])                       # trade-through: filled at 20 ms
    r = leg(b, tp, 10, 1, 1.0, 1000, markouts_ms=(100, 1000, 1980, 5000))
    assert r["markout_bp_100"] == pytest.approx((100.05 - 100.0) / 100.0 * 1e4)   # row 0 at 120 ms
    assert r["markout_bp_1000"] == pytest.approx((100.3 - 100.0) / 100.0 * 1e4)   # row 1 at 1020 ms
    assert r["markout_bp_1980"] == pytest.approx((99.1 - 100.0) / 100.0 * 1e4)    # row 2 exactly at 2000 ms
    assert np.isnan(r["markout_bp_5000"])                  # past the end of the book data
    s = leg(b, tape([(20, 100.2, 1, False)]), 10, -1, 1.0, 1000, markouts_ms=(1000,))
    assert s["markout_bp_1000"] == pytest.approx(-(100.3 - 100.1) / 100.1 * 1e4)
    none = leg(b, tape([]), 10, 1, 1.0, 1000, markouts_ms=(100,))
    assert np.isnan(none["markout_bp_100"])
    cols = QS.simulate_legs(b, tp, T0 + 10, 1, 1.0, 1000).columns
    assert list(cols) == QS.LEG_COLS + ["markout_bp_60000", "markout_bp_300000", "markout_bp_900000",
                                        "markout_bp_3600000", "wait_ms", "book_age_ms", "complete"]


def test_dust_remainder_counts_as_filled():
    assert (1.0 - 0.7) - 0.3 > 0.0                         # the float remainder this test exercises
    b = book([(0, 100.0, 0.0001, 100.1, 3), (20, 99.9, 1, 100.1, 3)])   # queue ahead 0 after 20 ms
    r = leg(b, tape([(30, 100.0, 0.7, True), (40, 100.0, 0.3, True)]), 10, 1, 1.0, 1000)
    assert r["fill_ts"] == T0 + 40 and r["filled_qty"] == 1.0 and r["maker_share"] == 1.0
    assert np.isnan(r["fallback_px"])


def test_complete_flag_and_book_age():
    b = book([(0, 100.0, 1, 100.1, 3), (500, 100.0, 1, 100.1, 3), (900, 100.0, 1, 100.1, 3)])
    tp = tape([(600, 100.1, 1, False)])
    # complete: the book reaches t_exp + max(markouts) and the tape reaches t_exp
    df = QS.simulate_legs(b, tp, T0 + 100, 1, 1.0, [400, 401], markouts_ms=(50, 400))
    assert list(df["complete"]) == [True, False]           # 500 + 400 = 900 is the last row; 901 is past it
    df = QS.simulate_legs(b, tp, T0 + np.array([100, 100, 550]), 1, 1.0, [500, 700, 300], markouts_ms=(50,))
    assert list(df["complete"]) == [True, False, False]    # t_exp 600: yes; 800: the tape ends at 600
    assert list(df["book_age_ms"]) == [100.0, 100.0, 50.0]
    none = QS.simulate_legs(b, tape([(900, 100.1, 1, False)]), T0 + 100, 1, 1.0, [800, 801], markouts_ms=())
    assert not any(c.startswith("markout") for c in none.columns) and list(none["complete"]) == [True, False]


def test_complete_covers_the_markout_horizon_like_the_1m_model():
    # maker_engine.simulate_legs: complete = rows m0 .. exp + max(markouts) exist; the tick model matches it
    b = book([(0, 100.0, 1, 100.1, 3), (1500, 100.0, 1, 100.1, 3)])
    tp = tape([(900, 99.9, 1, True), (1500, 100.0, 1, True)])
    r = leg(b, tp, 10, 1, 1.0, 1000, markouts_ms=(1000,))
    assert r["fill_ts"] == T0 + 900 and np.isnan(r["markout_bp_1000"])     # 1900 is past the book's end
    assert not r["complete"]                                               # both streams do reach t_exp = 1010
    assert leg(b, tp, 10, 1, 1.0, 1000, markouts_ms=(490,))["complete"]    # 1010 + 490 = 1500, the last row


def test_float_dust_in_the_queue_is_not_a_fill():
    assert 1.234 - 0.3 < 0.934 and 0.3 - 0.1 < 0.2 and 0.8 - 0.1 > 0.7      # the float residues exercised here
    # a seller hits exactly the displayed 0.934 after 0.3 of 1.234 traded: the queue is gone, nothing is left for us
    b = book([(0, 30000.0, 1.234, 30000.1, 2), (150, 30000.0, 0.934, 30000.1, 2), (400, 30000.0, 0.5, 30000.1, 2)])
    tp = tape([(100, 30000.0, 0.3, True), (200, 30000.0, 0.934, True)])
    for q in QS.QUEUE:
        r = leg(b, tp, 10, 1, 0.002, 1000, queue=q, markouts_ms=(60, 300))
        assert r["filled_qty"] == 0.0 and r["maker_share"] == 0.0
        assert np.isnan(r["first_fill_ts"]) and np.isnan(r["avg_maker_px"]) and np.isnan(r["markout_bp_60"])
        assert r["fallback_px"] == 30000.1 and r["saving_bp"] == 0.0
    b2 = book([(0, 100.0, 0.3, 100.1, 2)])
    r = leg(b2, tape([(20, 100.0, 0.1, True), (30, 100.0, 0.2, True)]), 10, 1, 0.002, 1000, markouts_ms=(60,))
    assert r["filled_qty"] == 0.0 and np.isnan(r["first_fill_ts"]) and np.isnan(r["markout_bp_60"])
    # the next print fills us from its first unit, and a real excess of one lot still fills
    r = leg(b2, tape([(20, 100.0, 0.1, True), (30, 100.0, 0.2, True), (40, 100.0, 0.002, True)]), 10, 1, 0.002, 1000)
    assert r["filled_qty"] == 0.002 and r["first_fill_ts"] == r["fill_ts"] == T0 + 40
    r = leg(b2, tape([(20, 100.0, 0.1, True), (30, 100.0, 0.201, True)]), 10, 1, 0.002, 1000)
    assert r["filled_qty"] == pytest.approx(0.001) and r["first_fill_ts"] == T0 + 30 and np.isnan(r["fill_ts"])
    # a residue on the other side (0.8 - 0.1 > 0.7) is emptied as well, so the next print fills exactly
    r = leg(book([(0, 100.0, 0.8, 100.1, 2)]),
            tape([(20, 100.0, 0.1, True), (30, 100.0, 0.7, True), (40, 100.0, 0.002, True)]), 10, 1, 0.002, 1000)
    assert r["filled_qty"] == 0.002 and r["maker_share"] == 1.0 and r["fill_ts"] == T0 + 40


def test_timestamps_are_shift_invariant_including_negative_ones():
    def run(off):
        b = QS.Book(off + np.array([0, 100]), np.full(2, 100.0), np.ones(2), np.full(2, 100.1), np.ones(2))
        t = QS.Tape(off + np.array([20, 30]), np.full(2, 100.0), np.array([1.5, 1.0]), np.ones(2, np.bool_))
        return QS.simulate_legs(b, t, off + 10, 1, 1.0, 50, markouts_ms=(5,))
    base = run(0)
    assert base["fill_ts"].iloc[0] == 30 and base["first_fill_ts"].iloc[0] == 20 and base["wait_ms"].iloc[0] == 20
    assert np.isnan(base["fallback_px"].iloc[0]) and base["markout_bp_5"].iloc[0] == pytest.approx(5.0)
    for off in (-1000, -10**12, T0):
        r = run(off)
        for c in ("fill_ts", "first_fill_ts"):
            r[c] -= off
        pd.testing.assert_frame_equal(r, base)


def test_book_crossed_flags_an_opposite_side_at_or_through_P():
    # the bid at P is cancelled away and sellers come down by posting asks, without one selling into a bid <= P.
    # The model fills only on prints (as specified), so it falls back below its own post price; the flag reports it.
    b = book([(0, 100.00, 5, 100.01, 2), (100, 99.99, 3, 100.00, 2), (200, 99.98, 3, 99.99, 2),
              (1000, 99.98, 3, 99.99, 2)])
    tp = tape([(300, 99.99, 0.5, False)])
    r = leg(b, tp, 10, 1, 1.0, 1000)
    assert r["book_crossed"] and r["level_vanished"]
    assert r["filled_qty"] == 0.0 and r["fallback_px"] == 99.99 < r["post_px"]
    assert leg(mirror_book(b), mirror_tape(tp), 10, -1, 1.0, 1000)["book_crossed"]
    # an ask reaching P only at the fallback row (exactly t_exp, not an event) is flagged too
    b2 = book([(0, 100.0, 5, 100.01, 2), (20, 99.99, 3, 100.01, 2), (1010, 99.99, 3, 100.0, 2)])
    r2 = leg(b2, tape([]), 10, 1, 1.0, 1000)
    assert r2["book_crossed"] and r2["fallback_px"] == 100.0
    # an open spread, and a full maker fill before the cross, are not flagged
    assert not leg(b2, tape([]), 10, 1, 1.0, 999)["book_crossed"]
    first = leg(b, tape([(50, 99.99, 1, True)]), 10, 1, 1.0, 1000)
    assert first["filled_qty"] == 1.0 and first["trade_through"] and not first["book_crossed"]


# ---------------------------------------------------------------- data in separate windows

H = 3_600_000


def gapped(seller_at=None):
    """1 s rows over [0, 2h) at 100.0 / 100.1 and over [10h, 12h) at 110.0 / 110.1; buyer-initiated prints, plus
    one seller-initiated print at 99.9 at `seller_at` (dt ms)."""
    t1, t2 = np.arange(0, 2 * H, 1000), np.arange(10 * H, 12 * H, 1000)
    ts = np.r_[t1, t2]
    bid = np.where(ts < 5 * H, 100.0, 110.0)
    b = QS.Book(T0 + ts, bid, np.full(len(ts), 5.0), bid + 0.1, np.full(len(ts), 5.0))
    tt = ts[::10] + 500
    px, bm = bid[::10] + 0.1, np.zeros(len(tt), np.bool_)
    if seller_at is not None:
        k = np.searchsorted(tt, seller_at)
        tt, px, bm = np.insert(tt, k, seller_at), np.insert(px, k, 99.9), np.insert(bm, k, True)
    return b, QS.Tape(T0 + tt, px, np.ones(len(tt)), bm), [(T0, T0 + 2 * H), (T0 + 10 * H, T0 + 12 * H)]


def test_windows_flag_legs_that_run_into_or_live_in_a_hole():
    b, tp, win = gapped()
    # A: t_exp = 2h30 lies in the hole: without windows the leg falls back at the 1:59:59 ask and looks complete
    a0 = leg(b, tp, int(1.5 * H), 1, 1.0, H)
    a = leg(b, tp, int(1.5 * H), 1, 1.0, H, windows_ms=win)
    assert a0["complete"] and a0["fallback_px"] == 100.1
    assert a["status"] == "ok" and not a["complete"]
    # B: live inside the hole: the 3 h old snapshot is not a book
    b0 = leg(b, tp, 5 * H, 1, 1.0, 15 * 60_000)
    bb = leg(b, tp, 5 * H, 1, 1.0, 15 * 60_000, windows_ms=win)
    assert b0["status"] == "ok" and b0["book_age_ms"] == 3 * H + 1000 and b0["complete"]
    assert bb["status"] == "no_book" and np.isnan(bb["saving_bp"]) and not bb["complete"]
    # C: a fill at 1:59:50: the 60 min markout would read the stale 1:59:59 mid; inside its window only 5 s fits
    b, tp, win = gapped(seller_at=2 * H - 10_000)
    r = leg(b, tp, 2 * H - 20_000, 1, 1.0, 15_000, markouts_ms=(5_000, 60_000, H), windows_ms=win)
    assert r["fill_ts"] == T0 + 2 * H - 10_000 and r["trade_through"]
    assert r["markout_bp_5000"] == pytest.approx(5.0)
    assert np.isnan(r["markout_bp_60000"]) and np.isnan(r["markout_bp_3600000"]) and not r["complete"]
    assert leg(b, tp, 2 * H - 20_000, 1, 1.0, 15_000, markouts_ms=(H,))["markout_bp_3600000"] == pytest.approx(5.0)
    # a leg wholly inside the second window is complete
    d = leg(b, tp, int(10.5 * H), -1, 1.0, 15 * 60_000, windows_ms=win)
    assert d["complete"] and d["post_px"] == 110.1


def test_windows_join_row_must_lie_in_the_legs_window():
    b, tp, _ = gapped()
    win = [(T0, T0 + 2 * H), (T0 + int(9.5 * H), T0 + 12 * H)]        # declared from 9h30, first row at 10h
    r = QS.simulate_legs(b, tp, T0 + np.array([int(9.75 * H), 10 * H, 10 * H + 500]), 1, 1.0, 1000,
                         markouts_ms=(1,), windows_ms=win)
    assert list(r["status"]) == ["no_book", "ok", "ok"] and list(r["complete"]) == [False, True, True]
    assert list(r["book_age_ms"].iloc[1:]) == [0.0, 500.0]


def test_windows_touching_or_overlapping_join():
    b, tp, _ = gapped()
    # a midnight-style split of the first stretch, given out of order, and an overlap in the second
    win = [(T0 + H, T0 + 2 * H), (T0 + 10 * H, T0 + 11 * H), (T0, T0 + H), (T0 + int(10.5 * H), T0 + 12 * H)]
    t = T0 + np.array([H - 300_000, 11 * H - 300_000])              # both legs span the joins
    kw = {"markouts_ms": (60_000, 300_000)}
    r = QS.simulate_legs(b, tp, t, 1, 1.0, 15 * 60_000, windows_ms=win, **kw)
    assert r["complete"].all() and (r["status"] == "ok").all()
    one = QS.simulate_legs(b, tp, t, 1, 1.0, 15 * 60_000, windows_ms=[(T0, T0 + 2 * H), (T0 + 10 * H, T0 + 12 * H)],
                           **kw)
    pd.testing.assert_frame_equal(r, one)
    assert not QS.simulate_legs(b, tp, t, 1, 1.0, 15 * 60_000, windows_ms=[(T0, T0 + H), (T0 + 10 * H, T0 + 11 * H)],
                                **kw)["complete"].any()


def test_windows_validation():
    b, tp, _ = gapped()
    for w in ([(T0 + 5, T0 + 5)], [(T0 + 6, T0 + 5)], [T0, T0 + 5], [(T0 + 0.5, T0 + 5)], [(T0, T0 + 5, T0 + 9)]):
        with pytest.raises(ValueError):
            QS.simulate_legs(b, tp, T0 + 10, 1, 1.0, 100, windows_ms=w)
    empty = QS.simulate_legs(b, tp, T0 + np.array([10, H]), 1, 1.0, 100, windows_ms=[])
    assert (empty["status"] == "no_book").all() and not empty["complete"].any()


def test_broadcasting_matches_single_legs():
    rng = np.random.default_rng(3)
    b, tp = random_streams(rng)
    t_live = T0 + rng.integers(0, 2000, 40)
    side = rng.choice([-1, 1], 40)
    many = QS.simulate_legs(b, tp, t_live, side, 2.5, 300, queue="prorata")
    one = pd.concat([QS.simulate_legs(b, tp, t, s, 2.5, 300, queue="prorata") for t, s in zip(t_live, side)],
                    ignore_index=True)
    pd.testing.assert_frame_equal(many, one)


def test_validation():
    b = book([(0, 100.0, 5, 100.1, 3), (10, 100.0, 5, 100.1, 3)])
    tp = tape([(5, 100.0, 1, True)])

    def run(bb=b, tt=tp, t=T0 + 1, side=1, qty=1.0, T=100, **kw):
        return QS.simulate_legs(bb, tt, t, side, qty, T, **kw)

    run()
    bad_books = [QS.Book(b.ts[::-1].copy(), b.bid_px, b.bid_qty, b.ask_px, b.ask_qty),
                 QS.Book(b.ts, np.array([100.0, np.nan]), b.bid_qty, b.ask_px, b.ask_qty),
                 QS.Book(b.ts, b.bid_px, np.array([5.0, 0.0]), b.ask_px, b.ask_qty),
                 QS.Book(b.ts, b.bid_px, b.bid_qty, np.array([100.1, -1.0]), b.ask_qty),
                 QS.Book(b.ts, b.bid_px, b.bid_qty, b.ask_px, np.array([3.0]))]
    for bb in bad_books:
        with pytest.raises(ValueError):
            run(bb=bb)
    bad_tapes = [QS.Tape(tp.ts, np.array([np.inf]), tp.qty, tp.buyer_maker),
                 QS.Tape(tp.ts, tp.px, np.array([0.0]), tp.buyer_maker),
                 QS.Tape(np.array([T0 + 5, T0 + 4]), np.array([100.0, 100.0]), np.ones(2), np.ones(2, bool))]
    for tt in bad_tapes:
        with pytest.raises(ValueError):
            run(tt=tt)
    with pytest.raises(ValueError):
        QS.Tape(tp.ts, tp.px, tp.qty, np.array(["true"]))
    with pytest.raises(ValueError):
        QS.Book(np.array([0.5, 1.0]), b.bid_px, b.bid_qty, b.ask_px, b.ask_qty)
    assert QS.Tape(tp.ts, tp.px, tp.qty, np.array([1])).buyer_maker.dtype == np.bool_
    for kw in ({"side": 0}, {"side": 2}, {"qty": 0.0}, {"qty": np.nan}, {"T": -1}, {"t": T0 + 0.5},
               {"queue": "fifo"}, {"markouts_ms": (0,)}, {"markouts_ms": (5, 5)}, {"fee_t": np.nan}):
        with pytest.raises(ValueError):
            run(**kw)


# ---------------------------------------------------------------- random cross-check against the literal rules

def random_streams(rng, n_book=400, n_tape=900, span=2000, tick=0.1):
    bts = np.sort(rng.integers(0, span, n_book))
    lvl = np.cumsum(rng.choice([-1, 0, 0, 0, 1], n_book))
    spread = rng.integers(1, 3, n_book)
    px = lambda k: 100.0 + tick * k  # noqa: E731
    b = QS.Book(T0 + bts, px(lvl), 0.5 * rng.integers(1, 21, n_book), px(lvl + spread),
                0.5 * rng.integers(1, 21, n_book))
    tts = np.sort(rng.integers(0, span, n_tape))
    j = np.clip(np.searchsorted(bts, tts, "right") - 1, 0, None)
    bm = rng.random(n_tape) < 0.5
    # sellers print around the bid, buyers around the ask: mostly at the touch, sometimes a tick through or inside
    off = rng.choice([-1, 0, 1], n_tape, p=[0.03, 0.87, 0.10])
    k = np.where(bm, lvl[j] + off, lvl[j] + spread[j] - off)
    t = QS.Tape(T0 + tts, px(k), 0.5 * rng.integers(1, 9, n_tape), bm)
    return b, t


def ref_leg(b, tp, t_live, side, qty, T, prorata, markouts):
    """The spec's rules, transcribed literally (pure Python, one leg)."""
    nanrow = {"status": "no_book", "trade_through": False, "level_vanished": False, "book_crossed": False,
              "complete": False}
    rows = [i for i in range(len(b.ts)) if b.ts[i] <= t_live]
    if not rows:
        return nanrow
    i0 = rows[-1]
    buy = side > 0
    P = b.bid_px[i0] if buy else b.ask_px[i0]
    taker = b.ask_px[i0] if buy else b.bid_px[i0]
    qa = b.bid_qty[i0] if buy else b.ask_qty[i0]
    q0, obs, traded, observed = qa, qa, 0.0, True
    eps = 1e-9 * P
    t_exp = t_live + T
    ev = sorted([(tp.ts[i], 0, i) for i in range(len(tp.ts)) if t_live < tp.ts[i] < t_exp]
                + [(b.ts[j], 1, j) for j in range(i0 + 1, len(b.ts)) if b.ts[j] < t_exp])
    rem, first, fill, through, vanished, crossed = qty, None, None, False, False, False
    for ts, kind, i in ev:
        if rem == 0:
            break
        if kind == 0:
            if bool(tp.buyer_maker[i]) != buy:
                continue
            px, tq = tp.px[i], tp.qty[i]
            f = 0.0
            if (px < P - eps) if buy else (px > P + eps):
                f = rem
                through = True
            elif abs(px - P) <= eps:
                tol = QS.QTY_REL * max(q0, tq)            # the implementation's queue-dust rule
                if qa <= 0 or tq - qa > tol:
                    f = min(tq - qa, rem)
                    qa = 0.0
                elif tq - qa >= -tol:
                    qa = 0.0
                else:
                    qa -= tq
                traded += tq
            if f > 0:
                rem -= f
                first = ts if first is None else first
                if rem <= QS.DUST * qty:              # the implementation's float-dust rule
                    rem = 0.0
                    fill = ts
        else:
            lv = b.bid_px[i] if buy else b.ask_px[i]
            lq = b.bid_qty[i] if buy else b.ask_qty[i]
            opp = b.ask_px[i] if buy else b.bid_px[i]
            crossed |= (opp <= P + eps) if buy else (opp >= P - eps)
            if (lv < P - eps) if buy else (lv > P + eps):
                qa, observed, vanished = 0.0, False, True
            elif abs(lv - P) <= eps:
                if prorata and observed:
                    e = obs - traded
                    if lq < e and e > 0:
                        qa -= (e - lq) * qa / e
                qa = min(qa, lq)
                obs, traded, observed = lq, 0.0, True
            else:
                observed = False
    filled = qty - rem
    share = filled / qty
    jx = [j for j in range(len(b.ts)) if b.ts[j] <= t_exp][-1]
    fb = (b.ask_px[jx] if buy else b.bid_px[jx]) if rem > 0 else np.nan
    crossed |= rem > 0 and ((fb <= P + eps) if buy else (fb >= P - eps))
    avg = P if rem == 0 else (fb if filled == 0 else (filled * P + rem * fb) / qty)
    out = {"status": "ok", "post_px": P, "taker_px": taker, "q_ahead0": q0, "filled_qty": filled,
           "maker_share": share, "fill_ts": np.nan if fill is None else fill,
           "first_fill_ts": np.nan if first is None else first, "avg_maker_px": P if filled > 0 else np.nan,
           "fallback_px": fb, "avg_px": avg, "saving_bp": (side * (taker - avg) / taker + share * (FT - FM)) * 1e4,
           "trade_through": through, "level_vanished": vanished, "book_crossed": crossed,
           "wait_ms": fill - t_live if fill is not None else T, "book_age_ms": t_live - b.ts[i0],
           "complete": b.ts[-1] >= t_exp + max(markouts) and tp.ts[-1] >= t_exp}
    mref = fill if fill is not None else first
    for k in markouts:
        v = np.nan
        if mref is not None and mref + k <= b.ts[-1]:
            jm = [j for j in range(len(b.ts)) if b.ts[j] <= mref + k][-1]
            v = side * ((b.bid_px[jm] + b.ask_px[jm]) / 2 - P) / P * 1e4
        out[f"markout_bp_{k}"] = v
    return out


@pytest.mark.parametrize("seed", [11, 12, 13])
@pytest.mark.parametrize("queue", list(QS.QUEUE))
def test_random_sequences_match_the_literal_rules(seed, queue):
    rng = np.random.default_rng(seed)
    b, tp = random_streams(rng)
    n = 150
    t_live = T0 + np.r_[-5, rng.integers(0, 2000, n - 1)]
    side = rng.choice([-1, 1], n)
    qty = 0.5 * rng.integers(1, 30, n)
    T = rng.integers(0, 600, n)
    mks = (30, 250)
    got = QS.simulate_legs(b, tp, t_live, side, qty, T, queue=queue, markouts_ms=mks)
    want = pd.DataFrame([ref_leg(b, tp, int(t_live[i]), int(side[i]), float(qty[i]), int(T[i]), queue == "prorata",
                                 mks) for i in range(n)])[got.columns]
    for c in got.columns:
        if c in ("status", "trade_through", "level_vanished", "book_crossed", "complete"):
            assert list(got[c]) == list(want[c]), c
        else:
            np.testing.assert_allclose(got[c].to_numpy(float), want[c].to_numpy(float), rtol=1e-12, atol=1e-9,
                                       equal_nan=True, err_msg=c)
    # the sample exercises every branch
    assert (got["status"] == "no_book").sum() >= 1
    assert got["trade_through"].any() and got["level_vanished"].any() and got["book_crossed"].any()
    assert got["complete"].any() and not got["complete"].all()
    assert (got["maker_share"] == 1).any() and (got["maker_share"] == 0).any()
    assert ((got["maker_share"] > 0) & (got["maker_share"] < 1)).any()


def stitch(parts, shifts):
    bs = [QS.Book(b.ts + d, b.bid_px, b.bid_qty, b.ask_px, b.ask_qty) for (b, _), d in zip(parts, shifts)]
    ts = [QS.Tape(t.ts + d, t.px, t.qty, t.buyer_maker) for (_, t), d in zip(parts, shifts)]
    cat = lambda xs, f: np.concatenate([getattr(x, f) for x in xs])  # noqa: E731
    return (bs, ts, QS.Book(*(cat(bs, f) for f in ("ts", "bid_px", "bid_qty", "ask_px", "ask_qty"))),
            QS.Tape(*(cat(ts, f) for f in ("ts", "px", "qty", "buyer_maker"))))


@pytest.mark.parametrize("queue", list(QS.QUEUE))
def test_stitched_windows_match_each_window_alone(queue):
    rng = np.random.default_rng(31)
    shifts = (0, 10_000)
    bs, ts, b, tp = stitch([random_streams(rng), random_streams(rng)], shifts)
    win = [(T0 + d, T0 + d + 2000) for d in shifts]
    n = 200
    side = rng.choice([-1, 1], 2 * n)
    qty = 0.5 * rng.integers(1, 30, 2 * n)
    T = rng.integers(0, 600, 2 * n)
    t_live = T0 + np.r_[rng.integers(-50, 2050, n), shifts[1] + rng.integers(-50, 2050, n)]
    kw = {"queue": queue, "markouts_ms": (30, 250)}
    got = QS.simulate_legs(b, tp, t_live, side, qty, T, windows_ms=win, **kw)
    for h in range(2):
        sl = slice(h * n, (h + 1) * n)
        g = got.iloc[sl].reset_index(drop=True)
        alone = QS.simulate_legs(bs[h], ts[h], t_live[sl], side[sl], qty[sl], T[sl], windows_ms=[win[h]], **kw)
        assert list(g["status"]) == list(alone["status"]) and list(g["complete"]) == list(alone["complete"])
        c = g["complete"].to_numpy()
        assert 0.3 * n < c.sum() < n
        pd.testing.assert_frame_equal(g[c].reset_index(drop=True), alone[c].reset_index(drop=True))
    # without windows the legs live in the hole join the first stretch's last row and look complete
    hole = T0 + 2500 + rng.integers(0, 5000, 50)
    assert (QS.simulate_legs(b, tp, hole, 1, 1.0, 100, windows_ms=win, **kw)["status"] == "no_book").all()
    stale = QS.simulate_legs(b, tp, hole, 1, 1.0, 100, **kw)
    assert (stale["status"] == "ok").all() and stale["complete"].all()
