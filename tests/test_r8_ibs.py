"""Round 8 IBS rule and the date-level relocation null. Synthetic data only: no real price is loaded here."""
import numpy as np
import pandas as pd
import pytest

from btcq.research import round8_calib as C
from btcq.strategies import r8_ibs as R


# ---------------------------------------------------------------- the rule

def test_grid_primaries_and_neighbours():
    assert len(R.CONFIGS) == 12
    assert R.CONFIGS[R.PRIMARIES["IBS-L"]] == {"dir": "L", "theta": 0.20, "hold": 1}
    assert R.CONFIGS[R.PRIMARIES["IBS-LS"]] == {"dir": "LS", "theta": 0.20, "hold": 1}
    for name, d in (("IBS-L", "L"), ("IBS-LS", "LS")):
        assert [R.CONFIGS[i] for i in R.NEIGHBOURS[name]] == [{"dir": d, "theta": 0.15, "hold": 1},
                                                               {"dir": d, "theta": 0.25, "hold": 1},
                                                               {"dir": d, "theta": 0.20, "hold": 2}]
    assert R.MARKETS == ["SPYUSDT", "QQQUSDT", "EWYUSDT"]
    assert R.SLIP == {"SPYUSDT": 3e-4, "QQQUSDT": 2e-4, "EWYUSDT": 2e-4}


def test_ibs_and_undefined_days():
    h = np.array([10.0, 10.0, 10.0, 10.0, np.nan, 10.0])
    lo = np.array([8.0, 8.0, 8.0, 10.0, 8.0, 11.0])
    c = np.array([8.4, 9.6, 8.0, 10.0, 9.0, 10.5])
    ib = R.ibs(h, lo, c)
    np.testing.assert_allclose(ib[:3], [0.2, 0.8, 0.0], atol=1e-12)
    assert np.isnan(ib[3:]).all()                    # H = L, a missing high, H < L: no signal


def test_signals_thresholds_are_inclusive_and_long_only_has_no_shorts():
    ib = np.array([0.0, 0.2, 0.2000001, 0.5, 0.7999999, 0.8, 1.0, np.nan])
    np.testing.assert_array_equal(R.signals(ib, 0.2, "L"), [1, 1, 0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(R.signals(ib, 0.2, "LS"), [1, 1, 0, 0, 0, -1, -1, 0])
    np.testing.assert_array_equal(R.signals(np.array([0.85, 0.15]), 0.15, "LS"), [-1, 1])
    with pytest.raises(ValueError):
        R.signals(ib, 0.2, "S")


def test_hold_two_follows_the_latest_signal_of_the_last_two_closes():
    sig = np.array([1, 0, 0, -1, 0, 1, -1, 0, 0], np.int8)
    np.testing.assert_array_equal(R.hold_positions(sig, 1), sig)
    np.testing.assert_array_equal(R.hold_positions(sig, 2), [1, 1, 0, -1, -1, 1, -1, -1, 0])
    assert R.episodes(R.hold_positions(sig, 2)) == 4
    assert R.episodes(np.array([0, 1, 1, -1, -1, 0, 1], np.int8)) == 3


def ohlc(n, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    hi = c * np.exp(np.abs(rng.normal(0, 0.006, n)))
    lo = c * np.exp(-np.abs(rng.normal(0, 0.006, n)))
    return hi, lo, c


def test_no_position_before_sigma_and_sizes():
    h, lo, c = ohlc(200)
    pos, size = R.positions(h, lo, c, R.CONFIGS[R.PRIMARIES["IBS-LS"]])
    assert (pos[:90] == 0).all() and np.isnan(size[:90]).all() and np.isfinite(size[90:]).all()
    assert (pos[90:] != 0).any() and (size[90:] <= 2.0).all()
    r = np.diff(np.log(c))
    sigma = r[100 - 90:100].std(ddof=0) * np.sqrt(252)
    assert size[100] == pytest.approx(min(0.25 / sigma, 2.0), rel=1e-12)


@pytest.mark.parametrize("t", [120, 150])
def test_the_rule_is_causal(t):
    h, lo, c = ohlc(200, 3)
    cfg = {"dir": "LS", "theta": 0.25, "hold": 2}
    pos, size = R.positions(h, lo, c, cfg)
    rng = np.random.default_rng(9)
    h2, lo2, c2 = h.copy(), lo.copy(), c.copy()
    k = rng.uniform(0.5, 1.5, 200 - t - 1)
    c2[t + 1:] *= k
    h2[t + 1:] = c2[t + 1:] * 1.02
    lo2[t + 1:] = c2[t + 1:] * 0.99
    pos2, size2 = R.positions(h2, lo2, c2, cfg)
    np.testing.assert_array_equal(pos[:t + 1], pos2[:t + 1])
    np.testing.assert_array_equal(size[:t + 1], size2[:t + 1])


# ---------------------------------------------------------------- strata and the null

def toy_markets(n=700, seed=0, starts=(0, 250, 300)):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2000-01-03", periods=n, tz="UTC").as_unit("ms").asi8.astype(np.int64)
    out = []
    for j, a in enumerate(starts):
        h, lo, c = ohlc(n - a, seed + 10 * j)
        keep = rng.random(n - a) > 0.01                  # a few holes
        keep[:5] = True
        out.append(R.MarketRows(f"M{j}", days[a:][keep], c[keep] * 0.999, h[keep], lo[keep], c[keep]))
    return out, days


def test_strata_panel_eras_gaps_and_window():
    mk, days = toy_markets()
    a, b = int(days[150]), int(days[650])
    sp = R.strata_panel(mk, a, b)
    assert sp.eligible.sum() > 0 and not sp.eligible[~sp.in_window].any()
    inw = np.flatnonzero(sp.in_window)
    assert sp.dates[inw[0]] == sp.dates[sp.dates >= a][0]
    assert sp.dates[inw[-1]] < sp.dates[np.flatnonzero(sp.dates < b)[-1]]      # the last window date is out
    assert set(np.unique(sp.era[sp.in_window])) <= {1, 3, 7}
    e1 = int(np.searchsorted(sp.dates, mk[1].ts[90]))
    assert sp.era[e1 - 1] & 2 == 0 and sp.era[e1] & 2 == 2           # market 1 is ready from its 91st row on
    fri = pd.to_datetime(sp.dates, unit="ms", utc=True).weekday == 4
    assert (sp.gap[:-1][fri[:-1]] == 1).all()
    for e in np.unique(sp.era[sp.in_window]):
        m = sp.in_window & (sp.era == e) & (sp.vol_bin != R.WARMUP)
        counts = np.bincount(sp.vol_bin[m], minlength=3)
        assert counts.max() - counts.min() <= 1
    # eligibility: every ready market has a sized next return
    ok = np.all(np.isfinite(sp.Y) | ~sp.ready, axis=1)
    np.testing.assert_array_equal(sp.eligible, sp.in_window & (sp.era > 0) & ok)


def test_strata_use_only_the_past():
    mk, days = toy_markets(seed=4)
    sp = R.strata_panel(mk, int(days[150]), int(days[650]))
    t = 400
    cut = int(days[t])
    mk2 = [R.MarketRows(m.sym, m.ts, m.open, m.high, m.low,
                        np.where(m.ts > cut, m.close * np.linspace(0.7, 1.4, len(m.ts)), m.close)) for m in mk]
    sp2 = R.strata_panel(mk2, int(days[150]), int(days[650]))
    i = int(np.searchsorted(sp.dates, cut))
    np.testing.assert_array_equal(sp.trend[:i + 1], sp2.trend[:i + 1])
    np.testing.assert_array_equal(sp.era, sp2.era)
    np.testing.assert_array_equal(sp.ewma_vol[:i + 1], sp2.ewma_vol[:i + 1])


def test_a_next_close_past_the_window_end_is_outside():
    mk, days = toy_markets(seed=4)
    end = int(days[500])
    q = mk[1]
    keep = q.ts != days[499]                                     # market 1 misses the window's last grid date
    mk[1] = R.MarketRows(q.sym, q.ts[keep], q.open[keep], q.high[keep], q.low[keep], q.close[keep])
    sp = R.strata_panel(mk, int(days[150]), end)
    last_own = int(np.searchsorted(sp.dates, mk[1].ts[mk[1].ts < end][-1]))
    assert np.isnan(sp.Y[last_own, 1])                           # its next close is at or after the end
    v = R.held_vectors(sp, mk, R.CONFIGS[R.PRIMARIES["IBS-LS"]])
    assert v[last_own, 1] == 0


def test_held_vectors_only_inside_the_window():
    mk, days = toy_markets(seed=1)
    sp = R.strata_panel(mk, int(days[200]), int(days[600]))
    v = R.held_vectors(sp, mk, R.CONFIGS[R.PRIMARIES["IBS-LS"]])
    assert not v[~sp.in_window].any() and v[sp.in_window].any()
    assert np.isfinite(sp.Y[v != 0]).all()


def test_identity_draws_reproduce_the_statistic_and_give_p_one(monkeypatch):
    mk, days = toy_markets(seed=2)
    sp = R.strata_panel(mk, int(days[150]), int(days[690]))
    v = R.held_vectors(sp, mk, R.CONFIGS[R.PRIMARIES["IBS-LS"]])
    held = np.flatnonzero((v != 0).any(axis=1))
    manual = sum(v[d, m] * sp.Y[d, m] for d in held for m in range(3) if v[d, m] != 0) / (v != 0).sum()
    monkeypatch.setattr(R, "relocation_draws", lambda sp_, h, n, s: (np.tile(h, (n, 1)).astype(np.int64), 0))
    out = R.null_test(sp, v, 50, 1)
    assert out["actual"] == pytest.approx(manual, rel=1e-12)
    assert (out["dist"] == out["actual"]).all() and out["p_value"] == 1.0


def test_a_draw_moves_whole_vectors_within_the_stratum():
    mk, days = toy_markets(seed=3)
    sp = R.strata_panel(mk, int(days[150]), int(days[690]))
    v = R.held_vectors(sp, mk, R.CONFIGS[R.PRIMARIES["IBS-LS"]])
    held = np.flatnonzero((v != 0).any(axis=1))
    J, kept = R.relocation_draws(sp, held, 40, 7)
    assert J.shape == (40, len(held)) and kept == 0
    assert (sp.stratum[J] == sp.stratum[held][None, :]).all() and sp.eligible[J].all()
    assert (J != held[None, :]).mean() > 0.5
    # the statistic of a draw is the held vectors evaluated, whole, at their destination dates
    r = 5
    vh = v[held]
    manual = sum(vh[h, m] * sp.Y[J[r, h], m] for h in range(len(held)) for m in range(3) if vh[h, m] != 0)
    assert R._stat_rows(vh, sp.Y, J[r:r + 1])[0] == pytest.approx(manual / (vh != 0).sum(), rel=1e-12)
    J2, _ = R.relocation_draws(sp, held, 40, 7)
    np.testing.assert_array_equal(J, J2)                             # seeded


def test_a_stratum_without_destinations_keeps_its_dates():
    mk, days = toy_markets(seed=5)
    sp = R.strata_panel(mk, int(days[150]), int(days[690]))
    v = R.held_vectors(sp, mk, R.CONFIGS[R.PRIMARIES["IBS-L"]])
    held = np.flatnonzero((v != 0).any(axis=1))
    g = sp.stratum[held[0]]
    sp.eligible[sp.stratum == g] = False
    J, kept = R.relocation_draws(sp, held, 10, 1)
    lone = sp.stratum[held] == g
    assert kept == int(lone.sum()) and (J[:, lone] == held[lone][None, :]).all()


def test_planted_edge_is_detected_and_noise_is_not():
    rng = np.random.default_rng([5, 5])
    mk, a, b = C.synthetic_markets(rng, "planted", 0.10)
    assert max(C.calibration_p(mk, a, b, 199, 3)) < 0.02
    rng = np.random.default_rng([5, 6])
    mk, a, b = C.synthetic_markets(rng, "iid")
    assert min(C.calibration_p(mk, a, b, 199, 3)) > 0.02


# ---------------------------------------------------------------- the calibration generator

def test_generator_shape_ramp_and_ohlc_consistency():
    rng = np.random.default_rng(1)
    mk, a, b = C.synthetic_markets(rng, "factor")
    assert [len(m.ts) for m in mk] == [C.N_DAYS - s for s in C.STARTS]
    for m in mk:
        assert (m.low <= np.minimum(m.open, m.close) + 1e-9).all() and (m.high >= np.maximum(m.open, m.close) - 1e-9).all()
        assert (m.high > m.low).all()
    sp = R.strata_panel(mk, a, b)
    days = sp.in_window.sum() + 1
    assert 5500 < days < 5700
    yrs1 = (np.searchsorted(sp.dates, mk[1].ts[90]) - np.searchsorted(sp.dates, a)) / 252
    yrs2 = (np.searchsorted(sp.dates, mk[2].ts[90]) - np.searchsorted(sp.dates, a)) / 252
    assert 5.9 < yrs1 < 6.1 and 7.1 < yrs2 < 7.3
    r = [m.close[1:] / m.close[:-1] - 1 for m in mk]
    n = min(len(x) for x in r)
    cc = np.corrcoef(np.vstack([x[-n:] for x in r]))
    assert 0.8 < cc[0, 1] < 0.9 and 0.62 < cc[0, 2] < 0.78


@pytest.mark.parametrize("sc", ["iid", "garch", "weekend"])
def test_the_expected_simple_return_does_not_depend_on_ibs(sc):
    """No-edge panels: next-day simple returns after low-IBS and other days have the same mean (to sampling error),
    pooled over a few panels."""
    lows, rest = [], []
    for i in range(6):
        mk, _, _ = C.synthetic_markets(np.random.default_rng([77, i]), sc)
        m = mk[0]
        ib = R.ibs(m.high, m.low, m.close)[:-1]
        r = m.close[1:] / m.close[:-1] - 1
        if sc == "weekend":                                   # compare within weekday-gap groups
            g = np.diff(m.ts) >= 2 * R.DAY_MS
            ib, r = ib[~g], r[~g]
        lows.append(r[ib <= 0.2])
        rest.append(r[ib > 0.2])
    lo, re = np.concatenate(lows), np.concatenate(rest)
    se = np.sqrt(lo.var() / len(lo) + re.var() / len(re))
    assert abs(lo.mean() - re.mean()) < 4 * se


def test_calibration_does_not_depend_on_chunking():
    a = C.run_rows(draws=49, workers=1, chunk=1, rows=[0, 7], reps=3)
    b = C.run_rows(draws=49, workers=1, chunk=2, rows=[0, 7], reps=3)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])


def test_registered_calibration_rows():
    kinds = [(sc, kind) for sc, _, _, kind in C.CAL_ROWS]
    assert [sc for sc, k in kinds if k == "no-edge"] == ["iid", "garch", "varprem", "regime", "factor", "weekend"]
    assert all(reps == 1000 for _, _, reps, kind in C.CAL_ROWS if kind == "no-edge")
    assert C.CAL_DRAWS == 499 and C.CAL_MAX_P05 == 0.07 and C.CAL_MAX_P025 == 0.035
    assert [e for sc, e, _, _ in C.CAL_ROWS if sc == "planted"] == [0.05, 0.10]
