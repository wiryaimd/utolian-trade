import numpy as np
import pytest

from btcq.backtest import stats


def test_dsr_matches_bailey_lopez_de_prado_example():
    # Paper example: annual SR 2.5, T=1250 daily obs, skew -3, kurt 10, N=100, V[SR]=0.5 (annual).
    var_daily = 0.5 / 250
    assert stats.expected_max_sr(var_daily, 100) == pytest.approx(0.1132, abs=5e-4)
    d = stats.dsr(2.5 / np.sqrt(250), 1250, -3.0, 10.0, var_daily, 100)
    assert d == pytest.approx(0.9004, abs=2e-3)


def test_psr_is_half_at_threshold_and_monotone():
    assert stats.psr(0.1, 500, 0.0, 3.0, sr_star=0.1) == pytest.approx(0.5)
    assert stats.psr(0.1, 500, 0, 3) > stats.psr(0.05, 500, 0, 3)


def test_pbo_noise_vs_real_edge():
    # A single noise matrix can land anywhere; averaged over seeds PBO must sit near 0.5.
    pbos = [stats.pbo_cscv(np.random.default_rng(s).normal(0, 0.01, size=(1600, 20)))[0]
            for s in range(20)]
    assert 0.4 < np.mean(pbos) < 0.65
    noise = np.random.default_rng(1).normal(0, 0.01, size=(1600, 20))
    edge = noise.copy()
    edge[:, 3] += 0.004  # one config with a large persistent edge
    pbo_edge, _ = stats.pbo_cscv(edge)
    assert pbo_edge < 0.05


def test_block_bootstrap_shapes_and_sanity():
    r = np.full(200, 0.01)
    tot, mdd = stats.block_bootstrap(r, n_sims=500)
    assert tot.shape == (500,) and np.allclose(tot, 1.01 ** 200 - 1)
    assert np.allclose(mdd, 0)


def test_random_entry_null_centres_on_minus_cost_for_random_walk():
    rng = np.random.default_rng(2)
    o = 100 * np.exp(np.cumsum(rng.normal(0, 1e-4, 200_000)))
    cand = np.arange(0, 190_000, 15)
    null = stats.random_entry_null(o, cand, np.full(300, 60), np.r_[np.ones(150), -np.ones(150)],
                                   cost=0.0012, n_perm=300)
    assert abs(null.mean() + 0.0012) < 2e-4


def test_grid_neighbors():
    grid = {"a": [1, 2, 3], "b": ["x", "y"]}
    cfg = stats.grid_configs(grid)
    nb = stats.grid_neighbors(grid, cfg)
    i = cfg.index({"a": 2, "b": "x"})
    assert sorted(cfg[j]["a"] for j in nb[i]) == [1, 2, 3]
    assert len(nb[i]) == 3


def test_effective_trials_bounds():
    rng = np.random.default_rng(4)
    base = rng.normal(size=(2000, 1))
    identical = np.repeat(base, 10, axis=1)
    assert stats.effective_trials(identical) == pytest.approx(1.0, abs=1e-6)
    indep = rng.normal(size=(2000, 10))
    assert 9.0 < stats.effective_trials(indep) <= 10.0
    # The zero-skill sampling variance of a Sharpe estimate is ~1/(T-1).
    assert stats.null_sr_variance(1001) == pytest.approx(1e-3)
