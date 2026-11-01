import numpy as np
from churnlens.metrics import top_k_metrics, psi


def test_perfect_ranking_top_10_percent():
    y = np.array([1] * 10 + [0] * 90)
    p = np.concatenate([np.linspace(1, 0.5, 10), np.linspace(0.4, 0, 90)])
    m = top_k_metrics(y, p, fraction=0.10)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0


def test_psi_zero_for_identical_distributions():
    x = np.random.default_rng(0).normal(size=1000)
    assert psi(x, x) < 1e-6


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(0, 1, 2000), rng.normal(1, 1, 2000)) > 0.1


def test_value_weighted_lift_perfect_top_captures_all_value():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    v = np.array([100, 50, 10, 10, 10, 10, 10, 10, 10, 10], dtype=float)
    from churnlens.metrics import value_weighted_lift
    m = value_weighted_lift(y, p, v, fraction=0.2)
    assert m["value_share_captured"] == 1.0
