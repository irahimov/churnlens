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
