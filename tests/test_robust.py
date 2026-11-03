"""Sanity tests for the VERITAS defense primitives."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from veritas.robust import consistency_flags, robust_mu, trimmed_mean, cap_weights
from veritas.metrics import detection_metrics


def test_sigma_inflation_is_flagged():
    # 5 honest clients: sigma ~ 2 * ||update||; one liar inflates sigma x1000
    norms = np.array([1.0, 1.1, 0.9, 1.05, 0.95, 1.0])
    sigmas = 2.0 * norms.copy()
    sigmas[5] *= 1000.0                      # liar
    flags, _ = consistency_flags(sigmas, norms, tau=3.0)
    assert flags[5] and not flags[:5].any()


def test_robust_mu_ignores_one_poison():
    mus = [1.0, 1.02, 0.98, 1.01, 0.99, 6.0]  # last is poisoned
    assert abs(robust_mu(mus, "median") - 1.0) < 0.1
    assert robust_mu(mus, "mean") > 1.5       # plain mean is corrupted


def test_trimmed_mean_drops_outlier_update():
    ups = [np.ones(4), np.ones(4)*1.1, np.ones(4)*0.9,
           np.ones(4), np.ones(4)*0.95, np.ones(4)*50]  # last is a norm outlier
    out = trimmed_mean(ups, trim=1)
    assert np.all(out < 2.0)


def test_cap_weights_bounds_influence():
    w = cap_weights([100, 1, 1, 1], cap=0.4)
    assert w.max() <= 0.4 + 1e-9 and abs(w.sum() - 1.0) < 1e-9


def test_detection_metrics():
    flags = [False, False, False, False, False, True]
    m = detection_metrics(flags, [5])
    assert m["tpr"] == 1.0 and m["fpr"] == 0.0
