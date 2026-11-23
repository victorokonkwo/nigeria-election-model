"""Unit tests for ensemble stacking."""

from __future__ import annotations

import numpy as np
import pytest

from ngforecast.models.ensemble.stacking import bayesian_stacking, ensemble_predict


class TestBayesianStacking:
    def test_weights_sum_to_one(self):
        preds = {
            "m1": np.random.rand(100),
            "m2": np.random.rand(100),
            "m3": np.random.rand(100),
        }
        obs = np.random.rand(100)
        weights = bayesian_stacking(preds, obs)
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-10)

    def test_simple_average(self):
        preds = {"a": np.ones(10), "b": np.ones(10)}
        obs = np.ones(10)
        weights = bayesian_stacking(preds, obs, method="simple_average")
        assert weights["a"] == pytest.approx(0.5)
        assert weights["b"] == pytest.approx(0.5)


class TestEnsemblePredict:
    def test_weighted_sum(self):
        preds = {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])}
        weights = {"a": 0.7, "b": 0.3}
        result = ensemble_predict(preds, weights)
        expected = np.array([0.7 * 1 + 0.3 * 3, 0.7 * 2 + 0.3 * 4])
        np.testing.assert_allclose(result, expected)


class TestDensityStacking:
    def test_prefers_model_with_higher_density(self):
        from ngforecast.models.ensemble.stacking import stack_densities

        rng = np.random.default_rng(0)
        good = rng.normal(-1.0, 0.1, 50)
        bad = rng.normal(-5.0, 0.1, 50)
        w = stack_densities(np.column_stack([good, bad]), ["good", "bad"])
        assert w["good"] > 0.95
        assert sum(w.values()) == pytest.approx(1.0)

    def test_complementary_models_both_get_weight(self):
        """Each model is right on half the data: stacking mixes them."""
        from ngforecast.models.ensemble.stacking import stack_densities

        a = np.r_[np.full(25, 0.0), np.full(25, -20.0)]
        b = np.r_[np.full(25, -20.0), np.full(25, 0.0)]
        w = stack_densities(np.column_stack([a, b]), ["a", "b"])
        assert w["a"] == pytest.approx(0.5, abs=0.05)

    def test_gaussian_logit_logdens_peaks_at_center(self):
        from ngforecast.models.ensemble.stacking import gaussian_logit_logdens

        rng = np.random.default_rng(1)
        draws = 1 / (1 + np.exp(-rng.normal(0.0, 0.5, (4000, 2))))
        ld = gaussian_logit_logdens(draws, np.array([0.5, 0.9]))
        assert ld[0] > ld[1]

    def test_mixture_draws_respects_weights(self):
        from ngforecast.models.ensemble.stacking import mixture_draws

        draws = {"a": np.zeros((100, 3)), "b": np.ones((100, 3))}
        out = mixture_draws(draws, {"a": 0.8, "b": 0.2}, 10_000, np.random.default_rng(2))
        assert out[:, 0].mean() == pytest.approx(0.2, abs=0.02)
