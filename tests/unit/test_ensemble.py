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
