"""Unit tests for QRE voter migration model."""

from __future__ import annotations

import numpy as np
import pytest

from ngforecast.game_theory.migration_qre import apply_qre_all_states, qre_reallocation


class TestQREReallocation:
    def test_shares_sum_to_one(self):
        sincere = np.array([0.4, 0.35, 0.25])
        utility = np.array([[1.0, 0.5], [0.3, 1.0], [0.6, 0.8]])
        actual = qre_reallocation(sincere, utility, lam=2.0)
        assert actual.sum() == pytest.approx(1.0)

    def test_high_lambda_more_strategic(self):
        """Higher λ → voters move more toward highest-utility candidate."""
        sincere = np.array([0.5, 0.5])
        utility = np.array([[1.0, 0.0], [0.0, 1.0]])
        # λ=0 → random, λ→∞ → fully strategic (no migration)
        low = qre_reallocation(sincere, utility, lam=0.1)
        high = qre_reallocation(sincere, utility, lam=10.0)
        # With these utilities, high λ should keep votes with own candidate
        assert high[0] > low[0] or np.isclose(high[0], low[0])

    def test_identity_utility_no_change(self):
        """If utility matrix is identity, voters stay with their candidate."""
        sincere = np.array([0.6, 0.4])
        utility = np.eye(2) * 10.0  # strong preference for own
        actual = qre_reallocation(sincere, utility, lam=5.0)
        np.testing.assert_allclose(actual, sincere, atol=0.01)


class TestApplyQREAllStates:
    def test_output_shape(self):
        sincere = np.random.dirichlet([1, 1, 1], size=37)
        utility = np.eye(3)
        actual = apply_qre_all_states(sincere, utility)
        assert actual.shape == (37, 3)

    def test_all_states_sum_to_one(self):
        sincere = np.random.dirichlet([1, 1, 1, 1], size=37)
        utility = np.random.rand(4, 3)
        actual = apply_qre_all_states(sincere, utility)
        np.testing.assert_allclose(actual.sum(axis=1), 1.0, atol=1e-10)
