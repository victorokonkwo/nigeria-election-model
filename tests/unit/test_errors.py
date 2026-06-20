"""Unit tests for correlated error draws."""

from __future__ import annotations

import numpy as np
import pytest

from ngforecast.simulation.errors import draw_correlated_errors


class TestCorrelatedErrors:
    def test_shape(self):
        errors = draw_correlated_errors(
            n_sims=100, n_states=37, zone_assignments=np.zeros(37, dtype=int)
        )
        assert errors.shape == (100, 37)

    def test_deterministic_with_seed(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        zones = np.array([0] * 7 + [1] * 6 + [2] * 7 + [3] * 5 + [4] * 6 + [5] * 6)
        e1 = draw_correlated_errors(100, 37, zones, rng=rng1)
        e2 = draw_correlated_errors(100, 37, zones, rng=rng2)
        np.testing.assert_array_equal(e1, e2)

    def test_zonal_correlation(self):
        """States in the same zone should be more correlated than across zones."""
        zones = np.array([0] * 7 + [1] * 6 + [2] * 7 + [3] * 5 + [4] * 6 + [5] * 6)
        errors = draw_correlated_errors(
            n_sims=10000,
            n_states=37,
            zone_assignments=zones,
            sigma_national=0.0,
            sigma_zonal=0.05,
            sigma_state=0.01,
            rng=np.random.default_rng(123),
        )
        # Within zone 0 (states 0-6)
        within_corr = np.corrcoef(errors[:, 0], errors[:, 1])[0, 1]
        # Across zones (state 0 in zone 0, state 7 in zone 1)
        across_corr = np.corrcoef(errors[:, 0], errors[:, 7])[0, 1]
        assert within_corr > across_corr

    def test_mean_near_zero(self):
        errors = draw_correlated_errors(
            n_sims=50000,
            n_states=37,
            zone_assignments=np.zeros(37, dtype=int),
            rng=np.random.default_rng(99),
        )
        assert abs(errors.mean()) < 0.01
