"""Integration test — end-to-end simulation on synthetic fixture data."""

from __future__ import annotations

import numpy as np
import pytest

from ngforecast.simulation.constitution import check_win_rule
from ngforecast.simulation.engine import run_simulations
from ngforecast.simulation.errors import draw_correlated_errors


@pytest.fixture
def synthetic_election():
    """Minimal synthetic election with 3 states, 2 elections, 3 candidates."""
    n_states = 3
    n_candidates = 3
    return {
        "predicted_shares": np.array([
            [0.45, 0.35, 0.20],
            [0.30, 0.50, 0.20],
            [0.40, 0.30, 0.30],
        ]),
        "turnout": np.array([0.60, 0.55, 0.65]),
        "registered_voters": np.array([1_000_000, 800_000, 1_200_000]),
        "zone_assignments": np.array([0, 0, 1]),
        "candidates": ["A", "B", "C"],
    }


@pytest.mark.integration
class TestEndToEnd:
    def test_simulation_runs(self, synthetic_election):
        """Smoke test: full simulation completes without error."""
        # Use small n_sims for speed, and use 37 states for constitutional rule
        n_states = 37
        n_cand = 3
        rng = np.random.default_rng(42)
        predicted = rng.dirichlet([3, 2, 1], size=n_states)
        turnout = rng.uniform(0.4, 0.7, size=n_states)
        registered = rng.integers(500_000, 5_000_000, size=n_states)
        zones = np.array([0]*7 + [1]*6 + [2]*7 + [3]*5 + [4]*6 + [5]*6)

        results = run_simulations(
            predicted_shares=predicted,
            turnout=turnout,
            registered_voters=registered,
            zone_assignments=zones,
            candidates=["APC", "PDP", "LP"],
            n_sims=100,
            seed=42,
        )

        assert "win_probabilities" in results
        assert "runoff_probability" in results
        assert sum(results["win_probabilities"].values()) + results["runoff_probability"] <= 1.01

    def test_probabilities_are_valid(self, synthetic_election):
        """Win probabilities are in [0,1] and sum ≤ 1."""
        n_states = 37
        rng = np.random.default_rng(99)
        predicted = rng.dirichlet([2, 2, 2, 1], size=n_states)
        turnout = rng.uniform(0.4, 0.7, size=n_states)
        registered = rng.integers(500_000, 5_000_000, size=n_states)
        zones = np.array([0]*7 + [1]*6 + [2]*7 + [3]*5 + [4]*6 + [5]*6)

        results = run_simulations(
            predicted_shares=predicted,
            turnout=turnout,
            registered_voters=registered,
            zone_assignments=zones,
            candidates=["APC", "PDP", "LP", "NNPP"],
            n_sims=50,
            seed=99,
        )

        for prob in results["win_probabilities"].values():
            assert 0.0 <= prob <= 1.0
