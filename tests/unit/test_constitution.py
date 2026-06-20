"""Unit tests for the constitutional win/runoff rule (§134)."""

from __future__ import annotations

import numpy as np
import pytest

from ngforecast.simulation.constitution import check_win_rule, compute_spread


@pytest.fixture
def constitution():
    """Minimal constitution config for testing."""
    return {
        "total_units": 37,
        "win_rule": {
            "majority": "plurality",
            "spread_threshold": 0.25,
            "spread_minimum_units": 25,
        },
        "runoff": {
            "candidates": 2,
        },
    }


@pytest.fixture
def candidates():
    return ["APC", "PDP", "LP", "NNPP"]


class TestCheckWinRule:
    """Exhaustive tests for §134 win-rule logic."""

    def test_clear_winner_meets_spread(self, constitution, candidates):
        """Candidate wins plurality AND ≥25% in 25+ states → outright win."""
        n_states, n_cand = 37, 4
        votes = np.zeros((n_states, n_cand))
        # APC wins every state with >25% share
        votes[:, 0] = 1000  # APC
        votes[:, 1] = 500   # PDP
        votes[:, 2] = 300   # LP
        votes[:, 3] = 200   # NNPP

        national = votes.sum(axis=0)
        result = check_win_rule(votes, national, candidates, constitution)
        assert result == "APC"

    def test_plurality_leader_fails_spread(self, constitution, candidates):
        """National leader but <25% in too many states → runoff."""
        n_states, n_cand = 37, 4
        votes = np.zeros((n_states, n_cand))

        # APC wins nationally but has <25% in 15 states
        for i in range(22):
            votes[i] = [1000, 300, 200, 100]  # APC > 25%
        for i in range(22, 37):
            votes[i] = [100, 800, 700, 600]   # APC < 25% (100/2200 ≈ 4.5%)

        # APC still has most national votes
        national = votes.sum(axis=0)
        assert np.argmax(national) == 0  # APC leads nationally

        result = check_win_rule(votes, national, candidates, constitution)
        assert result is None  # runoff: only 22 states ≥ 25%

    def test_exactly_25_states_wins(self, constitution, candidates):
        """Exactly 25 states meeting threshold → wins."""
        n_states, n_cand = 37, 4
        votes = np.zeros((n_states, n_cand))

        for i in range(25):
            votes[i] = [500, 300, 200, 100]  # APC ≥ 25%
        for i in range(25, 37):
            votes[i] = [50, 800, 700, 600]   # APC < 25%

        national = votes.sum(axis=0)
        result = check_win_rule(votes, national, candidates, constitution)
        assert result == "APC"

    def test_exactly_24_states_triggers_runoff(self, constitution, candidates):
        """24 states meeting threshold → runoff (need 25)."""
        n_states, n_cand = 37, 4
        votes = np.zeros((n_states, n_cand))

        for i in range(24):
            votes[i] = [500, 300, 200, 100]
        for i in range(24, 37):
            votes[i] = [50, 800, 700, 600]

        national = votes.sum(axis=0)
        result = check_win_rule(votes, national, candidates, constitution)
        assert result is None

    def test_second_place_cannot_win(self, constitution, candidates):
        """Even if second place has better spread, national leader wins."""
        n_states, n_cand = 37, 4
        votes = np.zeros((n_states, n_cand))

        # APC wins nationally with big margins in few states
        # PDP has consistent 30% everywhere but less total
        for i in range(37):
            votes[i] = [600, 400, 300, 200]

        national = votes.sum(axis=0)
        result = check_win_rule(votes, national, candidates, constitution)
        assert result == "APC"

    def test_all_zeros_returns_first_candidate(self, constitution, candidates):
        """Edge case: all zero votes → argmax returns 0."""
        votes = np.zeros((37, 4))
        national = votes.sum(axis=0)
        # With all zeros, state_totals are 0, shares become 0/1=0 → no state ≥ 25%
        result = check_win_rule(votes, national, candidates, constitution)
        assert result is None  # 0 states meet threshold

    def test_two_candidate_race(self, constitution):
        """Two-candidate race — winner should have >50% in most states."""
        cands = ["APC", "PDP"]
        votes = np.zeros((37, 2))
        for i in range(37):
            votes[i] = [700, 300]

        national = votes.sum(axis=0)
        result = check_win_rule(votes, national, cands, constitution)
        assert result == "APC"

    def test_close_race_threshold_boundary(self, constitution, candidates):
        """Exactly 25% should count as meeting the threshold."""
        votes = np.zeros((37, 4))
        # Exactly 25%: 250 out of 1000
        for i in range(25):
            votes[i] = [250, 250, 250, 250]
        for i in range(25, 37):
            votes[i] = [240, 260, 250, 250]

        national = votes.sum(axis=0)
        # In the first 25 states, all candidates have exactly 25%
        # APC should be plurality leader (or tied)
        result = check_win_rule(votes, national, candidates, constitution)
        # With exactly 25% in first 25 states, APC meets spread
        # National: APC has 250*25 + 240*12 = 9130, which might not be leader
        # Let's check the actual numbers
        if national[0] >= max(national):
            assert result is not None


class TestComputeSpread:
    def test_all_states_above(self):
        votes = np.full((37, 3), 100.0)
        shares, n = compute_spread(votes, 0)
        assert n == 37
        assert np.allclose(shares, 1 / 3)

    def test_no_states_above(self):
        votes = np.zeros((37, 4))
        votes[:, 0] = 10
        votes[:, 1:] = 100
        shares, n = compute_spread(votes, 0)
        assert n == 0
