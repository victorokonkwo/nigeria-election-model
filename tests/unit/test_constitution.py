"""Unit tests for §134 of the 1999 Constitution: win rule and second/third elections."""

from __future__ import annotations

import copy
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import yaml

from ngforecast.simulation.constitution import (
    _spread_requirement,
    check_win_rule,
    compute_spread,
    meets_spread,
    resolve_fct_index,
)
from ngforecast.simulation.runoff import select_runoff_candidates, simulate_runoff

CONFIG = Path(__file__).resolve().parents[2] / "config" / "constitution.yaml"
CANDIDATES = ["APC", "PDP", "LP", "NNPP"]
N_UNITS = 37
FCT = 36  # FCT is the last unit in config/constitution.yaml


@pytest.fixture
def constitution():
    with open(CONFIG) as f:
        return yaml.safe_load(f)


def with_rule(const: dict, **overrides) -> dict:
    c = copy.deepcopy(const)
    c["win_rule"].update(overrides)
    return c


def make_votes(leader_shares: np.ndarray, total: float = 1000.0) -> np.ndarray:
    """Votes where APC has the given share per unit; the rest split 50/30/20."""
    s = np.asarray(leader_shares, dtype=float)
    rest = (1 - s) * total
    votes = np.column_stack([s * total, rest * 0.5, rest * 0.3, rest * 0.2])
    assert np.argmax(votes.sum(axis=0)) == 0, "test setup: APC must lead nationally"
    return votes


def shares_with(
    n_strong: int, extra: tuple[float, ...] = (), weak: float = 0.10, fct: float | None = None
) -> np.ndarray:
    """n_strong non-FCT units at 60%, then `extra` units, rest at `weak`; FCT optionally set."""
    s = np.full(N_UNITS, weak)
    s[:n_strong] = 0.60
    for k, v in enumerate(extra):
        s[n_strong + k] = v
    if fct is not None:
        s[FCT] = fct
    return s


# ── Config ───────────────────────────────────


class TestConfig:
    def test_defaults_follow_binding_law(self, constitution):
        rule = constitution["win_rule"]
        assert rule["fct_interpretation"] == "fct_as_state"  # Obi v INEC (2023)
        assert rule["fractional_rule"] == "awolowo_v_shagari"  # SC.62/1979
        assert rule["spread_threshold"] == 0.25
        assert Fraction(rule["spread_fraction"]) == Fraction(2, 3)

    def test_second_election_within_seven_days(self, constitution):
        assert constitution["runoff"]["days_after"] == 7
        assert constitution["runoff"]["third_election"]["days_after"] == 7

    def test_fct_index_resolves(self, constitution):
        assert resolve_fct_index(constitution, N_UNITS) == FCT
        assert resolve_fct_index(constitution, 3) is None

    def test_two_thirds_is_exact(self, constitution):
        rule = constitution["win_rule"]
        assert _spread_requirement(rule, 37) == (24, Fraction(2, 3))
        assert _spread_requirement(rule, 36) == (24, Fraction(0))


# ── Default rule: FCT as 37th unit, Awolowo v Shagari fraction ──


class TestDefaultRule:
    def test_clear_winner(self, constitution):
        votes = make_votes(np.full(N_UNITS, 0.5))
        assert check_win_rule(votes, votes.sum(axis=0), CANDIDATES, constitution) == "APC"

    def test_25_full_units_wins(self, constitution):
        votes = make_votes(shares_with(25))
        assert check_win_rule(votes, votes.sum(axis=0), CANDIDATES, constitution) == "APC"

    def test_24_units_plus_two_thirds_of_quarter_wins(self, constitution):
        """24 units at ≥25% and a 25th at ≥16.67% satisfies 24⅔ (Awolowo v Shagari)."""
        votes = make_votes(shares_with(24, extra=[0.17]))
        assert check_win_rule(votes, votes.sum(axis=0), CANDIDATES, constitution) == "APC"

    def test_24_units_and_25th_below_two_thirds_of_quarter(self, constitution):
        votes = make_votes(shares_with(24, extra=[0.16]))
        assert check_win_rule(votes, votes.sum(axis=0), CANDIDATES, constitution) is None

    def test_23_units_fails_even_with_strong_fractional_units(self, constitution):
        votes = make_votes(shares_with(23, extra=[0.24, 0.24]))
        assert check_win_rule(votes, votes.sum(axis=0), CANDIDATES, constitution) is None

    def test_fct_counts_like_any_state(self, constitution):
        """No separate FCT requirement: 0% in FCT does not block a win."""
        votes = make_votes(shares_with(25, fct=0.0))
        assert check_win_rule(votes, votes.sum(axis=0), CANDIDATES, constitution) == "APC"

    def test_exactly_25_percent_counts(self, constitution):
        votes = np.tile([250.0, 250.0, 250.0, 250.0], (N_UNITS, 1))  # exactly 25% everywhere
        shares, n_above = compute_spread(votes, 0)
        assert n_above == N_UNITS
        assert meets_spread(shares, constitution)

    def test_plurality_leader_fails_spread(self, constitution):
        votes = np.zeros((N_UNITS, 4))
        votes[:22] = [1000, 300, 200, 100]
        votes[22:] = [100, 800, 700, 600]
        national = votes.sum(axis=0)
        assert np.argmax(national) == 0
        assert check_win_rule(votes, national, CANDIDATES, constitution) is None

    def test_all_zeros(self, constitution):
        votes = np.zeros((N_UNITS, 4))
        assert check_win_rule(votes, votes.sum(axis=0), CANDIDATES, constitution) is None


# ── Alternative readings (sensitivity analysis) ──


class TestFractionalRules:
    def test_ceil_needs_25_full_units(self, constitution):
        const = with_rule(constitution, fractional_rule="ceil")
        assert not meets_spread(shares_with(24, extra=[0.20]), const)
        assert meets_spread(shares_with(25), const)

    def test_floor_needs_24_full_units(self, constitution):
        const = with_rule(constitution, fractional_rule="floor")
        assert meets_spread(shares_with(24), const)
        assert not meets_spread(shares_with(23), const)


class TestFctInterpretations:
    def test_states_only_ignores_fct(self, constitution):
        const = with_rule(constitution, fct_interpretation="states_only")
        assert meets_spread(shares_with(24, fct=0.0), const)
        # 23 states + FCT is not enough: FCT does not count
        assert not meets_spread(shares_with(23, fct=0.9), const)

    def test_states_plus_fct_requires_fct(self, constitution):
        const = with_rule(constitution, fct_interpretation="states_plus_fct")
        assert not meets_spread(shares_with(24, fct=0.19), const)
        assert meets_spread(shares_with(24, fct=0.30), const)

    def test_fct_reading_without_fct_row_raises(self, constitution):
        const = with_rule(constitution, fct_interpretation="states_only")
        del const["units"]
        with pytest.raises(ValueError, match="FCT"):
            meets_spread(shares_with(24), const)

    def test_unknown_options_raise(self, constitution):
        with pytest.raises(ValueError):
            meets_spread(shares_with(30), with_rule(constitution, fct_interpretation="bogus"))
        with pytest.raises(ValueError):
            meets_spread(shares_with(30), with_rule(constitution, fractional_rule="bogus"))

    def test_legacy_fixed_unit_count(self):
        legacy = {"win_rule": {"spread_threshold": 0.25, "spread_minimum_units": 25}}
        assert meets_spread(shares_with(25), legacy)
        assert not meets_spread(shares_with(24, extra=[0.24]), legacy)


class TestComputeSpread:
    def test_all_states_above(self):
        shares, n = compute_spread(np.full((37, 3), 100.0), 0)
        assert n == 37
        assert np.allclose(shares, 1 / 3)

    def test_no_states_above(self):
        votes = np.zeros((37, 4))
        votes[:, 0] = 10
        votes[:, 1:] = 100
        _, n = compute_spread(votes, 0)
        assert n == 0


# ── §134(3)–(5): second and third elections ──


class TestRunoff:
    def test_second_candidate_is_most_units_won_not_runner_up(self):
        """§134(3)(b): LP leads in more units than PDP, so LP is the finalist
        even though PDP is the national runner-up."""
        votes = np.zeros((N_UNITS, 4))
        votes[:15] = [1000, 100, 50, 0]  # APC leads
        votes[15:17] = [100, 5000, 50, 0]  # PDP leads 2 big units
        votes[17:] = [300, 100, 400, 0]  # LP leads 20 small units
        national = votes.sum(axis=0)
        assert np.argsort(national)[-2] == 1  # PDP is national runner-up
        assert select_runoff_candidates(votes) == (0, 2)

    def test_tie_on_units_broken_by_national_votes(self):
        votes = np.zeros((4, 3))
        votes[0] = [100, 0, 0]
        votes[1] = [100, 0, 0]
        votes[2] = [0, 90, 0]
        votes[3] = [0, 0, 50]
        assert select_runoff_candidates(votes) == (0, 1)

    def test_runoff_with_broad_support_decided_in_second_election(self, constitution):
        votes = np.tile([500.0, 400.0, 100.0, 0.0], (N_UNITS, 1))
        result = simulate_runoff(votes, CANDIDATES, constitution, rng=np.random.default_rng(0))
        assert result.finalists == ("APC", "PDP")
        assert result.winner == "APC"
        assert result.decided_in == 2

    def test_concentrated_majority_goes_to_third_election(self, constitution):
        """Majority but < 25% in most units fails §134(4); §134(5) decides."""
        votes = np.zeros((N_UNITS, 2))
        votes[:5] = [100_000, 0]  # APC huge in 5 units
        votes[5:] = [10, 990]  # 1% elsewhere
        result = simulate_runoff(votes, ["APC", "PDP"], constitution, rng=np.random.default_rng(1))
        assert result.decided_in == 3
        assert result.winner == "APC"
