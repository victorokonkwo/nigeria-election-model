"""Spatial-competition Nash entry/withdrawal model."""

from __future__ import annotations

import numpy as np
from loguru import logger


def compute_entry_payoffs(
    candidate_positions: dict[str, np.ndarray],
    state_electorates: np.ndarray,
    entry_cost: float = 0.05,
) -> dict[str, float]:
    """Compute expected payoff for each candidate's entry decision.

    Models spatial competition where candidates choose to enter/stay
    based on expected vote share minus entry cost.

    Parameters
    ----------
    candidate_positions : {candidate_id: (n_states,) position vector}
    state_electorates : (n_states,) registered voters per state
    entry_cost : fixed cost of running (as fraction of total)

    Returns
    -------
    dict of {candidate: net_payoff} — positive means entry is rational
    """
    total_electorate = state_electorates.sum()
    payoffs = {}

    for cand, pos in candidate_positions.items():
        # Expected vote share (proximity model)
        distances = {}
        for other, other_pos in candidate_positions.items():
            if other != cand:
                distances[other] = np.linalg.norm(pos - other_pos)

        # Simplified: expected share proportional to inverse distance
        expected_share = np.sum(pos * state_electorates) / total_electorate
        payoffs[cand] = expected_share - entry_cost

    logger.info("Entry payoffs: {}", {k: f"{v:.4f}" for k, v in payoffs.items()})
    return payoffs


def nash_equilibrium_entry(
    candidate_positions: dict[str, np.ndarray],
    state_electorates: np.ndarray,
    entry_cost: float = 0.05,
) -> set[str]:
    """Find Nash equilibrium set of entrants.

    Iteratively remove candidates with negative payoff until stable.
    """
    active = set(candidate_positions.keys())

    for _ in range(len(candidate_positions)):
        active_positions = {c: candidate_positions[c] for c in active}
        payoffs = compute_entry_payoffs(active_positions, state_electorates, entry_cost)
        exits = {c for c, p in payoffs.items() if p < 0}

        if not exits:
            break
        active -= exits
        logger.info("Candidates exiting: {}", exits)

    logger.info("Nash equilibrium entrants: {}", active)
    return active
