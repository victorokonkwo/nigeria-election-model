"""Cooperative game theory — Shapley values, Banzhaf index, core stability."""

from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np
from loguru import logger


def shapley_values(
    players: list[str],
    coalition_value: dict[frozenset[str], float],
) -> dict[str, float]:
    """Compute Shapley values for a cooperative game.

    Parameters
    ----------
    players : list of player names (party IDs)
    coalition_value : mapping from frozenset of players → coalition value
        (e.g., predicted combined vote share)

    Returns
    -------
    dict of {player: shapley_value}
    """
    n = len(players)
    phi = defaultdict(float)

    for perm in itertools.permutations(players):
        coalition = frozenset()
        for player in perm:
            coalition_with = coalition | {player}
            marginal = coalition_value.get(coalition_with, 0.0) - coalition_value.get(
                coalition, 0.0
            )
            phi[player] += marginal
            coalition = coalition_with

    # Average over all permutations
    n_fact = np.math.factorial(n)
    for player in players:
        phi[player] /= n_fact

    logger.info("Shapley values computed for {} players", n)
    return dict(phi)


def banzhaf_index(
    players: list[str],
    coalition_value: dict[frozenset[str], float],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute normalized Banzhaf power index.

    A player is critical in a coalition if removing them drops
    the coalition below the winning threshold.
    """
    n = len(players)
    swing_count = defaultdict(int)

    for r in range(1, n + 1):
        for coalition in itertools.combinations(players, r):
            cs = frozenset(coalition)
            if coalition_value.get(cs, 0.0) >= threshold:
                for player in coalition:
                    without = cs - {player}
                    if coalition_value.get(without, 0.0) < threshold:
                        swing_count[player] += 1

    total = sum(swing_count.values())
    if total == 0:
        return {p: 0.0 for p in players}

    banzhaf = {p: swing_count[p] / total for p in players}
    logger.info("Banzhaf index: {}", banzhaf)
    return banzhaf


def is_core_stable(
    allocation: dict[str, float],
    coalition_value: dict[frozenset[str], float],
) -> bool:
    """Check if an allocation is in the core of the cooperative game.

    The core requires that no coalition can do better by deviating.
    """
    players = list(allocation.keys())
    for r in range(1, len(players) + 1):
        for coalition in itertools.combinations(players, r):
            cs = frozenset(coalition)
            coalition_payoff = sum(allocation[p] for p in coalition)
            if coalition_value.get(cs, 0.0) > coalition_payoff:
                return False
    return True
