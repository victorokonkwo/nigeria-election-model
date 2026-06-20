"""Tullock contest model → turnout multipliers by state/zone."""

from __future__ import annotations

import numpy as np
from loguru import logger


def tullock_turnout_multiplier(
    party_resources: dict[str, np.ndarray],
    contest_parameter: float = 1.0,
) -> np.ndarray:
    """Compute Tullock contest turnout multipliers by state.

    In a Tullock contest, each party's mobilization effort determines
    the probability of winning, which feeds back into expected turnout.

    Parameters
    ----------
    party_resources : {party_id: (n_states,) resource/effort vector}
    contest_parameter : r > 0, higher = more decisive contests

    Returns
    -------
    turnout_multiplier : (n_states,) values typically in [0.8, 1.2]
    """
    parties = list(party_resources.keys())
    n_states = next(iter(party_resources.values())).shape[0]

    # Total contest intensity per state
    total_effort = np.zeros(n_states)
    for party in parties:
        effort = np.maximum(party_resources[party], 0.0)
        total_effort += effort**contest_parameter

    # Competitiveness index: higher when contest is close
    # Uses Herfindahl-like measure — most competitive when shares are equal
    shares = np.zeros((len(parties), n_states))
    for i, party in enumerate(parties):
        effort = np.maximum(party_resources[party], 0.0)
        shares[i] = effort**contest_parameter / np.maximum(total_effort, 1e-10)

    hhi = np.sum(shares**2, axis=0)
    # Normalize: HHI = 1/K (perfectly competitive) → multiplier ~ 1.1
    #            HHI = 1 (monopoly) → multiplier ~ 0.9
    k = len(parties)
    competitiveness = (1.0 - hhi) / (1.0 - 1.0 / k) if k > 1 else np.ones(n_states)
    multiplier = 0.9 + 0.2 * competitiveness

    logger.info(
        "Tullock multipliers: mean={:.3f}, range=[{:.3f}, {:.3f}]",
        multiplier.mean(),
        multiplier.min(),
        multiplier.max(),
    )
    return multiplier
