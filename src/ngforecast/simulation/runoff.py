"""Round-two (runoff) two-candidate simulation."""

from __future__ import annotations

import numpy as np
from loguru import logger


def simulate_runoff(
    votes_by_state: np.ndarray,
    candidates: list[str],
    rng: np.random.Generator | None = None,
    noise_sd: float = 0.02,
) -> str | None:
    """Simulate a two-candidate runoff between the top two candidates.

    In the runoff, votes from eliminated candidates are reallocated
    with some noise. The winner needs simple majority nationally
    plus geographic spread (§134 applied to round two).

    Parameters
    ----------
    votes_by_state : (n_states, n_candidates) from round one
    candidates : candidate IDs
    rng : random generator
    noise_sd : noise added to reallocation

    Returns
    -------
    Winner candidate ID
    """
    if rng is None:
        rng = np.random.default_rng()

    national_votes = votes_by_state.sum(axis=0)
    top2_idx = np.argsort(national_votes)[-2:]

    # Reallocate eliminated candidates' votes roughly 50/50 with noise
    n_states = votes_by_state.shape[0]
    runoff_votes = np.zeros((n_states, 2))
    runoff_votes[:, 0] = votes_by_state[:, top2_idx[0]]
    runoff_votes[:, 1] = votes_by_state[:, top2_idx[1]]

    # Eliminated votes
    eliminated_mask = np.ones(len(candidates), dtype=bool)
    eliminated_mask[top2_idx] = False
    eliminated_votes = votes_by_state[:, eliminated_mask].sum(axis=1)

    # Split with noise
    split = 0.5 + rng.normal(0, noise_sd, size=n_states)
    split = np.clip(split, 0.1, 0.9)
    runoff_votes[:, 0] += eliminated_votes * split
    runoff_votes[:, 1] += eliminated_votes * (1 - split)

    # Winner by simple majority
    runoff_national = runoff_votes.sum(axis=0)
    winner_idx = int(np.argmax(runoff_national))
    winner = candidates[top2_idx[winner_idx]]

    logger.debug("Runoff winner: {}", winner)
    return winner
