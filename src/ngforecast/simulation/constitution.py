"""Constitutional win/runoff rule — §134 of the 1999 Constitution.

A candidate wins outright if:
  (a) they have the highest national vote total (plurality), AND
  (b) they secure ≥ 25% of votes in at least 25 of 37 states+FCT.

If no candidate satisfies both conditions, a runoff is triggered.
"""

from __future__ import annotations

import math

import numpy as np
from loguru import logger


def check_win_rule(
    votes_by_state: np.ndarray,
    national_votes: np.ndarray,
    candidates: list[str],
    constitution: dict,
) -> str | None:
    """Apply §134 win rule. Returns winner name or None (runoff).

    Parameters
    ----------
    votes_by_state : (n_states, n_candidates) vote counts per state
    national_votes : (n_candidates,) total national votes
    candidates : list of candidate IDs
    constitution : parsed constitution.yaml

    Returns
    -------
    winner candidate ID, or None if runoff is triggered
    """
    win_cfg = constitution["win_rule"]
    threshold = win_cfg["spread_threshold"]  # 0.25
    min_units = win_cfg["spread_minimum_units"]  # 25

    n_states, n_cand = votes_by_state.shape

    # (a) Plurality winner
    leader_idx = int(np.argmax(national_votes))

    # (b) Geographic spread: ≥ threshold in ≥ min_units states
    state_totals = votes_by_state.sum(axis=1)  # total valid votes per state
    state_totals_safe = np.maximum(state_totals, 1.0)
    leader_state_shares = votes_by_state[:, leader_idx] / state_totals_safe

    states_meeting_threshold = int(np.sum(leader_state_shares >= threshold))

    if states_meeting_threshold >= min_units:
        winner = candidates[leader_idx]
        logger.debug(
            "§134 win: {} — national leader + {}/{} states ≥ {:.0%}",
            winner,
            states_meeting_threshold,
            n_states,
            threshold,
        )
        return winner

    logger.debug(
        "§134 runoff: leader {} has {}/{} states ≥ {:.0%} (need {})",
        candidates[leader_idx],
        states_meeting_threshold,
        n_states,
        threshold,
        min_units,
    )
    return None


def compute_spread(
    votes_by_state: np.ndarray,
    candidate_idx: int,
) -> tuple[np.ndarray, int]:
    """Compute a candidate's vote share in each state and count of states ≥ 25%.

    Returns (state_shares, n_states_above_threshold).
    """
    state_totals = np.maximum(votes_by_state.sum(axis=1), 1.0)
    shares = votes_by_state[:, candidate_idx] / state_totals
    n_above = int(np.sum(shares >= 0.25))
    return shares, n_above
