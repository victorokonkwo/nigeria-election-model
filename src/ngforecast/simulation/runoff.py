"""Second and third presidential elections — §134(3)–(5).

§134(3): if nobody wins under §134(2), a second election is held between
  (a) the candidate with the highest number of votes, and
  (b) among the remaining candidates, the one with "a majority of votes in
      the highest number of States" — read here as the candidate who led
      (had the most votes) in the most units. Ties go to the higher national
      total. This is not necessarily the national runner-up.
§134(4): the second election is won by a majority of votes cast plus the
  same one-quarter / two-thirds spread as §134(2)(b).
§134(5): failing that, a third election between the same two candidates is
  won by a simple majority of votes cast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger

from ngforecast.simulation.constitution import meets_spread


@dataclass(frozen=True)
class RunoffResult:
    winner: str
    finalists: tuple[str, str]
    decided_in: int  # 2 = second election (§134(4)), 3 = third election (§134(5))


def select_runoff_candidates(votes_by_state: np.ndarray) -> tuple[int, int]:
    """Indices of the two §134(3) finalists: (national leader, most-units candidate)."""
    national = votes_by_state.sum(axis=0)
    leader = int(np.argmax(national))

    unit_winners = np.argmax(votes_by_state, axis=1)
    units_won = np.bincount(unit_winners, minlength=votes_by_state.shape[1]).astype(float)

    others = [j for j in range(votes_by_state.shape[1]) if j != leader]
    # Most units led; tie-break on national total.
    second = max(others, key=lambda j: (units_won[j], national[j]))
    return leader, second


def _two_candidate_vote(
    votes_by_state: np.ndarray,
    finalists: tuple[int, int],
    rng: np.random.Generator,
    noise_sd: float,
) -> np.ndarray:
    """Round votes for two finalists: their own votes plus a noisy split of the rest."""
    a, b = finalists
    n_states = votes_by_state.shape[0]
    round_votes = votes_by_state[:, [a, b]].astype(float).copy()

    eliminated = np.ones(votes_by_state.shape[1], dtype=bool)
    eliminated[[a, b]] = False
    eliminated_votes = votes_by_state[:, eliminated].sum(axis=1)

    split = np.clip(0.5 + rng.normal(0, noise_sd, size=n_states), 0.1, 0.9)
    round_votes[:, 0] += eliminated_votes * split
    round_votes[:, 1] += eliminated_votes * (1 - split)
    return round_votes


def simulate_runoff(
    votes_by_state: np.ndarray,
    candidates: list[str],
    constitution: dict,
    rng: np.random.Generator | None = None,
    noise_sd: float = 0.02,
    fct_index: int | None = None,
) -> RunoffResult:
    """Simulate the §134(3)–(5) second (and, if needed, third) election.

    Votes for eliminated candidates are split roughly 50/50 between the two
    finalists, with state-level noise. Each round draws fresh noise.

    Parameters
    ----------
    votes_by_state : (n_states, n_candidates) first-round votes
    candidates : candidate IDs
    constitution : parsed constitution.yaml (for the §134(4) spread test)
    rng : random generator
    noise_sd : SD of the state-level split of eliminated candidates' votes
    fct_index : row of the FCT (resolved from the config if omitted)
    """
    if rng is None:
        rng = np.random.default_rng()

    finalists = select_runoff_candidates(votes_by_state)
    names = (candidates[finalists[0]], candidates[finalists[1]])

    # §134(4): majority of votes cast + spread
    second = _two_candidate_vote(votes_by_state, finalists, rng, noise_sd)
    totals = second.sum(axis=0)
    lead = int(np.argmax(totals))
    has_majority = totals[lead] > 0.5 * totals.sum()
    unit_shares = second[:, lead] / np.maximum(second.sum(axis=1), 1.0)
    if has_majority and meets_spread(unit_shares, constitution, fct_index=fct_index):
        logger.debug("§134(4) second election won by {}", names[lead])
        return RunoffResult(winner=names[lead], finalists=names, decided_in=2)

    # §134(5): simple majority only
    third = _two_candidate_vote(votes_by_state, finalists, rng, noise_sd)
    lead = int(np.argmax(third.sum(axis=0)))
    logger.debug("§134(5) third election won by {}", names[lead])
    return RunoffResult(winner=names[lead], finalists=names, decided_in=3)
