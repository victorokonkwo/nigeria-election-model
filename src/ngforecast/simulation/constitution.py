"""Constitutional win rule — §134(2) of the 1999 Constitution (as amended).

A candidate wins outright if:
  (a) they have the highest number of votes cast nationally, AND
  (b) they have not less than one-quarter of the votes cast in each of at
      least two-thirds of all the States and the FCT.

How (b) counts the FCT, and how a fractional requirement (2/3 × 37 = 24⅔)
is met, are legal questions. Both are read from ``config/constitution.yaml``
(``win_rule.fct_interpretation`` and ``win_rule.fractional_rule``); see the
comments there and ``docs/decisions/adr-003-section-134-interpretation.md``.

Default (binding law as of the 2023 Supreme Court decision):
  * FCT is treated as the 37th unit (§299; Obi v INEC, 2023), so the
    requirement is 2/3 × 37 = 24⅔ units.
  * The fraction is met as in Awolowo v Shagari (1979): 24 units at ≥ 25%
    plus one further unit at ≥ ⅔ × 25% ≈ 16.67%.

If nobody wins, the second and third elections of §134(3)–(5) are handled in
``ngforecast.simulation.runoff``.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
from loguru import logger

FCT_INTERPRETATIONS = ("fct_as_state", "states_only", "states_plus_fct")
FRACTIONAL_RULES = ("awolowo_v_shagari", "ceil", "floor")

# Tolerance so that a share of exactly 0.25 counts despite float rounding.
_EPS = 1e-12


def resolve_fct_index(constitution: dict, n_rows: int) -> int | None:
    """Row index of the FCT in a (n_rows, …) state array, or None if unknown.

    Rows are assumed to follow the order of ``constitution["units"]``.
    """
    units = constitution.get("units")
    if not units or len(units) != n_rows:
        return None
    fct_code = constitution.get("win_rule", {}).get("fct_code", "FC")
    for i, unit in enumerate(units):
        if unit.get("code") == fct_code:
            return i
    return None


def _spread_requirement(win_cfg: dict, n_units: int) -> tuple[int, Fraction]:
    """Return (whole units, fractional remainder) required for the spread test."""
    if "spread_fraction" not in win_cfg and "spread_minimum_units" in win_cfg:
        # Legacy config: fixed number of whole units.
        return int(win_cfg["spread_minimum_units"]), Fraction(0)

    fraction = Fraction(str(win_cfg.get("spread_fraction", "2/3"))).limit_denominator(1000)
    required = fraction * n_units
    whole = math.floor(required)
    return whole, required - whole


def meets_spread(
    unit_shares: np.ndarray,
    constitution: dict,
    fct_index: int | None = None,
) -> bool:
    """Does a candidate with these per-unit vote shares satisfy §134(2)(b)?

    Parameters
    ----------
    unit_shares : (n_rows,) the candidate's share of votes cast in each state
        (and the FCT, if present).
    constitution : parsed constitution.yaml
    fct_index : row of the FCT in ``unit_shares``. Required for the
        ``states_only`` and ``states_plus_fct`` interpretations; if omitted it
        is resolved from ``constitution["units"]``.
    """
    win_cfg = constitution["win_rule"]
    threshold = float(win_cfg["spread_threshold"])
    interpretation = win_cfg.get("fct_interpretation", "fct_as_state")
    fractional_rule = win_cfg.get("fractional_rule", "awolowo_v_shagari")

    if interpretation not in FCT_INTERPRETATIONS:
        raise ValueError(
            f"Unknown fct_interpretation {interpretation!r}; expected one of {FCT_INTERPRETATIONS}"
        )
    if fractional_rule not in FRACTIONAL_RULES:
        raise ValueError(
            f"Unknown fractional_rule {fractional_rule!r}; expected one of {FRACTIONAL_RULES}"
        )

    shares = np.asarray(unit_shares, dtype=float)

    if interpretation == "fct_as_state":
        counted = shares
    else:
        if fct_index is None:
            fct_index = resolve_fct_index(constitution, len(shares))
        if fct_index is None:
            raise ValueError(
                f"fct_interpretation={interpretation!r} needs to know which row is the FCT; "
                "pass fct_index or supply constitution['units'] matching the rows."
            )
        counted = np.delete(shares, fct_index)
        if interpretation == "states_plus_fct" and shares[fct_index] < threshold - _EPS:
            return False

    whole, frac = _spread_requirement(win_cfg, len(counted))
    if frac and fractional_rule == "ceil":
        whole, frac = whole + 1, Fraction(0)
    elif fractional_rule == "floor":
        frac = Fraction(0)

    ranked = np.sort(counted)[::-1]
    if whole > len(ranked):
        return False
    if whole > 0 and ranked[whole - 1] < threshold - _EPS:
        return False
    # Awolowo v Shagari: the fractional unit needs frac × threshold.
    return not frac or (whole < len(ranked) and ranked[whole] >= float(frac) * threshold - _EPS)


def check_win_rule(
    votes_by_state: np.ndarray,
    national_votes: np.ndarray,
    candidates: list[str],
    constitution: dict,
    fct_index: int | None = None,
) -> str | None:
    """Apply §134(2). Returns the winner's ID, or None if a second election is needed.

    Parameters
    ----------
    votes_by_state : (n_states, n_candidates) vote counts per state (+ FCT)
    national_votes : (n_candidates,) total national votes
    candidates : list of candidate IDs
    constitution : parsed constitution.yaml
    fct_index : row of the FCT (resolved from the config if omitted)
    """
    # (a) Plurality winner
    leader_idx = int(np.argmax(national_votes))

    # (b) Geographic spread
    leader_shares, n_above = compute_spread(
        votes_by_state, leader_idx, threshold=constitution["win_rule"]["spread_threshold"]
    )
    if meets_spread(leader_shares, constitution, fct_index=fct_index):
        logger.debug(
            "§134(2) win: {} — national leader, {} units ≥ 25%", candidates[leader_idx], n_above
        )
        return candidates[leader_idx]

    logger.debug("§134(2) not met: leader {} has {} units ≥ 25%", candidates[leader_idx], n_above)
    return None


def compute_spread(
    votes_by_state: np.ndarray,
    candidate_idx: int,
    threshold: float = 0.25,
) -> tuple[np.ndarray, int]:
    """A candidate's vote share in each unit and the number of units ≥ threshold.

    Returns (unit_shares, n_units_at_or_above_threshold).
    """
    state_totals = np.maximum(votes_by_state.sum(axis=1), 1.0)
    shares = votes_by_state[:, candidate_idx] / state_totals
    n_above = int(np.sum(shares >= threshold - _EPS))
    return shares, n_above
