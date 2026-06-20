"""Reporting tables — summary tables for forecast output."""

from __future__ import annotations

import pandas as pd
from loguru import logger


def win_probability_table(forecast: dict) -> pd.DataFrame:
    """Create a formatted win probability table from forecast artifact."""
    rows = []
    for c in forecast.get("candidates", []):
        rows.append({
            "Candidate": c["candidate_id"],
            "Party": c["party"],
            "Win Prob": f"{c['win_probability']:.1%}",
            "Vote Share": f"{c['national_vote_share_mean']:.1%}",
            "95% CI": f"[{c['national_vote_share_ci_lower']:.1%}, {c['national_vote_share_ci_upper']:.1%}]",
            "States ≥25%": f"{c['states_above_25pct_mean']:.1f}",
        })

    df = pd.DataFrame(rows)
    logger.info("Win probability table: {} candidates", len(df))
    return df


def state_level_table(forecast: dict, candidate_id: str) -> pd.DataFrame:
    """Create state-level vote share table for a given candidate."""
    for c in forecast.get("candidates", []):
        if c["candidate_id"] == candidate_id:
            shares = c.get("state_vote_shares", {})
            df = pd.DataFrame(
                [{"State": s, "Vote Share": f"{v:.1%}"} for s, v in sorted(shares.items())]
            )
            return df

    logger.warning("Candidate {} not found in forecast", candidate_id)
    return pd.DataFrame()
