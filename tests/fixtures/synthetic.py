"""Synthetic fixture data for testing — 3 states, 2 elections."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_toy_panel() -> pd.DataFrame:
    """Create minimal synthetic panel for testing."""
    records = []
    states = ["ST1", "ST2", "ST3"]
    years = [2019, 2023]

    rng = np.random.default_rng(42)
    for year in years:
        for state in states:
            records.append({
                "year": year,
                "state": state,
                "vote_share": rng.uniform(0.2, 0.6),
                "turnout": rng.uniform(0.3, 0.7),
                "incumbent_party": rng.choice([0, 1]),
                "poverty_rate": rng.uniform(0.1, 0.8),
                "unemployment_rate": rng.uniform(0.05, 0.4),
                "violence_index": rng.uniform(0, 5),
                "vote_share_lag1": rng.uniform(0.2, 0.6) if year > 2019 else np.nan,
                "voter_turnout_lag1": rng.uniform(0.3, 0.7) if year > 2019 else np.nan,
            })
    return pd.DataFrame(records)


def make_toy_W() -> np.ndarray:
    """3×3 contiguity matrix: all connected."""
    W = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ], dtype=float)
    # Row-standardize
    W = W / W.sum(axis=1, keepdims=True)
    return W
