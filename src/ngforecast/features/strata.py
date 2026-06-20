"""Demographic strata for MRP poststratification."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger


def build_strata_frame(census_path: Path) -> pd.DataFrame:
    """Build demographic cell frame from census marginals.

    Each row is a unique (state, age_group, gender, urban_rural) cell
    with a population weight for poststratification.
    """
    census = pd.read_csv(census_path)
    required = {"state", "age_group", "gender", "urban_rural", "population"}
    missing = required - set(census.columns)
    if missing:
        raise ValueError(f"Census marginals missing columns: {missing}")

    # Compute cell proportions within each state
    state_totals = census.groupby("state")["population"].transform("sum")
    census["cell_weight"] = census["population"] / state_totals

    logger.info(
        "Strata frame: {} cells across {} states",
        len(census),
        census["state"].nunique(),
    )
    return census


def save_strata(strata: pd.DataFrame, dest: Path) -> None:
    """Write strata frame to parquet."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    strata.to_parquet(dest, index=False)
    logger.info("Strata saved to {}", dest)
