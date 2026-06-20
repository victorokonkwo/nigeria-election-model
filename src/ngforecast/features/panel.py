"""Assemble the dynamic state × election panel from cleaned sources."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger


def build_panel(
    results_path: Path,
    macro_dir: Path,
    violence_path: Path | None = None,
) -> pd.DataFrame:
    """Merge election results with macro and violence data into a panel.

    Returns a DataFrame indexed by (year, state) with vote shares,
    turnout, and all covariates needed for the fundamentals model.
    """
    # ── Election results ─────────────────────
    results = pd.read_parquet(results_path)
    logger.info("Results: {} rows", len(results))

    # ── Macro indicators ─────────────────────
    poverty = pd.read_csv(macro_dir / "poverty.csv")
    unemployment = pd.read_csv(macro_dir / "unemployment.csv")

    panel = results.copy()
    panel = panel.merge(poverty, on=["year", "state"], how="left")
    panel = panel.merge(unemployment, on=["year", "state"], how="left")

    # ── Violence index (optional) ────────────
    if violence_path and violence_path.exists():
        violence = pd.read_parquet(violence_path)
        panel = panel.merge(violence, on=["year", "state"], how="left")
        panel["violence_index"] = panel["violence_index"].fillna(0.0)

    # ── Lagged features ──────────────────────
    panel = panel.sort_values(["state", "year"])
    panel["vote_share_lag1"] = panel.groupby("state")["vote_share"].shift(1)
    panel["voter_turnout_lag1"] = panel.groupby("state")["turnout"].shift(1)

    logger.info("Panel assembled: {} rows × {} cols", *panel.shape)
    return panel


def save_panel(panel: pd.DataFrame, dest: Path) -> None:
    """Write panel to parquet."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(dest, index=False)
    logger.info("Panel saved to {}", dest)
