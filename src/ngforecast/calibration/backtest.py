"""Backtesting — train ≤ t-1, predict t for historical elections."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


def leave_future_out(
    panel: pd.DataFrame,
    target_years: list[int],
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate train/test splits for temporal backtesting.

    For each target year t, train on all years < t.
    """
    splits = []
    all_years = sorted(panel["year"].unique())

    for t in target_years:
        train = panel[panel["year"] < t].copy()
        test = panel[panel["year"] == t].copy()
        if train.empty or test.empty:
            logger.warning("Skipping year {} — insufficient data", t)
            continue
        splits.append((train, test))
        logger.info(
            "Backtest split: train years={}, test year={}",
            sorted(train["year"].unique()),
            t,
        )

    return splits


def run_backtest(
    panel: pd.DataFrame,
    target_years: list[int],
    fit_fn,
    predict_fn,
) -> pd.DataFrame:
    """Run full backtest loop and collect predictions vs actuals.

    Parameters
    ----------
    panel : full panel dataset
    target_years : years to hold out and predict
    fit_fn : callable(train_df) → model
    predict_fn : callable(model, test_df) → predictions array

    Returns
    -------
    DataFrame with columns: year, state, predicted, actual
    """
    splits = leave_future_out(panel, target_years)
    records = []

    for train, test in splits:
        model = fit_fn(train)
        preds = predict_fn(model, test)
        year = test["year"].iloc[0]

        for i, (_, row) in enumerate(test.iterrows()):
            records.append({
                "year": year,
                "state": row["state"],
                "predicted": float(preds[i]),
                "actual": float(row["vote_share"]),
            })

    results = pd.DataFrame(records)
    logger.info("Backtest complete: {} predictions across {} years", len(results), len(target_years))
    return results
