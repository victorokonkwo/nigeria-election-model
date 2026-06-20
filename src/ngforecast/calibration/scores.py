"""Scoring metrics — Brier, log score, RMSE, coverage."""

from __future__ import annotations

import numpy as np
from loguru import logger


def brier_score(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Brier score: mean squared error of probability predictions."""
    bs = float(np.mean((predicted - actual) ** 2))
    logger.info("Brier score: {:.6f}", bs)
    return bs


def log_score(predicted: np.ndarray, actual: np.ndarray, eps: float = 1e-10) -> float:
    """Mean log score (higher = better). Assumes binary-like actual in [0,1]."""
    p = np.clip(predicted, eps, 1 - eps)
    ls = float(np.mean(actual * np.log(p) + (1 - actual) * np.log(1 - p)))
    logger.info("Log score: {:.6f}", ls)
    return ls


def rmse(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Root mean squared error."""
    r = float(np.sqrt(np.mean((predicted - actual) ** 2)))
    logger.info("RMSE: {:.6f}", r)
    return r


def mae(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Mean absolute error."""
    m = float(np.mean(np.abs(predicted - actual)))
    logger.info("MAE: {:.6f}", m)
    return m


def coverage(
    lower: np.ndarray,
    upper: np.ndarray,
    actual: np.ndarray,
    nominal: float = 0.90,
) -> float:
    """Empirical coverage of prediction intervals.

    Parameters
    ----------
    lower, upper : (n,) lower and upper bounds of intervals
    actual : (n,) actual values
    nominal : nominal coverage level (for logging)

    Returns
    -------
    Fraction of actuals falling within [lower, upper]
    """
    in_interval = np.mean((actual >= lower) & (actual <= upper))
    cov = float(in_interval)
    logger.info("Coverage: {:.3f} (nominal: {:.3f})", cov, nominal)
    return cov


def score_summary(predicted: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    """Compute all scalar scores and return as dict."""
    return {
        "brier": brier_score(predicted, actual),
        "log_score": log_score(predicted, actual),
        "rmse": rmse(predicted, actual),
        "mae": mae(predicted, actual),
    }
