"""Ensemble model — Bayesian stacking of component forecasts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger


def bayesian_stacking(
    component_predictions: dict[str, np.ndarray],
    observed: np.ndarray,
    method: str = "bayesian_stacking",
) -> dict[str, float]:
    """Compute optimal stacking weights from leave-one-out predictions.

    Parameters
    ----------
    component_predictions : dict mapping model name → (n_obs,) predicted values
    observed : (n_obs,) actual outcomes
    method : "bayesian_stacking" | "simple_average" | "bma"

    Returns
    -------
    dict of {model_name: weight}
    """
    names = list(component_predictions.keys())
    K = len(names)

    if method == "simple_average":
        weights = {name: 1.0 / K for name in names}
        logger.info("Ensemble: simple average — {} components", K)
        return weights

    # Stack predictions into matrix (n_obs, K)
    pred_matrix = np.column_stack([component_predictions[n] for n in names])
    n = len(observed)

    # Log predictive density for each model (Gaussian approximation)
    residuals = pred_matrix - observed[:, None]
    sigma2 = np.var(residuals, axis=0)
    sigma2 = np.maximum(sigma2, 1e-10)
    log_lik = -0.5 * n * np.log(2 * np.pi * sigma2) - 0.5 * np.sum(
        residuals**2 / sigma2, axis=0
    )

    # Stacking via softmax of log-likelihoods
    log_lik -= log_lik.max()
    w = np.exp(log_lik)
    w /= w.sum()

    weights = dict(zip(names, w.tolist()))
    logger.info("Ensemble stacking weights: {}", weights)
    return weights


def ensemble_predict(
    component_predictions: dict[str, np.ndarray],
    weights: dict[str, float],
) -> np.ndarray:
    """Weighted combination of component predictions."""
    result = np.zeros_like(next(iter(component_predictions.values())))
    for name, pred in component_predictions.items():
        result += weights.get(name, 0.0) * pred
    return result
