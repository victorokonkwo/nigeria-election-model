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


# ── Stacking of predictive distributions (Yao, Vehtari, Simpson & Gelman 2018) ──


def gaussian_logit_logdens(draws: np.ndarray, actual: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Log predictive density of `actual` shares under each state's draws.

    Each state's predictive distribution is approximated by a Gaussian on the
    logit scale (shares are clipped to [eps, 1 - eps]). The Jacobian of the
    logit transform is common to all models, so it is omitted: it does not
    affect stacking weights.

    draws : (S, n) predictive draws of shares; actual : (n,) -> (n,) log densities
    """
    clip = lambda p: np.clip(p, eps, 1 - eps)  # noqa: E731
    z = np.log(clip(draws) / (1 - clip(draws)))
    y = np.log(clip(actual) / (1 - clip(actual)))
    mu, sd = z.mean(axis=0), np.maximum(z.std(axis=0), 1e-3)
    return -0.5 * np.log(2 * np.pi * sd**2) - 0.5 * ((y - mu) / sd) ** 2


def stack_densities(log_dens: np.ndarray, names: list[str]) -> dict[str, float]:
    """Stacking weights maximising sum_i log sum_k w_k p_k(y_i).

    log_dens : (n_obs, K) log predictive densities of held-out outcomes.
    """
    from scipy.optimize import minimize
    from scipy.special import logsumexp, softmax

    k = log_dens.shape[1]
    if k == 1:
        return {names[0]: 1.0}

    def neg(theta):
        logw = np.log(softmax(np.append(theta, 0.0)) + 1e-300)
        return -np.sum(logsumexp(log_dens + logw[None, :], axis=1))

    res = minimize(neg, np.zeros(k - 1), method="L-BFGS-B")
    w = softmax(np.append(res.x, 0.0))
    weights = dict(zip(names, w.tolist(), strict=True))
    logger.info("Stacking weights: {}", {n: round(v, 3) for n, v in weights.items()})
    return weights


def mixture_draws(
    draws: dict[str, np.ndarray], weights: dict[str, float], n_draws: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample whole-draw vectors from the weighted mixture of models' draws."""
    names = list(draws)
    p = np.array([weights.get(n, 0.0) for n in names])
    p = p / p.sum()
    choice = rng.choice(len(names), size=n_draws, p=p)
    out = np.empty((n_draws, draws[names[0]].shape[1]))
    for j, name in enumerate(names):
        idx = np.flatnonzero(choice == j)
        if idx.size:
            src = draws[name]
            out[idx] = src[rng.integers(0, src.shape[0], size=idx.size)]
    return out
