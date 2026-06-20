"""Quantal Response Equilibrium (QRE) voter migration model.

Models how voters reallocate from sincere preferences to actual votes
when candidates form coalitions or withdraw, with state-level heterogeneity.
"""

from __future__ import annotations

import numpy as np
from loguru import logger


def qre_reallocation(
    sincere_shares: np.ndarray,
    utility_matrix: np.ndarray,
    lam: float = 2.0,
) -> np.ndarray:
    """Compute QRE vote allocation for a single state.

    Parameters
    ----------
    sincere_shares : (n_candidates_original,) sincere vote shares
    utility_matrix : (n_candidates_original, n_candidates_actual) mapping
        from sincere preference to utility of voting for each actual candidate
    lam : rationality parameter (λ → ∞ = fully strategic, λ → 0 = random)

    Returns
    -------
    actual_shares : (n_candidates_actual,) vote shares after QRE migration
    """
    n_orig, n_actual = utility_matrix.shape

    # QRE choice probabilities: P(j|i) ∝ exp(λ * U(i,j))
    log_probs = lam * utility_matrix
    log_probs -= log_probs.max(axis=1, keepdims=True)  # numerical stability
    probs = np.exp(log_probs)
    probs /= probs.sum(axis=1, keepdims=True)

    # Actual shares = sum over original supporters weighted by migration probs
    actual_shares = sincere_shares @ probs
    actual_shares = np.maximum(actual_shares, 0.0)
    actual_shares /= actual_shares.sum()

    return actual_shares


def apply_qre_all_states(
    sincere_shares_by_state: np.ndarray,
    utility_matrix: np.ndarray,
    lam: float = 2.0,
) -> np.ndarray:
    """Apply QRE reallocation across all states.

    Parameters
    ----------
    sincere_shares_by_state : (n_states, n_candidates_original)
    utility_matrix : (n_candidates_original, n_candidates_actual)
    lam : rationality parameter

    Returns
    -------
    actual_shares_by_state : (n_states, n_candidates_actual)
    """
    n_states = sincere_shares_by_state.shape[0]
    n_actual = utility_matrix.shape[1]
    actual = np.zeros((n_states, n_actual))

    for i in range(n_states):
        actual[i] = qre_reallocation(sincere_shares_by_state[i], utility_matrix, lam)

    logger.info(
        "QRE migration applied: {} states, λ={}, {} → {} candidates",
        n_states,
        lam,
        sincere_shares_by_state.shape[1],
        n_actual,
    )
    return actual
