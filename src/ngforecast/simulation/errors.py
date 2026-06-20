"""Correlated error structure for Monte Carlo draws.

Generates national + zonal + idiosyncratic error terms with
the appropriate correlation structure for election simulation.
"""

from __future__ import annotations

import numpy as np
from loguru import logger


def draw_correlated_errors(
    n_sims: int,
    n_states: int,
    zone_assignments: np.ndarray,
    sigma_national: float = 0.02,
    sigma_zonal: float = 0.03,
    sigma_state: float = 0.04,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw correlated simulation errors at three levels.

    Error structure per sim s, state i:
        ε_{s,i} = η_s (national) + ζ_{s,z(i)} (zonal) + ν_{s,i} (idiosyncratic)

    Parameters
    ----------
    n_sims : number of Monte Carlo simulations
    n_states : number of geographic units (37)
    zone_assignments : (n_states,) integer zone index for each state
    sigma_national : SD of national shock
    sigma_zonal : SD of zonal shock
    sigma_state : SD of idiosyncratic state shock
    rng : numpy random generator

    Returns
    -------
    errors : (n_sims, n_states) correlated error draws
    """
    if rng is None:
        rng = np.random.default_rng()

    n_zones = int(zone_assignments.max()) + 1

    # National shock (same for all states in a sim)
    eta = rng.normal(0, sigma_national, size=(n_sims, 1))

    # Zonal shock (same for states in the same zone)
    zeta_raw = rng.normal(0, sigma_zonal, size=(n_sims, n_zones))
    zeta = zeta_raw[:, zone_assignments]

    # Idiosyncratic state shock
    nu = rng.normal(0, sigma_state, size=(n_sims, n_states))

    errors = eta + zeta + nu

    logger.info(
        "Drew {} × {} correlated errors (σ_nat={}, σ_zone={}, σ_state={})",
        n_sims,
        n_states,
        sigma_national,
        sigma_zonal,
        sigma_state,
    )
    return errors
