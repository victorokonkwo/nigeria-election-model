"""SUR turnout model — jointly estimates turnout and vote share."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


def fit_turnout_sur(
    panel: pd.DataFrame,
    covariates: list[str],
    out_dir: Path,
) -> dict:
    """Fit SUR turnout equation jointly with vote share.

    Uses seemingly unrelated regression to capture correlation
    between turnout and vote share at the state level.

    Returns coefficient dict for both equations.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare design matrices
    panel_clean = panel.dropna(subset=covariates + ["turnout", "vote_share"])
    X = panel_clean[covariates].values
    y_turnout = panel_clean["turnout"].values
    y_share = panel_clean["vote_share"].values

    # SUR via feasible GLS
    n = len(y_turnout)
    k = X.shape[1]

    # First-stage OLS for each equation
    XtX_inv = np.linalg.inv(X.T @ X)
    beta_turnout = XtX_inv @ X.T @ y_turnout
    beta_share = XtX_inv @ X.T @ y_share

    # Residuals
    e_turnout = y_turnout - X @ beta_turnout
    e_share = y_share - X @ beta_share

    # Cross-equation covariance
    sigma = np.array([
        [e_turnout @ e_turnout / n, e_turnout @ e_share / n],
        [e_share @ e_turnout / n, e_share @ e_share / n],
    ])

    results = {
        "coefficients_turnout": dict(zip(covariates, beta_turnout.tolist())),
        "coefficients_share": dict(zip(covariates, beta_share.tolist())),
        "sigma": sigma.tolist(),
        "n_obs": n,
    }

    import json

    dest = out_dir / "turnout_sur.json"
    dest.write_text(json.dumps(results, indent=2))
    logger.info("SUR turnout model fitted — {} obs, {} covariates", n, k)
    return results
