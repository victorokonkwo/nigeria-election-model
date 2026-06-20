"""Fit spatial panel model via R's splm package (subprocess bridge)."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from loguru import logger


def fit_spatial_panel(
    panel: pd.DataFrame,
    W: np.ndarray,
    model_config: dict,
    out_dir: Path,
) -> dict:
    """Fit SAR/SEM spatial panel in R and return coefficient estimates.

    Writes panel + W to temp files, calls R/spatial_panel.R,
    reads back the JSON results.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write inputs for R
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        panel_path = tmp / "panel.csv"
        w_path = tmp / "W.csv"
        result_path = tmp / "results.json"

        panel.to_csv(panel_path, index=False)
        np.savetxt(w_path, W, delimiter=",")

        covariates = ",".join(model_config.get("covariates", []))
        cmd = [
            "Rscript",
            "R/spatial_panel.R",
            str(panel_path),
            str(w_path),
            str(result_path),
            model_config.get("model", "SAR"),
            model_config.get("estimator", "ML"),
            model_config.get("effects", "twoways"),
            covariates,
        ]

        logger.info("Running spatial panel: {}", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if proc.returncode != 0:
            logger.error("R spatial panel failed:\n{}", proc.stderr)
            raise RuntimeError(f"Spatial panel estimation failed: {proc.stderr[:500]}")

        with open(result_path) as f:
            results = json.load(f)

    # Persist
    results_dest = out_dir / "fundamentals_coefficients.json"
    results_dest.write_text(json.dumps(results, indent=2))
    logger.info("Spatial panel fitted — {} coefficients saved", len(results.get("coefficients", {})))
    return results
