"""Fit GMM dynamic panel model via R's plm/pdynmc package."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


def fit_gmm(
    panel: pd.DataFrame,
    model_config: dict,
    out_dir: Path,
) -> dict:
    """Fit system-GMM or diff-GMM in R and return estimates."""
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        panel_path = tmp / "panel.csv"
        result_path = tmp / "results.json"

        panel.to_csv(panel_path, index=False)

        cmd = [
            "Rscript",
            "R/gmm.R",
            str(panel_path),
            str(result_path),
            model_config.get("model", "system_gmm"),
            str(model_config.get("lags", 2)),
        ]

        logger.info("Running GMM: {}", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if proc.returncode != 0:
            logger.error("R GMM failed:\n{}", proc.stderr)
            raise RuntimeError(f"GMM estimation failed: {proc.stderr[:500]}")

        with open(result_path) as f:
            results = json.load(f)

    results_dest = out_dir / "gmm_coefficients.json"
    results_dest.write_text(json.dumps(results, indent=2))
    logger.info("GMM fitted — {} coefficients", len(results.get("coefficients", {})))
    return results
