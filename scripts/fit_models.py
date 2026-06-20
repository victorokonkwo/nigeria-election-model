"""Fit all model layers — fundamentals, GMM, turnout, MRP, ensemble."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from loguru import logger

from ngforecast.features.weights import load_W


def main() -> None:
    load_dotenv()
    model_dir = Path("outputs/models")
    data_dir = Path("data/processed")

    with open("config/model.yaml") as f:
        model_cfg = yaml.safe_load(f)

    logger.info("Fitting models — output to {}", model_dir)

    panel = pd.read_parquet(data_dir / "panel.parquet")
    W = load_W(data_dir / "W.npz")

    # 1. Spatial panel fundamentals
    from ngforecast.models.fundamentals.fit import fit_spatial_panel

    fit_spatial_panel(panel, W, model_cfg["fundamentals"], model_dir / "fundamentals")

    # 2. GMM cross-check
    from ngforecast.models.gmm.fit import fit_gmm

    fit_gmm(panel, model_cfg["gmm"], model_dir / "gmm")

    # 3. Turnout SUR
    from ngforecast.models.turnout.fit import fit_turnout_sur

    fit_turnout_sur(panel, model_cfg["turnout"]["covariates"], model_dir / "turnout")

    # 4. MRP (if polls available)
    from ngforecast.data.polls import load_polls
    from ngforecast.models.mrp.fit import fit_mrp

    polls = load_polls(Path("data/raw/polls"))
    strata = pd.read_parquet(data_dir / "strata.parquet")
    fit_mrp(polls, strata, out_dir=model_dir / "mrp")

    logger.info("All models fitted")


if __name__ == "__main__":
    main()
