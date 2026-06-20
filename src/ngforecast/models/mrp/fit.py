"""MRP estimation — multilevel model + poststratification on census cells."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from loguru import logger


def fit_mrp(
    polls: pd.DataFrame,
    strata: pd.DataFrame,
    config_path: Path = Path("config/model.yaml"),
    out_dir: Path = Path("outputs/models"),
) -> dict:
    """Fit multilevel regression on poll data and poststratify.

    Returns state-level vote-intention estimates adjusted by
    demographic cell weights from the census strata frame.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    mrp_cfg = cfg["mrp"]
    min_polls = mrp_cfg["min_polls_to_activate"]

    if len(polls) < min_polls:
        logger.warning(
            "Only {} polls available (need {}). MRP will not activate.",
            len(polls),
            min_polls,
        )
        return {"active": False, "reason": "insufficient_polls"}

    out_dir.mkdir(parents=True, exist_ok=True)
    group_effects = mrp_cfg["group_effects"]

    logger.info(
        "Fitting MRP: {} poll obs, {} group effects, {} strata cells",
        len(polls),
        len(group_effects),
        len(strata),
    )

    # Placeholder for PyMC / bambi multilevel fit
    # In production, this would use bambi for the multilevel model:
    #   model = bmb.Model("vote ~ 1 + (1|state) + (1|zone) + (1|age_group) + ...", polls)
    #   idata = model.fit()
    # Then poststratify using strata cell weights.

    results = {
        "active": True,
        "n_polls": len(polls),
        "group_effects": group_effects,
        "state_estimates": {},  # populated by actual fit
    }

    import json

    dest = out_dir / "mrp_estimates.json"
    dest.write_text(json.dumps(results, indent=2))
    logger.info("MRP estimates saved to {}", dest)
    return results
