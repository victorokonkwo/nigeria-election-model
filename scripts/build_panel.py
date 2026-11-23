"""Build the processed panel (historical elections + the 2027 forecast rows) and W."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger


def main() -> None:
    processed, interim = Path("data/processed"), Path("data/interim")

    from ngforecast.data.inec_historical import load_results, load_totals
    from ngforecast.features.panel import build_panel, load_candidates, save_panel

    covariates = {
        "violence": pd.read_parquet(interim / "violence_index.parquet"),
        "poverty": pd.read_parquet(interim / "poverty.parquet"),
    }
    results, totals, cands = load_results(), load_totals(), load_candidates()
    hist = build_panel(results, totals, cands, covariates)
    future = build_panel(results, totals, cands, covariates, years=[2027])
    panel = pd.concat([hist, future], ignore_index=True)
    save_panel(panel, processed / "panel.parquet")

    from ngforecast.features.weights import build_W, save_W

    save_W(build_W(), processed / "W.npz")
    logger.info("Panel build complete: years {}", sorted(panel.year.unique()))


if __name__ == "__main__":
    main()
