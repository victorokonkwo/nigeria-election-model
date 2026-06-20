"""Build processed panel, W matrix, and strata frame from interim data."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from loguru import logger


def main() -> None:
    load_dotenv()
    processed = Path("data/processed")
    interim = Path("data/interim")
    external = Path("data/external")

    logger.info("Building panel from interim data")

    from ngforecast.features.panel import build_panel, save_panel

    panel = build_panel(
        results_path=interim / "results.parquet",
        macro_dir=interim,
        violence_path=interim / "violence_index.parquet",
    )
    save_panel(panel, processed / "panel.parquet")

    from ngforecast.features.weights import build_W, save_W

    W = build_W()
    save_W(W, processed / "W.npz")

    from ngforecast.features.strata import build_strata_frame, save_strata

    strata = build_strata_frame(external / "census_marginals.csv")
    save_strata(strata, processed / "strata.parquet")

    logger.info("Panel build complete")


if __name__ == "__main__":
    main()
