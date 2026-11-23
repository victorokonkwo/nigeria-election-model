"""Raw → interim: tidy, pre-election-dated covariates keyed by (year, state).

Writes data/interim/{poverty,macro,violence_index}.parquet.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from ngforecast.data._utils import load_election_dates, load_source_config


def main() -> None:
    raw, interim = Path("data/raw"), Path("data/interim")
    interim.mkdir(parents=True, exist_ok=True)
    dates = load_election_dates()

    from ngforecast.data.poverty import build_poverty, poverty_asof

    poverty = poverty_asof(build_poverty(raw / "poverty" / "nga_mpi_trends.csv"), dates)
    poverty.to_parquet(interim / "poverty.parquet", index=False)
    logger.info("poverty: {} rows, {} with data", len(poverty), poverty["mpi"].notna().sum())

    from ngforecast.data.macro import build_macro

    macro = build_macro(raw / "macro", dates)
    macro.to_parquet(interim / "macro.parquet", index=False)
    logger.info("macro: {} elections", len(macro))

    from ngforecast.data.acled import build_violence_index, load_acled

    window = load_source_config("acled")["window_months"]
    violence = build_violence_index(
        load_acled(raw / "acled" / "acled_political_violence.xlsx"), dates, window
    )
    violence.to_parquet(interim / "violence_index.parquet", index=False)
    logger.info("violence: {} state-years", len(violence))


if __name__ == "__main__":
    main()
