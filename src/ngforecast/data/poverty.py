"""State-level poverty — OPHI Multidimensional Poverty Index via HDX (keyless).

OPHI computes the MPI by state from NPC/DHS (2013, 2018) and NBS/UNICEF MICS
(2016-17, 2021) household surveys. There is no comparable state-level series
before 2013, so the 1999-2011 elections have no poverty covariate.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from ngforecast.data._utils import (
    download,
    hdx_resource_urls,
    load_source_config,
    write_provenance,
)
from ngforecast.data.geography import state_code


def fetch_poverty(out_dir: Path) -> Path:
    """Download the OPHI state-level MPI trends file into raw/poverty/."""
    cfg = load_source_config("poverty")
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = cfg["files"]["mpi_trends"]

    matches = [r for r in hdx_resource_urls(cfg["hdx_dataset"]) if r["url"].endswith(fname)]
    if not matches:
        raise RuntimeError(f"HDX dataset {cfg['hdx_dataset']} has no resource named {fname}")

    dest = out_dir / fname
    logger.info("Fetching OPHI MPI from {}", matches[0]["url"])
    meta = download(matches[0]["url"], dest)
    meta["last_modified"] = matches[0]["last_modified"]
    write_provenance(out_dir, cfg["citation"], [meta])
    return dest


def build_poverty(raw_path: Path) -> pd.DataFrame:
    """Tidy state x survey poverty table.

    Columns: state, survey, survey_end, mpi, headcount, severe_poverty
    (headcount and severe_poverty are fractions in [0, 1]).
    """
    raw = pd.read_csv(raw_path)
    raw = raw[raw["Admin 1 Name"].notna()].copy()  # drop national rows
    out = pd.DataFrame(
        {
            "state": raw["Admin 1 Name"].map(
                {n: state_code(n) for n in raw["Admin 1 Name"].unique()}
            ),
            "survey": raw["Survey"] + " " + raw["Start Date"].str[:4],
            "survey_end": pd.to_datetime(raw["End Date"], utc=True).dt.tz_localize(None),
            "mpi": raw["MPI"].astype(float),
            "headcount": raw["Headcount Ratio"].astype(float) / 100.0,
            "severe_poverty": raw["In Severe Poverty"].astype(float) / 100.0,
        }
    )
    return out.sort_values(["state", "survey_end"]).reset_index(drop=True)


def poverty_asof(poverty: pd.DataFrame, election_dates: dict[int, str]) -> pd.DataFrame:
    """For each state x election, the latest survey that ended before election day.

    Elections with no prior survey get NaN (not back-filled from later surveys,
    which would leak future information into the backtests).
    """
    rows = []
    for year, date in sorted(election_dates.items()):
        cutoff = pd.Timestamp(date)
        prior = poverty[poverty["survey_end"] < cutoff]
        latest = prior.sort_values("survey_end").groupby("state").tail(1)
        for state in sorted(poverty["state"].unique()):
            match = latest[latest["state"] == state]
            if match.empty:
                rows.append({"year": year, "state": state})
            else:
                r = match.iloc[0]
                rows.append(
                    {
                        "year": year,
                        "state": state,
                        "mpi": r["mpi"],
                        "poverty_headcount": r["headcount"],
                        "severe_poverty": r["severe_poverty"],
                        "poverty_survey": r["survey"],
                    }
                )
    return pd.DataFrame(rows)
