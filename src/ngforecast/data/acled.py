"""ACLED political violence — HDX admin2 x month aggregates (keyless).

ACLED's Terms of Use forbid redistributing its data. The downloaded file
stays in the gitignored data/ tree; only derived state-year indices are
published, with attribution ("ACLED, www.acleddata.com").
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from ngforecast.data._utils import (
    download,
    hdx_resource_urls,
    load_source_config,
    write_provenance,
)
from ngforecast.data.geography import state_code

_MONTHS = {
    m: i
    for i, m in enumerate(
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        start=1,
    )
}


def fetch_acled(out_dir: Path) -> Path:
    """Download ACLED political-violence events/fatalities by admin2 x month from HDX."""
    cfg = load_source_config("acled")
    out_dir.mkdir(parents=True, exist_ok=True)

    matches = [
        r
        for r in hdx_resource_urls(cfg["hdx_dataset"])
        if cfg["resource_match"] in r["url"].lower()
    ]
    if not matches:
        raise RuntimeError(f"No HDX resource matching {cfg['resource_match']!r}")

    dest = out_dir / "acled_political_violence.xlsx"
    logger.info("Fetching ACLED aggregates from {}", matches[0]["url"])
    meta = download(matches[0]["url"], dest, timeout=300)
    meta["last_modified"] = matches[0]["last_modified"]
    write_provenance(out_dir, cfg["citation"], [meta])
    return dest


def load_acled(path: Path) -> pd.DataFrame:
    """State x month events and fatalities. Columns: state, month, events, fatalities."""
    raw = pd.read_excel(path, sheet_name="Data")
    raw["month"] = pd.to_datetime(
        {"year": raw["Year"], "month": raw["Month"].map(_MONTHS), "day": 1}
    )
    if raw["month"].isna().any():
        raise ValueError("Unparseable ACLED month values")
    raw["state"] = raw["Admin1"].map({n: state_code(n) for n in raw["Admin1"].unique()})
    return (
        raw.groupby(["state", "month"], as_index=False)[["Events", "Fatalities"]]
        .sum()
        .rename(columns={"Events": "events", "Fatalities": "fatalities"})
    )


def build_violence_index(
    monthly: pd.DataFrame,
    election_dates: dict[int, str],
    window_months: int = 12,
) -> pd.DataFrame:
    """Violence in the `window_months` full months before each election month.

    The election month itself is excluded so the index is known before
    election day. For an election beyond the data, the window is the last
    `window_months` complete months (the final, possibly partial, month of the
    file is dropped). Returns state x year rows with events, fatalities,
    violence_index = log(1 + events) + log(1 + fatalities), and the window end.
    """
    first_month = monthly["month"].min()
    last_complete = monthly["month"].max()  # exclusive bound drops the partial last month
    rows = []
    for year, date in sorted(election_dates.items()):
        end = pd.Timestamp(date).to_period("M").to_timestamp()  # start of election month
        end = min(end, last_complete)
        start = end - pd.DateOffset(months=window_months)
        if start < first_month:
            logger.warning("ACLED does not cover the full window before {}", year)
            continue
        window = monthly[(monthly["month"] >= start) & (monthly["month"] < end)]
        agg = window.groupby("state")[["events", "fatalities"]].sum()
        agg = agg.reindex(sorted(monthly["state"].unique()), fill_value=0)
        for state, r in agg.iterrows():
            rows.append(
                {
                    "year": year,
                    "state": state,
                    "violence_events": int(r["events"]),
                    "violence_fatalities": int(r["fatalities"]),
                    "violence_index": float(np.log1p(r["events"]) + np.log1p(r["fatalities"])),
                    "violence_window_end": end,
                }
            )
    return pd.DataFrame(rows)
