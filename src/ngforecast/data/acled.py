"""ACLED data ingestion — political violence events for Nigeria."""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pandas as pd
from loguru import logger

from ngforecast.data._utils import load_source_config, stamp_vintage


def fetch_acled(out_dir: Path) -> Path:
    """Pull ACLED political violence data for Nigeria."""
    cfg = load_source_config("acled")
    out_dir.mkdir(parents=True, exist_ok=True)

    key = os.environ.get("ACLED_ACCESS_KEY", "")
    email = os.environ.get("ACLED_EMAIL", "")
    if not key or not email:
        logger.warning("ACLED credentials not set — request may fail")

    params = {
        **cfg["params"],
        "key": key,
        "email": email,
    }

    logger.info("Fetching ACLED data from {}", cfg["base_url"])
    with httpx.Client(timeout=120) as client:
        resp = client.get(cfg["base_url"], params=params)
        resp.raise_for_status()

    dest = out_dir / "acled_nigeria.json"
    dest.write_text(resp.text)
    stamp_vintage(out_dir, cfg["vintage"])
    logger.info("ACLED data saved to {}", dest)
    return dest


def load_acled(path: Path) -> pd.DataFrame:
    """Load ACLED events into a DataFrame."""
    return pd.read_json(path)


def build_violence_index(events: pd.DataFrame, by: str = "state") -> pd.DataFrame:
    """Aggregate ACLED events into a violence index by state and year."""
    if events.empty:
        return pd.DataFrame(columns=[by, "year", "violence_index"])

    grouped = (
        events.groupby([by, "year"])
        .agg(n_events=("event_id_cnty", "count"), fatalities=("fatalities", "sum"))
        .reset_index()
    )
    # Simple composite: log(1 + events) + log(1 + fatalities)
    import numpy as np

    grouped["violence_index"] = np.log1p(grouped["n_events"]) + np.log1p(
        grouped["fatalities"]
    )
    return grouped[[by, "year", "violence_index"]]
