"""NBS data ingestion — poverty, unemployment, CPI by state."""

from __future__ import annotations

from pathlib import Path

import httpx
import pandas as pd
from loguru import logger

from ngforecast.data._utils import load_source_config, stamp_vintage


def fetch_nbs(out_dir: Path) -> list[Path]:
    """Pull NBS macro indicators and write to raw/nbs/."""
    cfg = load_source_config("nbs")
    base = cfg["base_url"]
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    with httpx.Client(timeout=60) as client:
        for name, endpoint in cfg["endpoints"].items():
            url = f"{base}{endpoint}"
            logger.info("Fetching NBS {} from {}", name, url)
            resp = client.get(url)
            resp.raise_for_status()
            dest = out_dir / f"{name}.csv"
            dest.write_text(resp.text)
            paths.append(dest)

    stamp_vintage(out_dir, cfg["vintage"])
    return paths


def load_poverty(path: Path) -> pd.DataFrame:
    """Load state-level poverty data."""
    return pd.read_csv(path)


def load_unemployment(path: Path) -> pd.DataFrame:
    """Load state-level unemployment data."""
    return pd.read_csv(path)


def load_cpi(path: Path) -> pd.DataFrame:
    """Load CPI data."""
    return pd.read_csv(path)
