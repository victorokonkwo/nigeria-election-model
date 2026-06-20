"""Oil price and production data ingestion."""

from __future__ import annotations

from pathlib import Path

import httpx
import pandas as pd
from loguru import logger

from ngforecast.data._utils import load_source_config, stamp_vintage


def fetch_oil(out_dir: Path) -> list[Path]:
    """Pull oil price data from EIA / OPEC sources."""
    cfg = load_source_config("oil")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    with httpx.Client(timeout=60) as client:
        for source in cfg["sources"]:
            url = source["url"]
            name = source["name"]
            logger.info("Fetching oil data from {} ({})", name, url)
            resp = client.get(url)
            resp.raise_for_status()
            dest = out_dir / f"{name}.json"
            dest.write_text(resp.text)
            paths.append(dest)

    stamp_vintage(out_dir, cfg["vintage"])
    return paths


def load_brent(path: Path) -> pd.DataFrame:
    """Load Brent crude price series."""
    return pd.read_json(path)
