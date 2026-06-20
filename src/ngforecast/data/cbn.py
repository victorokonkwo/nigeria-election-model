"""CBN data ingestion — inflation, exchange rate, money supply."""

from __future__ import annotations

from pathlib import Path

import httpx
import pandas as pd
from loguru import logger

from ngforecast.data._utils import load_source_config, stamp_vintage


def fetch_cbn(out_dir: Path) -> list[Path]:
    """Pull CBN macro indicators and write to raw/cbn/."""
    cfg = load_source_config("cbn")
    base = cfg["base_url"]
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    with httpx.Client(timeout=60) as client:
        for name, endpoint in cfg["endpoints"].items():
            url = f"{base}{endpoint}"
            logger.info("Fetching CBN {} from {}", name, url)
            resp = client.get(url)
            resp.raise_for_status()
            dest = out_dir / f"{name}.json"
            dest.write_text(resp.text)
            paths.append(dest)

    stamp_vintage(out_dir, cfg["vintage"])
    return paths


def load_inflation(path: Path) -> pd.DataFrame:
    """Load inflation time series."""
    return pd.read_json(path)


def load_exchange_rate(path: Path) -> pd.DataFrame:
    """Load exchange rate time series."""
    return pd.read_json(path)
