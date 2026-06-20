"""INEC data ingestion — historical presidential results by state/LGA + voter register."""

from __future__ import annotations

from pathlib import Path

import httpx
import pandas as pd
from loguru import logger

from ngforecast.data._utils import load_source_config, stamp_vintage


def fetch_results(out_dir: Path) -> Path:
    """Pull presidential election results from INEC API and write to raw/."""
    cfg = load_source_config("inec")
    base = cfg["base_url"]
    out_dir.mkdir(parents=True, exist_ok=True)

    url = f"{base}{cfg['endpoints']['results']}"
    logger.info("Fetching INEC results from {}", url)

    with httpx.Client(timeout=60) as client:
        resp = client.get(url, headers=_auth_header(cfg))
        resp.raise_for_status()

    dest = out_dir / "presidential_results.csv"
    dest.write_text(resp.text)
    stamp_vintage(out_dir, cfg["vintage"])
    logger.info("INEC results saved to {}", dest)
    return dest


def fetch_register(out_dir: Path) -> Path:
    """Pull voter registration data from INEC API."""
    cfg = load_source_config("inec")
    base = cfg["base_url"]
    out_dir.mkdir(parents=True, exist_ok=True)

    url = f"{base}{cfg['endpoints']['register']}"
    logger.info("Fetching voter register from {}", url)

    with httpx.Client(timeout=60) as client:
        resp = client.get(url, headers=_auth_header(cfg))
        resp.raise_for_status()

    dest = out_dir / "voter_register.csv"
    dest.write_text(resp.text)
    logger.info("Voter register saved to {}", dest)
    return dest


def load_results(path: Path) -> pd.DataFrame:
    """Load cleaned presidential results into a DataFrame."""
    df = pd.read_csv(path)
    required = {"year", "state", "party", "votes"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"INEC results missing columns: {missing}")
    return df


def _auth_header(cfg: dict) -> dict[str, str]:
    """Build auth header from environment."""
    import os

    key = os.environ.get("INEC_API_KEY", "")
    if not key:
        logger.warning("INEC_API_KEY not set — request may fail")
    return {"Authorization": f"Bearer {key}"}
