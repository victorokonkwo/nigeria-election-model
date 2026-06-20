"""Pre-election polls ingestion (NOI Polls, Afrobarometer, ad hoc)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from ngforecast.data._utils import load_source_config


def load_polls(polls_dir: Path) -> pd.DataFrame:
    """Load and concatenate all available poll files from raw/polls/."""
    cfg = load_source_config("polls")
    frames = []

    for source in cfg["sources"]:
        source_dir = Path(source["path"])
        if not source_dir.exists():
            logger.warning("Poll source dir not found: {}", source_dir)
            continue
        for csv_file in sorted(source_dir.glob("*.csv")):
            logger.info("Loading poll: {}", csv_file.name)
            df = pd.read_csv(csv_file)
            df["source"] = source["name"]
            frames.append(df)

    if not frames:
        logger.warning("No poll data found")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Loaded {} poll observations from {} files", len(combined), len(frames))
    return combined
