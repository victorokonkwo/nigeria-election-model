"""Internal data utilities — config loading, vintage stamping."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


def load_source_config(source_name: str) -> dict:
    """Load a single source block from config/data_sources.yaml."""
    cfg_path = Path("config/data_sources.yaml")
    with open(cfg_path) as f:
        all_sources = yaml.safe_load(f)
    if source_name not in all_sources:
        raise KeyError(f"Source '{source_name}' not found in {cfg_path}")
    return all_sources[source_name]


def stamp_vintage(directory: Path, vintage: str) -> None:
    """Write a vintage metadata file into the given directory."""
    meta = {
        "vintage": vintage,
        "pulled_at": datetime.now(timezone.utc).isoformat(),
    }
    (directory / "_vintage.json").write_text(json.dumps(meta, indent=2))
