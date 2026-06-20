"""Append-only, timestamped forecast log — public auditable record."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from ngforecast.forecast.schema import ForecastLogEntry


LOG_PATH = Path("outputs/forecasts/forecast_log.jsonl")


def append_to_log(
    win_probabilities: dict[str, float],
    runoff_probability: float,
    model_version: str,
    data_vintage: str,
    notes: str = "",
    log_path: Path = LOG_PATH,
) -> ForecastLogEntry:
    """Append a new entry to the forecast log (JSONL format).

    This log is append-only and timestamped for auditability.
    Each line is a self-contained JSON object.
    """
    entry = ForecastLogEntry(
        forecast_date=datetime.now(timezone.utc),
        win_probabilities=win_probabilities,
        runoff_probability=runoff_probability,
        model_version=model_version,
        data_vintage=data_vintage,
        notes=notes,
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(entry.model_dump_json() + "\n")

    logger.info("Forecast log entry appended: {}", entry.forecast_date.isoformat())
    return entry


def read_log(log_path: Path = LOG_PATH) -> list[ForecastLogEntry]:
    """Read all entries from the forecast log."""
    if not log_path.exists():
        return []

    entries = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(ForecastLogEntry.model_validate_json(line))

    logger.info("Read {} entries from forecast log", len(entries))
    return entries
