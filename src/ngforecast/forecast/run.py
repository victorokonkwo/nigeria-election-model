"""Forecast runner — orchestrates one full dated forecast → artifact."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import typer
import yaml
from loguru import logger

from ngforecast.forecast.schema import CandidateForecast, ForecastArtifact

app = typer.Typer()


def run_forecast(
    config_path: Path = Path("config/config.yaml"),
    model_dir: Path = Path("outputs/models"),
    data_dir: Path = Path("data/processed"),
    out_dir: Path = Path("outputs/forecasts"),
    n_sims: int | None = None,
    seed: int | None = None,
) -> ForecastArtifact:
    """Execute a full forecast run and write the artifact."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    n_sims = n_sims or cfg["n_sims"]
    seed = seed or cfg["seed"]

    logger.info("Starting forecast: n_sims={}, seed={}", n_sims, seed)

    # Load fitted model outputs
    # Load processed panel + W matrix
    # Run simulation engine
    # Apply constitutional rule
    # Package results

    # Placeholder — in production, this calls the full pipeline
    artifact = ForecastArtifact(
        forecast_date=datetime.now(timezone.utc),
        election_date=cfg["election_date"],
        n_simulations=n_sims,
        seed=seed,
        runoff_probability=0.0,
        candidates=[],
    )

    # Write artifact
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = artifact.forecast_date.strftime("%Y-%m-%d")
    dest = out_dir / f"forecast_{date_str}.json"
    dest.write_text(artifact.model_dump_json(indent=2))

    # Also write as "latest"
    latest = out_dir / "latest.json"
    latest.write_text(artifact.model_dump_json(indent=2))

    logger.info("Forecast artifact written to {}", dest)
    return artifact


@app.command()
def main(
    n_sims: int = typer.Option(None, "--n-sims", help="Number of simulations"),
    seed: int = typer.Option(None, "--seed", help="Random seed"),
) -> None:
    """CLI entry point for forecast runner."""
    run_forecast(n_sims=n_sims, seed=seed)


if __name__ == "__main__":
    app()
