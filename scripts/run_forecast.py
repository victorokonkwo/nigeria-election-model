"""Run a dated forecast — thin wrapper around ngforecast.forecast.run."""

from __future__ import annotations

import typer
from dotenv import load_dotenv

from ngforecast.forecast.run import run_forecast

app = typer.Typer()


@app.command()
def main(
    n_sims: int = typer.Option(None, "--n-sims", help="Number of simulations"),
    seed: int = typer.Option(None, "--seed", help="Random seed"),
) -> None:
    """Produce a dated forecast artifact."""
    load_dotenv()
    run_forecast(n_sims=n_sims, seed=seed)


if __name__ == "__main__":
    app()
