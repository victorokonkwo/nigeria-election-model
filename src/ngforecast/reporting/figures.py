"""Reporting figures — publication-quality charts for forecast output."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger


def win_probability_bar(
    candidates: list[str],
    probabilities: list[float],
    dest: Path | None = None,
) -> plt.Figure:
    """Horizontal bar chart of win probabilities."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(candidates)))
    y_pos = range(len(candidates))

    ax.barh(y_pos, probabilities, color=colors, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(candidates)
    ax.set_xlabel("Win Probability")
    ax.set_title("Presidential Election — Win Probabilities")
    ax.set_xlim(0, 1)

    for i, p in enumerate(probabilities):
        ax.text(p + 0.01, i, f"{p:.1%}", va="center")

    plt.tight_layout()
    if dest:
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=150, bbox_inches="tight")
        logger.info("Win probability chart saved to {}", dest)
    return fig


def state_map(
    state_shares: dict[str, float],
    title: str = "Vote Share by State",
    dest: Path | None = None,
) -> plt.Figure:
    """Choropleth-style table of vote shares by state (placeholder for geopandas map)."""
    fig, ax = plt.subplots(figsize=(12, 8))
    states = list(state_shares.keys())
    shares = list(state_shares.values())

    ax.barh(states, shares, color="steelblue", edgecolor="white")
    ax.set_xlabel("Vote Share")
    ax.set_title(title)
    plt.tight_layout()

    if dest:
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=150, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    # When called as module: generate all figures from latest forecast
    import json

    forecast_path = Path("outputs/forecasts/latest.json")
    if not forecast_path.exists():
        logger.error("No forecast found at {}", forecast_path)
        raise SystemExit(1)

    with open(forecast_path) as f:
        forecast = json.load(f)

    fig_dir = Path("outputs/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    candidates = [c["candidate_id"] for c in forecast.get("candidates", [])]
    probs = [c["win_probability"] for c in forecast.get("candidates", [])]

    if candidates:
        win_probability_bar(candidates, probs, dest=fig_dir / "win_probabilities.png")

    # Calibration placeholder
    (fig_dir / "calibration.png").touch()
    logger.info("Figures generated in {}", fig_dir)
