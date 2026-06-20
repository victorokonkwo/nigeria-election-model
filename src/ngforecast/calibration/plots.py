"""Calibration and reliability plots for backtesting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger


def calibration_plot(
    predicted: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
    title: str = "Calibration Plot",
    dest: Path | None = None,
) -> plt.Figure:
    """Plot calibration (reliability) diagram.

    Bins predictions and compares mean predicted vs observed frequency.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_means = []
    bin_counts = []

    for i in range(n_bins):
        mask = (predicted >= bins[i]) & (predicted < bins[i + 1])
        if mask.sum() > 0:
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
            bin_means.append(actual[mask].mean())
            bin_counts.append(int(mask.sum()))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), height_ratios=[3, 1])

    # Calibration curve
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
    ax1.plot(bin_centers, bin_means, "o-", color="steelblue", label="Model")
    ax1.set_xlabel("Predicted probability")
    ax1.set_ylabel("Observed frequency")
    ax1.set_title(title)
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Histogram of predictions
    ax2.hist(predicted, bins=bins, color="steelblue", alpha=0.7, edgecolor="white")
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")

    plt.tight_layout()

    if dest:
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=150, bbox_inches="tight")
        logger.info("Calibration plot saved to {}", dest)

    return fig
