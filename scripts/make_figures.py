"""Paper figures from the result-of-record outputs (backtest v2, 2027 pre-registration).

Writes PDF + PNG to outputs/paper/figures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BACKTEST = Path("outputs/backtest/v2")
PREREG = Path("outputs/preregistered/2027")
OUT = Path("outputs/paper/figures")

# Reference categorical palette, slots 1-3 in fixed order (dataviz palette.md)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = {"stack": BLUE, "uniform_swing": ORANGE, "ps_null": AQUA}
LABELS = {
    "stack": "Prequential stack",
    "uniform_swing": "Uniform swing",
    "ps_null": "Probabilistic uniform swing",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "legend.frameon": False,
    }
)


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_backtest_skill() -> None:
    """Fig. 1: state-level CRPS by election, own vs true national share."""
    m = pd.read_csv(BACKTEST / "metrics.csv")
    years = [2015, 2019, 2023]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6), sharey=True)
    for ax, mode, title in zip(
        axes, ["forecast", "oracle"], ["Model's own national forecast", "True national share (oracle)"], strict=True
    ):
        width = 0.26
        for k, model in enumerate(SERIES):
            vals = [
                m[(m.model == model) & (m.national == mode) & (m.year == y)].state_crps_pp.iloc[0] for y in years
            ]
            x = np.arange(len(years)) + (k - 1) * width
            ax.bar(x, vals, width=width - 0.03, color=SERIES[model], label=LABELS[model], zorder=2)
        ax.set_xticks(range(len(years)), [str(y) for y in years])
        ax.set_title(title, fontsize=8, color=INK, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Mean state CRPS (pp, lower is better)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    save(fig, "fig1_backtest_crps")


def fig_backtest_intervals() -> None:
    """Fig. 2: stack 90% intervals vs outcomes, oracle national, per backtest year."""
    p = pd.read_parquet(BACKTEST / "predictions.parquet")
    p = p[(p.model == "stack") & (p.national == "oracle")]
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 4.2), sharex=True)
    for ax, year in zip(axes, [2015, 2019, 2023], strict=True):
        d = p[p.year == year].sort_values("q50").reset_index(drop=True)
        y = np.arange(len(d))
        inside = (d.actual >= d.q05) & (d.actual <= d.q95)
        ax.hlines(y, d.q05, d.q95, color=BLUE, lw=2, alpha=0.35, zorder=2)
        ax.plot(d.q50, y, "|", color=BLUE, ms=6, mew=1.5, zorder=3, label="Median, 90% interval")
        ax.plot(d.actual[inside], y[inside], "o", color=INK, ms=3, zorder=4, label="Outcome (inside)")
        ax.plot(
            d.actual[~inside], y[~inside], "o", mfc="white", mec=ORANGE, mew=1.2, ms=4, zorder=4, label="Outcome (outside)"
        )
        ax.axvline(0.25, color=INK2, lw=0.8, ls="--", zorder=1)
        ax.set_yticks(y, d.state, fontsize=5.5)
        ax.set_title(f"{year} (coverage {inside.mean():.0%})", fontsize=8, loc="left", color=INK)
        ax.set_xlim(0, 1)
        ax.grid(axis="y", visible=False)
    axes[1].set_xlabel("Incumbent-party share of valid votes")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03))
    fig.text(0.99, 0.005, "Dashed line: 25% (§134 threshold)", ha="right", fontsize=6.5, color=INK2)
    fig.tight_layout()
    save(fig, "fig2_backtest_intervals")


def fig_2027_conditional() -> None:
    """Fig. 3: P(APC meets §134 spread) as a function of APC's national share."""
    c = pd.read_csv(PREREG / "conditional.csv", keep_default_na=False)
    g = c.groupby("national_share").p_spread_met_default
    x = g.mean().index.to_numpy()
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.fill_between(x, g.min(), g.max(), color=BLUE, alpha=0.2, lw=0, label="Range over challenger-zone scenarios")
    ax.plot(x, g.mean(), color=BLUE, lw=2, marker="o", ms=4, label="Mean over scenarios")
    ax.axvline(0.366, color=INK2, lw=0.8, ls="--")
    ax.text(0.362, 0.85, "APC 2023:\n36.6%", fontsize=6.5, color=INK2, ha="right")
    ax.set_xlabel("APC national share of valid votes")
    ax.set_ylabel("P(≥25% in 24⅔ of 37 units)")
    ax.set_ylim(-0.02, 1.02)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax.legend(loc="lower right", fontsize=6.5)
    save(fig, "fig3_2027_spread_conditional")


def fig_2027_states() -> None:
    """Fig. 4: 2027 APC state-share forecast (main challenger from North-East)."""
    s = pd.read_csv(PREREG / "states.csv", keep_default_na=False)  # "NA" = Nasarawa
    d = s[s.challenger_zone == "North-East"].sort_values("median").reset_index(drop=True)
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(3.4, 5.0))
    ax.hlines(y, d.q05, d.q95, color=BLUE, lw=2, alpha=0.35)
    ax.plot(d["median"], y, "o", color=BLUE, ms=3.5)
    ax.axvline(0.25, color=INK2, lw=0.8, ls="--")
    ax.set_yticks(y, d.state, fontsize=6)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("APC share of valid votes, 2027 (median, 90% interval)")
    ax.grid(axis="y", visible=False)
    ax.text(0.26, len(d) - 0.5, "25%", fontsize=6.5, color=INK2)
    save(fig, "fig4_2027_states")


def main() -> None:
    fig_backtest_skill()
    fig_backtest_intervals()
    fig_2027_conditional()
    fig_2027_states()
    (OUT / "SOURCES.json").write_text(
        json.dumps({"backtest": str(BACKTEST), "preregistration": str(PREREG)}, indent=2)
    )
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
