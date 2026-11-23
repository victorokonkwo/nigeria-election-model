"""EXPLORATORY: reproduce the pre-registered 2027 forecast and add finer-grained readings.

1. Re-runs scripts/forecast_2027.py into a scratch directory and checks that
   states.csv and conditional.csv are byte-identical to the pre-registered
   outputs/preregistered/2027 files (the forecast is reproducible).
2. With the same fitted stack members and weights, computes P(spread met) on
   a 1-point grid of APC national shares (0.30-0.50) for every challenger-zone
   scenario, plus an exploratory "narrowed" variant in which each state's
   logit draws are scaled about their median by k (from
   outputs/exploratory/review/summary.json, fitted to 90% backtest coverage).

The pre-registered forecast is not modified. Writes outputs/exploratory/2027/.

Usage: python scripts/exploratory_2027.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
import forecast_2027 as fc  # noqa: E402
from run_backtest import CANDIDATES, sorted_W  # noqa: E402

from ngforecast.models.ensemble.stacking import mixture_draws  # noqa: E402
from ngforecast.models.fundamentals.bayes import SamplerConfig, fit, predict  # noqa: E402
from ngforecast.simulation.constitution import meets_spread  # noqa: E402

OUT = Path("outputs/exploratory/2027")
PREREG = Path("outputs/preregistered/2027")
GRID = np.round(np.arange(0.30, 0.5001, 0.01), 2)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def shrink(draws: np.ndarray, k: float) -> np.ndarray:
    z = np.log(draws / (1 - draws))
    m = np.median(z, axis=0)
    return 1 / (1 + np.exp(-(m[None, :] + k * (z - m[None, :]))))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. Reproduction check
    repro = OUT / "reproduction"
    fc.OUT = repro
    fc.main()
    check = {f: sha(repro / f) == sha(PREREG / f) for f in ("states.csv", "conditional.csv")}
    print("byte-identical to pre-registered:", check)

    # 2. Finer conditional grid with the same members and weights
    k = json.loads(Path("outputs/exploratory/review/summary.json").read_text())["recalibration"][
        "k_for_2027"
    ]
    panel = pd.read_parquet("data/processed/panel.parquet").sort_values(["year", "state"])
    train = panel[panel["inc_share"].notna()]
    test = panel[panel.year == 2027].sort_values("state")
    states = sorted(test.state)
    W = sorted_W(states)
    with open("config/constitution.yaml") as f:
        const = yaml.safe_load(f)
    fct = states.index(const["win_rule"]["fct_code"])
    weights = json.loads(fc.WEIGHTS.read_text())["next"]["weights"]
    active = {n: w for n, w in weights.items() if w > 1e-3}
    tot = sum(active.values())
    active = {n: w / tot for n, w in active.items()}
    fitted = {n: fit(train, W, states, cfg=SamplerConfig(), **CANDIDATES[n]) for n in active}
    rng = np.random.default_rng(2027)

    rows = []
    for zone in fc.ZONES:
        rows_z = fc.scenario_rows(test, zone)
        for g in GRID:
            runs = {
                n: predict(m, rows_z, W, seed=2027, national_override=float(g))
                for n, m in fitted.items()
            }
            mix = mixture_draws({n: r["state"] for n, r in runs.items()}, active, fc.N_DRAWS, rng)
            narrow = shrink(mix, k)
            rows.append(
                {
                    "challenger_zone": zone,
                    "national_share": float(g),
                    "units_at_25pct_median": float(np.median((mix >= 0.25).sum(axis=1))),
                    "p_spread_default": float(
                        np.mean([meets_spread(d, const, fct_index=fct) for d in mix])
                    ),
                    "p_spread_narrowed": float(
                        np.mean([meets_spread(d, const, fct_index=fct) for d in narrow])
                    ),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "conditional_fine.csv", index=False)
    (OUT / "summary.json").write_text(
        json.dumps({"reproduction_byte_identical": check, "k_narrowing": k}, indent=2)
    )
    print(
        out.groupby("national_share")[
            ["units_at_25pct_median", "p_spread_default", "p_spread_narrowed"]
        ]
        .agg(["min", "mean", "max"])
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
