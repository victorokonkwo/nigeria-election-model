"""EXPLORATORY: backtests of region models with a shared zone shock.

The v2 candidates treat states as conditionally independent given the national
draw, so the predicted count of units at 25% is too narrow and a shock shared
by a region (as in 2023) cannot be represented. This script adds, outside the
frozen protocol, `pp_region_zs` and `ps_region_zs`: the v2 `pp_region` and
`ps_region` with an election-specific shock shared by the states of each
geopolitical zone (bayes.fit_state(zone_shock=True)). Same targets, seeds and
scoring as run_backtest.py.

Writes outputs/exploratory/zone_shock/{metrics.csv, draws.npz, posterior.json}.

Usage: python scripts/exploratory_zone_shock.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from run_backtest import CANDIDATES, TARGETS, score, sorted_W  # noqa: E402

from ngforecast.models.fundamentals.bayes import SamplerConfig, fit, predict  # noqa: E402

OUT = Path("outputs/exploratory/zone_shock")
MODELS = {
    "pp_region_zs": {**CANDIDATES["pp_region"], "zone_shock": True},
    "ps_region_zs": {**CANDIDATES["ps_region"], "zone_shock": True},
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet("data/processed/panel.parquet")
    panel = panel[panel["inc_share"].notna()].sort_values(["year", "state"])
    states = sorted(panel["state"].unique())
    W = sorted_W(states)  # noqa: N806
    with open("config/constitution.yaml") as f:
        const = yaml.safe_load(f)
    fct = states.index(const["win_rule"]["fct_code"])
    cfg = SamplerConfig(draws=1000, tune=1000, chains=4)

    metrics, draws, post = [], {}, {}
    for t in TARGETS:
        train = panel[panel.year < t]
        test = panel[panel.year == t].sort_values("state")
        nat_true = float(test["nat_share"].iloc[0])
        for name, spec in MODELS.items():
            m = fit(train, W, states, cfg=cfg, **spec)
            post[f"{t}_{name}"] = {
                "omega_mean": float(m.state["omega"].mean()),
                "omega_sd": float(m.state["omega"].std()),
                "sigma_mean": float(m.state["sigma"].mean()),
                "diagnostics": m.diagnostics,
            }
            for mode in ("forecast", "oracle"):
                r = predict(
                    m, test, W, seed=t, national_override=nat_true if mode == "oracle" else None
                )
                draws[f"{t}_{mode}_{name}_state"] = r["state"].astype(np.float32)
                draws[f"{t}_{mode}_{name}_national"] = np.asarray(r["national"], dtype=np.float32)
                metrics.append(
                    {
                        "year": t,
                        "model": name,
                        "national": mode,
                        **score(r["state"], r["national"], test, const, fct),
                    }
                )
    np.savez_compressed(OUT / "draws.npz", **draws)
    m = pd.DataFrame(metrics)
    m.to_csv(OUT / "metrics.csv", index=False)
    (OUT / "posterior.json").write_text(json.dumps(post, indent=2))
    cols = ["state_crps_pp", "state_rmse_pp", "state_cov90", "p_spread_met"]
    print(m.groupby(["national", "model", "year"])[cols].mean().round(3).to_string())
    print(json.dumps(post, indent=1))


if __name__ == "__main__":
    main()
