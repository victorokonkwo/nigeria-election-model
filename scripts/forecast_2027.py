"""Pre-registered 2027 forecast: APC (incumbent-party) state vote shares and §134 spread.

Model: the prequential stack of docs/evaluation_protocol.md, with weights from
all three backtests (outputs/backtest/v2/weights.json["next"]), refitted on
every election with results (2003, 2011, 2015, 2019, 2023).

The opposition line-up for 2027 is not settled, so no probabilities are
assigned to it. Results are reported per scenario for the *main challenger's
home zone* (it drives the opp_home_zone / opp_home_region covariates), and
also conditional on a grid of APC national shares (protocol option 3), since
the national swing is the dominant, weakly-predictable source of error.

Outright victory also requires a national plurality, which depends on how the
opposition vote splits and is not modelled here; P(APC national share > 50%)
is reported as a sufficient condition.

Writes outputs/preregistered/2027/{forecast.json, states.csv, conditional.csv, MANIFEST.json}.
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from ngforecast.features.panel import region_of_zone
from ngforecast.models.ensemble.stacking import mixture_draws
from ngforecast.models.fundamentals.bayes import SamplerConfig, fit, predict
from ngforecast.simulation.constitution import meets_spread

from run_backtest import CANDIDATES, PROTOCOL, sorted_W  # noqa: E402  (same directory)

OUT = Path("outputs/preregistered/2027")
WEIGHTS = Path("outputs/backtest/v2/weights.json")
ZONES = ["North-Central", "North-East", "North-West", "South-East", "South-South", "South-West"]
NATIONAL_GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
N_DRAWS = 8000
READINGS = {
    "default (FCT as 37th unit; Awolowo v Shagari)": {},
    "states only (24 of 36)": {"fct_interpretation": "states_only"},
    "24 states + 25% in FCT": {"fct_interpretation": "states_plus_fct"},
    "25 whole units (ceil)": {"fractional_rule": "ceil"},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scenario_rows(test: pd.DataFrame, zone: str) -> pd.DataFrame:
    t = test.copy()
    t["opp_home_zone"] = (t["zone"] == zone).astype(int)
    t["opp_home_region"] = (t["region"] == region_of_zone(zone)).astype(int)
    return t


def spread_probs(draws: np.ndarray, const: dict, fct_index: int) -> dict[str, float]:
    out = {}
    for label, override in READINGS.items():
        c = copy.deepcopy(const)
        c["win_rule"].update(override)
        out[label] = float(np.mean([meets_spread(d, c, fct_index=fct_index) for d in draws]))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet("data/processed/panel.parquet").sort_values(["year", "state"])
    train = panel[panel["inc_share"].notna()]
    test = panel[panel.year == 2027].sort_values("state")
    states = sorted(test.state)
    W = sorted_W(states)
    with open("config/constitution.yaml") as f:
        const = yaml.safe_load(f)
    fct_index = states.index(const["win_rule"]["fct_code"])

    weights = json.loads(WEIGHTS.read_text())["next"]["weights"]
    active = {n: w for n, w in weights.items() if w > 1e-3}
    total = sum(active.values())
    active = {n: w / total for n, w in active.items()}
    logger.info("Stack weights (renormalised, >0.1%): {}", active)

    cfg = SamplerConfig()
    fitted = {n: fit(train, W, states, cfg=cfg, **CANDIDATES[n]) for n in active}
    rng = np.random.default_rng(2027)

    scenarios, state_tables, conditional = {}, [], []
    for zone in ZONES:
        rows = scenario_rows(test, zone)
        runs = {n: predict(m, rows, W, seed=2027) for n, m in fitted.items()}
        joint = {n: np.column_stack([r["state"], r["national"]]) for n, r in runs.items()}
        mix = mixture_draws(joint, active, N_DRAWS, rng)
        draws, nat = mix[:, :-1], mix[:, -1]

        q = np.percentile(draws, [5, 50, 95], axis=0)
        p25 = (draws >= 0.25).mean(axis=0)
        n25 = (draws >= 0.25).sum(axis=1)
        scenarios[zone] = {
            "national_share": {
                "median": float(np.median(nat)),
                "q05": float(np.percentile(nat, 5)),
                "q95": float(np.percentile(nat, 95)),
            },
            "p_national_majority": float(np.mean(nat > 0.5)),
            "units_at_25pct": {
                "median": float(np.median(n25)),
                "q05": float(np.percentile(n25, 5)),
                "q95": float(np.percentile(n25, 95)),
            },
            "p_spread_met": spread_probs(draws, const, fct_index),
        }
        state_tables.append(
            pd.DataFrame(
                {
                    "challenger_zone": zone,
                    "state": states,
                    "q05": q[0],
                    "median": q[1],
                    "q95": q[2],
                    "p_ge_25pct": p25,
                }
            )
        )

        # Conditional on the national share (shared by all stack members)
        for nat_share in NATIONAL_GRID:
            cruns = {
                n: predict(m, rows, W, seed=2027, national_override=nat_share)
                for n, m in fitted.items()
            }
            cmix = mixture_draws({n: r["state"] for n, r in cruns.items()}, active, N_DRAWS, rng)
            conditional.append(
                {
                    "challenger_zone": zone,
                    "national_share": nat_share,
                    "units_at_25pct_median": float(np.median((cmix >= 0.25).sum(axis=1))),
                    "p_spread_met_default": float(
                        np.mean([meets_spread(d, const, fct_index=fct_index) for d in cmix])
                    ),
                }
            )
        logger.info("Scenario {}: {}", zone, scenarios[zone]["p_spread_met"])

    pd.concat(state_tables).to_csv(OUT / "states.csv", index=False)
    pd.DataFrame(conditional).to_csv(OUT / "conditional.csv", index=False)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    dirty_files = subprocess.run(
        ["git", "status", "--porcelain", "--", "src", "scripts", "config"],
        capture_output=True,
        text=True,
    ).stdout.split("\n")
    dirty_files = [line[3:] for line in dirty_files if line.strip()]
    forecast = {
        "target": "APC (incumbent party) share of valid votes, 2027 presidential election",
        "election_date": "2027-02-27",
        "issued_at": datetime.now(UTC).isoformat(),
        "stack_weights": active,
        "diagnostics": {n: m.diagnostics for n, m in fitted.items()},
        "scenarios_by_main_challenger_zone": scenarios,
        "not_modelled": "national plurality vs a split opposition; scenario probabilities",
    }
    (OUT / "forecast.json").write_text(json.dumps(forecast, indent=2))
    manifest = {
        "git_commit": commit,
        "uncommitted_code_at_run": dirty_files,
        "protocol_sha256": sha256(PROTOCOL),
        "panel_sha256": sha256(Path("data/processed/panel.parquet")),
        "weights_sha256": sha256(WEIGHTS),
        "outputs_sha256": {
            f.name: sha256(f) for f in sorted(OUT.glob("*")) if f.name != "MANIFEST.json"
        },
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    summary = pd.DataFrame(
        {
            z: {
                "APC national median": s["national_share"]["median"],
                "90% low": s["national_share"]["q05"],
                "90% high": s["national_share"]["q95"],
                "P(>50%)": s["p_national_majority"],
                "units>=25% (median)": s["units_at_25pct"]["median"],
                "P(spread, default)": s["p_spread_met"][
                    "default (FCT as 37th unit; Awolowo v Shagari)"
                ],
            }
            for z, s in scenarios.items()
        }
    ).T
    print(summary.round(3).to_string())
    logger.info("2027 forecast written to {}", OUT)


if __name__ == "__main__":
    main()
