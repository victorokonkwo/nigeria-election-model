"""Leave-future-out backtests under docs/evaluation_protocol.md (v2).

For each target t ∈ {2015, 2019, 2023}: fit every candidate model on elections
< t, predict t, and build the headline *prequential stack* whose weights come
only from backtests of targets < t (equal weights for 2015). Every model is
scored with its own national forecast ("forecast") and with the true national
share ("oracle"). Stacking weights use oracle-mode densities.

Writes outputs/backtest/<tag>/{metrics.csv, predictions.parquet,
posterior.json, weights.json, run_config.json}; run_config records the
SHA-256 of the protocol file.

Usage: python scripts/run_backtest.py [--draws 1000] [--tag v2]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from scipy.special import expit

from ngforecast.calibration.scores import brier_score, crps_sample
from ngforecast.data.geography import (
    build_zone_block_matrix,
    load_constitution_units,
    row_standardize,
)
from ngforecast.features.panel import logit
from ngforecast.models.ensemble.stacking import (
    gaussian_logit_logdens,
    mixture_draws,
    stack_densities,
)
from ngforecast.models.fundamentals.bayes import SamplerConfig, fit, predict
from ngforecast.simulation.constitution import meets_spread

PROTOCOL = Path("docs/evaluation_protocol.md")
TARGETS = [2015, 2019, 2023]

HOME = [
    "home_state",
    "home_zone",
    "opp_home_zone",
    "home_state_lag",
    "home_zone_lag",
    "opp_home_zone_lag",
    "violence_z",
]
REGION = HOME + ["home_region", "opp_home_region", "home_region_lag", "opp_home_region_lag"]

# Candidate models — protocol §2. Do not edit without amending the protocol.
CANDIDATES: dict[str, dict] = {
    "ps_null": {"covariates": [], "spatial": False, "phi_fixed": 1.0},
    "ps_home": {"covariates": HOME, "spatial": False, "phi_fixed": 1.0},
    "ps_region": {"covariates": REGION, "spatial": False, "phi_fixed": 1.0},
    "sar_region": {"covariates": REGION, "spatial": True, "phi_fixed": 1.0},
    "pp_region": {"covariates": REGION, "spatial": False, "phi_prior": (1.0, 0.2)},
    "lvl_home": {"covariates": HOME, "spatial": False, "phi_prior": (0.0, 1.0)},
}


def sorted_W(states: list[str]) -> np.ndarray:
    codes = [u["code"] for u in load_constitution_units()]
    W = row_standardize(build_zone_block_matrix())
    order = [codes.index(s) for s in states]
    return W[np.ix_(order, order)]


def score(draws, nat, test, const, fct_index) -> dict:
    y = test["inc_share"].to_numpy()
    nat_true = float(test["nat_share"].iloc[0])
    med = np.median(draws, axis=0)
    lo, hi = np.percentile(draws, [5, 95], axis=0)
    p25 = (draws >= 0.25).mean(axis=0)
    spread = np.array([meets_spread(d, const, fct_index=fct_index) for d in draws])
    nlo, nhi = np.percentile(nat, [5, 95])
    return {
        "state_rmse_pp": 100 * float(np.sqrt(np.mean((med - y) ** 2))),
        "state_mae_pp": 100 * float(np.mean(np.abs(med - y))),
        "state_cov90": float(np.mean((y >= lo) & (y <= hi))),
        "state_crps_pp": 100 * crps_sample(draws, y),
        "brier_25pct": brier_score(p25, (y >= 0.25).astype(float)),
        "national_pred": float(np.median(nat)),
        "national_true": nat_true,
        "national_err_pp": 100 * float(np.median(nat) - nat_true),
        "national_in_90": bool(nlo <= nat_true <= nhi),
        "p_spread_met": float(spread.mean()),
        "spread_met_true": bool(meets_spread(y, const, fct_index=fct_index)),
    }


def uniform_swing(test: pd.DataFrame, nat: np.ndarray) -> np.ndarray:
    lag = logit(test["inc_share_lag"].to_numpy())
    swing = logit(nat) - logit(float(test["nat_share_lag"].iloc[0]))
    return expit(lag[None, :] + swing[:, None])


def summarise_posterior(m) -> dict:
    params = {
        k: {"mean": float(np.mean(v)), "sd": float(np.std(v))}
        for k, v in {**m.state, **{f"nat_{k}": v for k, v in m.national.items()}}.items()
        if np.ndim(v) == 1
    }
    beta = {
        c: {"mean": float(m.state["beta"][:, j].mean()), "sd": float(m.state["beta"][:, j].std())}
        for j, c in enumerate(m.covariates)
    }
    return {"params": params, "beta": beta, "diagnostics": m.diagnostics}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/processed/panel.parquet")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tag", default="v2")
    ap.add_argument(
        "--save-draws",
        action="store_true",
        help="also write draws.npz (state/national draws per year, mode, model); "
        "does not change any result",
    )
    args = ap.parse_args()

    out = Path("outputs/backtest") / args.tag
    out.mkdir(parents=True, exist_ok=True)
    protocol_sha = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    logger.info("Protocol {} sha256={}", PROTOCOL, protocol_sha[:12])

    panel = pd.read_parquet(args.panel)
    panel = panel[panel["inc_share"].notna()].sort_values(["year", "state"])
    states = sorted(panel["state"].unique())
    W = sorted_W(states)
    with open("config/constitution.yaml") as f:
        const = yaml.safe_load(f)
    fct_index = states.index(const["win_rule"]["fct_code"])
    cfg = SamplerConfig(draws=args.draws, tune=args.draws, chains=4)
    rng = np.random.default_rng(2027)

    metrics, preds, posterior, weights_log = [], [], {}, {}
    holdout_logdens: dict[int, dict[str, np.ndarray]] = {}  # oracle-mode, per past target
    saved_draws: dict[str, np.ndarray] = {}

    for t in TARGETS:
        train = panel[panel.year < t]
        test = panel[panel.year == t].sort_values("state")
        y, nat_true = test["inc_share"].to_numpy(), float(test["nat_share"].iloc[0])
        logger.info("Backtest {}: train {}", t, sorted(train.year.unique()))

        fitted = {name: fit(train, W, states, cfg=cfg, **spec) for name, spec in CANDIDATES.items()}
        posterior[t] = {name: summarise_posterior(m) for name, m in fitted.items()}

        # Prequential stacking weights: only targets < t
        past = [s for s in TARGETS if s < t]
        if past:
            ld = np.vstack(
                [np.column_stack([holdout_logdens[s][n] for n in CANDIDATES]) for s in past]
            )
            weights = stack_densities(ld, list(CANDIDATES))
        else:
            weights = {n: 1 / len(CANDIDATES) for n in CANDIDATES}
        weights_log[t] = {"fitted_on_targets": past, "weights": weights}

        for mode in ("forecast", "oracle"):
            override = nat_true if mode == "oracle" else None
            runs = {
                n: predict(m, test, W, seed=t, national_override=override)
                for n, m in fitted.items()
            }
            if mode == "oracle":
                holdout_logdens[t] = {
                    n: gaussian_logit_logdens(r["state"], y) for n, r in runs.items()
                }

            joint = {n: np.column_stack([r["state"], r["national"]]) for n, r in runs.items()}
            mix = mixture_draws(joint, weights, n_draws=4 * args.draws, rng=rng)
            runs["stack"] = {"state": mix[:, :-1], "national": mix[:, -1]}
            nat_draws = runs["ps_null"]["national"]
            runs["uniform_swing"] = {"state": uniform_swing(test, nat_draws), "national": nat_draws}

            if args.save_draws:
                for name, r in runs.items():
                    saved_draws[f"{t}_{mode}_{name}_state"] = r["state"].astype(np.float32)
                    saved_draws[f"{t}_{mode}_{name}_national"] = np.asarray(
                        r["national"], dtype=np.float32
                    )
            for name, r in runs.items():
                metrics.append(
                    {
                        "year": t,
                        "model": name,
                        "national": mode,
                        **score(r["state"], r["national"], test, const, fct_index),
                    }
                )
                q = np.percentile(r["state"], [5, 50, 95], axis=0)
                preds.append(
                    pd.DataFrame(
                        {
                            "year": t,
                            "model": name,
                            "national": mode,
                            "state": states,
                            "q05": q[0],
                            "q50": q[1],
                            "q95": q[2],
                            "actual": y,
                        }
                    )
                )

        lag = test["inc_share_lag"].to_numpy()
        metrics.append(
            {
                "year": t,
                "model": "persistence",
                "national": "forecast",
                "state_rmse_pp": 100 * float(np.sqrt(np.mean((lag - y) ** 2))),
                "state_mae_pp": 100 * float(np.mean(np.abs(lag - y))),
                "national_pred": float(test["nat_share_lag"].iloc[0]),
                "national_true": nat_true,
                "national_err_pp": 100 * float(test["nat_share_lag"].iloc[0] - nat_true),
            }
        )

    # Weights for the next (2027) forecast: stacking on all backtests (protocol §4)
    ld_all = np.vstack(
        [np.column_stack([holdout_logdens[s][n] for n in CANDIDATES]) for s in TARGETS]
    )
    weights_log["next"] = {
        "fitted_on_targets": TARGETS,
        "weights": stack_densities(ld_all, list(CANDIDATES)),
    }
    np.savez(
        out / "holdout_logdens.npz",
        **{f"{t}_{n}": holdout_logdens[t][n] for t in TARGETS for n in CANDIDATES},
    )

    if args.save_draws:
        np.savez_compressed(out / "draws.npz", **saved_draws)
    m = pd.DataFrame(metrics)
    m.to_csv(out / "metrics.csv", index=False)
    pd.concat(preds).to_parquet(out / "predictions.parquet", index=False)
    (out / "posterior.json").write_text(json.dumps(posterior, indent=2))
    (out / "weights.json").write_text(json.dumps(weights_log, indent=2))
    (out / "run_config.json").write_text(
        json.dumps(
            {
                "protocol": str(PROTOCOL),
                "protocol_sha256": protocol_sha,
                "candidates": {
                    k: {**v, "phi_prior": v.get("phi_prior")} for k, v in CANDIDATES.items()
                },
                "draws": args.draws,
                "chains": cfg.chains,
                "targets": TARGETS,
            },
            indent=2,
            default=list,
        )
    )
    cols = ["state_rmse_pp", "state_crps_pp", "state_cov90", "brier_25pct"]
    with pd.option_context("display.width", 200):
        for mode in ("forecast", "oracle"):
            print(f"\n== {mode} (mean over {TARGETS})")
            print(
                m[m.national == mode]
                .groupby("model")[cols]
                .mean()
                .round(2)
                .sort_values("state_crps_pp")
                .to_string()
            )
    print(
        "\nweights:",
        json.dumps(
            {t: {k: round(v, 3) for k, v in w["weights"].items()} for t, w in weights_log.items()}
        ),
    )
    logger.info("Backtest written to {}", out)


if __name__ == "__main__":
    main()
