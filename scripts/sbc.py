"""EXPLORATORY: simulation-based calibration (Talts et al. 2018) of the fitted models.

Replaces the earlier coarse parameter-recovery check. For each replication,
parameters are drawn from the prior, data are simulated on the real design
(the 2023-backtest training panel: 2003, 2011, 2015, 2019; region covariates;
observed lagged deviations), the model is refitted, and the rank of the true
value among thinned posterior draws is recorded. Ranks are uniform if the
inference is calibrated.

Models: the national swing model, and the state model in the `pp_region`
form (level model, phi ~ N(1, 0.2), the stack's dominant member).

Writes outputs/exploratory/sbc/{ranks.csv, summary.json}.

Usage: python scripts/sbc.py [--reps 100]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chisquare

sys.path.insert(0, str(Path(__file__).parent))
from run_backtest import REGION, sorted_W  # noqa: E402

from ngforecast.models.fundamentals.bayes import (  # noqa: E402
    SamplerConfig,
    fit_national,
    fit_state,
)

OUT = Path("outputs/exploratory/sbc")
L = 99  # thinned draws -> ranks 0..L
BINS = 10


def rank(true: float, draws: np.ndarray, rng) -> int:
    idx = np.linspace(0, len(draws) - 1, L).astype(int)
    d = draws[idx]
    less = int(np.sum(d < true))
    ties = int(np.sum(d == true))
    return less + int(rng.integers(0, ties + 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    panel = pd.read_parquet("data/processed/panel.parquet").sort_values(["year", "state"])
    train = panel[(panel.year < 2023) & panel.inc_share.notna()].copy()
    states = sorted(train.state.unique())
    W = sorted_W(states)
    years = sorted(train.year.unique())
    n, T, k = len(states), len(years), len(REGION)
    X = np.stack([train.loc[train.year == y, REGION].to_numpy(float) for y in years])
    d_lag = np.stack([train.loc[train.year == y, "dev_lag"].to_numpy() for y in years])
    nat = train.groupby("year").first()
    running = nat["incumbent_running"].to_numpy(float)
    j_home = REGION.index("home_region")
    j_opp = REGION.index("opp_home_region")

    cfg = SamplerConfig(draws=500, tune=500, chains=2)
    rows = []
    for r in range(args.reps):
        # national
        mu, gamma, s_n = rng.normal(0, 0.5), rng.normal(0, 0.5), abs(rng.normal(0, 0.5))
        swing = rng.normal(mu + gamma * running, s_n)
        nat_df = nat.copy()
        lag_logit = np.log(nat_df.nat_share_lag / (1 - nat_df.nat_share_lag))
        nat_df["nat_share"] = 1 / (1 + np.exp(-(lag_logit + swing)))
        tr_n = train.copy()
        tr_n["nat_share"] = tr_n.year.map(nat_df["nat_share"])
        post_n, diag_n = fit_national(tr_n, cfg)
        # state (pp_region form)
        phi = rng.normal(1, 0.2)
        beta = rng.normal(0, 0.5, k)
        alpha = rng.normal(0, 0.5)
        tau = abs(rng.normal(0, 0.3))
        u = rng.normal(0, tau, n) if tau > 0 else np.zeros(n)
        sig = abs(rng.normal(0, 0.5))
        dev = alpha + phi * d_lag + X @ beta + u[None, :] + rng.normal(0, sig, (T, n))
        tr_s = train.copy()
        for i, y in enumerate(years):
            tr_s.loc[tr_s.year == y, "dev"] = dev[i]
        post_s, diag_s = fit_state(
            tr_s, W, states, REGION, cfg, spatial=False, phi_fixed=None, phi_prior=(1.0, 0.2)
        )
        truth = {
            "nat_mu": (mu, post_n["mu"]),
            "nat_gamma": (gamma, post_n["gamma"]),
            "nat_sigma": (s_n, post_n["sigma"]),
            "alpha": (alpha, post_s["alpha"]),
            "phi": (phi, post_s["phi"]),
            "tau": (tau, post_s["tau"]),
            "sigma": (sig, post_s["sigma"]),
            "beta_home_region": (beta[j_home], post_s["beta"][:, j_home]),
            "beta_opp_home_region": (beta[j_opp], post_s["beta"][:, j_opp]),
        }
        for name, (t, d) in truth.items():
            rows.append(
                {
                    "rep": r,
                    "param": name,
                    "rank": rank(t, d, rng),
                    "divergences": diag_n["divergences"]
                    if name.startswith("nat")
                    else diag_s["divergences"],
                }
            )
        print(f"rep {r + 1}/{args.reps}", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "ranks.csv", index=False)

    ranks = pd.DataFrame(rows)
    summ = {}
    for name, g in ranks.groupby("param"):
        counts = np.histogram(g["rank"], bins=BINS, range=(0, L + 1))[0]
        summ[name] = {
            "n": int(len(g)),
            "chi2_p_uniform": float(chisquare(counts).pvalue),
            "bin_counts": counts.tolist(),
            "share_reps_with_divergences": float(np.mean(g.divergences > 0)),
        }
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
