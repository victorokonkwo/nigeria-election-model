"""Simulation study: two general lessons for forecasting geographic-threshold rules.

Synthetic country with K = 37 units in 6 zones (sizes as Nigeria's zones),
equal vote weights. In each simulated election the true incumbent logit share is

    logit y_i = logit N + d_i,   d_i = phi * d_lag_i + omega * z_zone(i) + sigma * e_i,

reconciled so that the mean share equals N exactly.

Study 1 (evaluation power). Forecaster A knows the data-generating process;
forecaster B uses uniform swing (phi = 1) with noise matched to its residual
variance. Both receive the same national forecast, logit N_hat = logit N + e,
e ~ N(0, s^2), with a calibrated predictive spread s. Over 3 elections we
record whether mean state CRPS ranks A above B, (a) given the true national
share ("oracle") and (b) with the national forecast ("forecast"), as s grows.

Study 2 (threshold counts). Given the true national share, forecaster C knows
the zone shock; forecaster D has the same marginal variance but independent
unit errors. Both have calibrated marginals; we score the predictive
distribution of the number of units with y >= 25% and the event "at least 25
units", at national shares near the threshold region.

Writes outputs/simulation/{study1.csv, study2.csv, summary.json}.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from ngforecast.models.fundamentals.bayes import _reconcile

OUT = Path("outputs/simulation")
ZONE_SIZES = [7, 6, 7, 5, 6, 6]  # NC (incl. FCT), NE, NW, SE, SS, SW
ZONE = np.repeat(np.arange(6), ZONE_SIZES)
K = len(ZONE)
PHI, OMEGA, SIGMA = 0.5, 0.8, 0.7  # logit scale: fitted pp_region_zs values (2023 backtest)
VAR_LAG = 2.0  # variance of lagged deviations
S = 400  # predictive draws


def crps(d: np.ndarray, y: np.ndarray) -> float:
    s = d.shape[0]
    t1 = np.mean(np.abs(d - y[None, :]), axis=0)
    x = np.sort(d, axis=0)
    i = np.arange(1, s + 1)[:, None]
    return float(np.mean(t1 - np.sum((2 * i - s - 1) * x, axis=0) / (s * s)))


def lagged(rng) -> np.ndarray:
    return rng.normal(0, 1.0, 6)[ZONE] + rng.normal(0, 1.0, K)


def truth(rng, d_lag, nat) -> np.ndarray:
    d = PHI * d_lag + OMEGA * rng.normal(0, 1, 6)[ZONE] + SIGMA * rng.normal(0, 1, K)
    return _reconcile((logit(nat) + d)[None, :], np.ones(K), np.array([nat]))[0]


def draws_dev(rng, d_lag, kind: str, phi_b: float = 1.0) -> np.ndarray:
    if kind == "A":  # true process
        return (
            PHI * d_lag[None, :]
            + OMEGA * rng.normal(0, 1, (S, 6))[:, ZONE]
            + SIGMA * rng.normal(0, 1, (S, K))
        )
    if kind == "B":  # wrong persistence phi_b (1 = uniform swing), calibrated residual noise
        resid_var = (phi_b - PHI) ** 2 * VAR_LAG + OMEGA**2 + SIGMA**2
        return phi_b * d_lag[None, :] + np.sqrt(resid_var) * rng.normal(0, 1, (S, K))
    if kind == "C":  # knows the zone shock
        return draws_dev(rng, d_lag, "A")
    if kind == "D":  # independent errors, same marginal variance
        return PHI * d_lag[None, :] + np.sqrt(OMEGA**2 + SIGMA**2) * rng.normal(0, 1, (S, K))
    raise ValueError(kind)


def predictive(rng, d_lag, kind, nat_draws, phi_b: float = 1.0) -> np.ndarray:
    dev = draws_dev(rng, d_lag, kind, phi_b)
    return _reconcile(logit(nat_draws)[:, None] + dev, np.ones(K), nat_draws)


def study1(rng, reps=300) -> pd.DataFrame:
    rows = []
    for phi_b, s in [(pb, s) for pb in (1.0, 0.75, 0.625) for s in (0.0, 0.25, 0.5, 0.75)]:
        for r in range(reps):
            tot = {
                ("A", "oracle"): 0.0,
                ("B", "oracle"): 0.0,
                ("A", "forecast"): 0.0,
                ("B", "forecast"): 0.0,
            }
            for _ in range(3):
                d_lag = lagged(rng)
                nat = rng.uniform(0.35, 0.6)
                y = truth(rng, d_lag, nat)
                nat_hat = expit(logit(nat) + rng.normal(0, s)) if s > 0 else nat
                nat_f = (
                    expit(logit(nat_hat) + s * rng.normal(0, 1, S)) if s > 0 else np.full(S, nat)
                )
                for kind in ("A", "B"):
                    tot[(kind, "oracle")] += crps(
                        predictive(rng, d_lag, kind, np.full(S, nat), phi_b), y
                    )
                    tot[(kind, "forecast")] += crps(predictive(rng, d_lag, kind, nat_f, phi_b), y)
            rows.append(
                {
                    "phi_b": phi_b,
                    "national_error_sd_logit": s,
                    "rep": r,
                    "oracle_A_better": tot[("A", "oracle")] < tot[("B", "oracle")],
                    "forecast_A_better": tot[("A", "forecast")] < tot[("B", "forecast")],
                    "oracle_gain_pp": 100 * (tot[("B", "oracle")] - tot[("A", "oracle")]) / 3,
                    "forecast_gain_pp": 100 * (tot[("B", "forecast")] - tot[("A", "forecast")]) / 3,
                }
            )
    return pd.DataFrame(rows)


def study2(rng, reps=3000) -> pd.DataFrame:
    rows = []
    for _ in range(reps):
        d_lag = lagged(rng)
        nat = rng.uniform(0.30, 0.45)
        y = truth(rng, d_lag, nat)
        a = int((y >= 0.25).sum())
        for kind in ("C", "D"):
            d = predictive(rng, d_lag, kind, np.full(S, nat))
            cnt = (d >= 0.25).sum(axis=1)
            lo, hi = np.percentile(cnt, [5, 95])
            p25 = (d >= 0.25).mean(axis=0)
            p_event = float(np.mean(cnt >= 25))
            rows.append(
                {
                    "forecaster": kind,
                    "count_pit": float(np.mean(cnt < a) + 0.5 * np.mean(cnt == a)),
                    "count_covered90": bool(lo <= a <= hi),
                    "count_sd": float(cnt.std()),
                    "unit_brier": float(np.mean((p25 - (y >= 0.25)) ** 2)),
                    "event": a >= 25,
                    "p_event": p_event,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(134)
    s1 = study1(rng)
    s1.to_csv(OUT / "study1.csv", index=False)
    s2 = study2(rng)
    s2.to_csv(OUT / "study2.csv", index=False)
    t1 = s1.groupby(["phi_b", "national_error_sd_logit"])[
        ["oracle_A_better", "forecast_A_better", "oracle_gain_pp", "forecast_gain_pp"]
    ].mean()
    t2 = s2.groupby("forecaster").agg(
        count_cov90=("count_covered90", "mean"),
        count_sd=("count_sd", "mean"),
        pit_outer10=("count_pit", lambda p: float(np.mean((p < 0.05) | (p > 0.95)))),
        unit_brier=("unit_brier", "mean"),
        event_brier=("p_event", lambda p: np.nan),
    )
    for kind in ("C", "D"):
        sub = s2[s2.forecaster == kind]
        t2.loc[kind, "event_brier"] = float(np.mean((sub.p_event - sub.event) ** 2))
    print(t1.round(3).to_string())
    print(t2.round(4).to_string())
    (OUT / "summary.json").write_text(
        json.dumps(
            {
                "params": {
                    "phi": PHI,
                    "omega": OMEGA,
                    "sigma": SIGMA,
                    "var_lag": VAR_LAG,
                    "K": K,
                    "draws": S,
                },
                "study1": t1.round(4).reset_index().to_dict(orient="records"),
                "study2": t2.round(4).reset_index().to_dict(orient="records"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
