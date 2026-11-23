"""EXPLORATORY (second review round): fair baselines, joint scores, block bootstrap.

Not part of the v2 protocol; see docs/exploratory_analyses.md. Inputs are the
saved draws of the v2 re-run (outputs/backtest/v2_draws/draws.npz, metrics
identical to v2) and, if present, the zone-shock backtest
(outputs/exploratory/zone_shock/draws.npz).

1. Baselines: additive (percentage-point) uniform swing; logit uniform swing
   reconciled to the national share by the same logit shift the models use;
   reconciled probabilistic uniform swing (PUS).
2. Joint scores over the 37 units: energy score and variogram score (p = 0.5),
   plus the ranked probability score and PIT of the count of units >= 25%,
   sharpness (mean 90% interval width) and state PIT histograms.
3. Paired CRPS differences with a zone-block bootstrap (6 zones resampled).
4. Monte Carlo SE of the stack (mixture resampling, 20 seeds).
5. A prequential stack without the region covariate family.
6. Regional error diagnosis.

Writes outputs/exploratory/review2/*.csv and summary.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.special import expit

from ngforecast.features.panel import logit
from ngforecast.models.ensemble.stacking import (
    gaussian_logit_logdens,
    mixture_draws,
    stack_densities,
)
from ngforecast.models.fundamentals.bayes import _reconcile
from ngforecast.simulation.constitution import meets_spread

V2D = Path("outputs/backtest/v2_draws/draws.npz")
ZSD = Path("outputs/exploratory/zone_shock/draws.npz")
OUT = Path("outputs/exploratory/review2")
TARGETS = [2015, 2019, 2023]
CANDS = ["ps_null", "ps_home", "ps_region", "sar_region", "pp_region", "lvl_home"]
NOREGION = ["ps_null", "ps_home", "lvl_home"]
RNG = np.random.default_rng(7)


def crps_states(d: np.ndarray, y: np.ndarray) -> np.ndarray:
    s = d.shape[0]
    t1 = np.mean(np.abs(d - y[None, :]), axis=0)
    x = np.sort(d, axis=0)
    i = np.arange(1, s + 1)[:, None]
    return t1 - np.sum((2 * i - s - 1) * x, axis=0) / (s * s)


def energy(d: np.ndarray, y: np.ndarray) -> float:
    d = 100 * d
    y = 100 * y
    t1 = np.mean(np.linalg.norm(d - y[None, :], axis=1))
    perm = RNG.permutation(len(d))
    t2 = np.mean(np.linalg.norm(d - d[perm], axis=1))
    return float(t1 - 0.5 * t2)


def variogram(d: np.ndarray, y: np.ndarray, p: float = 0.5) -> float:
    d = 100 * d[RNG.choice(len(d), 1500, replace=False)]
    y = 100 * y
    iu = np.triu_indices(len(y), 1)
    vy = np.abs(y[:, None] - y[None, :])[iu] ** p
    vd = np.mean(np.abs(d[:, :, None] - d[:, None, :]) ** p, axis=0)[iu]
    return float(np.sum((vy - vd) ** 2))


def count_scores(d: np.ndarray, y: np.ndarray) -> dict:
    cnt = (d >= 0.25).sum(axis=1)
    a = int((y >= 0.25).sum())
    ks = np.arange(0, 38)
    F = np.array([np.mean(cnt <= k) for k in ks])  # noqa: N806
    rps = float(np.sum((F - (a <= ks)) ** 2))
    return {
        "count_actual": a,
        "count_median": float(np.median(cnt)),
        "count_sd": float(cnt.std()),
        "count_pit": float(np.mean(cnt < a) + 0.5 * np.mean(cnt == a)),
        "count_rps": rps,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet("data/processed/panel.parquet").sort_values(["year", "state"])
    panel = panel[panel["inc_share"].notna()]
    with open("config/constitution.yaml") as f:
        const = yaml.safe_load(f)
    states = sorted(panel.state.unique())
    fct = states.index(const["win_rule"]["fct_code"])
    z = np.load(V2D)
    D = {k: z[k].astype(float) for k in z.files}  # noqa: N806
    has_zs = ZSD.exists()
    if has_zs:
        zz = np.load(ZSD)
        D.update({k: zz[k].astype(float) for k in zz.files})
    w_v2 = json.loads(Path("outputs/backtest/v2/weights.json").read_text())

    # ── build all draws, incl. new baselines and the no-region stack ──
    draws: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    pus_sigma = {}
    ld = {}
    for t in TARGETS:
        tr = panel[panel.year < t]
        resid = (logit(tr.inc_share) - logit(tr.inc_share_lag)) - (
            logit(tr.nat_share) - logit(tr.nat_share_lag)
        )
        pus_sigma[t] = float(np.sqrt(np.mean(resid**2)))
        test = panel[panel.year == t].sort_values("state")
        y = test.inc_share.to_numpy()
        lag = test.inc_share_lag.to_numpy()
        nat_lag = float(test.nat_share_lag.iloc[0])
        w = test.weight.to_numpy()
        for mode in ("forecast", "oracle"):
            md = {n: D[f"{t}_{mode}_{n}_state"] for n in CANDS + ["stack", "uniform_swing"]}
            if has_zs:
                for n in ("pp_region_zs", "ps_region_zs"):
                    md[n] = D[f"{t}_{mode}_{n}_state"]
            nat = D[f"{t}_{mode}_uniform_swing_national"]
            S = len(nat)  # noqa: N806
            md["us_additive"] = np.clip(lag[None, :] + (nat - nat_lag)[:, None], 0.001, 0.999)
            base = np.repeat(logit(lag)[None, :], S, axis=0)
            md["us_reconciled"] = _reconcile(base, w, nat)
            noisy = base + pus_sigma[t] * RNG.standard_normal(base.shape)
            md["pus_reconciled"] = _reconcile(noisy, w, nat)
            md["pus"] = expit(
                logit(lag)[None, :]
                + (logit(nat) - logit(nat_lag))[:, None]
                + pus_sigma[t] * RNG.standard_normal(base.shape)
            )
            draws[(t, mode)] = md
            if mode == "oracle":
                ld[t] = {n: gaussian_logit_logdens(md[n], y) for n in CANDS}

    # no-region prequential stack
    nr_weights = {}
    for t in TARGETS:
        past = [s for s in TARGETS if s < t]
        if past:
            m = np.vstack([np.column_stack([ld[s][n] for n in NOREGION]) for s in past])
            wts = stack_densities(m, NOREGION)
        else:
            wts = dict.fromkeys(NOREGION, 1 / 3)
        nr_weights[t] = wts
        for mode in ("forecast", "oracle"):
            draws[(t, mode)]["stack_noregion"] = mixture_draws(
                {n: draws[(t, mode)][n] for n in NOREGION}, wts, 4000, RNG
            )

    # ── scores ──
    rows, per_state = [], []
    for (t, mode), md in draws.items():
        test = panel[panel.year == t].sort_values("state")
        y = test.inc_share.to_numpy()
        for n, d in md.items():
            lo, med, hi = np.percentile(d, [5, 50, 95], axis=0)
            c = crps_states(d, y)
            pit = np.mean(d < y[None, :], axis=0)
            per_state.append(
                pd.DataFrame(
                    {
                        "year": t,
                        "mode": mode,
                        "model": n,
                        "state": test.state.to_numpy(),
                        "zone": test.zone.to_numpy(),
                        "region": test.region.to_numpy(),
                        "y": y,
                        "median": med,
                        "crps": c,
                        "pit": pit,
                    }
                )
            )
            rows.append(
                {
                    "year": t,
                    "mode": mode,
                    "model": n,
                    "crps": 100 * c.mean(),
                    "rmse": 100 * np.sqrt(np.mean((med - y) ** 2)),
                    "mae": 100 * np.mean(np.abs(med - y)),
                    "cov90": float(np.mean((y >= lo) & (y <= hi))),
                    "width90": 100 * float(np.mean(hi - lo)),
                    "energy": energy(d, y),
                    "variogram": variogram(d, y),
                    "p_spread": float(np.mean([meets_spread(x, const, fct_index=fct) for x in d])),
                    **count_scores(d, y),
                }
            )
    sc = pd.DataFrame(rows)
    sc.round(4).to_csv(OUT / "scores_by_year.csv", index=False)
    ps = pd.concat(per_state, ignore_index=True)
    ps.to_csv(OUT / "per_state.csv", index=False)
    mean = (
        sc.groupby(["mode", "model"])[
            [
                "crps",
                "rmse",
                "mae",
                "cov90",
                "width90",
                "energy",
                "variogram",
                "count_rps",
                "count_sd",
            ]
        ]
        .mean()
        .round(3)
    )
    mean.to_csv(OUT / "mean_scores.csv")

    # PIT histogram summary (pooled over years)
    pit = (
        ps.groupby(["mode", "model"])
        .pit.apply(lambda s: pd.Series(np.histogram(s, bins=10, range=(0, 1))[0]))
        .unstack()
    )
    pit.to_csv(OUT / "pit_histograms.csv")

    # ── zone-block bootstrap of paired CRPS differences ──
    refs = ["ps_null", "us_reconciled", "pus_reconciled", "uniform_swing", "pus"]
    brow = []
    for mode in ("oracle", "forecast"):
        for ref in refs + (["pp_region_zs"] if has_zs else []):
            per_year, boots_all = {}, []
            for t in TARGETS:
                a = ps[(ps.year == t) & (ps["mode"] == mode) & (ps.model == "stack")].reset_index()
                b = ps[(ps.year == t) & (ps["mode"] == mode) & (ps.model == ref)].reset_index()
                dlt = 100 * (a.crps - b.crps)
                zones = a.zone.to_numpy()
                zl = np.unique(zones)
                bs = []
                for _ in range(2000):
                    pick = RNG.choice(zl, len(zl), replace=True)
                    idx = np.concatenate([np.flatnonzero(zones == q) for q in pick])
                    bs.append(dlt.to_numpy()[idx].mean())
                per_year[t] = (dlt.mean(), np.percentile(bs, 5), np.percentile(bs, 95))
                boots_all.append(np.array(bs))
                brow.append(
                    {
                        "mode": mode,
                        "reference": ref,
                        "year": t,
                        "diff": dlt.mean(),
                        "lo90": np.percentile(bs, 5),
                        "hi90": np.percentile(bs, 95),
                    }
                )
            mb = np.mean(np.vstack(boots_all), axis=0)
            brow.append(
                {
                    "mode": mode,
                    "reference": ref,
                    "year": "mean",
                    "diff": np.mean([v[0] for v in per_year.values()]),
                    "lo90": np.percentile(mb, 5),
                    "hi90": np.percentile(mb, 95),
                    "years_stack_better": int(sum(v[0] < 0 for v in per_year.values())),
                }
            )
    pd.DataFrame(brow).round(3).to_csv(OUT / "paired_zone_bootstrap.csv", index=False)

    # ── Monte Carlo SE of the stack ──
    mc = []
    for t in TARGETS:
        wts = w_v2[str(t)]["weights"]
        y = panel[panel.year == t].sort_values("state").inc_share.to_numpy()
        for mode in ("oracle", "forecast"):
            vals = []
            for seed in range(20):
                r = np.random.default_rng(1000 + seed)
                mix = mixture_draws({n: draws[(t, mode)][n] for n in CANDS}, wts, 4000, r)
                vals.append(100 * crps_states(mix, y).mean())
            mc.append(
                {"year": t, "mode": mode, "crps_mean": np.mean(vals), "crps_mc_sd": np.std(vals)}
            )
    pd.DataFrame(mc).round(4).to_csv(OUT / "stack_mc_error.csv", index=False)

    # ── regional diagnosis ──
    reg = (
        ps[ps.model.isin(["stack", "us_reconciled", "pus_reconciled", "pp_region_zs"])]
        .assign(err=lambda d: 100 * (d["median"] - d.y))
        .groupby(["mode", "model", "year", "region"])
        .err.mean()
        .unstack()
        .round(2)
    )
    reg.to_csv(OUT / "regional_errors.csv")
    misses = []
    for t in TARGETS:
        s = ps[(ps.year == t) & (ps["mode"] == "oracle") & (ps.model == "stack")]
        misses.append(
            {
                "year": t,
                "sd_logit_median": float(logit(s["median"]).std()),
                "sd_logit_actual": float(logit(s.y).std()),
                "above25_predicted_below": sorted(s.state[(s.y >= 0.25) & (s["median"] < 0.25)]),
                "below25_predicted_above": sorted(s.state[(s.y < 0.25) & (s["median"] >= 0.25)]),
            }
        )
    summary = {
        "pus_sigma": pus_sigma,
        "noregion_weights": {
            t: {k: round(v, 3) for k, v in w.items()} for t, w in nr_weights.items()
        },
        "threshold_misses": misses,
        "has_zone_shock": has_zs,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    with pd.option_context(
        "display.width", 250, "display.max_columns", 40, "display.max_rows", 200
    ):
        print(mean)
        print(
            sc[
                sc.model.isin(
                    [
                        "stack",
                        "pp_region",
                        "pp_region_zs",
                        "ps_region_zs",
                        "us_reconciled",
                        "pus_reconciled",
                        "ps_null",
                        "sar_region",
                    ]
                )
            ][["year", "mode", "model", "crps", "energy", "count_pit", "count_sd", "p_spread"]]
            .round(3)
            .to_string()
        )
        print(pd.read_csv(OUT / "paired_zone_bootstrap.csv").to_string())
        print(pd.read_csv(OUT / "stack_mc_error.csv"))
        print(reg)
        print(pit)
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
