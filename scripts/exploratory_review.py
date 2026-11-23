"""EXPLORATORY analyses added in response to referee comments (not part of the v2 protocol).

Nothing here changes the result of record (outputs/backtest/v2) or the
pre-registered 2027 forecast. All analyses below were defined *after* the v2
results were seen and are labelled exploratory in the paper; see
docs/exploratory_analyses.md.

Inputs
- outputs/backtest/v2_draws/draws.npz: predictive draws from a re-run of the
  v2 backtest with --save-draws (same seeds; metrics must match v2 exactly,
  checked below).
- data/processed/panel.parquet, data/external/inec_historical/*.csv

Writes outputs/exploratory/review/{*.csv, summary.json}.

Usage: python scripts/exploratory_review.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import yaml
from scipy.special import expit
from scipy.stats import gaussian_kde

from ngforecast.features.panel import logit
from ngforecast.models.ensemble.stacking import (
    gaussian_logit_logdens,
    mixture_draws,
    stack_densities,
)
from ngforecast.simulation.constitution import meets_spread

DRAWS = Path("outputs/backtest/v2_draws")
V2 = Path("outputs/backtest/v2")
OUT = Path("outputs/exploratory/review")
TARGETS = [2015, 2019, 2023]
CANDS = ["ps_null", "ps_home", "ps_region", "sar_region", "pp_region", "lvl_home"]
SHOW = ["stack", "pus", "ps_null", "ps_region", "pp_region", "uniform_swing"]
RNG = np.random.default_rng(20261123)
N_BOOT = 2000


# ── helpers ──────────────────────────────────


def crps_per_state(draws: np.ndarray, y: np.ndarray) -> np.ndarray:
    s = draws.shape[0]
    t1 = np.mean(np.abs(draws - y[None, :]), axis=0)
    x = np.sort(draws, axis=0)
    i = np.arange(1, s + 1)[:, None]
    t2 = np.sum((2 * i - s - 1) * x, axis=0) / (s * s)
    return t1 - t2


def load() -> tuple[pd.DataFrame, dict, dict, int]:
    panel = pd.read_parquet("data/processed/panel.parquet").sort_values(["year", "state"])
    panel = panel[panel["inc_share"].notna()]
    with open("config/constitution.yaml") as f:
        const = yaml.safe_load(f)
    states = sorted(panel.state.unique())
    fct = states.index(const["win_rule"]["fct_code"])
    z = np.load(DRAWS / "draws.npz")
    draws = {k: z[k].astype(float) for k in z.files}
    return panel, draws, const, fct


def check_reproduction() -> bool:
    a = pd.read_csv(V2 / "metrics.csv")
    b = pd.read_csv(DRAWS / "metrics.csv")
    num = a.select_dtypes("number").columns
    return bool(np.allclose(a[num].fillna(-9), b[num].fillna(-9), atol=1e-9))


def pus_draws(panel: pd.DataFrame, t: int, nat: np.ndarray) -> tuple[np.ndarray, float]:
    """Pure probabilistic uniform swing: median = uniform swing (logit), plus
    N(0, s^2) noise, s = RMS of uniform-swing residuals in the training years."""
    tr = panel[panel.year < t]
    resid = (logit(tr.inc_share) - logit(tr.inc_share_lag)) - (
        logit(tr.nat_share) - logit(tr.nat_share_lag)
    )
    s = float(np.sqrt(np.mean(resid**2)))
    test = panel[panel.year == t].sort_values("state")
    lag = logit(test["inc_share_lag"].to_numpy())
    swing = logit(nat) - logit(float(test["nat_share_lag"].iloc[0]))
    z = RNG.standard_normal((len(nat), len(lag)))
    return expit(lag[None, :] + swing[:, None] + s * z), s


def shrink(draws: np.ndarray, k: float) -> np.ndarray:
    """Scale each state's logit draws about its median by k (k<1 narrows)."""
    z = logit(draws)
    m = np.median(z, axis=0)
    return expit(m[None, :] + k * (z - m[None, :]))


def coverage(draws: np.ndarray, y: np.ndarray) -> float:
    lo, hi = np.percentile(draws, [5, 95], axis=0)
    return float(np.mean((y >= lo) & (y <= hi)))


def fit_k(pairs: list[tuple[np.ndarray, np.ndarray]]) -> float:
    """Scale factor whose shrunk draws give 90% pooled coverage on earlier backtests."""
    grid = np.round(np.arange(0.30, 1.51, 0.01), 2)
    cov = np.array(
        [
            np.mean(np.concatenate([[coverage(shrink(d, k), y)] * len(y) for d, y in pairs]))
            for k in grid
        ]
    )
    return float(grid[np.argmin(np.abs(cov - 0.90))])


# ── national model prior sensitivity ─────────


NAT_VARIANTS = {
    "baseline: mu,gamma~N(0,0.5^2), sigma~HN(0.5)": (0.5, 0.5, 0.5, True),
    "tight: sd 0.25": (0.25, 0.25, 0.25, True),
    "wide: sd 1.0": (1.0, 1.0, 1.0, True),
    "sigma~HN(1.0) only": (0.5, 0.5, 1.0, True),
    "no incumbent-running term": (0.5, 0.0, 0.5, False),
}


def national_posterior(nat: pd.DataFrame, mu_sd, g_sd, s_sd, use_gamma, seed=2027):
    swing = (logit(nat["nat_share"]) - logit(nat["nat_share_lag"])).to_numpy()
    r = nat["incumbent_running"].to_numpy(float)
    with pm.Model():
        mu = pm.Normal("mu", 0, mu_sd)
        gamma = pm.Normal("gamma", 0, g_sd) if use_gamma else 0.0
        sigma = pm.HalfNormal("sigma", s_sd)
        pm.Normal("swing", mu + gamma * r, sigma, observed=swing)
        idata = pm.sample(
            1000, tune=1000, chains=4, target_accept=0.99, random_seed=seed, progressbar=False
        )
    post = idata.posterior
    mu_d = post["mu"].to_numpy().ravel()
    g_d = post["gamma"].to_numpy().ravel() if use_gamma else np.zeros_like(mu_d)
    s_d = post["sigma"].to_numpy().ravel()
    div = int(idata.sample_stats["diverging"].sum())
    return mu_d, g_d, s_d, div


def national_sensitivity(panel: pd.DataFrame) -> pd.DataFrame:
    nat_all = pd.read_parquet("data/processed/panel.parquet").groupby("year").first().reset_index()
    rows = []
    for label, (a, b, c, ug) in NAT_VARIANTS.items():
        for t in TARGETS + [2027]:
            tr = nat_all[(nat_all.year < t) & nat_all.nat_share.notna()]
            mu, g, s, div = national_posterior(tr, a, b, c, ug)
            row = nat_all[nat_all.year == t].iloc[0]
            r = float(row["incumbent_running"])
            rng = np.random.default_rng(t)
            nat = expit(logit(row["nat_share_lag"]) + rng.normal(mu + g * r, s))
            out = {
                "variant": label,
                "target": t,
                "median": float(np.median(nat)),
                "q05": float(np.percentile(nat, 5)),
                "q95": float(np.percentile(nat, 95)),
                "p_gt_50": float(np.mean(nat > 0.5)),
                "mu_mean": float(mu.mean()),
                "gamma_mean": float(g.mean()),
                "gamma_sd": float(g.std()),
                "sigma_mean": float(s.mean()),
                "divergences": div,
            }
            if t != 2027:
                out["true"] = float(row["nat_share"])
                out["err_pp"] = 100 * (out["median"] - out["true"])
                out["in_90"] = bool(out["q05"] <= out["true"] <= out["q95"])
            rows.append(out)
    return pd.DataFrame(rows)


# ── main ─────────────────────────────────────


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel, D, const, fct = load()
    summary: dict = {"reproduces_v2_metrics": check_reproduction()}

    # Per-state scores for every model, both modes
    per_state, all_draws = [], {}
    pus_sigma = {}
    for t in TARGETS:
        test = panel[panel.year == t].sort_values("state")
        y = test.inc_share.to_numpy()
        for mode in ("forecast", "oracle"):
            md = {n: D[f"{t}_{mode}_{n}_state"] for n in CANDS + ["stack", "uniform_swing"]}
            nat = D[f"{t}_{mode}_uniform_swing_national"]
            md["pus"], pus_sigma[t] = pus_draws(panel, t, nat)
            all_draws[(t, mode)] = md
            for n, d in md.items():
                lo, med, hi = np.percentile(d, [5, 50, 95], axis=0)
                per_state.append(
                    pd.DataFrame(
                        {
                            "year": t,
                            "mode": mode,
                            "model": n,
                            "state": test.state.to_numpy(),
                            "y": y,
                            "median": med,
                            "crps": crps_per_state(d, y),
                            "covered": (y >= lo) & (y <= hi),
                            "p25": (d >= 0.25).mean(axis=0),
                        }
                    )
                )
    ps = pd.concat(per_state, ignore_index=True)
    ps.to_csv(OUT / "per_state_scores.csv", index=False)
    summary["pus_sigma_logit"] = pus_sigma

    # (A) Mean scores incl. pure PUS
    agg = (
        ps.assign(se=lambda d: (d["median"] - d.y) ** 2, ae=lambda d: (d["median"] - d.y).abs())
        .groupby(["mode", "model", "year"])
        .agg(crps=("crps", "mean"), mse=("se", "mean"), mae=("ae", "mean"), cov=("covered", "mean"))
        .reset_index()
    )
    agg["rmse"] = np.sqrt(agg.mse)
    mean_scores = agg.groupby(["mode", "model"])[["crps", "rmse", "mae", "cov"]].agg(
        lambda s: s.mean()
    )
    mean_scores[["crps", "mae"]] *= 100
    mean_scores["rmse"] = agg.groupby(["mode", "model"]).rmse.mean() * 100
    mean_scores.round(3).to_csv(OUT / "mean_scores.csv")
    agg.assign(crps=agg.crps * 100, rmse=agg.rmse * 100, mae=agg.mae * 100).round(3).to_csv(
        OUT / "scores_by_year.csv", index=False
    )

    # (B) Paired CRPS differences, per election, bootstrap over states
    rows = []
    for mode in ("oracle", "forecast"):
        for ref in ["pus", "ps_null", "uniform_swing"]:
            pooled = []
            for t in TARGETS:
                a = ps[
                    (ps.year == t) & (ps["mode"] == mode) & (ps.model == "stack")
                ].crps.to_numpy()
                b = ps[(ps.year == t) & (ps["mode"] == mode) & (ps.model == ref)].crps.to_numpy()
                d = 100 * (a - b)
                pooled.append(d)
                bs = [d[RNG.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)]
                rows.append(
                    {
                        "mode": mode,
                        "reference": ref,
                        "year": t,
                        "diff_pp": d.mean(),
                        "lo90": np.percentile(bs, 5),
                        "hi90": np.percentile(bs, 95),
                        "share_states_stack_better": float(np.mean(d < 0)),
                    }
                )
            bs = [
                np.mean([p[RNG.integers(0, len(p), len(p))].mean() for p in pooled])
                for _ in range(N_BOOT)
            ]
            rows.append(
                {
                    "mode": mode,
                    "reference": ref,
                    "year": "mean",
                    "diff_pp": np.mean([p.mean() for p in pooled]),
                    "lo90": np.percentile(bs, 5),
                    "hi90": np.percentile(bs, 95),
                    "share_states_stack_better": float(np.mean(np.concatenate(pooled) < 0)),
                }
            )
    pd.DataFrame(rows).round(3).to_csv(OUT / "paired_crps_differences.csv", index=False)

    # (C) Forecast vs oracle decomposition of squared median error
    rows = []
    for t in TARGETS:
        for n in ["stack", "pus", "ps_null", "ps_region", "pp_region", "uniform_swing"]:
            f = ps[(ps.year == t) & (ps["mode"] == "forecast") & (ps.model == n)].set_index("state")
            o = ps[(ps.year == t) & (ps["mode"] == "oracle") & (ps.model == n)].set_index("state")
            e_o = o["median"] - o.y
            delta = f["median"] - o["median"]
            rows.append(
                {
                    "year": t,
                    "model": n,
                    "rmse_forecast": 100 * np.sqrt(np.mean((f["median"] - f.y) ** 2)),
                    "rmse_oracle": 100 * np.sqrt(np.mean(e_o**2)),
                    "rms_national_shift": 100 * np.sqrt(np.mean(delta**2)),
                    "cross_term": 1e4 * 2 * np.mean(e_o * delta),
                    "mean_slope_p1mp": float(np.mean(o["median"] * (1 - o["median"]))),
                    "corr_state_err_shift": float(np.corrcoef(e_o, delta)[0, 1]),
                }
            )
    pd.DataFrame(rows).round(3).to_csv(OUT / "forecast_oracle_decomposition.csv", index=False)

    # (D) Stacking: forecast-mode weights (prequential), weight bootstrap, KDE densities
    ld = {
        (t, mode): {
            n: gaussian_logit_logdens(
                all_draws[(t, mode)][n],
                panel[panel.year == t].sort_values("state").inc_share.to_numpy(),
            )
            for n in CANDS
        }
        for t in TARGETS
        for mode in ("oracle", "forecast")
    }
    # reproduce v2 weights from saved oracle densities
    v2w = json.loads((V2 / "weights.json").read_text())

    def weights_from(mode: str, past: list[int], idx=None) -> dict:
        mats = []
        for s in past:
            m = np.column_stack([ld[(s, mode)][n] for n in CANDS])
            mats.append(m if idx is None else m[idx[s]])
        return stack_densities(np.vstack(mats), CANDS)

    wrows = []
    for label, past in [("2019", [2015]), ("2023", [2015, 2019]), ("2027", TARGETS)]:
        for mode in ("oracle", "forecast"):
            w = weights_from(mode, past)
            wrows.append({"for": label, "density_mode": mode, **{n: round(w[n], 3) for n in CANDS}})
        # bootstrap over states (oracle mode, as in the protocol)
        boots = []
        for _ in range(200):
            idx = {s: RNG.integers(0, 37, 37) for s in past}
            boots.append([weights_from("oracle", past, idx)[n] for n in CANDS])
        boots = np.array(boots)
        for q, name in [(5, "boot_q05"), (50, "boot_median"), (95, "boot_q95")]:
            wrows.append(
                {
                    "for": label,
                    "density_mode": f"oracle {name}",
                    **dict(zip(CANDS, np.round(np.percentile(boots, q, axis=0), 3), strict=True)),
                }
            )
        wrows.append(
            {
                "for": label,
                "density_mode": "oracle P(weight>0.5)",
                **dict(zip(CANDS, np.round((boots > 0.5).mean(axis=0), 3), strict=True)),
            }
        )
        # KDE densities (oracle) instead of Gaussian-on-logit
        mats = []
        for s in past:
            y = logit(panel[panel.year == s].sort_values("state").inc_share.to_numpy())
            cols = []
            for n in CANDS:
                z = logit(np.clip(all_draws[(s, "oracle")][n], 1e-3, 1 - 1e-3))
                cols.append([np.log(gaussian_kde(z[:, i])(y[i])[0] + 1e-300) for i in range(37)])
            mats.append(np.array(cols).T)
        wk = stack_densities(np.vstack(mats), CANDS)
        wrows.append(
            {"for": label, "density_mode": "oracle KDE", **{n: round(wk[n], 3) for n in CANDS}}
        )
    pd.DataFrame(wrows).to_csv(OUT / "stacking_weights_robustness.csv", index=False)
    summary["v2_next_weights"] = v2w["next"]["weights"]

    # forecast-mode-weighted stack, scored prequentially
    rows = []
    for t in TARGETS:
        past = [s for s in TARGETS if s < t]
        w = weights_from("forecast", past) if past else dict.fromkeys(CANDS, 1 / 6)
        y = panel[panel.year == t].sort_values("state").inc_share.to_numpy()
        for mode in ("forecast", "oracle"):
            mix = mixture_draws({n: all_draws[(t, mode)][n] for n in CANDS}, w, 4000, RNG)
            lo, med, hi = np.percentile(mix, [5, 50, 95], axis=0)
            rows.append(
                {
                    "year": t,
                    "mode": mode,
                    "crps": 100 * crps_per_state(mix, y).mean(),
                    "rmse": 100 * np.sqrt(np.mean((med - y) ** 2)),
                    "cov90": float(np.mean((y >= lo) & (y <= hi))),
                    "weights": json.dumps({k: round(v, 3) for k, v in w.items() if v > 1e-3}),
                }
            )
    pd.DataFrame(rows).round(3).to_csv(OUT / "stack_forecast_mode_weights.csv", index=False)

    # (E) §134 spread: calibration of the count of units >= 25%, recalibration
    rows, recal = [], {}
    for t in TARGETS:
        test = panel[panel.year == t].sort_values("state")
        y = test.inc_share.to_numpy()
        actual_count = int((y >= 0.25).sum())
        for mode in ("oracle", "forecast"):
            for n in ["stack", "pus", "ps_null", "uniform_swing"]:
                d = all_draws[(t, mode)][n]
                cnt = (d >= 0.25).sum(axis=1)
                p = float(np.mean([meets_spread(x, const, fct_index=fct) for x in d]))
                rows.append(
                    {
                        "year": t,
                        "mode": mode,
                        "model": n,
                        "actual_units_ge25": actual_count,
                        "pred_units_median": float(np.median(cnt)),
                        "pred_units_q05": float(np.percentile(cnt, 5)),
                        "pred_units_q95": float(np.percentile(cnt, 95)),
                        "pit_count": float(
                            np.mean(cnt < actual_count) + 0.5 * np.mean(cnt == actual_count)
                        ),
                        "p_spread": p,
                        "spread_met": bool(meets_spread(y, const, fct_index=fct)),
                    }
                )
        # Prequential recalibration of the stack's oracle state spread
        past = [s for s in TARGETS if s < t]
        pairs = [
            (
                all_draws[(s, "oracle")]["stack"],
                panel[panel.year == s].sort_values("state").inc_share.to_numpy(),
            )
            for s in past
        ]
        k = fit_k(pairs) if pairs else 1.0
        d = shrink(all_draws[(t, "oracle")]["stack"], k)
        recal[t] = {
            "k": k,
            "cov90": coverage(d, y),
            "crps": 100 * crps_per_state(d, y).mean(),
            "p_spread": float(np.mean([meets_spread(x, const, fct_index=fct) for x in d])),
            "pred_units_median": float(np.median((d >= 0.25).sum(axis=1))),
        }
    spread = pd.DataFrame(rows)
    spread.round(3).to_csv(OUT / "spread_calibration.csv", index=False)
    all_pairs = [
        (
            all_draws[(s, "oracle")]["stack"],
            panel[panel.year == s].sort_values("state").inc_share.to_numpy(),
        )
        for s in TARGETS
    ]
    recal["k_for_2027"] = fit_k(all_pairs)
    summary["recalibration"] = recal
    st = spread[(spread.model == "stack")]
    summary["spread_scores"] = {
        mode: {
            "brier_stack": float(np.mean((1 - st[st["mode"] == mode].p_spread) ** 2)),
            "log_stack": float(np.mean(-np.log(np.clip(st[st["mode"] == mode].p_spread, 1e-6, 1)))),
            "brier_always_met": 0.0,
        }
        for mode in ("oracle", "forecast")
    }

    # (F) Brier skill score for each state's 25% event vs state climatology
    clim_rows = []
    for t in TARGETS:
        tr = panel[panel.year < t]
        clim = tr.groupby("state").inc_share.apply(lambda s: float(np.mean(s >= 0.25)))
        test = panel[panel.year == t].sort_values("state")
        ev = (test.inc_share.to_numpy() >= 0.25).astype(float)
        clim_rows.append(
            pd.DataFrame(
                {
                    "year": t,
                    "state": test.state,
                    "event": ev,
                    "clim": clim.loc[test.state].to_numpy(),
                }
            )
        )
    clim = pd.concat(clim_rows)
    bs_ref = float(np.mean((clim.clim - clim.event) ** 2))
    bss = {}
    for mode in ("oracle", "forecast"):
        for n in SHOW + ["ps_home", "sar_region", "lvl_home"]:
            sub = ps[(ps["mode"] == mode) & (ps.model == n)].merge(clim, on=["year", "state"])
            bs = float(np.mean((sub.p25 - sub.event) ** 2))
            bss[f"{mode}:{n}"] = {"brier": bs, "bss_vs_climatology": 1 - bs / bs_ref}
    summary["brier25"] = {
        "climatology_brier": bs_ref,
        "models": bss,
        "event_rate": float(clim.event.mean()),
    }

    # (G) Logit-shift weights: lagged valid-vote weights vs actual
    totals = pd.read_csv("data/external/inec_historical/state_totals.csv")
    from ngforecast.data.geography import state_code

    totals["state"] = totals.state_name.map(state_code)
    rows = []
    for t in TARGETS:
        test = panel[panel.year == t].sort_values("state")
        w_lag = test.weight.to_numpy() / test.weight.sum()
        v = totals[totals.year == t].set_index("state").loc[test.state, "valid"].to_numpy(float)
        w_act = v / v.sum()
        y = test.inc_share.to_numpy()
        rows.append(
            {
                "year": t,
                "true_national": float(test.nat_share.iloc[0]),
                "lag_weighted_actual": float(w_lag @ y),
                "gap_pp": 100 * float(w_lag @ y - test.nat_share.iloc[0]),
                "actual_weighted_actual": float(w_act @ y),
            }
        )
    pd.DataFrame(rows).round(4).to_csv(OUT / "logit_shift_weights.csv", index=False)

    # (H) National: naive no-change benchmark, two-party share, fragmentation
    res = pd.read_csv("data/external/inec_historical/results_long.csv")
    nat_votes = res.groupby(["year", "party"]).votes.sum()
    rows = []
    for t in [2003, 2011, 2015, 2019, 2023]:
        v = nat_votes.loc[t]
        v = v[v.index != "Others"]
        sh = v / res[res.year == t].groupby("party").votes.sum().sum()
        enp = 1 / float(np.sum(sh**2))
        inc = panel[panel.year == t].incumbent_party.iloc[0]
        other = v.drop(inc).sort_values(ascending=False)
        rows.append(
            {
                "year": t,
                "incumbent": inc,
                "inc_share": float(sh[inc]),
                "main_opponent": other.index[0],
                "inc_two_party_share": float(v[inc] / (v[inc] + other.iloc[0])),
                "enp_votes": enp,
                "top_two_share": float((v[inc] + other.iloc[0]) / v.sum()),
            }
        )
    frag = pd.DataFrame(rows)
    frag.round(4).to_csv(OUT / "fragmentation.csv", index=False)
    m = pd.read_csv(V2 / "metrics.csv")
    nat = m[(m.model == "ps_null") & (m.national == "forecast")][
        ["year", "national_pred", "national_true"]
    ]
    lagn = panel.groupby("year").nat_share_lag.first()
    nat["naive_pred"] = nat.year.map(lagn)
    nat["model_err_pp"] = 100 * (nat.national_pred - nat.national_true)
    nat["naive_err_pp"] = 100 * (nat.naive_pred - nat.national_true)
    nat.round(4).to_csv(OUT / "national_benchmark.csv", index=False)
    summary["national_benchmark"] = {
        "model_mae": float(nat.model_err_pp.abs().mean()),
        "naive_mae": float(nat.naive_err_pp.abs().mean()),
        "model_rmse": float(np.sqrt(np.mean(nat.model_err_pp**2))),
        "naive_rmse": float(np.sqrt(np.mean(nat.naive_err_pp**2))),
    }

    # (I) National prior sensitivity
    sens = national_sensitivity(panel)
    sens.round(4).to_csv(OUT / "national_prior_sensitivity.csv", index=False)

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(mean_scores.round(2))
        print(pd.read_csv(OUT / "paired_crps_differences.csv"))
        print(pd.read_csv(OUT / "forecast_oracle_decomposition.csv"))
        print(pd.read_csv(OUT / "stacking_weights_robustness.csv"))
        print(pd.read_csv(OUT / "stack_forecast_mode_weights.csv"))
        print(spread.round(3))
        print(pd.read_csv(OUT / "logit_shift_weights.csv"))
        print(frag.round(3))
        print(nat.round(3))
        print(sens.round(3))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
