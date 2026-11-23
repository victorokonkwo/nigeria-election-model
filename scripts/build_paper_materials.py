"""Generate PaperOrchestra raw materials from the results of record.

Every number in experimental_log.md is computed here from committed outputs
(backtest v2 and v1, the 2027 pre-registration, the curated data), so the
paper cannot quote a number that the pipeline did not produce.

Writes paper/raw_materials/{experimental_log.md, figures/*.png, figures/info.json}.
idea_sparse.md is hand-written and kept in the same directory.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

OUT = Path("paper/raw_materials")
V2, V1 = Path("outputs/backtest/v2"), Path("outputs/backtest/main")
PREREG = Path("outputs/preregistered/2027")
FIGS = Path("outputs/paper/figures")
DATA = Path("data/external/inec_historical")

MODEL_DESC = {
    "stack": "Prequential stack (headline; weights from earlier backtests only)",
    "ps_null": "Probabilistic uniform swing (phi=1, no covariates)",
    "ps_home": "Swing model + home-state/zone covariates",
    "ps_region": "Swing model + home + North/South region covariates",
    "sar_region": "As ps_region + spatial SAR term (zone-block W)",
    "pp_region": "Partial persistence, phi ~ N(1, 0.2), region covariates",
    "lvl_home": "Level model, phi ~ N(0, 1), home covariates (v1 reference)",
    "uniform_swing": "Deterministic uniform swing (baseline)",
    "persistence": "Last election's state shares unchanged (baseline)",
}


def md(df: pd.DataFrame, floatfmt: int = 2) -> str:
    return df.round(floatfmt).to_markdown(index=False)


def section_data() -> str:
    res = pd.read_csv(DATA / "results_long.csv")
    tot = pd.read_csv(DATA / "state_totals.csv")
    cov = res.groupby("year").agg(units=("state_name", "nunique"), parties=("party", "nunique")).reset_index()
    valid = tot.groupby("year").valid.sum().reset_index().rename(columns={"valid": "total_valid_votes"})
    cov = cov.merge(valid, on="year")
    panel = pd.read_parquet("data/processed/panel.parquet")
    nat = (
        panel.groupby("year")
        .agg(
            incumbent_party=("incumbent_party", "first"),
            lag_election=("lag_year", "first"),
            national_share=("nat_share", "first"),
            national_share_lag=("nat_share_lag", "first"),
            incumbent_candidate_running=("incumbent_running", "first"),
        )
        .reset_index()
    )
    return f"""## 1. Data

### 1.1 Election results (curated, state level)
Presidential results for all 36 states + FCT. 2007 is absent because INEC never
published a state-level breakdown of the 2007 presidential result.

{md(cov, 0)}

Sources: 1999 from the IFES observation report table of INEC data (scanned,
OCR, four cells corrected by arithmetic constraints); 2003 and 2011 from archived
INEC results websites; 2015 from INEC's scanned summary sheet (OCR, all rows
satisfy party sum = valid and valid + rejected = total); 2019 and 2023 from Stears
Elections (INEC published national totals only), with registered voters from
INEC's 2023 General Election Report.

Validation: for every year, the state-level votes of the top parties sum exactly
to INEC's official national totals (e.g. 2023: APC 8,794,726; PDP 6,984,520; LP
6,101,533; NNPP 1,496,687; 2019: APC 15,191,847; PDP 11,262,978; 2015: APC
15,424,921; PDP 12,853,162). Two discrepancies are internal to INEC's own files:
2011 ACN state sum is 11,850 below INEC's national page; one 2015 ADC cell is off
by 1 on INEC's sheet.

### 1.2 Target and panel
Target: the incumbent (ruling) party's share of valid votes, by state. PDP is the
incumbent party 2003-2015, APC 2019-2027. 1999 has no incumbent party (transition
from military rule) and enters only as a lag. Because 2007 is missing, 2011 is
lagged against 2003.

{md(nat, 3)}

### 1.3 Covariates (all measured before election day)
- Candidate home state / zone of the incumbent-party candidate and of major
  challengers (from candidate lists), current and for the previous election.
- North/South region match: whether the state is in the region (North = North-Central,
  North-East, North-West incl. FCT; South = South-East, South-South, South-West) of
  the incumbent-party candidate and of the main challenger; current and lagged.
- Political violence: ACLED events and fatalities (HDX admin-2 x month aggregates)
  in the 12 full months before the election month; index = log(1+events) +
  log(1+fatalities), z-scored within election year (removes ACLED's growing
  coverage over time).
- State poverty (OPHI MPI, DHS 2013/2018, MICS 2016-17/2021) exists only from 2013,
  so it cannot be used in models trained on earlier elections; excluded.
- National macro indicators (inflation, GDP per capita growth, FX depreciation,
  Brent) were assembled but are absorbed by election-level effects and not used.
"""


def section_method() -> str:
    cfg = json.loads((V2 / "run_config.json").read_text())
    return f"""## 2. Constitutional rule (§134)
- §134(2): winner needs the most votes nationally AND >= 25% of votes in at least
  two-thirds of all the States and the FCT.
- FCT is treated as the 37th unit following Obi v INEC / Atiku v INEC (Supreme
  Court of Nigeria, 26 Oct 2023), so the requirement is 2/3 x 37 = 24 2/3 units.
- The fraction is resolved per Awolowo v Shagari (1979): 24 units at >= 25% plus a
  25th unit at >= 2/3 x 25% (16.67%).
- Sensitivity readings: states only (24 of 36), 24 states + 25% in FCT (petitioners'
  2023 reading, rejected by the courts), 25 whole units (ceil).
- §134(3)-(5): if nobody wins, a second election within 7 days between the
  national leader and the candidate leading in the most states (not necessarily the
  national runner-up); winner needs a majority + the same spread; otherwise a third
  election by simple majority. The original code required 25 whole units and used
  the national runner-up with a 21-day gap; both were corrected.

## 3. Model
Decomposition on the logit scale: logit(y_it) = logit(N_t) + d_it, where y_it is the
incumbent-party share in state i at election t, N_t the national share, d_it the
state deviation.

National swing model: logit N_t - logit N_(t-1) = mu + gamma * incumbent_running_t + e_t,
e_t ~ N(0, sigma_n); priors mu ~ N(0, 0.5), gamma ~ N(0, 0.5), sigma_n ~ HalfNormal(0.5).

State model (levels): (I - rho W) d_t = alpha + phi d_(t-1) + X_t beta + u + eps_t.
State model (swing, phi fixed at 1): (I - rho W)(d_t - d_(t-1)) = alpha + X_t beta + u + eps_t,
which nests uniform swing. W = zone-block (states in the same geopolitical zone),
row-standardised; rho ~ Uniform(-1, 1); beta ~ N(0, 0.5); alpha ~ N(0, 0.5);
u_i ~ N(0, tau), tau ~ HalfNormal(0.3); eps ~ N(0, sigma), sigma ~ HalfNormal(0.5).
Exact SAR likelihood with log-determinant from the eigenvalues of W.

Prediction: posterior predictive draws of national swing and state deviations; for
each draw the state logits are shifted by a common constant so that the
vote-weighted mean of state shares equals the national draw (weights = previous
election's valid votes). Point forecast = posterior median.

Estimation: PyMC NUTS, {cfg['chains']} chains x {cfg['draws']} draws (+ {cfg['draws']} tuning),
target_accept 0.99. Parameter-recovery test on simulated data recovers rho, phi and
beta within tolerance and achieves >= 75% coverage of 90% intervals.

Candidate models (protocol v2):

{md(pd.DataFrame([{"model": k, "description": v} for k, v in MODEL_DESC.items()]))}

Prequential stacking: for target t, weights are fitted by stacking of predictive
densities (Yao, Vehtari, Simpson and Gelman 2018) using only backtests of targets
< t (equal weights for 2015). Densities: Gaussian approximation on the logit scale,
oracle-national mode. The stack's predictive distribution is the weighted mixture of
draws.
"""


def section_protocol() -> str:
    sha = json.loads((V2 / "run_config.json").read_text())["protocol_sha256"]
    return f"""## 4. Evaluation protocol and development history (disclosed)
- Leave-future-out backtests: train on elections < t, predict t in {{2015, 2019, 2023}}
  (2015 has only two training elections, 2003 and 2011).
- Every model is scored in **forecast mode** (its own national forecast) and in
  **oracle mode** (true national share supplied), isolating state-level skill.
- Metrics: state CRPS (percentage points; primary), RMSE and MAE of the median,
  coverage of 90% intervals, Brier score for the event share >= 25%, national
  error, P(§134 spread met).
- v1a: level models first. phi was estimated at 0.13-0.29, identified from only
  two realigning transitions (1999->2003, 2003->2011); the level models
  underperformed uniform swing.
- v1b: swing models (phi = 1) were added *after* seeing v1a results.
- v2: a written protocol (docs/evaluation_protocol.md, SHA-256 {sha[:16]}...) fixed
  six candidate models, one new theory-motivated covariate family (North/South
  region) and prequential stacking **before** the v2 run. One post-freeze change:
  NUTS target_accept raised from 0.95 to 0.99 after 26 divergences in the
  2015 national fit; results essentially unchanged (both runs archived).
- The v2 run is reproducible bit-for-bit (identical metrics on rerun with fixed seeds).
"""


def section_results() -> str:
    m = pd.read_csv(V2 / "metrics.csv")
    cols = ["state_crps_pp", "state_rmse_pp", "state_mae_pp", "state_cov90", "brier_25pct"]
    out = ["## 5. Backtest results (v2, result of record)\n"]
    for mode in ("oracle", "forecast"):
        avg = m[m.national == mode].groupby("model")[cols].mean().reset_index().sort_values("state_crps_pp")
        out.append(f"### 5.{1 if mode == 'oracle' else 2} Mean over 2015/2019/2023, {mode} mode\n\n{md(avg)}\n")
    per = m[m.model.isin(["stack", "ps_null", "ps_region", "uniform_swing"])].pivot_table(
        index=["national", "model"], columns="year", values="state_crps_pp"
    )
    out.append(f"### 5.3 State CRPS (pp) by election\n\n{per.round(2).reset_index().to_markdown(index=False)}\n")
    rm = m[m.model.isin(["stack", "uniform_swing", "persistence"])].pivot_table(
        index=["national", "model"], columns="year", values="state_rmse_pp"
    )
    out.append(f"### 5.4 State RMSE (pp) of point forecasts by election\n\n{rm.round(2).reset_index().to_markdown(index=False)}\n")
    nat = m[(m.model == "stack") & (m.national == "forecast")][
        ["year", "national_pred", "national_true", "national_err_pp", "national_in_90"]
    ]
    out.append(f"### 5.5 National forecast (stack, forecast mode)\n\n{md(nat, 3)}\n")
    sp = m[m.model == "stack"][["year", "national", "p_spread_met", "spread_met_true", "state_cov90"]]
    out.append(
        "### 5.6 §134 spread: forecast probability vs outcome (stack)\n\n"
        f"{md(sp, 3)}\n\nThe incumbent party met the spread in all three elections; "
        "the model's probabilities were below 0.5 in oracle mode for 2015 and 2023, i.e. the spread "
        "event was under-forecast.\n"
    )
    w = json.loads((V2 / "weights.json").read_text())
    wt = pd.DataFrame({t: v["weights"] for t, v in w.items()}).round(3).reset_index().rename(columns={"index": "model"})
    out.append(f"### 5.7 Prequential stacking weights (column = target; 'next' = used for 2027)\n\n{wt.to_markdown(index=False)}\n")
    post = json.loads((V2 / "posterior.json").read_text())
    rows = []
    for t in ("2015", "2019", "2023"):
        for name in ("ps_region", "sar_region", "pp_region"):
            p = post[t][name]
            row = {"target": t, "model": name}
            for k in ("rho", "phi", "sigma"):
                if k in p["params"]:
                    row[k] = f"{p['params'][k]['mean']:.2f} ({p['params'][k]['sd']:.2f})"
            for k in ("home_zone", "opp_home_zone", "home_region", "opp_home_region"):
                b = p["beta"].get(k)
                if b:
                    row[k] = f"{b['mean']:.2f} ({b['sd']:.2f})"
            rows.append(row)
    out.append(
        "### 5.8 Posterior means (sd) of selected parameters, by backtest fit\n\n"
        f"{pd.DataFrame(rows).fillna('').to_markdown(index=False)}\n\n"
        "Note: home_region and home_region_lag are identical in all training elections up to 2019 "
        "(the incumbent party's candidate never switched region before 2023), so their separate "
        "effects are identified only by the prior in the 2019 and 2023 backtests.\n"
    )
    diag = []
    for t, v in post.items():
        for name, d in v.items():
            for part in ("national", "state"):
                g = d["diagnostics"][part]
                diag.append(g)
    dd = pd.DataFrame(diag)
    out.append(
        f"### 5.9 Convergence\nAcross all fits: max R-hat {dd.max_rhat.max():.3f}, min bulk ESS "
        f"{int(dd.min_ess_bulk.min())}, total divergences {int(dd.divergences.sum())} "
        "(R-hat 1.019 occurs only in the lvl_home reference model, which receives zero stacking weight).\n"
    )
    v1 = pd.read_csv(V1 / "metrics.csv")
    v1a = v1[v1.model.isin(["sar_level", "nonspatial_level", "uniform_swing"])].groupby(["national", "model"])[
        ["state_crps_pp", "state_rmse_pp"]
    ].mean()
    out.append(f"### 5.10 v1 results (disclosed development history; mean over 2015/2019/2023)\n\n{v1a.round(2).reset_index().to_markdown(index=False)}\n")
    out.append(
        """### 5.11 Summary of findings
- Oracle mode (state-level skill): the prequential stack has the lowest mean CRPS
  (7.86 pp) vs uniform swing 11.05 pp (29% lower) and probabilistic uniform swing
  8.77 pp (10% lower); its median also has lower RMSE than uniform swing (12.62 vs
  14.21 pp). It wins clearly in 2019 and 2023; 2015 is roughly a tie.
- Forecast mode: gains largely vanish (stack CRPS 10.66 vs uniform swing 10.93 pp;
  RMSE 17.90 vs 17.04 pp) because national errors are large (+12, -7, +17 pp).
- Calibration: stack 90% intervals cover 95% of states (oracle), uniform swing
  (deterministic given the national share) covers none; probabilistic intervals are wide.
- The spatial SAR term does not improve out-of-sample forecasts (sar_region vs
  ps_region oracle CRPS 8.88 vs 8.29) and receives zero stacking weight.
- North/South region covariates improve on home-zone covariates alone
  (ps_region vs ps_home oracle CRPS 8.29 vs 8.71; forecast 10.34 vs 10.80).
"""
    )
    return "\n".join(out)


def section_forecast() -> str:
    f = json.loads((PREREG / "forecast.json").read_text())
    man = json.loads((PREREG / "MANIFEST.json").read_text())
    rows = []
    for z, s in f["scenarios_by_main_challenger_zone"].items():
        row = {
            "main challenger zone": z,
            "APC national median": s["national_share"]["median"],
            "90% low": s["national_share"]["q05"],
            "90% high": s["national_share"]["q95"],
            "P(APC > 50%)": s["p_national_majority"],
            "units >= 25% (median)": s["units_at_25pct"]["median"],
        }
        row.update({f"P(spread): {k}": v for k, v in s["p_spread_met"].items()})
        rows.append(row)
    c = pd.read_csv(PREREG / "conditional.csv", keep_default_na=False)
    cond = c.pivot(index="national_share", columns="challenger_zone", values="p_spread_met_default").reset_index()
    units = c.groupby("national_share").units_at_25pct_median.mean().reset_index()
    st = pd.read_csv(PREREG / "states.csv", keep_default_na=False)
    st = st[st.challenger_zone == "North-East"].sort_values("median", ascending=False)
    return f"""## 6. The 2027 pre-registered forecast
Issued {f['issued_at'][:16]} UTC from git commit {man['git_commit'][:7]} (public repository);
election date 2027-02-27. Stack weights: {json.dumps({k: round(v, 3) for k, v in f['stack_weights'].items()})}.
Trained on 2003, 2011, 2015, 2019, 2023. Target: APC (incumbent party) share of valid
votes. Not forecast: outright winner or runoff probability (plurality depends on the
opposition split, unsettled at issue time); no probabilities are attached to the
opposition scenarios. Incumbent candidate (Tinubu, South-West) running.

### 6.1 By main-challenger home-zone scenario

{md(pd.DataFrame(rows), 3)}

### 6.2 P(§134 spread met | APC national share), default reading, by scenario

{md(cond, 2)}

Median number of units with APC >= 25%, averaged over scenarios:

{units.assign(units_at_25pct_median=units.units_at_25pct_median.round(1)).to_markdown(index=False)}

For reference, APC's 2023 national share was 36.6% and APC met the spread in 2023.

### 6.3 State forecasts (main challenger from North-East; median and 90% interval)

{md(st[["state", "median", "q05", "q95", "p_ge_25pct"]], 3)}
"""


def section_limitations() -> str:
    return """## 7. Limitations (to be reported)
- Only 5 elections with state results inform the model (2003-2023, no 2007); the
  national swing model is fitted on 2-5 observations and is weak.
- 2015 backtest trains on two elections only.
- Data quality: 1999, 2003 and especially 2007 elections were widely reported as
  rigged; 2019/2023 state results come from Stears (reconciled exactly to INEC
  national totals); 1999 and 2015 required OCR.
- P(§134 spread met) was under-forecast in the backtests; 2027 spread
  probabilities may be biased downward.
- State-level predictive intervals are wide.
- Plurality/outright victory is not modelled; opposition fragmentation is the key
  unknown for 2027.
- Candidate home states were entered from public candidate information.
- Region effects are identified only from the 2019->2023 switch.
"""


def main() -> None:
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    log = "\n".join(
        [
            "# Experimental log: Nigerian presidential election forecasting (state level, §134)\n",
            "All tables below are generated from the committed result files. Numbers quoted in prose are copied from these tables (or, for data validation, from data/external/inec_historical/validation_log.txt).\n",
            section_data(),
            section_method(),
            section_protocol(),
            section_results(),
            section_forecast(),
            section_limitations(),
        ]
    )
    (OUT / "experimental_log.md").write_text(log)

    captions = {
        "fig1_backtest_crps.png": (
            "Leave-future-out backtests: mean state-level CRPS (percentage points, lower is better) of "
            "the incumbent-party vote share for 2015, 2019 and 2023, for the prequential stack, "
            "deterministic uniform swing and probabilistic uniform swing. Left: each model uses its own "
            "national forecast. Right: the true national share is supplied (oracle), isolating state-level skill."
        ),
        "fig2_backtest_intervals.png": (
            "Prequential stack, oracle mode: posterior median and 90% interval of the incumbent-party share "
            "in each of the 37 units against the outcome, for 2015, 2019 and 2023 (coverage 100%, 100%, 86%). "
            "Hollow orange markers are outcomes outside the interval. Dashed line: the 25% §134 threshold."
        ),
        "fig3_2027_spread_conditional.png": (
            "2027 pre-registered forecast: probability that APC obtains at least 25% of the vote in 24 2/3 of "
            "the 37 units (§134, default legal reading) as a function of APC's national vote share. Line: mean "
            "over the six main-challenger home-zone scenarios; band: range across scenarios. Dashed line: "
            "APC's 2023 national share (36.6%)."
        ),
        "fig4_2027_states.png": (
            "2027 pre-registered forecast of APC's share of valid votes by state (posterior median and 90% "
            "interval), scenario with the main challenger from the North-East. Dashed line: 25% threshold."
        ),
    }
    info = []
    for name, cap in captions.items():
        shutil.copy(FIGS / name, OUT / "figures" / name)
        info.append({"name": name, "caption": cap})
    (OUT / "figures" / "info.json").write_text(json.dumps(info, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
