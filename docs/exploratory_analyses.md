# Exploratory analyses (post-review, 2026-11-23)

These analyses were added in response to a referee-style review of the IJF
manuscript, **after** the v2 results of record were seen. They are not part of
the frozen protocol (`docs/evaluation_protocol.md`, unchanged so its hash
stays valid), and they change neither the result of record
(`outputs/backtest/v2`) nor the pre-registered 2027 forecast
(`outputs/preregistered/2027`). The paper labels them as exploratory.

## Reproducibility checks
- `scripts/run_backtest.py --tag v2_draws --save-draws` re-runs v2 with the
  same seeds and also saves predictive draws. Its `metrics.csv` matches v2
  exactly (checked in `exploratory_review.py`, `reproduces_v2_metrics: true`).
- `scripts/exploratory_2027.py` re-runs `forecast_2027.py` into
  `outputs/exploratory/2027/reproduction/`; `states.csv` and `conditional.csv`
  are byte-identical to the pre-registered files.

## Analyses and outputs (`outputs/exploratory/`)
| Review point | Analysis | Script | Output |
|---|---|---|---|
| Baselines | Pure probabilistic uniform swing (PUS): uniform-swing median + logit noise with training RMS | `exploratory_review.py` | `review/mean_scores.csv`, `scores_by_year.csv` |
| Baselines | Naive no-change national benchmark | same | `review/national_benchmark.csv` |
| Inference | Per-election paired CRPS differences, 90% intervals resampling states | same | `review/paired_crps_differences.csv` |
| Mode reversal | Forecast-vs-oracle decomposition of state errors | same | `review/forecast_oracle_decomposition.csv` |
| Stacking | Forecast-mode weights, KDE densities, resampled-state weight uncertainty | same | `review/stacking_weights_robustness.csv`, `stack_forecast_mode_weights.csv` |
| §134 spread | Calibration of the count of units ≥ 25%; prequential interval narrowing | same | `review/spread_calibration.csv`, `summary.json` |
| Brier 25% | Brier skill score vs state climatology | same | `review/summary.json` |
| Logit shift | Lagged vs actual valid-vote weights | same | `review/logit_shift_weights.csv` |
| National model | Prior sensitivity (5 variants) | same | `review/national_prior_sensitivity.csv` |
| Target | Two-party share and effective number of parties | same | `review/fragmentation.csv` |
| SBC | Simulation-based calibration, national model + `pp_region` state model | `sbc.py` | `sbc/ranks.csv`, `sbc/summary.json` |
| 2027 | 1-point conditional grid; narrowed variant (k = 0.51) | `exploratory_2027.py` | `2027/conditional_fine.csv` |

## Main findings
- Oracle mode: stack CRPS 7.86 vs PUS 9.12 (−14%); stack beats PUS in all
  three elections (2023 inconclusive). Forecast mode: no model reliably beats
  the baselines.
- National: model MAE 12.1 vs naive no-change 11.5 (RMSE 12.7 vs 13.6).
  2019→2023: incumbent share −19.0 pts, two-party share −1.7 pts, ENP 2.1→3.5.
- Mode reversal: reconciled models pass national errors on in full; logit
  uniform swing (unreconciled) attenuates them (state medians moved
  7.5/4.4/13.6 pts vs 14.6/8.4/19.9 for the stack).
- §134 spread under-forecast because the count of units ≥ 25% was
  under-predicted (outcome at the 83rd/98th/99.9th percentile); dispersion of
  state deviations fell from 1.65–2.06 (2003–2015) to 1.06/1.26 (2019/2023).
  Narrowing intervals raises P(spread) but worsens coverage (84%, 68%).
- Stacking weights robust in direction (pp_region > 0.5 in 97% of resamples
  for 2027) but uncertain in size.

## Second review round (2026-11-23)
| Review point | Analysis | Script | Output |
|---|---|---|---|
| Fair baselines | Additive, reconciled logit uniform swing; reconciled PUS | `exploratory_review2.py` | `review2/mean_scores.csv`, `scores_by_year.csv` |
| Joint evaluation | Energy score, variogram score (p = 0.5), count RPS/PIT, interval width, PIT histograms | same | `review2/*` |
| Inference | Zone-block bootstrap (6 zones); Monte Carlo SE of the stack | same | `review2/paired_zone_bootstrap.csv`, `stack_mc_error.csv` |
| Hindsight covariate | Prequential stack without the region family | same | `review2/mean_scores.csv` |
| 2023 spread miss | Regional errors, threshold misses | same | `review2/regional_errors.csv`, `summary.json` |
| Dependence | Zone-shock variants of pp_region / ps_region (`bayes.fit(zone_shock=True)`) | `exploratory_zone_shock.py` | `zone_shock/` |
| Generality / power | Simulation study (power of 3-election comparisons; threshold-count calibration) | `simulation_study.py` | `outputs/simulation/` |
| 2027 forecast | Decomposition of state forecasts vs 2023 | `exploratory_2027_decomp.py` | `2027/decomposition.csv` |
| Reproducibility | Clean checkout of the public commit rebuilds `panel.parquet` and the forecast `states.csv`/`conditional.csv` byte for byte | (manual, git worktree) | — |

Corrections to the first round:
- The 2023 spread miss is a regional bias: northern states were under-predicted, southern
  states over-predicted, and the medians were less dispersed than the outcomes. It was not
  "the vote became more even".
- Unreconciled uniform swing attenuating national errors does not explain the forecast-mode
  reversal, because reconciled uniform swing does as well. The reversal is consistent with
  chance (simulation study).

Second pre-registered forecast: `docs/evaluation_protocol_v3.md`, `scripts/forecast_2027_v3.py`,
`outputs/preregistered/2027_v3/`. v2 remains primary.
