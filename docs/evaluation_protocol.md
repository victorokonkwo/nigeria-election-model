# Evaluation Protocol v2 — Model Development Without Test-Set Selection

Status: **frozen before the v2 runs.** The SHA-256 of this file is recorded in
every `outputs/backtest/<tag>/run_config.json`. Results produced under a
different hash are not v2 results.

## 1. History (disclosed)
- **v1a** (first run): level models (`sar_level`, `nonspatial_level`). The
  persistence parameter φ was identified from only 1999→2003 and 2003→2011
  (two regional realignments) and the models underperformed uniform swing.
- **v1b**: swing models (φ = 1) added *after* seeing v1a. Their 2015/2019/2023
  scores are therefore not clean out-of-sample evidence for choosing them.

v2 exists to obtain an honest estimate despite this history.

## 2. Candidate models (fixed list; no additions after freezing)
All share the national swing model and the logit deviation decomposition.

| id | φ | spatial ρ | state covariates |
|---|---|---|---|
| `ps_null`   | 1 (fixed)   | no  | none (probabilistic uniform swing: intercept, state effect, noise) |
| `ps_home`   | 1 (fixed)   | no  | home state/zone (incumbent & challengers), current + lag, violence_z |
| `ps_region` | 1 (fixed)   | no  | `ps_home` + North/South region match (§3), current + lag |
| `sar_region`| 1 (fixed)   | yes | as `ps_region` |
| `pp_region` | ~ N(1, 0.2) | no  | as `ps_region` (partial persistence, shrunk toward uniform swing) |
| `lvl_home`  | ~ N(0, 1)   | no  | as `ps_home` (v1 level model, kept for reference) |

Baselines: `uniform_swing` (deterministic) and `persistence`.

## 3. The one new covariate family (theory-motivated, not tuned)
Nigeria's dominant electoral cleavage is North vs South (zones NC, NE, NW incl.
FCT vs SE, SS, SW), the axis of the zoning/rotation convention. Defined
pre-election from `config/candidates.yaml`:
- `home_region` = 1 if the state is in the incumbent-party candidate's region.
- `opp_home_region` = 1 if the state is in the **main challenger's** region.
  The main challenger is the first entry of `major_challengers`, which lists
  parties by their (or their predecessors') previous-election national share —
  knowable before election day.
- `_lag` versions refer to the same party's candidate and main opponent at the
  previous election with results.

## 4. Headline model: prequential stacking
For target election t, each candidate is fitted on elections < t. The ensemble
weights for t are chosen by **stacking of predictive densities** (Yao et al.
2018) using only the backtests of targets < t:
- 2015: no earlier backtest → equal weights.
- 2019: weights from the 2015 backtest.
- 2023: weights from the 2015 and 2019 backtests.
- 2027: weights from 2015, 2019 and 2023.

Densities are evaluated on the logit scale with a Gaussian approximation of each
model's state-level predictive draws, in **oracle-national mode** (the national
component is shared by all candidates, so weights reflect state-level skill).
The stacked predictive distribution is the weighted mixture of draws. Because
the weights for t never see t, the stack's score at t is out-of-sample.

## 5. Metrics
- **Primary:** mean state-level CRPS (percentage points) of the prequential
  stack, in forecast mode and oracle mode, averaged over 2015, 2019, 2023.
  Compared against `uniform_swing` and `ps_null`.
- Secondary: state RMSE/MAE of the posterior median, 90% interval coverage,
  Brier score for the 25% threshold, national error, P(§134 spread met).
- 2015 is reported but is a weak test (two training elections).

## 6. Rules
- All candidate models and both baselines are reported, whatever they show.
- No model, covariate, prior or metric is added or changed after freezing,
  except bug fixes, which are listed in §7 with their effect on results.
- The 2027 forecast uses the stack with weights from all three backtests and is
  pre-registered (forecast log) before election day.

## 7. Post-freeze changes
1. **Sampler setting (computational, not a model change).** The first v2 run
   (`outputs/backtest/v2_ta095`, NUTS `target_accept` 0.95) had 26 divergences
   in the 2015 national fit (two observations) and 45 in the 2019 `lvl_home`
   state fit. Rerun with `target_accept` 0.99, model list, priors and weights
   procedure unchanged. Both runs are kept; the 0.99 run is the result of
   record. The protocol hash in `run_config.json` of the 0.99 run therefore
   differs from that of the 0.95 run only by this section.
