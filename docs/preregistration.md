# Pre-Registration: 2027 Nigerian Presidential Election

## Version
2.0 — forecast issued 2026-11-23T09:31Z from commit `4c18fbf`
(artifacts and SHA-256 hashes: `outputs/preregistered/2027/MANIFEST.json`).
Version 1.0 was a template that was never frozen or issued.

## What is forecast
The **APC (incumbent-party) share of valid votes** in each of the 36 states and
the FCT, and nationally, for the presidential election of 27 Feb 2027, plus the
probability that APC meets the §134(2)(b) spread requirement.

**Not forecast:** who wins outright, or the runoff probability. A win also
needs a national plurality, which depends on how the opposition vote splits.
The opposition line-up was not settled at issue time, so results are given
separately for each zone the main challenger could come from, with no
probabilities attached to those scenarios.

## Model (frozen)
- Model specification: `docs/evaluation_protocol.md` (v2, with the logged
  change in §7). The forecast uses the prequential stack with weights from the
  2015/2019/2023 backtests: `pp_region` 0.81, `ps_region` 0.19.
- Training data: state results for 2003, 2011, 2015, 2019 and 2023 (with 1999
  used as a lag; no state data exists for 2007). Sources are in
  `data/external/inec_historical/`.
- §134 is implemented in `src/ngforecast/simulation/constitution.py`. The
  default reading is FCT as the 37th unit (*Obi v INEC*, 2023), with the 24⅔
  requirement met per *Awolowo v Shagari* (1979). Three alternative readings
  are reported as sensitivity checks (ADR-003).

## Issued forecast (summary; full detail in `outputs/preregistered/2027/`)
| Main challenger's zone | APC national median (90% interval) | P(APC > 50%) | P(spread met), default reading |
|---|---|---|---|
| North-Central | 32.4% (18.8–50.8) | 0.05 | 0.25 |
| North-East | 32.4% (19.1–50.6) | 0.05 | 0.25 |
| North-West | 32.5% (19.2–50.4) | 0.05 | 0.29 |
| South-East | 32.5% (18.8–50.7) | 0.05 | 0.26 |
| South-South | 32.5% (18.8–50.8) | 0.05 | 0.24 |
| South-West | 32.4% (19.1–50.8) | 0.05 | 0.28 |

`conditional.csv` gives P(spread met) as a function of APC's national share,
which is the recommended way to read the forecast because the national swing is
weakly predictable. Two weaknesses were known at issue time:
- In the backtests, P(spread met) was **too low** when the true national share
  was supplied (2015: 0.46, 2023: 0.38, though the spread was met both times).
- The state 90% intervals are wide.

## Evaluation (after the election)
Once INEC's state results are published, the forecast will be scored against
them, under the scenario that matches the actual main challenger:
1. **State CRPS** (pp) of APC share, compared against uniform swing and
   probabilistic uniform swing, both computed from the same 2023 baseline.
2. **Coverage** of the state 90% intervals.
3. **National error** of APC share and whether it falls inside the 90% interval.
4. **Log score / Brier** for the §134 spread event, and for each state's 25% event.
5. All of the above also conditional on the true national share (oracle mode).

## Commitments
- No post-hoc changes to the 2027 forecast. Any later forecast (for example
  after candidate nominations) is a new, separately dated artifact, and this
  one remains the pre-registered one.
- All code, data and outputs are versioned in git. This forecast counts as
  pre-registered only once the commit containing it is public and
  timestamped, i.e. pushed or deposited (OSF/Zenodo), before 27 Feb 2027.

## Amendment 1 (2026-11-23): evaluation rules fixed before the election
This amendment changes no forecast number. It fixes, in advance, how the
forecast will be scored. It was written after a review of the paper.

**Scope and validity**
- The forecast assumes that APC's candidate is the sitting president
  (`incumbent_running = 1`). If APC nominates someone else, the forecast is
  still scored and the failed assumption is reported.

**Scenario selection**
- The scenario scored is the home zone of the *main challenger*: the
  presidential candidate whose party, or coalition of parties, had the highest
  combined 2023 national vote among the candidates on the 2027 ballot.
- If that cannot be determined from the ballot (for example, a coalition of
  parties that did not exist in 2023), the opposition candidate with the most
  2027 votes is used, and the choice is reported as outcome-informed.
- All six scenarios are also reported.

**Baselines** (all computed from the 2023 results)
1. Uniform swing, logit scale (as in `scripts/run_backtest.py`).
2. Probabilistic uniform swing, reconciled to the national share, with logit
   noise equal to the RMS of uniform-swing residuals over all training
   elections (as in `scripts/exploratory_review2.py`).
3. `ps_null`, fitted on all five elections with the v2 sampler settings.
4. National no-change forecast: APC repeats 36.6%.

**Scores and decision rules.** Every score is computed in forecast mode and in
oracle mode, with the true national share supplied. The forecast is judged to
have:
- **Beaten the baselines on state skill (H1)** if its oracle-mode state CRPS
  is below that of every baseline 1–3.
- **Beaten no-change on the national vote (H2)** if its absolute national
  error is below that of baseline 4. Coverage of the national 90% interval is
  also reported.
- **Calibrated state intervals (H3)** if 30–36 of the 37 outcomes, a band
  around the nominal 33, lie inside the 90% intervals.
- **Got the §134 spread event right (H4)**, assessed by the log score of
  P(spread met):
  - in forecast mode, compared with a climatological reference of 0.92
    ((5 + 1/2)/(5 + 1): the spread was met in all five elections with state
    results);
  - in oracle mode, using the conditional curve evaluated at the true
    national share.

**Joint scores.** The energy score over the 37 units, and the ranked
probability score of the count of units at 25%, are also reported.

With one election, none of these outcomes is conclusive, and they will be
reported as such.
