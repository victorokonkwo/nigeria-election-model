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
