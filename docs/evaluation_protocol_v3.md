# Evaluation Protocol v3: one change, a shared zone shock

Status: **written 2026-11-23, after the v2 results and the exploratory
post-review analyses were seen.** v3 does not replace v2. The v2 forecast
(`outputs/preregistered/2027`, protocol `docs/evaluation_protocol.md`) remains
the primary pre-registered forecast. v3 is a second, separately dated forecast
whose purpose is to test, out of sample, one improvement suggested by the
backtests.

## 1. Motivation (post hoc, disclosed)
- The v2 candidate models treat states as conditionally independent given the
  national draw.
- In the 2023 backtest the stack's errors were shared within regions: northern
  states were under-predicted by 7.9 points and southern states over-predicted
  by 9.0 points, with the true national share supplied.
- The predicted number of units at 25% was therefore too narrow. Its standard
  deviation was about 2 units, and the 2023 outcome fell at the 99.9th
  percentile.
- Adding an election-specific shock shared within geopolitical zones improved
  the exploratory backtests (`outputs/exploratory/zone_shock`).
- That improvement was found after the outcomes were seen, so it is not
  evidence. The 2027 election is the test.

## 2. The single change
Every stack member keeps its v2 specification and gains one term in the state
model:

`mean_it += omega * z_{zone(i),t}`, with `omega ~ HalfNormal(0.3)` and
`z ~ N(0, 1)`.

At prediction, new zone shocks are drawn. This is implemented as
`ngforecast.models.fundamentals.bayes.fit(..., zone_shock=True)`.

## 3. Fixed elements
- **Stack members and weights:** the v2 weights for 2027, re-normalised over
  members above 0.1% (`pp_region` 0.812, `ps_region` 0.188, from
  `outputs/backtest/v2/weights.json["next"]`). No re-weighting.
- **Unchanged from v2:** training data, national model, priors, sampler,
  scenarios, legal readings and outputs (`scripts/forecast_2027_v3.py`).

## 4. Evaluation
The v3 forecast is evaluated in the same way as v2
(`docs/preregistration.md`, including Amendment 1), and the two are compared
head to head. The comparison uses:
- state CRPS;
- the energy score over the 37 units;
- the ranked probability score of the count of units at 25%;
- the log score of the §134 spread event.

All comparisons are made in both forecast mode and oracle mode. Whichever
forecast scores better, v2 remains the pre-registered primary forecast.

## 5. Issue record
- **Issued forecast:** the v3 forecast was issued from commit `e97e877` (issue
  time in `outputs/preregistered/2027_v3/forecast.json`). A rerun from that
  commit reproduces its `states.csv` and `conditional.csv` byte for byte; see
  `outputs/exploratory/reproduction_check.json`.
- **Earlier draft run:** a local run before that commit (never published)
  differed slightly:
  - up to 1.5 points in state medians and up to 0.005 in scenario spread
    probabilities;
  - the sampler chains also differed;
  - the cause was not identified, and other model fits were running at the
    same time.
- The forecast of record is the one issued from `e97e877`.
