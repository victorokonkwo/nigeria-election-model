# ADR-003: Interpretation of the §134 Spread Requirement and Runoff

## Status
Accepted (supersedes the earlier hard-coded "25 of 37 units" rule)

## Context
§134(2)(b) of the 1999 Constitution requires the winner to have "not less than one-quarter of the votes cast at the election in each of at least two-thirds of all the States in the Federation and the Federal Capital Territory, Abuja." The code previously required ≥ 25% in ⌈37 × 2/3⌉ = 25 whole units. That is stricter than any reading a Nigerian court has adopted, and it inflated the simulated runoff probability, which is the model's headline output.

Two questions decide the rule:

1. **How is the FCT counted?** In *Obi v INEC* and *Atiku v INEC* (Supreme Court, 26 Oct 2023) the Court upheld the Presidential Election Petition Court. Applying §299, it held that the FCT is treated as one of the states (the 37th unit) and that 25% in the FCT is not a separate requirement.
2. **How is a fractional requirement met?** Two-thirds of 37 units is 24⅔. In *Awolowo v Shagari* (SC.62/1979, 6–1), two-thirds of 19 states was held to be 12⅔ states. Shagari's 25% in 12 states plus 19.94% in Kano (more than ⅔ × 25%) was enough. The 2023 courts did not need to decide the fraction, since the winner cleared 25% in 29 units.

The second election had two further errors. The code chose the national runner-up as the second finalist, but §134(3)(b) names the remaining candidate "who has a majority of votes in the highest number of States." It also gave 21 days where §134(4)–(5) say seven, and it had no third election.

## Decision
- **Default:** `fct_interpretation: fct_as_state` with `fractional_rule: awolowo_v_shagari`. A win needs 24 units at ≥ 25% plus a 25th unit at ≥ 16.67%.
- **Sensitivity options**, to be reported in the paper:
  - `states_only`: 24 of 36 states, with the FCT ignored.
  - `states_plus_fct`: 24 states plus 25% in the FCT, which was the petitioners' 2023 reading and was rejected.
  - `ceil`: 25 whole units, the old behaviour.
  - `floor`: 24 whole units.
- **Second election (§134(3)–(4)):** the national leader faces the remaining candidate who led in the most units. Ties go to the higher national total. The winner needs a majority plus the spread above. If that fails, a **third election (§134(5))** is decided by simple majority.

## Consequences
- Simulated runoff probabilities will be lower than under the old rule. Any forecast produced before this change is not comparable and should be noted in the forecast log.
- "Majority of votes in the highest number of States" is read as *led (had the most votes) in the most units*. This reading is untested in court and should be stated as an assumption.
- The pre-registration should freeze the default rule and list the sensitivity variants before the 2027 forecast is published.
