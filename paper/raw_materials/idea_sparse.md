# Idea: How predictable are Nigerian presidential elections at the state level? A pre-registered probabilistic forecast under the §134 constitutional rule

## Problem
Nigeria's 1999 Constitution (§134) requires a presidential winner to have the most
votes nationally AND at least 25% of the vote in two-thirds of the 36 states and the
FCT; otherwise second and third elections follow. Whether a candidate clears this
geographic spread is a state-level question, yet there is no public, validated
state-level results dataset spanning the Fourth Republic and no evaluated
probabilistic state-level forecasting model for Nigeria. Existing commentary relies
on polls (badly wrong in 2023) or informal reasoning.

## Approach
1. Build and validate the first public state-level presidential results dataset for
   1999-2023 (no state breakdown exists for 2007), reconciled exactly to INEC's
   official national totals.
2. Formalise §134 in code, including the legally contested points: how the FCT
   counts (settled by the Supreme Court in 2023: FCT is the 37th unit) and how the
   fractional 24 2/3 units requirement is met (Awolowo v Shagari, 1979), plus the
   second/third-election mechanics, with alternative readings as sensitivity checks.
3. Model the incumbent party's state vote share as a national swing plus state
   deviations on the logit scale, in PyMC, with candidate home-state/zone and
   North/South region covariates, a spatial autoregressive variant, and a
   probabilistic uniform-swing benchmark.
4. Evaluate with leave-future-out backtests (2015, 2019, 2023), scoring both with
   the model's own national forecast and with the true national share (oracle),
   using proper scoring rules (CRPS), coverage and Brier scores.
5. Guard against overfitting with so few elections: a written, hashed evaluation
   protocol frozen before the final run, and *prequential* stacking of predictive
   densities, whose weights for each target election use only earlier backtests.
6. Issue a pre-registered forecast for the February 2027 election (APC vote shares
   and the probability of meeting the §134 spread), publicly timestamped before the
   election, conditional on opposition scenarios and on the national vote share.

## Key findings (see experimental log for numbers)
- Conditional on the national vote, the prequential stack clearly beats uniform
  swing on probabilistic accuracy (CRPS) and also on point accuracy, and is far
  better calibrated.
- With the model's own national forecast, the gains largely vanish: the national
  swing is barely predictable from 2-5 past elections and dominates forecast error.
- The spatial (zone-block SAR) term does not help out of sample; the North/South
  region cleavage does.
- The §134 spread event was under-forecast in backtests.
- Honest development history (a first model that underperformed, a post-hoc
  respecification, then a frozen protocol) is disclosed as part of the contribution.

## Contributions
(i) Validated open dataset; (ii) formal, legally grounded implementation of §134;
(iii) a rigorous, overfitting-resistant evaluation of state-level forecastability in
a data-poor democracy, including informative negative results; (iv) a
pre-registered 2027 forecast to be scored after the election.

## Target venue
International Journal of Forecasting (methodology + evaluation + pre-registration).
Relevant literatures: election forecasting (fundamentals, uniform swing, state-level
Bayesian models such as Linzer 2013; Heidemanns, Gelman and Morris 2020), proper
scoring rules (Gneiting and Raftery 2007), stacking (Yao et al. 2018), spatial
econometrics (LeSage and Pace 2009), forecasting in data-poor settings, African and
Nigerian elections and ethno-regional voting, pre-registration in political science.
