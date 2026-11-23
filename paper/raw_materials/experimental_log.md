# Experimental log: Nigerian presidential election forecasting (state level, §134)

All tables below are generated from the committed result files. Numbers quoted in prose are copied from these tables (or, for data validation, from data/external/inec_historical/validation_log.txt).

## 1. Data

### 1.1 Election results (curated, state level)
Presidential results for all 36 states + FCT. 2007 is absent because INEC never
published a state-level breakdown of the 2007 presidential result.

|   year |   units |   parties |   total_valid_votes |
|-------:|--------:|----------:|--------------------:|
|   1999 |      37 |         2 |            29848441 |
|   2003 |      37 |        20 |            39480489 |
|   2011 |      37 |        20 |            38199219 |
|   2015 |      37 |        14 |            28587564 |
|   2019 |      37 |         3 |            27324583 |
|   2023 |      37 |         5 |            24025940 |

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

|   year | incumbent_party   |   lag_election |   national_share |   national_share_lag |   incumbent_candidate_running |
|-------:|:------------------|---------------:|-----------------:|---------------------:|------------------------------:|
|   2003 | PDP               |           1999 |            0.619 |                0.628 |                             1 |
|   2011 | PDP               |           2003 |            0.589 |                0.619 |                             1 |
|   2015 | PDP               |           2011 |            0.45  |                0.589 |                             1 |
|   2019 | APC               |           2015 |            0.556 |                0.54  |                             1 |
|   2023 | APC               |           2019 |            0.366 |                0.556 |                             0 |
|   2027 | APC               |           2023 |          nan     |                0.366 |                             1 |

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

## 2. Constitutional rule (§134)
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

Estimation: PyMC NUTS, 4 chains x 1000 draws (+ 1000 tuning),
target_accept 0.99. Parameter-recovery test on simulated data recovers rho, phi and
beta within tolerance and achieves >= 75% coverage of 90% intervals.

Candidate models (protocol v2):

| model         | description                                                       |
|:--------------|:------------------------------------------------------------------|
| stack         | Prequential stack (headline; weights from earlier backtests only) |
| ps_null       | Probabilistic uniform swing (phi=1, no covariates)                |
| ps_home       | Swing model + home-state/zone covariates                          |
| ps_region     | Swing model + home + North/South region covariates                |
| sar_region    | As ps_region + spatial SAR term (zone-block W)                    |
| pp_region     | Partial persistence, phi ~ N(1, 0.2), region covariates           |
| lvl_home      | Level model, phi ~ N(0, 1), home covariates (v1 reference)        |
| uniform_swing | Deterministic uniform swing (baseline)                            |
| persistence   | Last election's state shares unchanged (baseline)                 |

Prequential stacking: for target t, weights are fitted by stacking of predictive
densities (Yao, Vehtari, Simpson and Gelman 2018) using only backtests of targets
< t (equal weights for 2015). Densities: Gaussian approximation on the logit scale,
oracle-national mode. The stack's predictive distribution is the weighted mixture of
draws.

## 4. Evaluation protocol and development history (disclosed)
- Leave-future-out backtests: train on elections < t, predict t in {2015, 2019, 2023}
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
- v2: a written protocol (docs/evaluation_protocol.md, SHA-256 cdeb897b8f097411...) fixed
  six candidate models, one new theory-motivated covariate family (North/South
  region) and prequential stacking **before** the v2 run. One post-freeze change:
  NUTS target_accept raised from 0.95 to 0.99 after 26 divergences in the
  2015 national fit; results essentially unchanged (both runs archived).
- The v2 run is reproducible bit-for-bit (identical metrics on rerun with fixed seeds).

## 5. Backtest results (v2, result of record)

### 5.1 Mean over 2015/2019/2023, oracle mode

| model         |   state_crps_pp |   state_rmse_pp |   state_mae_pp |   state_cov90 |   brier_25pct |
|:--------------|----------------:|----------------:|---------------:|--------------:|--------------:|
| stack         |            7.86 |           12.62 |           9.9  |          0.95 |          0.12 |
| ps_region     |            8.29 |           15.06 |          11.39 |          0.99 |          0.13 |
| pp_region     |            8.33 |           14.06 |          11.33 |          0.95 |          0.11 |
| ps_home       |            8.71 |           15.48 |          11.88 |          0.99 |          0.14 |
| ps_null       |            8.77 |           14.95 |          11.89 |          0.99 |          0.14 |
| sar_region    |            8.88 |           15.34 |          11.68 |          1    |          0.14 |
| lvl_home      |           10.67 |           19.18 |          14.7  |          0.9  |          0.13 |
| uniform_swing |           11.05 |           14.21 |          11.05 |          0    |          0.14 |

### 5.2 Mean over 2015/2019/2023, forecast mode

| model         |   state_crps_pp |   state_rmse_pp |   state_mae_pp |   state_cov90 |   brier_25pct |
|:--------------|----------------:|----------------:|---------------:|--------------:|--------------:|
| ps_region     |           10.34 |           17.61 |          14.29 |          0.95 |          0.12 |
| ps_null       |           10.39 |           16.64 |          13.36 |          0.99 |          0.13 |
| stack         |           10.66 |           17.9  |          15.01 |          0.96 |          0.13 |
| ps_home       |           10.8  |           18.47 |          14.84 |          0.97 |          0.13 |
| sar_region    |           10.9  |           18.4  |          15.05 |          0.98 |          0.13 |
| uniform_swing |           10.93 |           17.04 |          13.67 |          0.47 |          0.13 |
| pp_region     |           11.1  |           19.4  |          15.84 |          0.95 |          0.13 |
| lvl_home      |           13.09 |           23.98 |          19.15 |          0.91 |          0.14 |
| persistence   |          nan    |           17.94 |          14.75 |        nan    |        nan    |

### 5.3 State CRPS (pp) by election

| national   | model         |   2015 |   2019 |   2023 |
|:-----------|:--------------|-------:|-------:|-------:|
| forecast   | ps_null       |   8.8  |   9.64 |  12.74 |
| forecast   | ps_region     |   8.48 |   9.07 |  13.46 |
| forecast   | stack         |   8.95 |   8.24 |  14.77 |
| forecast   | uniform_swing |  10.55 |   8.59 |  13.67 |
| oracle     | ps_null       |   6.87 |   9.46 |   9.98 |
| oracle     | ps_region     |   6.47 |   8.42 |   9.98 |
| oracle     | stack         |   6.83 |   7.36 |   9.39 |
| oracle     | uniform_swing |   7.58 |  11.96 |  13.61 |

### 5.4 State RMSE (pp) of point forecasts by election

| national   | model         |   2015 |   2019 |   2023 |
|:-----------|:--------------|-------:|-------:|-------:|
| forecast   | persistence   |  16.94 |  13.99 |  22.88 |
| forecast   | stack         |  15.61 |  12.42 |  25.68 |
| forecast   | uniform_swing |  15.74 |  13.53 |  21.87 |
| oracle     | stack         |  10.7  |  10.88 |  16.27 |
| oracle     | uniform_swing |  10.28 |  14.3  |  18.04 |

### 5.5 National forecast (stack, forecast mode)

|   year |   national_pred |   national_true |   national_err_pp | national_in_90   |
|-------:|----------------:|----------------:|------------------:|:-----------------|
|   2015 |           0.568 |           0.45  |            11.867 | True             |
|   2019 |           0.483 |           0.556 |            -7.269 | True             |
|   2023 |           0.537 |           0.366 |            17.048 | True             |

### 5.6 §134 spread: forecast probability vs outcome (stack)

|   year | national   |   p_spread_met | spread_met_true   |   state_cov90 |
|-------:|:-----------|---------------:|:------------------|--------------:|
|   2015 | forecast   |          0.928 | True              |         1     |
|   2015 | oracle     |          0.462 | True              |         1     |
|   2019 | forecast   |          0.637 | True              |         1     |
|   2019 | oracle     |          0.985 | True              |         1     |
|   2023 | forecast   |          0.886 | True              |         0.892 |
|   2023 | oracle     |          0.379 | True              |         0.865 |

The incumbent party met the spread in all three elections; the model's probabilities were below 0.5 in oracle mode for 2015 and 2023, i.e. the spread event was under-forecast.

### 5.7 Prequential stacking weights (column = target; 'next' = used for 2027)

| model      |   2015 |   2019 |   2023 |   next |
|:-----------|-------:|-------:|-------:|-------:|
| ps_null    |  0.167 |    0   |      0 |  0     |
| ps_home    |  0.167 |    0   |      0 |  0     |
| ps_region  |  0.167 |    0.3 |      0 |  0.188 |
| sar_region |  0.167 |    0   |      0 |  0     |
| pp_region  |  0.167 |    0.7 |      1 |  0.812 |
| lvl_home   |  0.167 |    0   |      0 |  0     |

### 5.8 Posterior means (sd) of selected parameters, by backtest fit

|   target | model      | phi         | sigma       | home_zone   | opp_home_zone   | home_region   | opp_home_region   | rho         |
|---------:|:-----------|:------------|:------------|:------------|:----------------|:--------------|:------------------|:------------|
|     2015 | ps_region  | 1.00 (0.00) | 1.40 (0.12) | 0.91 (0.37) | -0.55 (0.34)    | 0.34 (0.41)   | -0.48 (0.35)      |             |
|     2015 | sar_region | 1.00 (0.00) | 1.07 (0.09) | 0.33 (0.34) | -0.25 (0.29)    | 0.13 (0.40)   | -0.19 (0.35)      | 0.63 (0.07) |
|     2015 | pp_region  | 0.31 (0.11) | 1.07 (0.10) | 0.95 (0.33) | -0.79 (0.30)    | 0.64 (0.40)   | -0.49 (0.35)      |             |
|     2019 | ps_region  | 1.00 (0.00) | 1.30 (0.09) | 0.95 (0.33) | -0.47 (0.29)    | 0.26 (0.40)   | -0.59 (0.35)      |             |
|     2019 | sar_region | 1.00 (0.00) | 0.95 (0.07) | 0.36 (0.28) | -0.19 (0.24)    | 0.10 (0.39)   | -0.23 (0.33)      | 0.64 (0.06) |
|     2019 | pp_region  | 0.41 (0.09) | 1.04 (0.08) | 1.20 (0.30) | -0.62 (0.26)    | 0.52 (0.39)   | -0.56 (0.35)      |             |
|     2023 | ps_region  | 1.00 (0.00) | 1.19 (0.07) | 0.86 (0.29) | -0.44 (0.25)    | 0.26 (0.37)   | -0.75 (0.22)      |             |
|     2023 | sar_region | 1.00 (0.00) | 0.89 (0.05) | 0.34 (0.26) | -0.16 (0.20)    | 0.11 (0.36)   | -0.31 (0.18)      | 0.62 (0.05) |
|     2023 | pp_region  | 0.49 (0.06) | 0.98 (0.06) | 1.11 (0.27) | -0.40 (0.21)    | 0.54 (0.37)   | -0.67 (0.19)      |             |

Note: home_region and home_region_lag are identical in all training elections up to 2019 (the incumbent party's candidate never switched region before 2023), so their separate effects are identified only by the prior in the 2019 and 2023 backtests.

### 5.9 Convergence
Across all fits: max R-hat 1.019, min bulk ESS 325, total divergences 0 (R-hat 1.019 occurs only in the lvl_home reference model, which receives zero stacking weight).

### 5.10 v1 results (disclosed development history; mean over 2015/2019/2023)

| national   | model            |   state_crps_pp |   state_rmse_pp |
|:-----------|:-----------------|----------------:|----------------:|
| forecast   | nonspatial_level |           13.07 |           24    |
| forecast   | sar_level        |           13.06 |           24.7  |
| forecast   | uniform_swing    |           10.96 |           17.07 |
| oracle     | nonspatial_level |           10.61 |           19.08 |
| oracle     | sar_level        |           10.66 |           19.64 |
| oracle     | uniform_swing    |           11.05 |           14.21 |

### 5.11 Summary of findings
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

## 6. The 2027 pre-registered forecast
Issued 2026-11-23T09:31 UTC from git commit 4c18fbf (public repository);
election date 2027-02-27. Stack weights: {"ps_region": 0.188, "pp_region": 0.812}.
Trained on 2003, 2011, 2015, 2019, 2023. Target: APC (incumbent party) share of valid
votes. Not forecast: outright winner or runoff probability (plurality depends on the
opposition split, unsettled at issue time); no probabilities are attached to the
opposition scenarios. Incumbent candidate (Tinubu, South-West) running.

### 6.1 By main-challenger home-zone scenario

| main challenger zone   |   APC national median |   90% low |   90% high |   P(APC > 50%) |   units >= 25% (median) |   P(spread): default (FCT as 37th unit; Awolowo v Shagari) |   P(spread): states only (24 of 36) |   P(spread): 24 states + 25% in FCT |   P(spread): 25 whole units (ceil) |
|:-----------------------|----------------------:|----------:|-----------:|---------------:|------------------------:|-----------------------------------------------------------:|------------------------------------:|------------------------------------:|-----------------------------------:|
| North-Central          |                 0.324 |     0.188 |      0.508 |          0.055 |                      20 |                                                      0.246 |                               0.238 |                               0.066 |                              0.201 |
| North-East             |                 0.324 |     0.191 |      0.506 |          0.053 |                      20 |                                                      0.249 |                               0.231 |                               0.107 |                              0.203 |
| North-West             |                 0.325 |     0.192 |      0.504 |          0.053 |                      20 |                                                      0.288 |                               0.271 |                               0.13  |                              0.234 |
| South-East             |                 0.325 |     0.188 |      0.507 |          0.054 |                      20 |                                                      0.264 |                               0.233 |                               0.14  |                              0.204 |
| South-South            |                 0.325 |     0.188 |      0.508 |          0.054 |                      19 |                                                      0.238 |                               0.212 |                               0.134 |                              0.189 |
| South-West             |                 0.324 |     0.191 |      0.508 |          0.054 |                      20 |                                                      0.281 |                               0.252 |                               0.156 |                              0.227 |

### 6.2 P(§134 spread met | APC national share), default reading, by scenario

|   national_share |   North-Central |   North-East |   North-West |   South-East |   South-South |   South-West |
|-----------------:|----------------:|-------------:|-------------:|-------------:|--------------:|-------------:|
|             0.3  |            0    |         0    |         0.01 |         0.01 |          0    |         0.01 |
|             0.35 |            0.14 |         0.13 |         0.21 |         0.17 |          0.14 |         0.25 |
|             0.4  |            0.65 |         0.65 |         0.77 |         0.63 |          0.56 |         0.69 |
|             0.45 |            0.96 |         0.95 |         0.98 |         0.9  |          0.88 |         0.91 |
|             0.5  |            1    |         1    |         1    |         0.98 |          0.97 |         0.98 |
|             0.55 |            1    |         1    |         1    |         1    |          1    |         1    |
|             0.6  |            1    |         1    |         1    |         1    |          1    |         1    |

Median number of units with APC >= 25%, averaged over scenarios:

|   national_share |   units_at_25pct_median |
|-----------------:|------------------------:|
|              0.3 |                    18   |
|              0.4 |                    21.3 |
|              0.4 |                    24.3 |
|              0.4 |                    27.3 |
|              0.5 |                    29.3 |
|              0.6 |                    31.3 |
|              0.6 |                    32.8 |

For reference, APC's 2023 national share was 36.6% and APC met the spread in 2023.

### 6.3 State forecasts (main challenger from North-East; median and 90% interval)

| state   |   median |   q05 |   q95 |   p_ge_25pct |
|:--------|---------:|------:|------:|-------------:|
| ON      |    0.657 | 0.203 | 0.933 |        0.926 |
| EK      |    0.643 | 0.203 | 0.931 |        0.928 |
| LA      |    0.637 | 0.154 | 0.936 |        0.892 |
| OG      |    0.604 | 0.182 | 0.919 |        0.906 |
| OY      |    0.595 | 0.157 | 0.915 |        0.887 |
| OS      |    0.536 | 0.134 | 0.888 |        0.858 |
| RI      |    0.423 | 0.097 | 0.838 |        0.752 |
| ZA      |    0.382 | 0.077 | 0.847 |        0.695 |
| CR      |    0.363 | 0.066 | 0.796 |        0.671 |
| BY      |    0.335 | 0.056 | 0.781 |        0.63  |
| AK      |    0.334 | 0.059 | 0.776 |        0.646 |
| SO      |    0.322 | 0.066 | 0.797 |        0.62  |
| ED      |    0.306 | 0.046 | 0.763 |        0.592 |
| KT      |    0.3   | 0.061 | 0.775 |        0.591 |
| KE      |    0.295 | 0.057 | 0.769 |        0.58  |
| JI      |    0.294 | 0.06  | 0.751 |        0.579 |
| KW      |    0.287 | 0.056 | 0.745 |        0.569 |
| EB      |    0.28  | 0.042 | 0.73  |        0.551 |
| IM      |    0.279 | 0.049 | 0.742 |        0.547 |
| KO      |    0.274 | 0.054 | 0.727 |        0.544 |
| NI      |    0.253 | 0.045 | 0.695 |        0.505 |
| DE      |    0.245 | 0.034 | 0.687 |        0.494 |
| BO      |    0.23  | 0.039 | 0.713 |        0.468 |
| KD      |    0.224 | 0.044 | 0.667 |        0.442 |
| KN      |    0.223 | 0.041 | 0.664 |        0.445 |
| BE      |    0.216 | 0.039 | 0.653 |        0.434 |
| NA      |    0.177 | 0.03  | 0.593 |        0.358 |
| YO      |    0.174 | 0.029 | 0.61  |        0.351 |
| PL      |    0.17  | 0.028 | 0.572 |        0.344 |
| BA      |    0.157 | 0.025 | 0.577 |        0.317 |
| FC      |    0.133 | 0.02  | 0.518 |        0.251 |
| TA      |    0.129 | 0.02  | 0.518 |        0.251 |
| GO      |    0.126 | 0.02  | 0.509 |        0.23  |
| AD      |    0.123 | 0.019 | 0.506 |        0.232 |
| AB      |    0.114 | 0.009 | 0.514 |        0.231 |
| EN      |    0.076 | 0.005 | 0.405 |        0.142 |
| AN      |    0.068 | 0.004 | 0.366 |        0.114 |

## 7. Limitations (to be reported)
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
