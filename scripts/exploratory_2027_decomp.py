"""EXPLORATORY: why do 2027 state forecasts differ from 2023 outcomes?

Refits the stack's dominant member (pp_region, v2 spec) on all five elections
and decomposes each state's predicted change in logit share from 2023 into
(posterior means; North-East challenger scenario, the scenario of Fig. 4):

    national swing    logit N_2027(median) - logit N_2023
    mean reversion    (phi - 1) * d_2023
    covariates        X_2027 beta
    intercept+state   alpha + u_i
    reconciliation    common shift c that restores the national share

Writes outputs/exploratory/2027/decomposition.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).parent))
import forecast_2027 as fc  # noqa: E402
from run_backtest import CANDIDATES, sorted_W  # noqa: E402

from ngforecast.features.panel import logit  # noqa: E402
from ngforecast.models.fundamentals.bayes import SamplerConfig, fit  # noqa: E402


def main() -> None:
    panel = pd.read_parquet("data/processed/panel.parquet").sort_values(["year", "state"])
    train = panel[panel["inc_share"].notna()]
    test = fc.scenario_rows(panel[panel.year == 2027].sort_values("state"), "North-East")
    states = sorted(test.state)
    W = sorted_W(states)  # noqa: N806
    m = fit(train, W, states, cfg=SamplerConfig(), **CANDIDATES["pp_region"])
    st, nt = m.state, m.national
    beta = st["beta"].mean(axis=0)
    X = test[m.covariates].to_numpy(float)  # noqa: N806
    nat23 = float(test.nat_share_lag.iloc[0])
    swing = float(np.mean(nt["mu"] + nt["gamma"] * test.incumbent_running.iloc[0]))
    nat27 = expit(logit(nat23) + swing)
    d23 = test.dev_lag.to_numpy()
    comp = pd.DataFrame(
        {
            "state": states,
            "share_2023": test.inc_share_lag.to_numpy(),
            "national_swing": swing,
            "mean_reversion": (float(st["phi"].mean()) - 1) * d23,
            "covariates": X @ beta,
            "intercept_state": float(st["alpha"].mean()) + st["u"].mean(axis=0),
        }
    )
    base = (
        logit(comp.share_2023.to_numpy())
        + comp[["national_swing", "mean_reversion", "covariates", "intercept_state"]]
        .sum(axis=1)
        .to_numpy()
    )
    w = test.weight.to_numpy() / test.weight.sum()
    c = brentq(lambda c: w @ expit(base + c) - nat27, -5, 5)
    comp["reconciliation"] = c
    comp["pred_share_postmean"] = expit(base + c)
    prereg = pd.read_csv(fc.OUT / "states.csv", keep_default_na=False)
    prereg = prereg[prereg.challenger_zone == "North-East"].set_index("state")
    comp["prereg_stack_median"] = comp.state.map(prereg["median"])
    for k, v in {"phi": st["phi"], "alpha": st["alpha"]}.items():
        print(k, round(float(v.mean()), 3))
    print(dict(zip(m.covariates, beta.round(3), strict=True)))
    out = Path("outputs/exploratory/2027")
    out.mkdir(parents=True, exist_ok=True)
    comp.round(4).to_csv(out / "decomposition.csv", index=False)
    print(comp.round(3).to_string())


if __name__ == "__main__":
    main()
