"""Assemble the state x election panel for the incumbent-party model.

Target: the incumbent (ruling) party's share of valid votes in each state.
The model works in logit space and splits it into
    logit(y_it) = logit(N_t) + d_it
where N_t is the national incumbent-party share and d_it the state's
deviation from it. Every predictor is known before election day:
    * lagged shares of the same party (or the parties it was formed from)
    * candidates' home states/zones (config/candidates.yaml)
    * violence in the 12 months before the election (z-scored within year,
      which removes ACLED's growing coverage over time)
    * the latest pre-election poverty survey (2015 onwards only)
    * vote weights from the previous election's valid votes
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from ngforecast.data.geography import build_zone_map

SHARE_CLIP = (0.005, 0.995)
NORTHERN_ZONES = {"North-Central", "North-East", "North-West"}  # FCT is North-Central


def region_of_zone(zone: str) -> str:
    return "North" if zone in NORTHERN_ZONES else "South"


def logit(p):
    p = np.clip(p, *SHARE_CLIP)
    return np.log(p / (1 - p))


def load_candidates(path: Path = Path("config/candidates.yaml")) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["elections"] = {int(k): v for k, v in cfg["elections"].items()}
    cfg["lineage"] = {
        party: {int(y): parties for y, parties in years.items()}
        for party, years in (cfg.get("lineage") or {}).items()
    }
    return cfg


def _party_set(party: str, year: int, lineage: dict) -> list[str]:
    """Parties whose votes count as `party` in `year` (handles mergers)."""
    return lineage.get(party, {}).get(year, [party])


def _majors(election: dict) -> list[str]:
    if election.get("incumbent_party"):
        return [election["incumbent_party"], *election.get("major_challengers", [])]
    return list(election["candidates"])


def _home_zones(parties: list[str], election: dict, zone_of: dict[str, str]) -> set[str]:
    cands = election["candidates"]
    return {zone_of[cands[p]["home_state"]] for p in parties if p in cands}


def _main_opponent(exclude: list[str], election: dict) -> dict | None:
    """Candidate of the first-listed major party not in `exclude`.

    `major_challengers` is ordered by previous-election strength, so this is
    knowable before election day (see docs/evaluation_protocol.md §3).
    """
    for p in _majors(election):
        if p not in exclude and p in election["candidates"]:
            return election["candidates"][p]
    return None


def _party_candidate(party_set: list[str], election: dict, national: pd.Series) -> dict | None:
    """The candidate who carried `party_set` in an election (largest party if merged)."""
    present = [p for p in party_set if p in election["candidates"]]
    if not present:
        return None
    best = max(present, key=lambda p: national.get(p, 0.0))
    return election["candidates"][best]


def build_panel(
    results: pd.DataFrame,
    totals: pd.DataFrame | None,
    candidates: dict,
    covariates: dict[str, pd.DataFrame] | None = None,
    years: list[int] | None = None,
) -> pd.DataFrame:
    """Build the (year, state) panel.

    Parameters
    ----------
    results : long table with columns year, state (2-letter code), party, votes
    totals : optional table year, state, registered, accredited, valid, rejected
    candidates : parsed config/candidates.yaml (see ``load_candidates``)
    covariates : optional {"violence": df, "poverty": df} keyed on (year, state)
    years : elections to include as targets (default: every election with an
        incumbent party and results for the previous election)
    """
    zone_of = build_zone_map()
    elections = candidates["elections"]
    lineage = candidates["lineage"]
    covariates = covariates or {}

    votes = results.pivot_table(
        index=["year", "state"], columns="party", values="votes", aggfunc="sum", fill_value=0
    )
    valid = votes.sum(axis=1)
    if totals is not None and "valid" in totals:
        t = totals.set_index(["year", "state"])["valid"].dropna()
        valid.loc[t.index.intersection(valid.index)] = t
    result_years = sorted(votes.index.get_level_values("year").unique())

    def share(party_set: list[str], year: int) -> tuple[pd.Series, float]:
        v = votes.loc[year]
        cols = [p for p in party_set if p in v.columns]
        num = v[cols].sum(axis=1) if cols else pd.Series(0.0, index=v.index)
        return num / valid.loc[year], float(num.sum() / valid.loc[year].sum())

    def national_by_party(year: int) -> pd.Series:
        return votes.loc[year].sum() / valid.loc[year].sum()

    if years is None:
        years = [
            y
            for y in result_years
            if elections.get(y, {}).get("incumbent_party") and _prev(y, result_years)
        ]

    rows = []
    for year in years:
        el = elections[year]
        party = el["incumbent_party"]
        # Previous election *with results* (2007 has no state data, so 2011
        # lags against 2003); the gap is recorded for diagnostics.
        prev = _prev(year, result_years)
        el_prev = elections[prev]
        lag_set = _party_set(party, prev, lineage)

        has_result = year in result_years
        if has_result:
            y_share, y_nat = share([party], year)
        lag_share, lag_nat = share(lag_set, prev)
        w = valid.loc[prev] / valid.loc[prev].sum()

        cand = el["candidates"][party]
        prev_cand = _party_candidate(lag_set, el_prev, national_by_party(prev))
        opp_zones = _home_zones(el.get("major_challengers", []), el, zone_of)
        prev_opp = [p for p in _majors(el_prev) if p not in lag_set]
        prev_opp_zones = _home_zones(prev_opp, el_prev, zone_of)
        region = {s: region_of_zone(z) for s, z in zone_of.items()}
        main_opp = _main_opponent([party], el)
        prev_main_opp = _main_opponent(lag_set, el_prev)
        inc_region = region[cand["home_state"]]
        opp_region = region[main_opp["home_state"]] if main_opp else None
        prev_inc_region = region[prev_cand["home_state"]] if prev_cand else None
        prev_opp_region = region[prev_main_opp["home_state"]] if prev_main_opp else None

        for state in sorted(zone_of):
            zone = zone_of[state]
            row = {
                "year": year,
                "state": state,
                "zone": zone,
                "incumbent_party": party,
                "incumbent_running": int(bool(el.get("incumbent_running"))),
                "lag_year": prev,
                "weight": float(w.get(state, np.nan)),
                "inc_share_lag": float(lag_share.get(state, np.nan)),
                "nat_share_lag": lag_nat,
                "home_state": int(cand["home_state"] == state),
                "home_zone": int(zone_of[cand["home_state"]] == zone),
                "opp_home_zone": int(zone in opp_zones),
                "home_state_lag": int(bool(prev_cand) and prev_cand["home_state"] == state),
                "home_zone_lag": int(bool(prev_cand) and zone_of[prev_cand["home_state"]] == zone),
                "opp_home_zone_lag": int(zone in prev_opp_zones),
                "region": region[state],
                "home_region": int(region[state] == inc_region),
                "opp_home_region": int(region[state] == opp_region),
                "home_region_lag": int(region[state] == prev_inc_region),
                "opp_home_region_lag": int(region[state] == prev_opp_region),
            }
            if has_result:
                row["inc_share"] = float(y_share.get(state, np.nan))
                row["nat_share"] = y_nat
            rows.append(row)

    panel = pd.DataFrame(rows)
    panel["dev_lag"] = logit(panel["inc_share_lag"]) - logit(panel["nat_share_lag"])
    if "inc_share" in panel:
        panel["dev"] = logit(panel["inc_share"]) - logit(panel["nat_share"])

    if totals is not None and {"registered", "accredited"} <= set(totals.columns):
        t = totals.assign(turnout=totals["accredited"] / totals["registered"])
        panel = panel.merge(t[["year", "state", "turnout"]], on=["year", "state"], how="left")

    if "violence" in covariates:
        v = covariates["violence"][["year", "state", "violence_index"]].copy()
        v["violence_z"] = v.groupby("year")["violence_index"].transform(
            lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else 0.0
        )
        panel = panel.merge(v, on=["year", "state"], how="left")

    if "poverty" in covariates:
        p = covariates["poverty"][["year", "state", "poverty_headcount", "poverty_survey"]]
        panel = panel.merge(p, on=["year", "state"], how="left")

    logger.info("Panel assembled: {} rows × {} cols, years {}", *panel.shape, years)
    return panel


def _prev(year: int, years: list[int]) -> int | None:
    earlier = [y for y in years if y < year]
    return max(earlier) if earlier else None


def save_panel(panel: pd.DataFrame, dest: Path) -> None:
    """Write panel to parquet."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(dest, index=False)
    logger.info("Panel saved to {}", dest)
