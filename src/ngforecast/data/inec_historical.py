"""Curated state-level presidential results, 1999-2023 (no 2007).

Built from archived INEC publications (1999 via IFES, 2003, 2011, 2015) and
Stears Elections (2019, 2023), reconciled to INEC's official national totals.
See data/external/inec_historical/sources.md.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ngforecast.data.geography import state_code

DEFAULT_DIR = Path("data/external/inec_historical")

# Normalise party labels to the keys used in config/candidates.yaml
PARTY_ALIASES = {"APP/AD (Falae)": "APP/AD"}


def _with_codes(df: pd.DataFrame) -> pd.DataFrame:
    codes = {n: state_code(n) for n in df["state_name"].unique()}
    return df.assign(state=df["state_name"].map(codes)).drop(columns="state_name")


def load_results(directory: Path = DEFAULT_DIR) -> pd.DataFrame:
    """Long table: year, state (2-letter code), party, votes."""
    df = pd.read_csv(directory / "results_long.csv")
    df["party"] = df["party"].replace(PARTY_ALIASES)
    return _with_codes(df)[["year", "state", "party", "votes"]]


def load_totals(directory: Path = DEFAULT_DIR) -> pd.DataFrame:
    """year, state, registered, accredited, valid, rejected (NaN where unpublished)."""
    return _with_codes(pd.read_csv(directory / "state_totals.csv"))
