"""Data validation — pandera schemas and integrity checks."""

from __future__ import annotations

import pandera as pa
from pandera import Column, DataFrameSchema


# ── Schema: election results panel ───────────
results_schema = DataFrameSchema(
    {
        "year": Column(int, pa.Check.isin([1999, 2003, 2007, 2011, 2015, 2019, 2023])),
        "state": Column(str, pa.Check.str_length(2, 2)),  # two-letter code
        "party": Column(str, nullable=False),
        "votes": Column(int, pa.Check.ge(0)),
    },
    strict=False,
    coerce=True,
)


# ── Schema: processed panel ──────────────────
panel_schema = DataFrameSchema(
    {
        "year": Column(int),
        "state": Column(str),
        "vote_share": Column(float, pa.Check.in_range(0, 1)),
        "turnout": Column(float, pa.Check.in_range(0, 1)),
        "incumbent_party": Column(int, pa.Check.isin([0, 1])),
        "poverty_rate": Column(float, pa.Check.ge(0), nullable=True),
        "unemployment_rate": Column(float, pa.Check.ge(0), nullable=True),
        "violence_index": Column(float, nullable=True),
    },
    strict=False,
    coerce=True,
)


def validate_results(df) -> None:
    """Validate raw election results against schema."""
    results_schema.validate(df, lazy=True)


def validate_panel(df) -> None:
    """Validate processed panel against schema."""
    panel_schema.validate(df, lazy=True)


def check_state_coverage(df, expected_n: int = 37) -> None:
    """Assert each election year has all 37 units."""
    for year, group in df.groupby("year"):
        n_states = group["state"].nunique()
        if n_states != expected_n:
            raise ValueError(
                f"Year {year}: expected {expected_n} states, got {n_states}"
            )
