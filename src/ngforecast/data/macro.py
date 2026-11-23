"""National macro indicators — World Bank WDI and Brent crude (FRED), keyless.

All indicators are national, so in a model with election fixed effects they
only enter through interactions (e.g. with incumbency). Each election uses
values known before election day: the last full calendar year before the
election for annual series, and the 12 months before election day for Brent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from loguru import logger

from ngforecast.data._utils import download, load_source_config, write_provenance


def fetch_macro(out_dir: Path) -> list[Path]:
    """Download WDI indicators (JSON) and the Brent series (CSV) into raw/macro/."""
    cfg = load_source_config("macro")
    out_dir.mkdir(parents=True, exist_ok=True)
    files, paths = [], []

    wb = cfg["world_bank"]
    for name, code in wb["indicators"].items():
        url = f"{wb['base_url']}/{code}?format=json&per_page=200"
        dest = out_dir / f"{name}.json"
        logger.info("Fetching WDI {} ({})", name, code)
        files.append(download(url, dest))
        paths.append(dest)

    dest = out_dir / "brent.csv"
    logger.info("Fetching Brent from FRED")
    files.append(download(cfg["fred"]["url"], dest))
    paths.append(dest)

    write_provenance(out_dir, cfg["citation"], files)
    return paths


def load_wdi(path: Path) -> pd.Series:
    """WDI JSON → Series indexed by calendar year (NaNs dropped)."""
    payload = json.loads(path.read_text())
    if len(payload) < 2 or not payload[1]:
        raise ValueError(f"No data in WDI file {path}")
    s = pd.Series({int(r["date"]): r["value"] for r in payload[1]}, dtype=float)
    return s.dropna().sort_index()


def load_brent(path: Path) -> pd.Series:
    """FRED Brent CSV → daily Series (missing days dropped)."""
    df = pd.read_csv(path)
    date_col = df.columns[0]
    s = pd.to_numeric(df["DCOILBRENTEU"], errors="coerce")
    s.index = pd.to_datetime(df[date_col])
    return s.dropna()


def build_macro(raw_dir: Path, election_dates: dict[int, str]) -> pd.DataFrame:
    """One row per election with pre-election national macro conditions.

    Columns: year, macro_ref_year, inflation_cpi, gdp_pc_growth,
    fx_depreciation, brent_12m, brent_change. For an election still in the
    future, the latest published year / 12 months of Brent are used.
    """
    inflation = load_wdi(raw_dir / "inflation_cpi.json")
    growth = load_wdi(raw_dir / "gdp_pc_growth.json")
    fx = load_wdi(raw_dir / "fx_official.json")
    brent = load_brent(raw_dir / "brent.csv")

    rows = []
    for year, date in sorted(election_dates.items()):
        # Last full calendar year before the election; if not yet published
        # (e.g. a forecast made before year-end), the latest year available.
        prev = max(y for y in inflation.index if y <= year - 1)
        day = min(pd.Timestamp(date), brent.index.max() + pd.Timedelta(days=1))
        last12 = brent[(brent.index >= day - pd.DateOffset(months=12)) & (brent.index < day)]
        prior12 = brent[
            (brent.index >= day - pd.DateOffset(months=24))
            & (brent.index < day - pd.DateOffset(months=12))
        ]
        brent_12m = last12.mean() if len(last12) else float("nan")
        brent_prior = prior12.mean() if len(prior12) else float("nan")
        rows.append(
            {
                "year": year,
                "macro_ref_year": prev,
                "inflation_cpi": inflation.get(prev, float("nan")),
                "gdp_pc_growth": growth.get(prev, float("nan")),
                "fx_depreciation": (
                    100.0 * (fx[prev] / fx[prev - 1] - 1)
                    if prev in fx.index and prev - 1 in fx.index
                    else float("nan")
                ),
                "brent_12m": brent_12m,
                "brent_change": 100.0 * (brent_12m / brent_prior - 1),
            }
        )
    return pd.DataFrame(rows)
