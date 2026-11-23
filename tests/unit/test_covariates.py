"""Tests for covariate builders: no look-ahead, correct windows, name mapping."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ngforecast.data.acled import build_violence_index
from ngforecast.data.macro import build_macro
from ngforecast.data.poverty import build_poverty, poverty_asof

DATES = {2019: "2019-02-23", 2023: "2023-02-25"}


@pytest.fixture
def monthly_violence():
    months = pd.date_range("2017-01-01", "2023-06-01", freq="MS")
    return pd.DataFrame(
        [
            {"state": s, "month": m, "events": 1, "fatalities": 2}
            for s in ("LA", "BO")
            for m in months
        ]
    )


class TestViolenceIndex:
    def test_window_is_12_months_before_election_month(self, monthly_violence):
        out = build_violence_index(monthly_violence, DATES, window_months=12)
        row = out[(out.year == 2019) & (out.state == "LA")].iloc[0]
        assert row.violence_events == 12  # Feb 2018 – Jan 2019
        assert row.violence_window_end == pd.Timestamp("2019-02-01")

    def test_election_month_excluded(self, monthly_violence):
        df = monthly_violence.copy()
        df.loc[df.month == "2019-02-01", "events"] = 1000
        out = build_violence_index(df, DATES)
        assert out[(out.year == 2019) & (out.state == "LA")].violence_events.iloc[0] == 12

    def test_future_election_uses_last_complete_months(self, monthly_violence):
        out = build_violence_index(monthly_violence, {2027: "2027-02-27"})
        row = out[out.state == "LA"].iloc[0]
        assert row.violence_window_end == pd.Timestamp("2023-06-01")  # partial June dropped
        assert row.violence_events == 12

    def test_index_formula(self, monthly_violence):
        out = build_violence_index(monthly_violence, DATES)
        r = out.iloc[0]
        assert r.violence_index == pytest.approx(np.log1p(12) + np.log1p(24))

    def test_window_before_data_start_skipped(self, monthly_violence):
        out = build_violence_index(monthly_violence, {2017: "2017-03-01"})
        assert out.empty


class TestPoverty:
    @pytest.fixture
    def raw_csv(self, tmp_path):
        rows = []
        for survey, start, end, head in [
            ("DHS", "2013-01-01", "2013-12-31", 40.0),
            ("DHS", "2018-01-01", "2018-12-31", 30.0),
            ("MICS", "2021-01-01", "2021-12-31", 20.0),
        ]:
            for name in ("Lagos", "FCT", None):
                rows.append(
                    {
                        "Admin 1 Name": name,
                        "MPI": head / 200,
                        "Headcount Ratio": head,
                        "In Severe Poverty": head / 2,
                        "Survey": survey,
                        "Start Date": f"{start} 00:00:00+00:00",
                        "End Date": f"{end} 23:59:59+00:00",
                    }
                )
        path = tmp_path / "mpi.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_tidy_drops_national_and_maps_codes(self, raw_csv):
        tidy = build_poverty(raw_csv)
        assert set(tidy.state) == {"LA", "FC"}
        assert tidy.headcount.max() == pytest.approx(0.40)

    def test_asof_uses_latest_prior_survey(self, raw_csv):
        out = poverty_asof(build_poverty(raw_csv), {2011: "2011-04-16", 2019: "2019-02-23"})
        la = out[out.state == "LA"].set_index("year")
        assert np.isnan(la.loc[2011, "mpi"])  # no survey before 2011: not back-filled
        assert la.loc[2019, "poverty_survey"] == "DHS 2018"
        assert la.loc[2019, "poverty_headcount"] == pytest.approx(0.30)


class TestMacro:
    @pytest.fixture
    def raw_dir(self, tmp_path):
        def wdi(values):
            return json.dumps([{}, [{"date": str(y), "value": v} for y, v in values.items()]])

        (tmp_path / "inflation_cpi.json").write_text(wdi({2017: 16.5, 2018: 12.1, 2022: 18.8}))
        (tmp_path / "gdp_pc_growth.json").write_text(wdi({2018: -0.4, 2022: 2.2}))
        (tmp_path / "fx_official.json").write_text(
            wdi({2017: 300.0, 2018: 306.0, 2021: 400.0, 2022: 426.0})
        )
        days = pd.date_range("2016-01-01", "2023-12-31", freq="D")
        price = np.where(days < "2018-02-23", 50.0, 70.0)
        pd.DataFrame({"observation_date": days, "DCOILBRENTEU": price}).to_csv(
            tmp_path / "brent.csv", index=False
        )
        return tmp_path

    def test_uses_year_before_election(self, raw_dir):
        out = build_macro(raw_dir, {2019: "2019-02-23"}).iloc[0]
        assert out.macro_ref_year == 2018
        assert out.inflation_cpi == pytest.approx(12.1)
        assert out.fx_depreciation == pytest.approx(2.0)

    def test_brent_window_is_12_months_before_election(self, raw_dir):
        out = build_macro(raw_dir, {2019: "2019-02-23"}).iloc[0]
        assert out.brent_12m == pytest.approx(70.0)
        assert out.brent_change == pytest.approx(40.0)

    def test_future_election_falls_back_to_latest_year(self, raw_dir):
        out = build_macro(raw_dir, {2027: "2027-02-27"}).iloc[0]
        assert out.macro_ref_year == 2022
        assert out.inflation_cpi == pytest.approx(18.8)
