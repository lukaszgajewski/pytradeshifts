import pandas as pd
import numpy as np
import pytest

from src.input_output import data
from src import convert_supply_yearly_to_monthly


@pytest.fixture
def dataset():
    return convert_supply_yearly_to_monthly.load_data(
        data["output"]["yearly"], data["input"]["seasonality"]
    )


def test_load_data(dataset):
    monthly_domestic_supply, monthly_seasonality = dataset
    assert isinstance(monthly_domestic_supply, pd.DataFrame)
    assert monthly_domestic_supply.index.equals(monthly_seasonality.index)
    monthly_domestic_supply.replace([np.inf, -np.inf], np.nan, inplace=True)
    assert ~monthly_domestic_supply.isna().any(axis=None)
    assert (monthly_domestic_supply >= 0).all(axis=None)
    assert isinstance(monthly_seasonality, pd.DataFrame)
    monthly_seasonality.replace([np.inf, -np.inf], np.nan, inplace=True)
    assert ~monthly_seasonality.isna().any(axis=None)
    assert (monthly_seasonality >= 0).all(axis=None)


def test_compute_year_one(dataset):
    year_one = convert_supply_yearly_to_monthly.compute_year_one(*dataset)
    assert isinstance(year_one, list)
    assert all([isinstance(vec, pd.Series) for vec in year_one])
    assert len(year_one) == 8  # May-December
    assert all(
        [
            ~vec.replace([np.inf, -np.inf], np.nan).isna().any(axis=None)
            for vec in year_one
        ]
    )
    assert all([(vec >= 0).all(axis=None) for vec in year_one])


def test_compute_other_years(dataset):
    other_years = convert_supply_yearly_to_monthly.compute_other_years(*dataset)
    assert isinstance(other_years, list)
    assert all([isinstance(vec, pd.Series) for vec in other_years])
    assert len(other_years) == 108  # 12 months x 9 years
    assert all(
        [
            ~vec.replace([np.inf, -np.inf], np.nan).isna().any(axis=None)
            for vec in other_years
        ]
    )
    assert all([(vec >= 0).all(axis=None) for vec in other_years])
