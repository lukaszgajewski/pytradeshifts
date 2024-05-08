import pandas as pd
import numpy as np
import pytest
from src.input_output import load_fao_zip, data
from src import fao_trade_zip_to_pkl, fao_production_zip_to_pkl
from src.fao_pkl_to_total_caloric_trade_and_production import (
    compute_calories,
    correct_reexports,
    prebalance,
    reindex_trade_and_production,
    format_trade_and_production,
)


@pytest.fixture
def oceania_dataset_filtered() -> tuple[pd.DataFrame, pd.DataFrame]:
    return fao_trade_zip_to_pkl.filter_data(
        fao_trade_zip_to_pkl.convert_to_ISO3(
            load_fao_zip("data/testing/Trade_DetailedTradeMatrix_E_Oceania.zip"),
            data["input"]["country_codes"],
        ),
        data["input"]["nutrition"],
        data["input"]["yield_reduction"],
    ), fao_production_zip_to_pkl.filter_data(
        fao_production_zip_to_pkl.convert_to_ISO3(
            load_fao_zip("data/testing/Production_Crops_Livestock_E_Oceania.zip"),
            data["input"]["country_codes"],
        ),
        data["input"]["nutrition"],
        data["input"]["yield_reduction"],
    )


@pytest.fixture
def oceania_dataset_formatted(oceania_dataset_filtered):
    trade_data, production_data = oceania_dataset_filtered
    country_list = ["AUS", "NZL", "PNG", "FJI"]
    trade_data, production_data = compute_calories(
        trade_data, production_data, data["input"]["nutrition"]
    )
    trade_data = trade_data[trade_data["Item"] == "Apples"]
    production_data = production_data[production_data["Item"] == "Apples"]
    trade_data, production_data = format_trade_and_production(
        trade_data, production_data, country_list
    )
    return trade_data, production_data


@pytest.fixture
def oceania_dataset_formatted_random_item(oceania_dataset_filtered):
    trade_data, production_data = oceania_dataset_filtered
    country_list = ["AUS", "NZL", "PNG", "FJI"]
    trade_data, production_data = compute_calories(
        trade_data, production_data, data["input"]["nutrition"]
    )
    item = np.random.choice(production_data["Item"].unique())
    trade_data = trade_data[trade_data["Item"] == item]
    production_data = production_data[production_data["Item"] == item]
    trade_data, production_data = format_trade_and_production(
        trade_data, production_data, country_list
    )
    return trade_data, production_data


def test_compute_calories(oceania_dataset_filtered):
    trade_data, production_data = oceania_dataset_filtered
    trade_data, production_data = compute_calories(
        trade_data, production_data, data["input"]["nutrition"]
    )
    assert isinstance(trade_data, pd.DataFrame)
    assert len(trade_data.columns) == 4
    assert "Dry Caloric Tonnes" in trade_data.columns
    trade_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    assert ~trade_data.isna().any(axis=None)
    assert trade_data.dtypes["Dry Caloric Tonnes"] == np.dtype("float64")
    assert (trade_data["Dry Caloric Tonnes"] >= 0).all(axis=None)
    assert isinstance(production_data, pd.DataFrame)
    assert len(production_data.columns) == 3
    assert "Dry Caloric Tonnes" in production_data.columns
    production_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    assert ~production_data.isna().any(axis=None)
    assert production_data.dtypes["Dry Caloric Tonnes"] == np.dtype("float64")
    assert (production_data["Dry Caloric Tonnes"] >= 0).all(axis=None)


def test_reindex(oceania_dataset_filtered):
    trade_data, production_data = oceania_dataset_filtered
    country_list = ["AUS", "NZL", "PNG", "FJI"]
    trade_data, production_data = reindex_trade_and_production(
        trade_data.drop_duplicates(subset=["Reporter ISO3"]).pivot(
            values=data["input"]["year_flag"],
            index="Reporter ISO3",
            columns="Partner ISO3",
        ),
        production_data.groupby(["ISO3"]).sum(numeric_only=True).squeeze(),
        country_list,
    )
    assert isinstance(trade_data, pd.DataFrame)
    assert trade_data.shape == (len(country_list), len(country_list))
    assert trade_data.columns.equals(trade_data.index)
    assert trade_data.columns.equals(pd.Index(country_list))
    assert ~trade_data.isna().any(axis=None)
    assert isinstance(production_data, pd.Series)
    assert len(production_data) == len(country_list)
    assert production_data.index.equals(trade_data.index)
    assert ~production_data.isna().any(axis=None)


def test_format(oceania_dataset_formatted):
    trade_data, production_data = oceania_dataset_formatted
    country_list = ["AUS", "NZL", "PNG", "FJI"]
    assert isinstance(trade_data, pd.DataFrame)
    assert trade_data.shape == (len(country_list), len(country_list))
    assert trade_data.columns.equals(trade_data.index)
    assert trade_data.columns.equals(pd.Index(country_list))
    assert ~trade_data.isna().any(axis=None)
    assert isinstance(production_data, pd.Series)
    assert len(production_data) == len(country_list)
    assert production_data.index.equals(trade_data.index)
    assert ~production_data.isna().any(axis=None)


def test_reexport(oceania_dataset_formatted):
    """
    This (removing net zero countries, prebalancing and reexport itself) has
    been tested thoroughly in the PyTradeShifts project:
    https://github.com/allfed/pytradeshifts
    """
    country_list = ["AUS", "NZL", "PNG", "FJI"]
    trade_data, production_data = reindex_trade_and_production(
        *correct_reexports(*prebalance(*oceania_dataset_formatted)), country_list
    )
    assert isinstance(trade_data, pd.DataFrame)
    assert trade_data.shape == (len(country_list), len(country_list))
    assert trade_data.columns.equals(trade_data.index)
    assert trade_data.columns.equals(pd.Index(country_list))
    trade_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    assert ~trade_data.isna().any(axis=None)
    assert (trade_data >= 0).all(axis=None)
    assert isinstance(production_data, pd.Series)
    assert len(production_data) == len(country_list)
    assert production_data.index.equals(trade_data.index)
    production_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    assert ~production_data.isna().any(axis=None)
    assert (production_data >= 0).all(axis=None)


def test_reexport_random_item(oceania_dataset_formatted_random_item):
    """
    This (removing net zero countries, prebalancing and reexport itself) has
    been tested thoroughly in the PyTradeShifts project:
    https://github.com/allfed/pytradeshifts
    """
    country_list = ["AUS", "NZL", "PNG", "FJI"]
    trade_data, production_data = reindex_trade_and_production(
        *correct_reexports(*prebalance(*oceania_dataset_formatted_random_item)),
        country_list
    )
    assert isinstance(trade_data, pd.DataFrame)
    assert trade_data.shape == (len(country_list), len(country_list))
    assert trade_data.columns.equals(trade_data.index)
    assert trade_data.columns.equals(pd.Index(country_list))
    trade_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    assert ~trade_data.isna().any(axis=None)
    assert (trade_data >= 0).all(axis=None)
    assert isinstance(production_data, pd.Series)
    assert len(production_data) == len(country_list)
    assert production_data.index.equals(trade_data.index)
    production_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    assert ~production_data.isna().any(axis=None)
    assert (production_data >= 0).all(axis=None)
