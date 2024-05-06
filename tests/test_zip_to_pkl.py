import pandas as pd
import numpy as np
from src.input_output import load_fao_zip, data
from src import fao_trade_zip_to_pkl, fao_production_zip_to_pkl
import pytest


@pytest.fixture
def oceania_dataset_with_iso3() -> tuple[pd.DataFrame, pd.DataFrame]:
    return fao_trade_zip_to_pkl.convert_to_ISO3(
        load_fao_zip("data/testing/Trade_DetailedTradeMatrix_E_Oceania.zip"),
        data["input"]["country_codes"],
    ), fao_production_zip_to_pkl.convert_to_ISO3(
        load_fao_zip("data/testing/Production_Crops_Livestock_E_Oceania.zip"),
        data["input"]["country_codes"],
    )


def test_load_zip():
    trade_data = load_fao_zip("data/testing/Trade_DetailedTradeMatrix_E_Oceania.zip")
    assert isinstance(trade_data, pd.core.frame.DataFrame)
    assert not trade_data.empty
    production_data = load_fao_zip(
        "data/testing/Production_Crops_Livestock_E_Oceania.zip"
    )
    assert isinstance(production_data, pd.core.frame.DataFrame)
    assert not production_data.empty


def test_convert_to_ISO3(oceania_dataset_with_iso3):
    trade_data, production_data = oceania_dataset_with_iso3
    assert "Reporter ISO3" in trade_data.columns
    assert "Partner ISO3" in trade_data.columns
    assert "not found" not in trade_data["Reporter ISO3"].values
    assert "not found" not in trade_data["Partner ISO3"].values
    assert "ISO3" in production_data.columns
    assert "not found" not in production_data["ISO3"].values


def test_data_filtering(oceania_dataset_with_iso3):
    trade_data, production_data = oceania_dataset_with_iso3
    og_trade_shape = np.array(trade_data.shape)
    og_production_shape = np.array(production_data.shape)
    trade_data = fao_trade_zip_to_pkl.filter_data(
        trade_data, data["input"]["nutrition"], data["input"]["yield_reduction"]
    )
    assert (og_trade_shape >= trade_data.shape).all()
    assert len(trade_data.columns) == 4
    assert ~trade_data.isna().any(axis=None)
    assert isinstance(trade_data.index, pd.core.indexes.range.RangeIndex)
    production_data = fao_production_zip_to_pkl.filter_data(
        production_data, data["input"]["nutrition"], data["input"]["yield_reduction"]
    )
    assert (og_production_shape >= production_data.shape).all()
    assert len(production_data.columns) == 3
    assert ~production_data.isna().any(axis=None)
    assert isinstance(production_data.index, pd.core.indexes.range.RangeIndex)
