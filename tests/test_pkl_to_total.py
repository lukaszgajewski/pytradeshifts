import pandas as pd
import numpy as np
import pytest
from src.input_output import load_fao_zip, data
from src import fao_trade_zip_to_pkl, fao_production_zip_to_pkl
from src.fao_pkl_to_total_caloric_trade_and_production import compute_calories


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


def test_compute_calories(oceania_dataset_filtered):
    trade_data, production_data = oceania_dataset_filtered
    trade_data, production_data = compute_calories(
        trade_data, production_data, data["input"]["nutrition"]
    )
    assert isinstance(trade_data, pd.DataFrame)
    assert len(trade_data.columns) == 4
    assert "Dry Caloric Tonnes" in trade_data.columns
    assert ~trade_data.isna().any(axis=None)
    assert trade_data.dtypes["Dry Caloric Tonnes"] == np.dtype("float64")
    assert (trade_data["Dry Caloric Tonnes"] >= 0).all(axis=None)
    assert isinstance(production_data, pd.DataFrame)
    assert len(production_data.columns) == 3
    assert "Dry Caloric Tonnes" in production_data.columns
    assert ~production_data.isna().any(axis=None)
    assert production_data.dtypes["Dry Caloric Tonnes"] == np.dtype("float64")
    assert (production_data["Dry Caloric Tonnes"] >= 0).all(axis=None)
