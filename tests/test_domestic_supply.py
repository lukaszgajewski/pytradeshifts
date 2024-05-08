import pandas as pd
from src.input_output import data
from src import compute_domestic_supply


def test_load_data():
    total_trade, total_production, yield_reduction = compute_domestic_supply.load_data(
        data["intermidiary"]["caloric_trade"],
        data["intermidiary"]["caloric_production"],
        data["input"]["yield_reduction"],
    )
    assert isinstance(total_trade, pd.DataFrame)
    assert total_trade.index.equals(total_trade.columns)
    assert ~total_trade.isna().any(axis=None)
    assert (total_trade >= 0).all(axis=None)
    assert isinstance(total_production, pd.Series)
    assert total_production.index.equals(total_trade.index)
    assert ~total_production.isna().any(axis=None)
    assert (total_production >= 0).all(axis=None)
    assert isinstance(yield_reduction, pd.DataFrame)
    assert yield_reduction.index.equals(total_production.index)
    assert ~yield_reduction.isna().any(axis=None)
    assert (yield_reduction >= 0).all(axis=None)
