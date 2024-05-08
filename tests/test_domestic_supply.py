import pandas as pd
import numpy as np
import pytest
from src.input_output import data
from src import compute_domestic_supply


@pytest.fixture
def dataset():
    return compute_domestic_supply.load_data(
        data["intermidiary"]["caloric_trade"],
        data["intermidiary"]["caloric_production"],
        data["input"]["yield_reduction"],
    )


def test_load_data(dataset):
    total_trade, total_production, yield_reduction = dataset
    assert isinstance(total_trade, pd.DataFrame)
    assert total_trade.index.equals(total_trade.columns)
    total_trade.replace([np.inf, -np.inf], np.nan, inplace=True)
    assert ~total_trade.isna().any(axis=None)
    assert (total_trade >= 0).all(axis=None)
    assert isinstance(total_production, pd.Series)
    assert total_production.index.equals(total_trade.index)
    total_production.replace([np.inf, -np.inf], np.nan, inplace=True)
    assert ~total_production.isna().any(axis=None)
    assert (total_production >= 0).all(axis=None)
    assert isinstance(yield_reduction, pd.DataFrame)
    assert yield_reduction.index.equals(total_production.index)
    yield_reduction.replace([np.inf, -np.inf], np.nan, inplace=True)
    assert ~yield_reduction.isna().any(axis=None)
    assert (yield_reduction >= 0).all(axis=None)


def test_compute_domestic_supply(dataset):
    total_trade, total_production, _ = dataset
    ds = compute_domestic_supply.compute_domestic_supply(total_trade, total_production)
    assert isinstance(ds, pd.Series)
    assert ds.index.equals(total_production.index)
    ds.replace([np.inf, -np.inf], np.nan, inplace=True)
    assert ~ds.isna().any(axis=None)
    assert (ds >= 0).all(axis=None)


def test_compute_reduced_supply(dataset):
    total_trade, total_production, yield_reduction = dataset
    reduced_ds = compute_domestic_supply.compute_reduced_supply_yearly(
        total_trade, total_production, yield_reduction
    )
    assert isinstance(reduced_ds, pd.DataFrame)
    assert reduced_ds.index.equals(total_trade.index)
    assert len(reduced_ds.columns) == len(yield_reduction.columns)
    reduced_ds.replace([np.inf, -np.inf], np.nan, inplace=True)
    assert ~reduced_ds.isna().any(axis=None)
    assert (reduced_ds >= 0).all(axis=None)
