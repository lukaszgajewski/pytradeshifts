
from src.preprocessing import read_in_raw_trade_data
from src.preprocessing import read_in_raw_production_data
from src.preprocessing import extract_relevant_trade_data
from src.preprocessing import extract_relevant_production_data
import pandas as pd

import pytest


def test_returns_correct_dataframe_trade():
    # Only read in the Oceanian subset of the data for testing purposes.
    # This is to reduce the time it takes to run the tests.
    result = read_in_raw_trade_data(testing=True)
    assert isinstance(result, pd.DataFrame)
    # This is the shape of the Oceanian subset of the data.
    assert result.shape == (215548, 84)
    # The first country in the dataset is Australia.
    assert result.iloc[0, 2] == "Australia"


def test_extracts_right_data_trade():
    # Only read in the Oceanian subset of the data for testing purposes.
    # This is to reduce the time it takes to run the tests.
    result = read_in_raw_trade_data(testing=True)
    # Extract only the relevant data for the trade model.
    result = extract_relevant_trade_data(result, ["Maize (corn)", "Wheat", "Rice, paddy (rice milled equivalent)"], year=2018)
    assert isinstance(result, pd.DataFrame)
    # This is the shape of the Oceanian subset of the data.
    assert result.shape == (139, 6)
    # The first country in the dataset is Australia.
    assert result.iloc[0, 0] == "Australia"
    # The first item in the dataset is Bananas.
    assert result.iloc[0, 3] == "Maize"


def test_returns_correct_dataframe_production():
    result = read_in_raw_production_data()
    assert isinstance(result, pd.DataFrame)
    # This is the shape of the complete dataset.
    assert result.shape == (79297, 70)
    # The first country in the dataset is Afghanistan.
    assert result.iloc[0, 2] == "Afghanistan"


def test_extracts_right_data_production():
    result = read_in_raw_production_data()
    # Extract only the relevant data for the production model.
    result = extract_relevant_production_data(result, ["Maize (corn)", "Wheat", "Rice, paddy (rice milled equivalent)"], year=2018)
    assert isinstance(result, pd.DataFrame)
    # This is the shape of the complete dataset.
    assert result.shape == (362, 5)
    # The first country in the dataset is Afghanistan.
    assert result.iloc[0, 0] == "Afghanistan"
    # The first item in the dataset is Bananas.
    assert result.iloc[0, 1] == "Maize"




