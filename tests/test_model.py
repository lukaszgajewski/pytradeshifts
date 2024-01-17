from src.model import PyTradeShifts
import pandas as pd
import os
import numpy as np


def loading(region):
    """
    Just loads the data to make sure it is loaded correctly.
    """
    Wheat2018 = PyTradeShifts("Wheat", "Y2018", region=region, testing=True)
    Wheat2018.load_data()

    # Make sure the data is loaded correctly
    assert isinstance(Wheat2018.trade_matrix, pd.DataFrame)
    assert isinstance(Wheat2018.production_data, pd.Series)


def test_loading_oceania():
    """
    Loads the data for Oceania.
    """
    loading("Oceania")


def test_loading_global():
    """
    Loads the data for the world.
    """
    loading("Global")


def removing_countries_with_zeros(region):
    """
    Runs the model with prebalancing and compares the results with the
    results from the R script.
    """
    Wheat2018 = PyTradeShifts("Wheat", "Y2018", region=region, testing=True)

    # Load the data
    Wheat2018.load_data()

    # Remove countries with both zero trade and zero production
    Wheat2018.remove_net_zero_countries()

    if region == "Global":
        region = "All_Data"

    # Load the data from the R script
    wheat2018_from_R = pd.read_csv(
        "data"
        + os.sep
        + "validation_data_from_Hedlung_2022"
        + os.sep
        + region
        + os.sep
        + "NEW_trade_wheat_2018.csv",
        index_col=0,
    )

    # Compare the results
    assert wheat2018_from_R.shape == Wheat2018.trade_matrix.shape
    # Comparing the sum this has only be correct to 2 decimal places
    assert round(wheat2018_from_R.sum().sum(), 2) == round(
        Wheat2018.trade_matrix.sum().sum(), 2
    )


def test_removing_countries_with_zeros_oceania():
    """
    Runs the model with prebalancing and compares the results with the
    results from the R script.
    """
    removing_countries_with_zeros("Oceania")


def test_removing_countries_with_zeros_global():
    """
    Runs the model with prebalancing and compares the results with the
    results from the R script.
    """
    removing_countries_with_zeros("Global")


def prebalancing(region):
    """
    Runs the model with prebalancing and compares the results with the
    results from the R script.
    """
    Wheat2018 = PyTradeShifts("Wheat", "Y2018", region=region, testing=True)

    # Load the data
    Wheat2018.load_data()

    # Remove countries with both zero trade and zero production
    Wheat2018.remove_net_zero_countries()

    # Run the prebalancing
    Wheat2018.prebalance()

    if region == "Global":
        region = "All_Data"
    # Load the data from the R script
    wheat2018_from_R = pd.read_csv(
        "data"
        + os.sep
        + "validation_data_from_Hedlung_2022"
        + os.sep
        + region
        + os.sep
        + "TEST_prebalanced_trade_wheat_2018.csv",
        index_col=0,
    )

    # Compare the results
    assert wheat2018_from_R.shape == Wheat2018.trade_matrix.shape
    # Comparing the sum this has only be correct to 2 decimal places
    assert round(wheat2018_from_R.sum().sum(), 2) == round(
        Wheat2018.trade_matrix.sum().sum(), 2
    )


def test_prebalancing_oceania():
    """
    Runs the model with prebalancing and compares the results with the
    results from the R script.
    """
    prebalancing("Oceania")


def test_prebalancing_global():
    """
    Runs the model with prebalancing and compares the results with the
    results from the R script.
    """
    prebalancing("Global")


def reexport(region):
    """
    Runs the model with prebalancing and compares the results with the
    results from the R script.
    """
    Wheat2018 = PyTradeShifts("Wheat", "Y2018", region=region, testing=True)

    # Load the data
    Wheat2018.load_data()

    # Remove countries with zero trade and zero production
    Wheat2018.remove_net_zero_countries()

    # Run the prebalancing
    Wheat2018.prebalance()

    # Reexport
    Wheat2018.correct_reexports()

    if region == "Global":
        region = "All_Data"

    # Load the data from the R script
    wheat2018_from_R = pd.read_csv(
        "data"
        + os.sep
        + "validation_data_from_Hedlung_2022"
        + os.sep
        + region
        + os.sep
        + "NEW_Ex_wheat_2018.csv",
        index_col=0,
    )

    # Compare the results
    assert wheat2018_from_R.shape == Wheat2018.trade_matrix.shape
    # Comparing the sum this has only be correct to 2 decimal places
    assert round(wheat2018_from_R.sum().sum(), 2) == round(
        Wheat2018.trade_matrix.sum().sum(), 2
    )
    # check if the median is the same for both
    assert round(wheat2018_from_R.median().median(), 2) == round(
        Wheat2018.trade_matrix.median().median(), 2
    )


def test_reexport_oceania():
    """
    Runs the model with prebalancing and compares the results with the
    results from the R script.
    """
    reexport("Oceania")


def test_reexport_global():
    """
    Runs the model with prebalancing and compares the results with the
    results from the R script.
    """
    reexport("Global")


def removing_low_trade_countries(region):
    """
    Runs the model with prebalancing, prebalancing, removing re-exports
    and removing countries with low trade and check if this worked out. 
    """
    Wheat2018 = PyTradeShifts("Wheat", "Y2018", region=region, testing=True)

    # Load the data
    Wheat2018.load_data()
    
    # Remove countries with zero trade
    Wheat2018.remove_net_zero_countries()

    # Run the prebalancing
    Wheat2018.prebalance()

    # Reexport
    Wheat2018.correct_reexports()

    # Remove countries with low trade
    Wheat2018.remove_below_percentile()

    print(Wheat2018.trade_matrix.shape)

    if region == "Oceania":
        assert Wheat2018.threshold == 0.1
        assert Wheat2018.trade_matrix.shape == (1, 2)
    elif region == "Global":
        assert Wheat2018.threshold == 165.372
        assert Wheat2018.trade_matrix.shape == (1, 2)


def test_removing_low_trade_countries_oceania():
    """
    Runs the model with prebalancing, prebalancing, removing re-exports
    and removing countries with low trade and check if this worked out.
    """
    removing_low_trade_countries("Oceania")


def test_removing_low_trade_countries_global():
    """
    Runs the model with prebalancing, prebalancing, removing re-exports
    and removing countries with low trade and check if this worked out. 
    """
    removing_low_trade_countries("Global")


if __name__ == "__main__":
    reexport("Global")
