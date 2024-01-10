from src.model import PyTradeShifts
import pandas as pd
import os


def loading(region):
    """
    Just loads the data to make sure it is loaded correctly.
    """
    Wheat2021 = PyTradeShifts(
        "Wheat",
        "Y2021",
        region=region,
        testing=True
    )
    Wheat2021.load_data()

    # Make sure the data is loaded correctly
    assert isinstance(Wheat2021.trade_matrix, pd.DataFrame)
    assert isinstance(Wheat2021.production_data, pd.Series)


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


def prebalancing(region):
    """
    Runs the model with prebalancing and compares the results with the
    results from the R script.
    """
    Wheat2021 = PyTradeShifts(
        "Wheat",
        "Y2021",
        region=region,
        testing=True
    )

    # Load the data
    Wheat2021.load_data()

    # Run the prebalancing
    Wheat2021.prebalance()

    if region == "Global":
        region = "All_Data"
    # Load the data from the R script
    wheat2021_from_R = pd.read_csv(
        "data" + os.sep +
        "validation_data_from_Hedlung_2022" + os.sep +
        region + os.sep +
        "TEST_prebalanced_trade_wheat_2021.csv",
        index_col=0,
    )

    # Compare the results
    assert wheat2021_from_R.shape == Wheat2021.trade_matrix.shape
    # Comparing the sum this has only be correct to 2 decimal places
    assert round(wheat2021_from_R.sum().sum(), 2) == round(Wheat2021.trade_matrix.sum().sum(), 2)


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


def removing_countries_with_zeros_and_reexport(region):
    """
    Runs the model with prebalancing and compares the results with the
    results from the R script.
    """
    Wheat2021 = PyTradeShifts(
        "Wheat",
        "Y2021",
        region=region,
        testing=True
    )

    # Load the data
    Wheat2021.load_data()

    # Run the prebalancing
    Wheat2021.prebalance()

    # Remove countries with zero trade
    Wheat2021.remove_net_zero_countries()

    # Reexport
    Wheat2021.correct_reexports()

    if region == "Global":
        region = "All_Data"

    # Load the data from the R script
    wheat2021_from_R = pd.read_csv(
        "data" + os.sep +
        "validation_data_from_Hedlung_2022" + os.sep +
        region + os.sep +
        "NEW_Ex_wheat_2021.csv",
        index_col=0,
    )

    # Compare the results
    assert wheat2021_from_R.shape == Wheat2021.trade_matrix.shape
    # Comparing the sum this has only be correct to 2 decimal places
    assert round(wheat2021_from_R.sum().sum(), 2) == round(Wheat2021.trade_matrix.sum().sum(), 2)


def test_removing_countries_with_zeros_and_reexport_oceania():
    """
    Runs the model with prebalancing and compares the results with the
    results from the R script.
    """
    removing_countries_with_zeros_and_reexport("Oceania")


def test_removing_countries_with_zeros_and_reexport_global():
    """
    Runs the model with prebalancing and compares the results with the
    results from the R script.
    """
    removing_countries_with_zeros_and_reexport("Global")


if __name__ == "__main__":
    test_removing_countries_with_zeros_and_reexport_global()
