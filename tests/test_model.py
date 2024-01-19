from src.model import PyTradeShifts
import pandas as pd
import os
import numpy as np


def loading(region):
    """
    Just loads the data to make sure it is loaded correctly.
    """
    Wheat2018 = PyTradeShifts("Wheat", 2018, region=region, testing=True)
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
    Wheat2018 = PyTradeShifts("Wheat", 2018, region=region, testing=True)

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
    Wheat2018 = PyTradeShifts("Wheat", 2018, region=region, testing=True)

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
    Wheat2018 = PyTradeShifts("Wheat", 2018, region=region, testing=True)

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
    Wheat2018 = PyTradeShifts("Wheat", 2018, region=region, testing=True)

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


def test_build_graph():
    """
    Builds a graph from the trade matrix and checks if it has the
    same dimensions as the trade matrix.
    """
    Wheat2018 = PyTradeShifts("Wheat", 2018, region="Global", testing=True)

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

    # Build the graph
    Wheat2018.build_graph()

    assert Wheat2018.trade_graph is not None

    # Check is the graph has the same number of
    # nodes as the trade matrix has unique countries in rows and columns
    assert Wheat2018.trade_graph.number_of_nodes() == len(
        np.unique(Wheat2018.trade_matrix.index)
    )


def test_find_communities():
    """
    Builds a graph from the trade matrix and finds the trade communities.
    """
    Wheat2018 = PyTradeShifts("Wheat", 2018, region="Global", testing=True)

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

    # Build the graph
    Wheat2018.build_graph()

    # Find the communities
    Wheat2018.find_trade_communities()

    assert Wheat2018.trade_communities is not None

    # The resulting communities should be a list of sets
    assert isinstance(Wheat2018.trade_communities, list)
    assert isinstance(Wheat2018.trade_communities[0], set)

    # At least one community should contain both Brazil and Chile
    assert any(
        {"Brazil", "Chile"} <= community for community in Wheat2018.trade_communities
    )
    # Another community should contain both Germany and France
    assert any(
        {"Germany", "France"} <= community for community in Wheat2018.trade_communities
    )
    # Canada and the US should be in the same community
    assert any(
        {"Canada", "United States"} <= community
        for community in Wheat2018.trade_communities
    )


def test_apply_scenario():
    """
    Builds a graph from the trade matrix and finds the trade communities.
    """
    Wheat2018 = PyTradeShifts(
        "Wheat",
        2018,
        region="Global",
        testing=True,
        scenario_name="ISIMIP",
        scenario_file_name="ISIMIP_wheat_Hedlung.csv",
    )

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

    # Apply the scenario
    Wheat2018.apply_scenario()

    print(Wheat2018.trade_matrix.shape)

    # Build the graph
    Wheat2018.build_graph()

    # Find the communities
    Wheat2018.find_trade_communities()

    # Check if the amount of trade for some countries has been reduced by the right amount
    aus_ban = 1.50777499999999998 * 91600.6390029294
    assert round(Wheat2018.trade_matrix.loc["Australia", "Bangladesh"], 1) == round(
        aus_ban, 1
    )

    # Check if a country which is not in the scenario has been removed
    assert "Indonesia" not in Wheat2018.trade_matrix.index


if __name__ == "__main__":
    test_apply_scenario()
