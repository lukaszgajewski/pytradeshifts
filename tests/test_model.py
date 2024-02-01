from src.model import PyTradeShifts
import pandas as pd
import os
import numpy as np
from src.preprocessing import rename_countries
import pytest


def get_pytradeshifts_post_reexport(
    crop: str,
    base_year: int,
    region: str,
    beta: float,
) -> PyTradeShifts:
    """
    Provides PyTradeShifts object after re-export correction for comparison
    with validation data.

    Arguments:
        crop (str): The crop to build the trade matrix for.
        base_year (int): The base_year to extract data for. The trade communities
            are built relative to this year.
        region (str): The region to extract data for.
        beta (float, optional): The parameter to use for the distance cost.
            If 0, no distance cost is applied.

    Returns:
        PyTradeShifts object
    """
    Wheat2018 = PyTradeShifts(
        crop=crop,
        base_year=base_year,
        region=region,
        beta=beta,
        testing=True,
    )
    # Load the data
    Wheat2018.load_data()
    # Remove countries with zero trade and zero production
    Wheat2018.remove_net_zero_countries()
    # Run the prebalancing
    Wheat2018.prebalance()
    # Reexport
    Wheat2018.correct_reexports()
    return Wheat2018


@pytest.mark.parametrize(
    ("crop", "base_year", "region"),
    [
        ("Wheat", 2018, "Global"),
        ("Wheat", 2018, "Oceania"),
    ],
)
class TestGeneralPyTradeShifts:
    """
    Testing class containing tests for PyTradeShifts that are parametrisable
    by crop, base_year and region.
    """

    def test_loading(self, crop: str, base_year: int, region: str) -> None:
        """
        Loads the data to make sure it is loaded correctly.

        Arguments:
            crop (str): The crop to build the trade matrix for.
            base_year (int): The base_year to extract data for. The trade communities
                are built relative to this year.
            region (str): The region to extract data for.
            beta (float, optional): The parameter to use for the distance cost.
                If 0, no distance cost is applied.

        Returns:
            None
        """
        pts = PyTradeShifts(crop=crop, base_year=base_year, region=region, testing=True)
        pts.load_data()
        assert isinstance(pts.trade_matrix, pd.DataFrame)
        assert isinstance(pts.production_data, pd.Series)

    def test_build_graph(self, crop: str, base_year: int, region: str) -> None:
        """
        Builds a graph from the trade matrix and checks if it has the
        same dimensions as the trade matrix.

        Arguments:
            crop (str): The crop to build the trade matrix for.
            base_year (int): The base_year to extract data for. The trade communities
                are built relative to this year.
            region (str): The region to extract data for.
            beta (float, optional): The parameter to use for the distance cost.
                If 0, no distance cost is applied.

        Returns:
            None
        """
        pts = get_pytradeshifts_post_reexport(crop, base_year, region, 0.0)
        # Remove countries with low trade
        pts.remove_below_percentile()
        # Set the diagonal to zero
        np.fill_diagonal(pts.trade_matrix.values, 0)
        # Build the graph
        pts.build_graph()
        assert pts.trade_graph is not None
        # Check is the graph has the same number of
        # nodes as the trade matrix has unique countries in rows and columns
        assert pts.trade_graph.number_of_nodes() == len(
            np.unique(pts.trade_matrix.index)
        )

    def test_apply_distance_cost_zero(
        self, crop: str, base_year: int, region: str
    ) -> None:
        """
        Applies the gravity model of trade modification to simulate higher transport
        costs for beta=0.
        Checks if the trade matrix format remains unchanged, and if the values
        post transformation are unchanged.

        Arguments:
            crop (str): The crop to build the trade matrix for.
            base_year (int): The base_year to extract data for. The trade communities
                are built relative to this year.
            region (str): The region to extract data for.
            beta (float, optional): The parameter to use for the distance cost.
                If 0, no distance cost is applied.

        Returns:
            None
        """
        pts = get_pytradeshifts_post_reexport(crop, base_year, region, 0.0)
        # Set the diagonal to zero
        np.fill_diagonal(pts.trade_matrix.values, 0)
        pre_apply_matrix = pts.trade_matrix.copy()
        pts.apply_distance_cost()
        # beta = 0 so nothing should have happened
        assert (pre_apply_matrix == pts.trade_matrix).all(axis=None)

    def test_apply_distance_cost_positive(
        self, crop: str, base_year: int, region: str
    ) -> None:
        """
        Applies the gravity model of trade modification to simulate higher transport
        costs for beta>0.
        Checks if the trade matrix format remains unchanged, and if the values
        post transformation are lower.

        Arguments:
            crop (str): The crop to build the trade matrix for.
            base_year (int): The base_year to extract data for. The trade communities
                are built relative to this year.
            region (str): The region to extract data for.
            beta (float, optional): The parameter to use for the distance cost.
                If 0, no distance cost is applied.

        Returns:
            None
        """
        pts = get_pytradeshifts_post_reexport(crop, base_year, region, 2.0)
        # Set the diagonal to zero
        np.fill_diagonal(pts.trade_matrix.values, 0)
        pre_apply_matrix = pts.trade_matrix.copy()
        pts.apply_distance_cost()
        # check that index remains unchanged
        # if it changed it means we're missing some regions in the distance matrix
        assert (pre_apply_matrix.index == pts.trade_matrix.index).all(axis=None), set(
            pre_apply_index
        ).difference(pts.trade_matrix.index)
        # check that the shape didn't change
        assert pre_apply_matrix.shape == pts.trade_matrix.shape, (
            pre_apply_shape,
            pts.trade_matrix.shape,
        )
        # beta = 2 so all values should be <= than before
        assert (pre_apply_matrix >= pts.trade_matrix).all(axis=None)

    def test_apply_distance_cost_negative(
        self, crop: str, base_year: int, region: str
    ) -> None:
        """
        Applies the gravity model of trade modification to simulate higher transport
        costs for beta>0.
        Checks if the trade matrix format remains unchanged, and if the values
        post transformation are higher.

        Arguments:
            crop (str): The crop to build the trade matrix for.
            base_year (int): The base_year to extract data for. The trade communities
                are built relative to this year.
            region (str): The region to extract data for.
            beta (float, optional): The parameter to use for the distance cost.
                If 0, no distance cost is applied.

        Returns:
            None
        """
        pts = get_pytradeshifts_post_reexport(crop, base_year, region, -2.0)
        # Set the diagonal to zero
        np.fill_diagonal(pts.trade_matrix.values, 0)
        pre_apply_matrix = pts.trade_matrix.copy()
        pts.apply_distance_cost()
        # check that index remains unchanged
        # if it changed it means we're missing some regions in the distance matrix
        assert (pre_apply_matrix.index == pts.trade_matrix.index).all(axis=None), set(
            pre_apply_index
        ).difference(pts.trade_matrix.index)
        # check that the shape didn't change
        assert pre_apply_matrix.shape == pts.trade_matrix.shape, (
            pre_apply_shape,
            pts.trade_matrix.shape,
        )
        # beta = -2 so all values should be >= than before
        assert (pre_apply_matrix <= pts.trade_matrix).all(axis=None)


def get_region_datafile_name(region: str) -> str:
    """
    Provides data file appropriate region name

    Arguments:
        region (str): The region to extract data for.

    Returns:
        str: region name as found in data files, e.g., "All_Data"
    """
    return "All_Data" if region == "Global" else region


@pytest.mark.parametrize(
    ("region"),
    [
        ("Global"),
        ("Oceania"),
    ],
)
class TestWheat2018PyTradeShifts:
    """
    Testing class containing tests for PyTradeShifts that are parametrisable
    by region only and are specifically compared against 2018 Wheat yields.
    """

    def test_removing_countries_with_zeros(self, region: str) -> None:
        """
        Runs removal of neto zero countries, checks if countries with zero
        production have non-zero trade, and compares with results from the R script

        Arguments:
            region (str): The region to extract data for.

        Returns:
            None
        """
        Wheat2018 = PyTradeShifts(
            crop="Wheat", base_year=2018, region=region, testing=True
        )
        # Load the data
        Wheat2018.load_data()
        # Remove countries with both zero trade and zero production
        Wheat2018.remove_net_zero_countries()
        # Load the data from the R script
        wheat2018_from_R = pd.read_csv(
            "data"
            + os.sep
            + "validation_data_from_Hedlung_2022"
            + os.sep
            + get_region_datafile_name(region)
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
        # Check that countries with zero production have some trade
        assert (
            Wheat2018.trade_matrix.loc[Wheat2018.production_data == 0, :].sum(axis=1)
            + Wheat2018.trade_matrix.loc[
                :,
                Wheat2018.production_data == 0,
            ].sum(axis=0)
            != 0
        ).all()

    def test_prebalancing(self, region: str) -> None:
        """
        Runs prebalancing and compares the results with the results from the
        R script.

        Arguments:
            region (str): The region to extract data for.

        Returns:
            None
        """
        Wheat2018 = PyTradeShifts(
            crop="Wheat", base_year=2018, region=region, testing=True
        )
        # Load the data
        Wheat2018.load_data()
        # Remove countries with both zero trade and zero production
        Wheat2018.remove_net_zero_countries()
        # Run the prebalancing
        Wheat2018.prebalance()
        # Load the data from the R script
        wheat2018_from_R = pd.read_csv(
            "data"
            + os.sep
            + "validation_data_from_Hedlung_2022"
            + os.sep
            + get_region_datafile_name(region)
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

    def test_reexport(self, region: str) -> None:
        """
        Runs the re-export algorithm and compares the results with the results
        from the R script.

        Arguments:
            region (str): The region to extract data for.

        Returns:
            None
        """
        Wheat2018 = get_pytradeshifts_post_reexport("Wheat", 2018, region, 0.0)
        # Load the data from the R script
        wheat2018_from_R = pd.read_csv(
            "data"
            + os.sep
            + "validation_data_from_Hedlung_2022"
            + os.sep
            + get_region_datafile_name(region)
            + os.sep
            + "NEW_Ex_wheat_2018.csv",
            index_col=0,
        )
        # Convert all entries in index and columns to strings
        wheat2018_from_R.index = wheat2018_from_R.index.astype(int)
        wheat2018_from_R.columns = wheat2018_from_R.columns.astype(int)
        # Rename the R names to match the Python names
        wheat2018_from_R = rename_countries(
            wheat2018_from_R,
            get_region_datafile_name(region),
            "Trade_DetailedTradeMatrix_E",
            "Area Code",
        )
        # Sort the index and columns
        wheat2018_from_R.sort_index(inplace=True)
        wheat2018_from_R = wheat2018_from_R[sorted(wheat2018_from_R.columns)]
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
        # Compare the sorted trade matrix with the sorted R matrix
        trade_matrix_sorted = Wheat2018.trade_matrix.sort_index()
        trade_matrix_sorted = trade_matrix_sorted[sorted(trade_matrix_sorted.columns)]
        assert wheat2018_from_R.index.equals(trade_matrix_sorted.index)
        assert wheat2018_from_R.columns.equals(trade_matrix_sorted.columns)
        # Check if the values if they are rounded to 2 decimal places
        # First round the whole matrix to 2 decimal places
        wheat2018_from_R = wheat2018_from_R.round(2)
        trade_matrix_sorted = trade_matrix_sorted.round(2)
        assert (wheat2018_from_R == trade_matrix_sorted).all(axis=None)


def test_removing_countries() -> None:
    """
    Runs the model and removes countries to check if this works
    """
    Wheat2018 = PyTradeShifts(
        "Wheat",
        2018,
        region="Global",
        testing=True,
        countries_to_remove=["Australia", "Bangladesh"],
    )

    # Load the data
    Wheat2018.load_data()

    # Remove countries with zero trade and zero production
    Wheat2018.remove_net_zero_countries()

    # Run the prebalancing
    Wheat2018.prebalance()

    # Reexport
    Wheat2018.correct_reexports()

    assert "Australia" in list(Wheat2018.trade_matrix.index)
    assert "Bangladesh" in list(Wheat2018.trade_matrix.index)

    shape_before = Wheat2018.trade_matrix.shape

    # Remove countries
    Wheat2018.remove_countries()

    assert "Australia" not in list(Wheat2018.trade_matrix.index)
    assert "Bangladesh" not in list(Wheat2018.trade_matrix.index)

    assert Wheat2018.trade_matrix.shape[0] == shape_before[0] - 2


def test_removing_countries_except():
    """
    Tests if removing countries except works
    """
    Wheat2018 = PyTradeShifts(
        "Wheat",
        2018,
        region="Global",
        testing=True,
        countries_to_keep=["Australia", "Bangladesh"],
    )

    # Load the data
    Wheat2018.load_data()

    # Remove countries with zero trade and zero production
    Wheat2018.remove_net_zero_countries()

    # Run the prebalancing
    Wheat2018.prebalance()

    # Reexport
    Wheat2018.correct_reexports()

    assert "Australia" in list(Wheat2018.trade_matrix.index)
    assert "Bangladesh" in list(Wheat2018.trade_matrix.index)
    assert "Germany" in list(Wheat2018.trade_matrix.index)

    # Remove countries
    Wheat2018.remove_countries_except()

    assert "Australia" in list(Wheat2018.trade_matrix.index)
    assert "Bangladesh" in list(Wheat2018.trade_matrix.index)
    assert "Germany" not in list(Wheat2018.trade_matrix.index)

    assert Wheat2018.trade_matrix.shape == (2, 2)


def test_removing_low_trade_countries() -> None:
    """
    Runs the model with prebalancing, prebalancing, removing re-exports
    and removing countries with low trade and check if this worked out.
    """
    # Remove the countries which don't have ISIMIP data, as
    # Johanna Hedlund did in her analysis for determining the global threshold
    ISIMIP = pd.read_csv(
        "data/scenario_files/ISIMIP_climate/ISIMIP_wheat_Hedlung.csv", index_col=0
    )
    # Get only those countries with NaNs in the ISIMIP data
    nan_indices = ISIMIP.index[ISIMIP.iloc[:, 0].isnull()].tolist()

    Wheat2018 = PyTradeShifts(
        "Wheat", 2018, region="Global", testing=True, countries_to_remove=nan_indices
    )

    # Load the data
    Wheat2018.load_data()

    # Remove countries with zero trade
    Wheat2018.remove_net_zero_countries()

    # Run the prebalancing
    Wheat2018.prebalance()

    # Reexport
    Wheat2018.correct_reexports()

    # Set diagonal to zero
    np.fill_diagonal(Wheat2018.trade_matrix.values, 0)

    # Remove countries
    Wheat2018.remove_countries()

    shape_before = Wheat2018.trade_matrix.shape

    # Calculate the threshold
    threshold = np.percentile(
        Wheat2018.trade_matrix.values[Wheat2018.trade_matrix.values > 0],
        Wheat2018.percentile * 100,
    )

    # Remove countries with low trade
    Wheat2018.remove_below_percentile()

    # Check if countries with low trade have been removed
    assert Wheat2018.trade_matrix.shape[0] < shape_before[0]

    # Check if all the values below the threshold have been removed
    # Find the smallest value in the trade matrix which is not zero
    smallest_value = np.min(
        Wheat2018.trade_matrix.values[Wheat2018.trade_matrix.values > 0]
    )
    assert smallest_value > threshold


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

    # Set diagonal to zero
    np.fill_diagonal(Wheat2018.trade_matrix.values, 0)

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


def test_apply_scenario() -> None:
    """
    Builds a graph from the trade matrix and finds the trade communities.
    """
    Wheat2018 = PyTradeShifts(
        "Wheat",
        2018,
        region="Global",
        testing=True,
        scenario_name="ISIMIP",
        scenario_file_name="ISIMIP_climate" + os.sep + "ISIMIP_wheat_Hedlung.csv",
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
