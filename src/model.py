import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import geopandas as gpd
import country_converter as coco
from matplotlib.colors import ListedColormap
import seaborn as sns
from src.preprocessing import main as preprocessing_main
from src.utils import plot_winkel_tripel_map

plt.style.use(
    "https://raw.githubusercontent.com/allfed/ALLFED-matplotlib-style-sheet/main/ALLFED.mplstyle"
)


class PyTradeShifts:
    """
    Class to build the trade matrix, calculate the trade shifts
    and plot them. This combines all the methods that are needed
    to easily run this from a jupyter notebook.

    Arguments:
        crop (str): The crop to build the trade matrix for.
        base_year (int): The base_year to extract data for. The trade communities
            are built relative to this year.
        percentile (float): The percentile to use for removing countries with
            low trade.
        region (str): The region to extract data for.
        testing (bool): Whether to run the methods or not. This is only used for
            testing purposes.
        scenario_name (str, optional): The name of the scenario to apply.
            If None, no scenario is applied.
        scenario_file_name (str, optional): The path to the scenario file.
            If None, no scenario is applied.
        with_preprocessing (bool, optional): Whether to run the preprocessing
            or not.

    Returns:
        None
    """

    def __init__(
        self,
        crop,
        base_year,
        percentile=0.75,
        region="Global",
        testing=False,
        scenario_name=None,
        scenario_file_name=None,
        with_preprocessing=False,
    ):
        # Save the arguments
        self.crop = crop
        self.base_year = "Y"+str(base_year)
        self.percentile = percentile
        self.region = region
        self.scenario_name = scenario_name
        self.scenario_file_name = scenario_file_name
        # State variables to keep track of the progress
        self.prebalanced = False
        self.reexports_corrected = False
        self.no_trade_removed = False
        self.scenario_run = False
        # variables to keep track of the results
        self.trade_graph = None
        self.trade_matrix = None
        self.production_data = None
        self.threshold = None
        self.trade_communities = None

        # Don't run the methods if we are testing, so we can test them individually
        if not testing:
            if with_preprocessing:
                preprocessing_main(self.crop, self.base_year, self.region)
            # Read in the data
            self.load_data()
            # Remove countries with all zeroes in trade and production
            self.remove_net_zero_countries()
            # Prebalance the trade matrix
            self.prebalance()
            # Remove re-exports
            self.correct_reexports()
            # Remove countries with low trade
            self.remove_below_percentile()
            if scenario_name is not None:
                self.apply_scenario()
            # Build the graph
            self.build_graph()
            # Find the trade communities
            self.find_trade_communities()
            # Plot the trade communities
            self.plot_trade_communities()

    def load_data(self):
        """
        Loads the data into a pandas dataframe and cleans it
        of countries with trade below a certain percentile.

        Arguments:
            None

        Returns:
            None
        """
        assert self.trade_matrix is None
        # Read in the data
        trade_matrix = pd.read_csv(
            "."
            + os.sep
            + "data"
            + os.sep
            + "preprocessed_data"
            + os.sep
            + f"{self.crop}_{self.base_year}_{self.region}_trade.csv",
            index_col=0,
        )

        production_data = pd.read_csv(
            "."
            + os.sep
            + "data"
            + os.sep
            + "preprocessed_data"
            + os.sep
            + f"{self.crop}_{self.base_year}_{self.region}_production.csv",
            index_col=0,
        ).squeeze()

        print(f"Loaded data for {self.crop} in {self.base_year}.")

        # Retain only the countries where we have production data and trade data
        countries = np.intersect1d(trade_matrix.index, production_data.index)
        trade_matrix = trade_matrix.loc[countries, countries]
        production_data = production_data.loc[countries]
        # Make sure this worked
        assert trade_matrix.shape[0] == production_data.shape[0]

        # Save the data
        self.trade_matrix = trade_matrix
        self.production_data = production_data

    def remove_below_percentile(
        self
    ):
        """
        Removes countries with trade below a certain percentile.

        Arguments:
            None

        Returns:
            None
        """
        # Make sure the data is loaded and no threshold is calculated yet
        assert self.trade_matrix is not None
        assert self.percentile is not None
        assert self.threshold is None
        # Calculate the percentile out of all values in the trade matrix. This
        # only considers the values above 0.
        threshold = np.percentile(
            self.trade_matrix.values[self.trade_matrix.values > 0], self.percentile * 100
        )
        # Set all values to 0 which are below the threshold
        self.trade_matrix[self.trade_matrix < threshold] = 0

        # b_ signifies boolean here, these are filtering masks
        row_sums = self.trade_matrix.sum(axis=1)
        col_sums = self.trade_matrix.sum(axis=0)

        b_filter = ~(row_sums.eq(0) & col_sums.eq(0))
        # Filter out the countries with all zeroes in trade
        self.trade_matrix = self.trade_matrix.loc[b_filter, b_filter]

        print(f"Removed countries with trade below the {int(self.percentile*100)}th percentile.")

        # Save threshold for testing purposes
        self.threshold = threshold

    def prebalance(self, precision=10**-3):
        """
        This implementation also includes pre-balancing to ensure that countries don't
        export more than they produce and import.

        Arguments:
            precision (float, optional): Specifies precision of the prebalancing.

        Returns:
            None
        """
        assert self.prebalanced is False
        self.prebalanced = True

        # this is virtually 1:1 as in Croft et al.
        test = (
            self.production_data
            + self.trade_matrix.sum(axis=0)
            - self.trade_matrix.sum(axis=1)
        )
        while (test <= -precision).any():
            sf = (
                self.production_data + self.trade_matrix.sum(axis=0)
            ) / self.trade_matrix.sum(axis=1)
            multiplier = np.where(test < 0, sf, 1)
            self.trade_matrix = pd.DataFrame(
                np.diag(multiplier) @ self.trade_matrix.values,
                index=self.trade_matrix.index,
                columns=self.trade_matrix.columns,
            )
            test = (
                self.production_data
                + self.trade_matrix.sum(axis=0)
                - self.trade_matrix.sum(axis=1)
            )

    def remove_net_zero_countries(self):
        """
        Return production and trading data with "all zero" countries removed.
        "All zero" countries are such states that has production = 0 and sum of rows
        in trade matrix = 0, and sum of columns = 0.

        Arguments:
            None

        Returns:
            None
        """
        assert self.no_trade_removed is False
        self.no_trade_removed = True

        # b_ signifies boolean here, these are filtering masks
        row_sums = self.trade_matrix.sum(axis=1)
        col_sums = self.trade_matrix.sum(axis=0)

        b_filter = ~(row_sums.eq(0) & col_sums.eq(0) & (self.production_data == 0))

        # Filter out the countries with all zeroes
        self.production_data = self.production_data[b_filter]
        self.trade_matrix = self.trade_matrix.loc[b_filter, b_filter]

    def correct_reexports(self):
        """
        Removes re-exports from the trade matrix.
        This is a Python implementation of the R/Matlab code from:
        Croft, S. A., West, C. D., & Green, J. M. H. (2018).
        "Capturing the heterogeneity of sub-national production
        in global trade flows."

        Journal of Cleaner Production, 203, 1106–1118.

        https://doi.org/10.1016/j.jclepro.2018.08.267


        Input to this function should be prebalanced and have countries with all zeroes
        removed.

        Arguments:
            None

        Returns:
            None
        """
        assert self.prebalanced is True
        assert self.no_trade_removed is True
        assert self.reexports_corrected is False
        self.reexports_corrected = True

        # I know that the variable names here are confusing, but this is a conversion
        # by the original R code from Johanna Hedlung/Croft et al.. The variable names are the
        # same as in the R code and we leave them this way, so we can more easily
        # compare the two pieces of code if something goes wrong.
        self.trade_matrix = self.trade_matrix.T
        self.production_data = self.production_data.fillna(0)

        x = self.production_data + self.trade_matrix.sum(axis=1)
        y = np.linalg.inv(np.diag(x))
        A = self.trade_matrix @ y
        R = np.linalg.inv(np.identity(len(A)) - A) @ np.diag(self.production_data)
        c = np.diag(y @ (x - self.trade_matrix.sum(axis=0)))
        R = (c @ R).T
        R[~np.isfinite(R)] = 0
        # we could consider parametrising this but Croft had this hard coded
        R[R < 0.001] = 0

        self.trade_matrix = pd.DataFrame(
            R, index=self.trade_matrix.index, columns=self.trade_matrix.columns
        )

    def apply_scenario(self):
        """
        Loads the scenario files unifies the names and applies the scenario to the trade matrix.
        by multiplying the trade matrix with the scenario scalar.

        Arguments:
            None

        Returns:
            None
        """
        assert self.scenario_name is not None
        assert self.scenario_file_name is not None
        assert self.scenario_run is False
        self.scenario_run = True

        # Read in the scenario data
        scenario_data = pd.read_csv(
            "."
            + os.sep
            + "data"
            + os.sep
            + "scenario_files"
            + os.sep
            + self.scenario_file_name,
            index_col=0,
        )
        # Drop all NaNs
        scenario_data = scenario_data.dropna()

        cc = coco.CountryConverter()
        # Convert the country names to the same format as in the trade matrix
        scenario_data.index = cc.pandas_convert(pd.Series(scenario_data.index), to="name_short")

        # Only keep the countries that are in the trade matrix index, trade matrix columns and
        # the scenario data
        countries = np.intersect1d(
            np.intersect1d(self.trade_matrix.index, self.trade_matrix.columns),
            scenario_data.index,
        )
        self.trade_matrix = self.trade_matrix.loc[countries, countries]
        scenario_data = scenario_data.loc[countries]

        # Sort the indices
        self.trade_matrix = self.trade_matrix.sort_index(axis=0).sort_index(axis=1)
        scenario_data = scenario_data.sort_index()

        # Make sure the indices + columns are the same
        assert self.trade_matrix.index.equals(self.trade_matrix.columns)
        assert self.trade_matrix.index.equals(scenario_data.index)

        # Multiply all the columns with the scenario data
        self.trade_matrix = self.trade_matrix.mul(scenario_data.values, axis=1)

    def build_graph(self):
        """
        Builds a directed and weighted graph from the trade matrix.

        Arguments:
            None

        Returns:
            None
        """
        assert self.trade_graph is None
        # only build the graph if all the prep is done
        assert self.prebalanced is True
        assert self.reexports_corrected is True
        assert self.no_trade_removed is True
        assert self.threshold is not None

        # Build the graph
        # Initialize a directed graph
        trade_graph = nx.DiGraph()

        # Iterate over the dataframe to add nodes and edges
        for source_country in self.trade_matrix.index:
            for destination_country in self.trade_matrix.columns:
                # Don't add self-loops
                if source_country == destination_country:
                    continue
                # Get the trade amount
                trade_amount = self.trade_matrix.loc[source_country, destination_country]

                # Add nodes if not already in the graph
                trade_graph.add_node(source_country)
                trade_graph.add_node(destination_country)

                # Add edge if trade amount is non-zero
                if trade_amount != 0:
                    trade_graph.add_edge(source_country, destination_country, weight=trade_amount)

        self.trade_graph = trade_graph

    def find_trade_communities(self, keep_singletons=False):
        """
        Finds the trade communities in the trade graph using the Louvain algorithm.

        Arguments:
            None

        Returns:
            None
        """
        assert self.trade_graph is not None
        assert self.trade_communities is None
        # Find the communities
        trade_communities = nx.community.louvain_communities(self.trade_graph, seed=1)
        # Remove all the communities with only one country and print the names of the
        # communities that are removed
        if keep_singletons:
            print("Keeping communities with only one country.")
        else:
            for community in list(trade_communities):
                if len(community) == 1:
                    trade_communities.remove(community)
                    print(f"Removed community {community} with only one country.")

        self.trade_communities = trade_communities

    def plot_trade_communities(self):
        """
        Plots the trade communities in the trade graph on a world map.

        Arguments:
            save (bool, optional): Whether to save the plot or not.

        Returns:
            None
        """
        assert self.trade_communities is not None

        # get the world map
        world = gpd.read_file(
            "." +
            os.sep
            + "data"
            + os.sep
            + "geospatial_references"
            + os.sep
            + "ne_110m_admin_0_countries"
            + os.sep
            + "ne_110m_admin_0_countries.shp"
        )
        world = world.to_crs('+proj=wintri')  # Change projection to Winkel Tripel

        # Create a dictionary with the countries and which community they belong to
        # The communities are numbered from 0 to n
        country_community = {}
        for i, community in enumerate(self.trade_communities):
            for country in community:
                # Convert to standard short names
                country_community[country] = i

        cc = coco.CountryConverter()
        world["names_short"] = cc.pandas_convert(pd.Series(world["ADMIN"]), to="name_short")

        # Join the country_community dictionary to the world dataframe
        world["community"] = world["names_short"].map(country_community)

        # Plot the world map and color the countries according to their community
        cmap = ListedColormap(sns.color_palette("deep", len(self.trade_communities)).as_hex())
        fig, ax = plt.subplots(figsize=(10, 6))
        world.plot(
            ax=ax,
            column="community",
            cmap=cmap,
            missing_kwds={"color": "lightgrey"},
            legend=False,
        )

        plot_winkel_tripel_map(ax)

        # Add a title with self.scenario_name if applicable
        ax.set_title(
            f"Trade communities for {self.crop} with base year {self.base_year[1:]}"
            + (f" in scenario: {self.scenario_name}" if self.scenario_name is not None else "(no scenario)")
        )

        # save the plot
        plt.savefig(
            "."
            + os.sep
            + "results"
            + os.sep
            + "figures"
            + os.sep
            + f"{self.crop}_{self.base_year}_{self.region}_"
            + (f"_{self.scenario_name}" if self.scenario_name is not None else "no_scenario")
            + "_trade_communities.png",
            dpi=300,
            bbox_inches="tight",
        )
