import pandas as pd
import numpy as np
import os


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

    Returns:
        None
    """
    def __init__(self, crop, base_year, percentile=0.75):
        self.base_year = base_year
        self.crop = crop
        self.percentile = percentile
        # Read in the data
        self.trade_data, self.production_data = self.load_data(
            self.crop,
            self.base_year,
        )
        self.trade_matrix = self.build_trade_matrix()


    def load_data(self, crop, base_year):
        """
        Loads the data into a pandas dataframe and cleans it
        of countries with trade below a certain percentile.

        Arguments:
            crop (str): The crop to build the trade matrix for.
            base_year (int): The base_year to extract data for.

        Returns:
            pd.DataFrame: The trade data with countries with low trade removed
                and only the relevant crop.
        """
        # Read in the data
        trade_data = pd.read_csv(
            "." + os.sep +
            "data" + os.sep +
            f"trade_data_only_relevant_{base_year}.csv"
        )

        production_data = pd.read_csv(
            "." + os.sep +
            "data" + os.sep +
            f"production_data_only_relevant_{base_year}.csv"
        )

        # Only use crop of interest
        crop_trade_data = trade_data[trade_data["Item"] == crop]
        production_data_data = production_data[production_data["Item"] == crop]

        # Reshape the production data to a vector
        production_data_data.index = production_data_data["Area"]
        production_data_data = production_data_data["Production"]

        return crop_trade_data, production_data_data

    def remove_above_percentile(self, trade_matrix, percentile):
        """
        Removes countries with trade below a certain percentile.

        Arguments:
            crop_trade_data (pd.DataFrame): The trade data with countries with low trade removed
                and only the relevant crop.
            percentile (float): The percentile to use for removing countries with
                low trade.

        Returns:
            pd.DataFrame: The trade data with countries with low trade removed
                and only the relevant crop.
        """
        # Calculate the percentile out of all values in the trade matrix
        threshold = np.percentile(trade_matrix.values, percentile*100)
        # Set all values to 0 which are below the threshold
        trade_matrix[trade_matrix < threshold] = 0
        # Remove all countries which have no trade
        trade_matrix = trade_matrix.loc[trade_matrix.sum(axis=1) > 0, :]
        trade_matrix = trade_matrix.loc[:, trade_matrix.sum(axis=0) > 0]

        return trade_matrix

    def build_trade_matrix(self):
        """
        Builds the trade matrix for the given crop and base_year.

        Arguments:
            None

        Returns:
            pd.DataFrame: The trade matrix.
        """
        # Build the trade matrix
        trade_matrix = self.trade_data.pivot_table(
            index="Reporter Countries",
            columns="Partner Countries",
            values="Quantity",
            aggfunc="sum",
            fill_value=0
        )

        # Make it a quadratic matrix
        index = trade_matrix.index.union(trade_matrix.columns)
        trade_matrix = trade_matrix.reindex(index=index, columns=index, fill_value=0)

        return trade_matrix

    def remove_re_exports(self, percentile=0.75):
        """
        Removes re-exports from the trade matrix.
        This is a Python implementation of the Matlab code from:
        Croft, S. A., West, C. D., & Green, J. M. H. (2018).
        "Capturing the heterogeneity of sub-national production
        in global trade flows."

        Journal of Cleaner Production, 203, 1106–1118.

        https://doi.org/10.1016/j.jclepro.2018.08.267

        Arguments:
            None

        Returns:
            pd.DataFrame: The trade matrix without re-exports.
        """
        # Retain only the countries where we have production data and trade data
        countries = np.intersect1d(self.trade_matrix.index, self.production_data.index)
        trade_matrix = self.trade_matrix.loc[countries, countries]
        production_data = self.production_data.loc[countries]

        # Remove re-exports
        # Convert the trade matrix to a numpy array
        trade_matrix = trade_matrix.to_numpy()
        # Convert the production data to a numpy array
        production_data = production_data.to_numpy()
        # Balance the trade data
        trade_matrix_balanced = self.balance_trade_data(trade_matrix, production_data)
        self.trade_matrix = pd.DataFrame(trade_matrix_balanced, index=countries, columns=countries)

        # Remove countries with trade which is below the percentile
        # of the total trade of all countries for this crop
        self.trade_matrix = self.remove_above_percentile(self.trade_matrix, percentile)

    def balance_trade_data(self, trade_matrix, production_data, tolerance=1e-3, max_iterations=10000):
        """
        Balance the trade data using an iterative approach.

        Arguments:
            trade_matrix (numpy.ndarray): The bilateral trade matrix.
            production_data (numpy.ndarray): A vector of production values for each country.
            tolerance (float, optional): A tolerance threshold for trade imbalances. Defaults to 0.001.
            max_iterations (int, optional): The maximum number of iterations. Defaults to 10000.

        Returns:
            numpy.ndarray: The balanced bilateral trade matrix.
        """

        iteration = 0  # Initialize the iteration counter

        while True:
            # Print every 1000 iterations
            if iteration % 1000 == 0:
                print(f"Iteration {iteration}")
            # Calculate row sums of the bilateral trade matrix
            row_sums = trade_matrix.sum(axis=1)

            # Calculate column sums of the bilateral trade matrix
            col_sums = trade_matrix.sum(axis=0)

            # Calculate the trade imbalances
            trade_imbalances = production_data + col_sums - row_sums

            # Check if all trade imbalances are below the tolerance
            if iteration > max_iterations:
                if all(trade_imbalances <= -tolerance):
                    # Calculate scaling factors to balance the trade
                    scaling_factors = (production_data + col_sums) / row_sums
                    # Replace negative scaling factors with 1
                    scaling_factors[scaling_factors < 0] = 1
                    # Update the bilateral trade matrix with scaled values
                    trade_matrix = trade_matrix.mul(scaling_factors, axis=0)
                else:
                    break

            iteration += 1  # Increment the iteration counter

        return trade_matrix
