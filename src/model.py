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
        self.trade_data, self.production_data = self.load_data(
            self.crop,
            self.base_year,
            self.percentile
        )
        self.trade_matrix = self.build_trade_matrix()

    def load_data(self, crop, base_year, percentile):
        """
        Loads the data into a pandas dataframe and cleans it
        of countries with trade below a certain percentile.

        Arguments:
            crop (str): The crop to build the trade matrix for.
            base_year (int): The base_year to extract data for.
            percentile (float): The percentile to use for removing countries with
                low trade.

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

        # Remove countries with trade which is below the percentile
        # of the total trade of all countries for this crop
        crop_trade_data = self.remove_above_percentile(crop_trade_data, percentile)

        # Reshape the production data to a vector
        production_data_data.index = production_data_data["Area"]
        production_data_data = production_data_data["Production"]

        return crop_trade_data, production_data_data

    def remove_above_percentile(self, crop_trade_data, percentile):
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
        return crop_trade_data[
                    crop_trade_data["Quantity"] > crop_trade_data["Quantity"].quantile(percentile)
                ]

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

    def remove_re_exports(self):
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
        trade_corrected = self.correct_for_reexports(trade_matrix, production_data)
        # Estimate trade-related matrices
        trade_matrix = self.estimate_trade_matrices(trade_corrected, production_data)

        # Set the corrected trade matrix as the trade matrix
        self.trade_matrix = trade_matrix
    
    def correct_for_reexports(self, trade_matrix, production_data, max_iter=10000):
        """
        Correct for re-exports in the wheat trade data.

        Args:
            trade_matrix (pd.DataFrame): Pivot table of wheat exports data.
            production_data (pd.DataFrame): Crop production data.
            max_iter (int): Maximum number of iterations for prebalancing.

        Returns:
            pd.DataFrame: Corrected wheat trade data.
        """
        # Prebalancing
        loop = 0
        while True:
            # The variable 'trade_imbalance' is an indicator to assess the balance of the trade.
            # It quantifies the discrepancies in trade flows and production values for each country
            # More specifically:
            # - For each country, 'trade_imbalance' computes the difference between total production
            #  and total wheat exports.
            # - It also calculates the difference between the total wheat imports and total
            # wheat exports for the same country.
            # - The 'trade_imbalance' array is constructed by subtracting these two differences,
            # which effectively measures the mismatch between production, exports, and imports.
            # - In the context of prebalancing, the 'while' loop iterates as long as any element
            # in 'trade_imbalance' is less than or equal to -1e-3, indicating an unbalanced trade.
            # - During each iteration, the code adjusts the matrix based on scaling factors until
            # 'trade_imbalance' values are sufficiently close to zero, signifying a balanced trade.
            trade_imbalance = (
                production_data +
                trade_matrix.sum(axis=1) -
                trade_matrix.sum(axis=0)
            )

            if not any(trade_imbalance <= -1e-3) or loop > max_iter:
                break

            # Calculate scaling factors for prebalancing
            scaling_factors = (
                production_data +
                trade_matrix.sum(axis=1)
            ) / trade_matrix.sum(axis=0)

            # Determine the multiplier to adjust the matrix
            multiplier = np.where(trade_imbalance < 0, scaling_factors, 1)

            # Apply the multiplier to the wheat trade matrix
            trade_matrix = np.diag(multiplier) @ trade_matrix

            loop += 1

        return trade_matrix
    
    def estimate_trade_matrices(self, trade_matrix, production_data):
        """
        Estimate trade-related matrices from trade data.

        Args:
            trade_matrix (pd.DataFrame): Trade data.
            production_data (pd.DataFrame): Crop production data.

        Returns:
            pd.DataFrame: Trade without re-exports.
        """
        # Calculate the total production and trade values for balancing
        domestic_supply = (
            production_data +
            trade_matrix.sum(axis=0)
        )

        # Create a diagonal matrix using crop production values
        production_diag = np.diag(production_data)

        # Calculate matrix A representing trade relationships
        trade_relationships = trade_matrix @ np.linalg.inv(np.diag(domestic_supply))

        # Calculate trade flows
        trade_flows = np.linalg.inv(
            np.identity(trade_relationships.shape[0]) -
            trade_relationships
        ) @ production_diag

        # Calculate consumption shares
        consumption_share = np.linalg.inv(
            np.diag(domestic_supply)
        ) @ (domestic_supply - trade_matrix.sum(axis=1))

        # Calculate consumption from each country to country
        trade_without_re_exports = np.diag(consumption_share) @ trade_flows

        # Data cleaning
        trade_without_re_exports[np.isnan(trade_without_re_exports)] = 0
        trade_without_re_exports[trade_without_re_exports < 0.001] = 0

        return trade_without_re_exports
