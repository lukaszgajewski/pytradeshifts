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

        print(f"Loaded data for {crop} in {base_year}.")

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

        print(f"Removed countries with trade below the {percentile*100}th percentile.")

        return trade_matrix

    def build_trade_matrix(self):
        """
        Builds the trade matrix for the given crop and base_year.

        Arguments:
            None

        Returns:
            pd.DataFrame: The trade matrix.
        """
        # Pivot the DataFrame to get a matrix representation
        trade_matrix = self.trade_data.pivot_table(
            index='Reporter Countries', columns='Partner Countries', values='Quantity', fill_value=0
        )

        # Ensure the trade_matrix is quadratic
        trade_matrix = trade_matrix.add(trade_matrix.transpose(), fill_value=0)
        trade_matrix = trade_matrix.reindex(index=trade_matrix.columns, columns=trade_matrix.index, fill_value=0)

        # Replace nan values with 0
        trade_matrix = trade_matrix.fillna(0)
        print("Finished building the trade matrix.")

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

        In addition to the original Matlab code, this implementation
        also includes pre-balancing to ensure that countries don't
        export more than they produce and import.

        Arguments:
            None

        Returns:
            pd.DataFrame: The trade matrix without re-exports.
        """
        # Retain only the countries where we have production data and trade data
        countries = np.intersect1d(self.trade_matrix.index, self.production_data.index)
        trade_matrix = self.trade_matrix.loc[countries, countries]
        production_data = self.production_data.loc[countries]
        # Make sure this worked
        assert trade_matrix.shape[0] == production_data.shape[0]

        print("Ignoring countries without production data.")
 
        trade_matrix = self.prebalance(production_data, trade_matrix)
        production_data, trade_matrix = self.remove_net_zero_countries(
            production_data,
            trade_matrix
        )
        trade_matrix = self.correct_reexports(production_data, trade_matrix)

        # Remove countries with trade which is below the percentile
        # of the total trade of all countries for this crop
        trade_matrix = self.remove_above_percentile(trade_matrix, self.percentile)
        self.trade_matrix = trade_matrix
        
    def prebalance(self, production_data, trade_matrix, precision=10**-3):
        test = production_data + trade_matrix.sum(axis=0) - trade_matrix.sum(axis=1)
        while (test <= -precision).any():
            sf = (production_data + trade_matrix.sum(axis=0)) / trade_matrix.sum(axis=1)
            multiplier = ((test < 0) * sf).replace(0, 1)
            trade_matrix = pd.DataFrame(
                np.diag(multiplier) @ trade_matrix.values,
                index=trade_matrix.index,
                columns=trade_matrix.columns,
            )
            test = production_data + trade_matrix.sum(axis=0) - trade_matrix.sum(axis=1)
        return trade_matrix

    def remove_net_zero_countries(self, production_data, trade_matrix):
        b_zero_prod = production_data == 0
        b_zero_colsum = trade_matrix.sum(axis=0) == 0
        b_zero_rowsum = trade_matrix.sum(axis=1) == 0
        b_filter = ~(b_zero_prod & b_zero_rowsum & b_zero_colsum)
        return production_data[b_filter], trade_matrix.loc[b_filter, b_filter]

    def correct_reexports(self, production_data, trade_matrix):
        """
        I know that the variable names here are confusing, but this is a conversion
        by the original R code from Johanna Hedlung. The variable names are the
        same as in the R code and we leave them this way, so we can more easily
        compare the two pieces of code if something goes wrong.
        """
        trade_matrix = trade_matrix.T
        production_data = production_data.fillna(0)

        x = production_data + trade_matrix.sum(axis=1)
        y = np.linalg.inv(np.diag(x))
        A = trade_matrix @ y
        R = np.linalg.inv(np.identity(len(A)) - A) @ np.diag(production_data)
        c = np.diag(y @ (x - trade_matrix.sum(axis=0)))
        R = (c @ R).T
        R[~np.isfinite(R)] = 0
        R[R < 0.001] = 0

        return pd.DataFrame(R, index=trade_matrix.index, columns=trade_matrix.columns)

