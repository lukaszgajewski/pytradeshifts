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
        pass


    def prebalance(self, vec_prod, mat_trad, precision=10**-3):
        test = vec_prod + mat_trad.sum(axis=0) - mat_trad.sum(axis=1)
        while (test <= -precision).any():
            sf = (vec_prod + mat_trad.sum(axis=0)) / mat_trad.sum(axis=1)
            multiplier = ((test < 0) * sf).replace(0, 1)
            mat_trad = pd.DataFrame(
                np.diag(multiplier) @ mat_trad.values,
                index=mat_trad.index,
                columns=mat_trad.columns,
            )
            test = vec_prod + mat_trad.sum(axis=0) - mat_trad.sum(axis=1)
        return mat_trad


    def remove_net_zero_countries(self, vec_prod, mat_trad):
        b_zero_prod = vec_prod == 0
        b_zero_colsum = mat_trad.sum(axis=0) == 0
        b_zero_rowsum = mat_trad.sum(axis=1) == 0
        b_filter = ~(b_zero_prod & b_zero_rowsum & b_zero_colsum)
        return vec_prod[b_filter], mat_trad.loc[b_filter, b_filter]


    def correct_reexports(self, vec_prod, mat_trad):
        mat_trad = mat_trad.T
        vec_prod = vec_prod.fillna(0)

        x = vec_prod + mat_trad.sum(axis=1)
        y = np.linalg.inv(np.diag(x))
        A = mat_trad @ y
        R = np.linalg.inv(np.identity(len(A)) - A) @ np.diag(vec_prod)
        c = np.diag(y @ (x - mat_trad.sum(axis=0)))
        R = (c @ R).T
        R[~np.isfinite(R)] = 0
        R[R < 0.001] = 0

        return pd.DataFrame(R, index=mat_trad.index, columns=mat_trad.columns)

