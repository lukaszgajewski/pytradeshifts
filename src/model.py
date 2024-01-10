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
    def __init__(self, crop, base_year, percentile=0.75, region="Global", testing=False):
        self.base_year = base_year
        self.crop = crop
        self.percentile = percentile
        self.region = region

        # Don't run the methods if we are testing, so we can test them individually
        if not testing:
            # Read in the data
            self.load_data()
            # Prebalance the trade matrix
            self.prebalance(self.production_data, self.trade_matrix)

    def load_data(
            self,
    ):
        """
        Loads the data into a pandas dataframe and cleans it
        of countries with trade below a certain percentile.

        Arguments:
            None

        Returns:
            None
        """
        # Read in the data
        trade_matrix = pd.read_csv(
            "." + os.sep +
            "data" + os.sep +
            "preprocessed_data" + os.sep +
            f"{self.crop}_{self.base_year}_{self.region}_trade.csv",
            index_col=0,
        )

        production_data = pd.read_csv(
            "." + os.sep +
            "data" + os.sep +
            "preprocessed_data" + os.sep +
            f"{self.crop}_{self.base_year}_{self.region}_production.csv",
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


    def remove_above_percentile(
            self,
            trade_matrix: pd.DataFrame,
            percentile: float = 0.75
    ) -> pd.DataFrame:
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


    def prebalance(
            self,
            precision=10**-3
    ):
        """
        This implementation also includes pre-balancing to ensure that countries don't
        export more than they produce and import.

        Arguments:
            precision (float, optional): Specifies precision of the prebalancing.

        Returns:
            None
        """
        # Make a local copy of the trade matrix and production data
        trade_matrix = self.trade_matrix.copy()
        production_data = self.production_data.copy()

        # this is virtually 1:1 as in Croft et al.
        test = production_data + trade_matrix.sum(axis=0) - trade_matrix.sum(axis=1)
        while (test <= -precision).any():
            sf = (production_data + trade_matrix.sum(axis=0)) / trade_matrix.sum(
                axis=1
            )
            multiplier = ((test < 0) * sf).replace(0, 1)
            trade_matrix = pd.DataFrame(
                np.diag(multiplier) @ trade_matrix.values,
                index=trade_matrix.index,
                columns=trade_matrix.columns,
            )
            test = production_data + trade_matrix.sum(axis=0) - trade_matrix.sum(
                axis=1
            )

    def remove_net_zero_countries(
            self,
    ):
        """
        Return production and trading data with "all zero" countries removed.
        "All zero" countries are such states that has production = 0 and sum of rows
        in trade matrix = 0, and sum of columns = 0.

        Arguments:
            None

        Returns:
            None
        """
        # Make a local copy of the trade matrix and production data
        trade_matrix = self.trade_matrix.copy()
        production_data = self.production_data.copy()

        # b_ signifies boolean here, these are filtering masks
        b_zero_prod = production_data == 0
        b_zero_colsum = trade_matrix.sum(axis=0) == 0
        b_zero_rowsum = trade_matrix.sum(axis=1) == 0
        b_filter = ~(b_zero_prod & b_zero_rowsum & b_zero_colsum)

        # Filter out the countries with all zeroes
        self.production_data = production_data[b_filter]
        self.trade_matrix = trade_matrix.loc[b_filter, b_filter]

    def correct_reexports(
            self,
    ):
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
        # Make a local copy of the trade matrix and production data
        trade_matrix = self.trade_matrix.copy()
        production_data = self.production_data.copy()

        # I know that the variable names here are confusing, but this is a conversion
        # by the original R code from Johanna Hedlung/Croft et al.. The variable names are the
        # same as in the R code and we leave them this way, so we can more easily
        # compare the two pieces of code if something goes wrong.
        trade_matrix = trade_matrix.T
        production_data = production_data.fillna(0)

        x = production_data + trade_matrix.sum(axis=1)
        y = np.linalg.inv(np.diag(x))
        A = trade_matrix @ y
        R = np.linalg.inv(np.identity(len(A)) - A) @ np.diag(production_data)
        c = np.diag(y @ (x - trade_matrix.sum(axis=0)))
        R = (c @ R).T
        R[~np.isfinite(R)] = 0
        # we could consider parametrising this but Croft had this hard coded
        R[R < 0.001] = 0

        self.trade_matrix = pd.DataFrame(R, index=trade_matrix.index, columns=trade_matrix.columns)
