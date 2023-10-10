import pandas as pd
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
        self.trade_data = self.load_data(self.crop, self.base_year, self.percentile)
        self.trade_matrix = self.build_trade_matrix()
       # self.trade_matrix = self.remove_re_exports()

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

        # Only use crop of interest
        crop_trade_data = trade_data[trade_data["Item"] == crop]

        # Remove countries with trade which is below the percentile
        # of the total trade of all countries for this crop
        crop_trade_data = crop_trade_data[
            crop_trade_data["Quantity"] > crop_trade_data["Quantity"].quantile(percentile)
        ]
        return crop_trade_data


    def build_trade_matrix(self):
        """
        Builds the trade matrix for the given crop and base_year.

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
        return trade_matrix
