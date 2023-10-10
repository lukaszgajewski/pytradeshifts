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
        percentile (float): The percentile to use for removing countries with
            low trade.
    
    Returns:
        None
    """
    def __init__(self, crop, percentile=0.75):
        self.trade_data = self.load_data(crop, percentile)
        self.trade_data = self.remove_re_exports()
        self.trade_matrix = self.build_trade_matrix()

    def load_data(self, crop, percentile):
        pass

    def remove_countries_with_low_trade(self, trade_data, percentile=0.75):
        pass


