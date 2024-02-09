import matplotlib.pyplot as plt
from src.model import PyTradeShifts
from src.utils import all_equal
import numpy as np
import geopandas as gpd
import os
import country_converter as coco
from matplotlib.colors import ListedColormap
import seaborn as sns
import pandas as pd


plt.style.use(
    "https://raw.githubusercontent.com/allfed/ALLFED-matplotlib-style-sheet/main/ALLFED.mplstyle"
)


class Postprocessing:
    """
    This class is used to postprocess the results of the scenarios. This should work
    for an arbitrary number of scenarios. It should be possible to compare the scenarios
    and generate a report of the results.

    Arguments:
        scenarios (list): List of scenarios to compare. Each scenario should be an instance
        of PyTradeShifts.

    Returns:
        None
    """

    def __init__(self, scenarios: list[PyTradeShifts]):
        self.scenarios = scenarios
        # check if community detection is uniform for all objects
        # there might be a case where it is desired so we allow it
        # but most times this is going to be undesirable hence the warning
        if not all_equal((sc.cd_algorithm for sc in scenarios)):
            print("Warning: Inconsistent community detection algorithms detected.")
        if not all_equal((sc.cd_kwargs for sc in scenarios)):
            print("Warning: Inconsistent community detection parameters detected.")
        self._calculate_stuff()

    def _calculate_stuff(self):
        """
        Hidden method to calculate stuff, like comparing scenarios

        """
        pass

    def plot(self):
        """
        Plots the results of the scenarios. This could be something like comparing on the
        world map where the scenarios differ or a visual comparison of the stability of
        the graphs of the scenarions.

        Not sure if this needs to be a method or could also just be in report.
        """
        _, axs = plt.subplots(len(self.scenarios), 1, sharex=True, tight_layout=True)
        for ax, sc in zip(axs, self.scenarios):
            sc._plot_trade_communities(ax)
        plt.show()

    def report(self):
        """
        This method generates a report of the results of the scenarios. This could be a
        pdf or a markdown file. This should contain the results of the scenarios and
        the comparison of the scenarios, like with plots or tables.
        """

        pass
