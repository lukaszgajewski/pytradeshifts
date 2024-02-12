from geopandas import base
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
from operator import itemgetter


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
        base_scenario (int): index of a scenario to plot a difference map against.
        Accepts values in range: [1, number of scenarios].

    Returns:
        None
    """

    def __init__(
        self,
        scenarios: list[PyTradeShifts],
        base_scenario=1,
        anchor_countries: list[str] = [],
    ):
        self.scenarios = scenarios
        self.base_scenario = base_scenario - 1  # 0-index the list, 1-index for UX
        self.anchor_countries = anchor_countries
        # check if community detection is uniform for all objects
        # there might be a case where it is desired so we allow it
        # but most times this is going to be undesirable hence the warning
        if not all_equal((sc.cd_algorithm for sc in scenarios)):
            print("Warning: Inconsistent community detection algorithms detected.")
        if not all_equal((sc.cd_kwargs for sc in scenarios)):
            print("Warning: Inconsistent community detection parameters detected.")
        if anchor_countries:
            self.arrange_communities()

    def _calculate_stuff(self):
        """
        Hidden method to calculate stuff, like comparing scenarios

        """
        pass

    def _find_new_order(self, scenario) -> list[set[str]]:
        # make sure there are communities computed
        assert scenario.trade_communities is not None
        # find location of anchor countries in communities' list
        anchor_idx = {}
        for anchor in self.anchor_countries:
            for idx, community in enumerate(scenario.trade_communities):
                if anchor in community:
                    anchor_idx[anchor] = idx
                    break
        # make sure indices are unqiue, they wouldn't be if user passed
        # two or more countries from the same community as anchors
        # TODO: handle this gracefully
        assert len(anchor_idx.values()) == len(set(anchor_idx.values()))
        # create new arrangement
        new_order = list(anchor_idx.values())
        # append remaining (i.e., un-anchored) community indices
        new_order.extend(
            set(range(len(scenario.trade_communities))).difference(new_order)
        )
        # make sure we've got that right
        assert len(new_order) == len(scenario.trade_communities)
        assert len(new_order) == len(set(new_order))
        # get communities in the new order
        print(new_order)
        return list(itemgetter(*new_order)(scenario.trade_communities))

    def arrange_communities(self) -> None:
        """
        TODO: order communities based on anchors such that the colour differences
        between plots don't seem random like they do now.
        """
        for scenario in self.scenarios:
            scenario.trade_communities = self._find_new_order(scenario)

    def community_diff(self):
        """
        TODO: plot a map showing countries that changed communities.
        Problem: how to actually detect that? based on anchors? Or by jaccard index?
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
