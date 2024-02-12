from geopandas import base
import matplotlib.pyplot as plt
from src.model import PyTradeShifts
from src.utils import all_equal, jaccard_index, plot_winkel_tripel_map
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
        anchor_countries: list[str] = [],
    ):
        self.scenarios = scenarios
        # we could make this user-specified but it's going to make the interface
        # and the code more complicated, let's just inform in the docs
        # that the first passed scenario is considered the base
        self.base_scenario = 0
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
        return list(itemgetter(*new_order)(scenario.trade_communities))

    def arrange_communities(self) -> None:
        """
        TODO: order communities based on anchors such that the colour differences
        between plots don't seem random like they do now.
        """
        for scenario in self.scenarios:
            scenario.trade_communities = self._find_new_order(scenario)

    def _find_community_diff(self) -> dict[int, dict[str, list[set[str]]]]:
        jaccard_indices = {
            scenario_idx: {}
            for scenario_idx, _ in enumerate(self.scenarios)
            if scenario_idx != self.base_scenario
        }
        base_scenario_country_list = self.scenarios[
            self.base_scenario
        ].production_data.index
        for scenario_idx, scenario in enumerate(self.scenarios):
            if scenario_idx == self.base_scenario:
                continue
            for country in base_scenario_country_list:
                new_community = None
                for comm in scenario.trade_communities:
                    if country in comm:
                        new_community = comm
                        break
                if new_community is None:
                    continue
                old_community = None
                for comm in self.scenarios[self.base_scenario].trade_communities:
                    if country in comm:
                        old_community = comm
                        break
                jaccard_indices[scenario_idx][country] = jaccard_index(
                    new_community, old_community
                )
        return jaccard_indices

    def _plot_jaccard_map(self, ax, scenario, jaccard) -> None:
        """
        TODO: move to util
        """
        assert scenario.trade_communities is not None
        # get the world map
        world = gpd.read_file(
            "."
            + os.sep
            + "data"
            + os.sep
            + "geospatial_references"
            + os.sep
            + "ne_110m_admin_0_countries"
            + os.sep
            + "ne_110m_admin_0_countries.shp"
        )
        world = world.to_crs("+proj=wintri")  # Change projection to Winkel Tripel

        cc = coco.CountryConverter()
        world["names_short"] = cc.pandas_convert(
            pd.Series(world["ADMIN"]), to="name_short"
        )

        # Join the country_community dictionary to the world dataframe
        world["jaccard_index"] = world["names_short"].map(jaccard)
        world["jaccard_distance"] = 1 - world["jaccard_index"]

        world.plot(
            ax=ax,
            column="jaccard_distance",
            missing_kwds={"color": "lightgrey"},
            legend=True,
            # TODO: shrink doesn't work as well for more than two scenarios
            legend_kwds={"shrink": 0.35, "label": "Jaccard distance"},
        )

        plot_winkel_tripel_map(ax)

        # Add a title with self.scenario_name if applicable
        ax.set_title(
            f"Difference vs. base scenario for {scenario.crop} with base year {scenario.base_year[1:]}"
            + (
                f" in scenario: {scenario.scenario_name}"
                if scenario.scenario_name is not None
                else " (no scenario)"
            )
        )

    def plot_community_diff(self):
        """
        TODO
        """
        jaccard_indices = self._find_community_diff()
        _, axs = plt.subplots(
            len(self.scenarios) - 1, 1, sharex=True, tight_layout=True
        )
        # if there are only two scenarios axs will be just an ax object
        # convert to a list to comply with other cases
        try:
            len(axs)
        except TypeError:
            axs = [axs]
        non_base_scenarios = [
            (idx, sc)
            for idx, sc in enumerate(self.scenarios)
            if idx != self.base_scenario
        ]

        for ax, (idx, sc) in zip(axs, non_base_scenarios):
            self._plot_jaccard_map(ax, sc, jaccard_indices[idx])
        plt.show()

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
