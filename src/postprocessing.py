from networkx import (
    to_numpy_array as nx_to_numpy_array,
    to_pandas_adjacency as nx_to_pandas_adjacency,
)
import matplotlib.pyplot as plt
from src.model import PyTradeShifts
from src.utils import all_equal, jaccard_index, plot_jaccard_map
import numpy as np
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
        self.base_scenario = 0  # TODO refactor this
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
        self._compute_frobenius_distance()
        self._compute_entropy_rate()

    def _compute_frobenius_distance(self) -> None:
        """
        TODO
        """
        elligible_countries = [
            set(scenario.trade_graph.nodes()).intersection(
                self.scenarios[0].trade_graph.nodes()
            )
            for scenario in self.scenarios[1:]
        ]
        self.frobenius = [
            np.linalg.norm(
                nx_to_numpy_array(
                    self.scenarios[0].trade_graph, nodelist=elligible_nodes
                )
                - nx_to_numpy_array(scenario.trade_graph, nodelist=elligible_nodes)
            )
            for scenario, elligible_nodes in zip(
                self.scenarios[1:], elligible_countries
            )
        ]

    def _compute_entropy_rate(self) -> None:
        """
        TODO: this is yanked from an old project and is likely to be incorrect
        """
        for scenario in self.scenarios:
            right_stochastic_matrix = nx_to_pandas_adjacency(scenario.trade_graph)
            right_stochastic_matrix = right_stochastic_matrix.div(
                right_stochastic_matrix.sum(axis=0)
            )
            right_stochastic_matrix.fillna(0, inplace=True)
            x = right_stochastic_matrix.values
            y = np.linalg.eig(x.T)
            _, z = min(zip(y[0], y[1].T), key=lambda v: abs(v[0] - 1.0))
            z /= np.sum(z)
            # equivalent to -np.sum(z*np.sum(np.nan_to_num(x * np.log(x)), axis=0))
            entropy_rate = -np.sum(x * z * np.nan_to_num(np.log(x)))
            print(entropy_rate)

        # transition_matrix.div(transition_matrix.sum(axis="columns"), axis="rows")

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
                        new_community = comm - {country}
                        break
                if new_community is None:
                    continue
                old_community = None
                for comm in self.scenarios[self.base_scenario].trade_communities:
                    if country in comm:
                        old_community = comm - {country}
                        break
                jaccard_indices[scenario_idx][country] = jaccard_index(
                    new_community, old_community
                )
        return jaccard_indices

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
            plot_jaccard_map(ax, sc, jaccard_indices[idx])
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
