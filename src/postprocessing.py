import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from src.model import PyTradeShifts
from src.utils import (
    all_equal,
    jaccard_index,
    plot_jaccard_map,
    get_right_stochastic_matrix,
    get_stationary_probability_vector,
    get_entropy_rate,
    get_dict_min_max,
    plot_degree_map,
)
import numpy as np
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
        frobenius: str | None = "relative",
    ):
        self.scenarios = scenarios
        # we could make this user-specified but it's going to make the interface
        # and the code more complicated, let's just inform in the docs
        # that the first passed scenario is considered the base
        self.base_scenario = 0  # TODO refactor this
        self.anchor_countries = anchor_countries
        self.frobenius_in_plot = frobenius
        # check if community detection is uniform for all objects
        # there might be a case where it is desired so we allow it
        # but most times this is going to be undesirable hence the warning
        if not all_equal((sc.cd_algorithm for sc in scenarios)):
            print("Warning: Inconsistent community detection algorithms detected.")
        if not all_equal((sc.cd_kwargs for sc in scenarios)):
            print("Warning: Inconsistent community detection parameters detected.")

        # in order to compute matrix distances we need matrices to be
        # of the same shape, this is a helper member variable that allows that
        self.elligible_countries = [
            set(scenario.trade_graph.nodes()).intersection(
                self.scenarios[0].trade_graph.nodes()
            )
            for scenario in self.scenarios[1:]
        ]

        if anchor_countries:
            self.arrange_communities()
        self._compute_frobenius_distance()
        self._compute_entropy_rate_distance()
        self._compute_stationary_markov_distance()
        self._format_distance_dataframe()
        self._compute_centrality_measures()

    def _compute_frobenius_distance(self) -> None:
        """
        TODO
        """
        self.frobenius = [
            np.linalg.norm(
                nx.to_numpy_array(
                    self.scenarios[0].trade_graph, nodelist=elligible_nodes
                )
                - nx.to_numpy_array(scenario.trade_graph, nodelist=elligible_nodes)
            )
            for scenario, elligible_nodes in zip(
                self.scenarios[1:], self.elligible_countries
            )
        ]

    def _compute_stationary_markov_distance(self) -> None:
        """
        TODO
        """
        graphs_with_elligible_nodes = [
            (
                nx.subgraph(self.scenarios[0].trade_graph, nbunch=elligible_nodes),
                nx.subgraph(scenario.trade_graph, nbunch=elligible_nodes),
            )
            for scenario, elligible_nodes in zip(
                self.scenarios[1:], self.elligible_countries
            )
        ]
        vecs = [
            [
                get_stationary_probability_vector(
                    get_right_stochastic_matrix(trade_matrix)
                )
                for trade_matrix in trade_matrix_pair
            ]
            for trade_matrix_pair in graphs_with_elligible_nodes
        ]
        self.markov = [np.linalg.norm(base_vec - vec) for (base_vec, vec) in vecs]

    def _compute_entropy_rate_distance(self) -> None:
        """
        TODO
        """
        entropy_rates = [get_entropy_rate(scenario) for scenario in self.scenarios]
        # compute difference from base scenario
        self.entropy_rate = [er - entropy_rates[0] for er in entropy_rates[1:]]

    def _format_distance_dataframe(self) -> None:
        df = pd.DataFrame(
            zip(
                range(1, len(self.scenarios)),
                self.frobenius,
                self.markov,
                self.entropy_rate,
            ),
            columns=["Scenario ID", "Frobenius", "Markov", "Entropy rate"],
        )
        self.distance_df = df

    def print_distance_metrics(self) -> None:
        """
        TODO
        """
        df = self.distance_df.copy()
        df.set_index("Scenario ID", drop=True, inplace=True)
        print(
            "***| Distance metrics vs. the base scenario (ID=0 on the list of scenarios) |***"
        )
        print(df.to_markdown(tablefmt="fancy_grid"))

    def plot_distance_metrics(self) -> None:
        """
        TODO
        """
        df = self.distance_df.copy()
        match self.frobenius_in_plot:
            case "relative":
                df[df.columns[1:]] = df[df.columns[1:]] / df[df.columns[1:]].max()
            case "ignore":
                df.drop(columns=["Frobenius"], inplace=True)
            case None:
                pass
            case _:
                print(
                    "Unrecognised option for plotting Frobenius, defaulting to 'relative'."
                )
                df[df.columns[1:]] = df[df.columns[1:]] / df[df.columns[1:]].max()
        df = df.melt(id_vars="Scenario ID", value_vars=df.columns[1:])
        sns.barplot(
            df,
            x="Scenario ID",
            y="value",
            hue="variable",
        )
        plt.ylabel("Distance")
        plt.title("Distance metrics vs. the base scenario")
        plt.show()

    def _compute_centrality_measures(self) -> None:
        """
        TODO
        """
        self.in_degree = [
            nx.in_degree_centrality(scenario.trade_graph) for scenario in self.scenarios
        ]
        self.out_degree = [
            nx.out_degree_centrality(scenario.trade_graph)
            for scenario in self.scenarios
        ]

    def plot_degree_maps(self) -> None:
        """
        TODO
        """
        _, axs = plt.subplots(
            2 * len(self.scenarios), 1, sharex=True, tight_layout=True
        )
        # if there is only one scenario axs will be just an ax object
        # convert to a list to comply with other cases
        try:
            len(axs)
        except TypeError:
            axs = [axs]

        idx = 0
        for scenario, in_degree, out_degree in zip(
            self.scenarios, self.in_degree, self.out_degree
        ):
            plot_degree_map(axs[idx], scenario, in_degree, label="in-degree")
            plot_degree_map(axs[idx + 1], scenario, out_degree, label="out-degree")
            idx += 2
        plt.show()

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
        TODO
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
        TODO
        This method generates a report of the results of the scenarios. This could be a
        pdf or a markdown file. This should contain the results of the scenarios and
        the comparison of the scenarios, like with plots or tables.
        """

        pass
