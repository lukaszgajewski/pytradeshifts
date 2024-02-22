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
    This class is used to postprocess the results of the scenarios.
    It works for an arbitrary number of scenarios.
    It compares the scenarios and generates a report of the results.

    Arguments:
        scenarios (list): List of scenarios to compare. Each scenario should be an instance
            of PyTradeShifts. The first scenario in the list is considered the
            'base scenario' and will be used as such for all comparison computations.
        anchor_countries (list, optional): List of country names to be used as anchores in plotting.
            The expected behaviour here is that the specified countries will retain
            their colour across all community plots.
        testing (bool): Whether to run the methods or not. This is only used for
            testing purposes.

    Returns:
        None
    """

    def __init__(
        self,
        scenarios: list[PyTradeShifts],
        anchor_countries: list[str] = [],
        testing=False,
    ):
        self.scenarios = scenarios
        # we could make this user-specified but it's going to make the interface
        # and the code more complicated, let's just inform in the docs
        # that the first passed scenario is considered the base
        self.anchor_countries = anchor_countries
        # check if community detection is uniform for all objects
        # there might be a case where it is desired so we allow it
        # but most times this is going to be undesirable hence the warning
        if not all_equal((sc.cd_algorithm for sc in scenarios)):
            print("Warning: Inconsistent community detection algorithms detected.")
        if not all_equal((sc.cd_kwargs for sc in scenarios)):
            print("Warning: Inconsistent community detection parameters detected.")

        if not testing:
            self.run()

    def run(self) -> None:
        """
        Executes all computation required for reporting.

        Arguments:
            None

        Returns:
            None
        """
        # in order to compute matrix distances we need matrices to be
        # of the same shape, this is a helper member variable that allows that
        self.elligible_countries = [
            set(scenario.trade_graph.nodes()).intersection(
                self.scenarios[0].trade_graph.nodes()
            )
            for scenario in self.scenarios[1:]
        ]
        if self.anchor_countries:
            self._arrange_communities()
        self._find_community_difference()
        self._compute_frobenius_distance()
        self._compute_entropy_rate_distance()
        self._compute_stationary_markov_distance()
        self._format_distance_dataframe()
        self._compute_centrality()
        self._compute_global_centrality_metrics()
        self._compute_community_centrality_metrics()

    def _compute_frobenius_distance(self) -> None:
        """
        Computes the Frobenius distance between the base scenario graph
        and each other scenario as the Frobenius norm of A-A',
        where A is the adjacency matrix of the base scenario, and A' of some
        other scenario.
        https://mathworld.wolfram.com/FrobeniusNorm.html

        As the matrices amongst scenarios might not match in shape,
        we conform all adjacency matrices to the common nodes of the base scenario
        and the other scenario being considered.

        Arguments:
            None

        Returns:
            None
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
        Computes the 'Markov' distance between the base scenario graph
        and each other scenario.
        This is the Eucledean distance between stationary probability distribution
        vectors assuming a Markov random walk on the trade graphs.

        As the graph nodes amongst scenarios might not match, we conform all
        adjacency matrices to the common nodes of the base scenario
        and the other scenario being considered.

        Arguments:
            None

        Returns:
            None
        """
        subgraphs_with_elligible_nodes = [
            (
                nx.subgraph(self.scenarios[0].trade_graph, nbunch=elligible_nodes),
                nx.subgraph(scenario.trade_graph, nbunch=elligible_nodes),
            )
            for scenario, elligible_nodes in zip(
                self.scenarios[1:], self.elligible_countries
            )
        ]
        stationary_distribution_vectors = [
            [
                get_stationary_probability_vector(
                    get_right_stochastic_matrix(trade_matrix)
                )
                for trade_matrix in trade_matrix_pair
            ]
            for trade_matrix_pair in subgraphs_with_elligible_nodes
        ]
        self.markov = [
            np.linalg.norm(base_scenario_vec - vec)
            for (base_scenario_vec, vec) in stationary_distribution_vectors
        ]

    def _compute_entropy_rate_distance(self) -> None:
        """
        Computes the 'entropy rate' distance between the base scenario
        and each other scenario.
        This is a simple difference between entropy rates computed for each graph
        assuming a Markov random walk as the process.

        Arguments:
            None

        Returns:
            None
        """
        entropy_rates = [get_entropy_rate(scenario) for scenario in self.scenarios]
        # compute difference from base scenario
        self.entropy_rate = [er - entropy_rates[0] for er in entropy_rates[1:]]

    def _format_distance_dataframe(self) -> None:
        """
        Creates a dataframe containing all computed graph difference  metrics.

        Arguments:
            None

        Returns:
            None
        """
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

    def print_distance_metrics(self, tablefmt="fancy_grid", **kwargs) -> None:
        """
        Prints the graph distance metrics in a neat tabulated form.

        Arguments:
            tablefmt (str): table format as expected by the tabulate package.
            **kwargs: any other keyworded arguments recognised by the tabulate package.

        Returns:
            None
        """
        df = self.distance_df.copy()
        df.set_index("Scenario ID", drop=True, inplace=True)
        print("***| Graph distance to the base scenario |***")
        print(df.to_markdown(tablefmt=tablefmt, **kwargs))

    def plot_distance_metrics(self, frobenius: str | None = None, **kwargs) -> None:
        """
        Plots the distance metrics as a bar plot.

        Arguments:
            frobenius (str | None, optional): Flag controlling the behaviour of
                                                graph difference metrics.
                If frobenius == "relative" *all* metrics are normalised relative
                to the highest found value in each category; if "ignore" then
                frobenius will not be included in the the plot; if None, nothing
                special happens -- in this case the Frobenius metrics is very likely
                to completety overshadow other values in the plot.
            **kwargs: any keyworded arguments recognised by seaborn barplot.

        Returns:
            None

        """
        df = self.distance_df.copy()
        match frobenius:
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
            **kwargs,
        )
        plt.ylabel("Distance")
        plt.title("Graph distance to the base scenario")
        plt.show()

    def _compute_centrality(self) -> None:
        """
        Computes the in-degree and out-degree centrality for each node in each scenario.
        By in/out-degree centrality we understand sum of in/out-coming edge weights
        to/from a given node, divided by the total in/out-flow of the graph
        (sum of all in/out-coming edge weights in the graph).

        Arguments:
            None

        Returns:
            None
        """
        self.in_degree = []
        self.out_degree = []
        for scenario in self.scenarios:
            in_degrees = list(scenario.trade_graph.in_degree(weight="weight"))
            total_in_degrees = sum(map(lambda t: t[1], in_degrees))
            in_degrees = dict(
                map(lambda t: (t[0], t[1] / total_in_degrees), in_degrees)
            )
            self.in_degree.append(in_degrees)
            out_degrees = list(scenario.trade_graph.out_degree(weight="weight"))
            total_out_degrees = sum(map(lambda t: t[1], out_degrees))
            out_degrees = dict(
                map(lambda t: (t[0], t[1] / total_out_degrees), out_degrees)
            )
            self.out_degree.append(out_degrees)

    def _compute_global_centrality_metrics(self) -> None:
        """
        Computes global centrality metrics, i.e., for each scenario it finds
        the highest/lowst in/out-degree centrality nodes.

        Arguments:
            None

        Returns:
            None
        """
        centrality_metrics = []
        for idx, (in_d, out_d) in enumerate(zip(self.in_degree, self.out_degree)):
            in_min_max = get_dict_min_max(in_d)
            out_min_max = get_dict_min_max(out_d)
            centrality_metrics.append([idx, *in_min_max, *out_min_max])
        self.global_centrality_metrics = centrality_metrics

    def print_global_centrality_metrics(self, tablefmt="fancy_grid", **kwargs) -> None:
        """
        Prints the global centrality metrics in a neat tabulated form.

        Arguments:
            tablefmt (str): table format as expected by the tabulate package.
            **kwargs: any other keyworded arguments recognised by the tabulate package.

        Returns:
            None
        """
        df = pd.DataFrame(
            self.global_centrality_metrics,
            columns=[
                "Scenario\nID",
                "Smallest\nin-degree\ncountry",
                "Smallest\nin-degree\nvalue",
                "Largest\nin-degree\ncountry",
                "Largest\nin-degree\nvalue",
                "Smallest\nout-degree\ncountry",
                "Smallest\nout-degree\nvalue",
                "Largest\nout-degree\ncountry",
                "Largest\nout-degree\nvalue",
            ],
        )
        print("***| Degree centrality metrics for each scenario |***")
        print(df.to_markdown(tablefmt=tablefmt, **kwargs))

    def _compute_community_centrality_metrics(self) -> None:
        """
        Computes local centrality metrics, i.e., for each scenario it finds
        the highest/lowst in/out-degree centrality nodes in each community.

        Arguments:
            None

        Returns:
            None
        """
        centrality_metrics = []
        for scenario_id, scenario in enumerate(self.scenarios):
            per_community_centrality_metrics = []
            for comm_id, community in enumerate(scenario.trade_communities):
                in_d = self.in_degree[scenario_id]
                out_d = self.out_degree[scenario_id]
                in_d = {k: v for k, v in in_d.items() if k in community}
                out_d = {k: v for k, v in out_d.items() if k in community}

                in_min_max = get_dict_min_max(in_d)
                out_min_max = get_dict_min_max(out_d)
                per_community_centrality_metrics.append(
                    [comm_id, *in_min_max, *out_min_max]
                )
            centrality_metrics.append(per_community_centrality_metrics)
        self.community_centrality_metrics = centrality_metrics

    def print_per_community_centrality_metrics(
        self, tablefmt="fancy_grid", **kwargs
    ) -> None:
        """
        Prints the local centrality metrics (per community) in a neat tabulated form.

        Arguments:
            tablefmt (str): table format as expected by the tabulate package.
            **kwargs: any other keyworded arguments recognised by the tabulate package.

        Returns:
            None
        """
        for scenario_id, per_community_centrality_metrics in enumerate(
            self.community_centrality_metrics
        ):
            df = pd.DataFrame(
                per_community_centrality_metrics,
                columns=[
                    "Community\nID",
                    "Smallest\nin-degree\ncountry",
                    "Smallest\nin-degree\nvalue",
                    "Largest\nin-degree\ncountry",
                    "Largest\nin-degree\nvalue",
                    "Smallest\nout-degree\ncountry",
                    "Smallest\nout-degree\nvalue",
                    "Largest\nout-degree\ncountry",
                    "Largest\nout-degree\nvalue",
                ],
            )
            print(
                f"***| Degree centrality metrics for the scenario with ID: {scenario_id} |***"
            )
            print(df.to_markdown(tablefmt=tablefmt, **kwargs))

    def plot_centrality_maps(
        self, figsize: tuple[float, float] | None = None, shrink=0.15, **kwargs
    ) -> None:
        """
        Plots world maps for each scenario, with each country coloured by their
        degree centrality in the trade graph.

        Arguments:
            figsize (tuple[float, float] | None, optional): the composite figure
                size as expected by the matplotlib subplots routine.
            label (str): label to be put on the colour bar and the title (e.g., 'in-degree')
            shrink (float, optional): colour bar shrink parameter
            **kwargs (optional): any additional keyworded arguments recognised
                by geopandas' plot function.

        Returns:
            None
        """
        _, axs = plt.subplots(
            2 * len(self.scenarios),
            1,
            sharex=True,
            tight_layout=True,
            figsize=(5, 2 * len(self.scenarios) * 5) if figsize is None else figsize,
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
            plot_degree_map(
                axs[idx],
                scenario,
                in_degree,
                label="in-degree",
                shrink=shrink,
                **kwargs,
            )
            plot_degree_map(
                axs[idx + 1],
                scenario,
                out_degree,
                label="out-degree",
                shrink=shrink,
                **kwargs,
            )
            idx += 2
        plt.show()

    def _find_new_order(self, scenario: PyTradeShifts) -> list[set[str]]:
        """
        Computes the new order of communities such that the anchor countries'
        communities are always in the same position across scenarios.

        Arguments:
            scenario (PyTradeShifts): a PyTradeShifts object instance.

        Returns:
            list[set[str]]: List containing communities in the new order.
        """
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
        assert len(anchor_idx.values()) == len(
            set(anchor_idx.values())
        ), "Two or more countries from the same community have been passed as anchores."
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

    def _arrange_communities(self) -> None:
        """
        Orders communities in each scenario based on anchor countries such that
        the colours amongst community plots are more consistent.

        Arguments:
            None

        Returns:
            None
        """
        for scenario in self.scenarios:
            scenario.trade_communities = self._find_new_order(scenario)

    def _find_community_difference(self) -> None:
        """
        For each country and scenario, computes community similarity score
        (as Jaccard Index) in comparison with the base scenario communities.

        Arguments:
            None

        Returns:
            None
        """
        # initialise the similarity score dictionary
        jaccard_indices = {
            scenario_idx: {} for scenario_idx, _ in enumerate(self.scenarios[1:], 1)
        }
        # compute the scores
        for scenario_idx, scenario in enumerate(self.scenarios[1:], 1):
            for country in self.scenarios[0].trade_matrix.index:
                # this assumes that a country can be only in one community
                # i.e. there is no community overlap
                # find the community in which the country is the current scenario
                new_community = next(
                    filter(
                        lambda community: country in community,
                        scenario.trade_communities,
                    ),
                    None,
                )
                if new_community is None:
                    print(
                        f"Warning: {country} has no community in scenario {scenario_idx}."
                    )
                    print(
                        "Skipping community similarity index computation for this country."
                    )
                    continue
                else:
                    # we want to compare how the community changed from the perspective
                    # of the country so we need to exclude the country itself from
                    # the comparison
                    new_community = new_community - {country}

                # find the community in which the country is the base scenario
                original_community = next(
                    filter(
                        lambda community: country in community,
                        self.scenarios[0].trade_communities,
                    ),
                    None,
                )
                if original_community is None:
                    print(
                        f"Warning: {country} has no community in base scenario. Skipping."
                    )
                    print(
                        "Skipping community similarity index computation for this country."
                    )
                    continue
                else:
                    original_community = original_community - {country}

                jaccard_indices[scenario_idx][country] = jaccard_index(
                    new_community, original_community
                )
        self.jaccard_indices = jaccard_indices

    def plot_community_difference(
        self,
        similarity=False,
        figsize: tuple[float, float] | None = None,
        shrink=1.0,
        **kwargs,
    ):
        """
        Plots the world map for each scenario where each country's colour is the
        dissimilarity score with the base scenario of their communities.

        Arguments:
            similarity (bool, optional): whether to plot Jaccard index or distance.
                If True similarity (index) will be used, if False, distance = (1-index).
                Defualt is False.
            figsize (tuple[float, float] | None, optional): the composite figure
                size as expected by the matplotlib subplots routine.
            shrink (float, optional): colour bar shrink parameter
            **kwargs (optional): any additional keyworded arguments recognised
                by geopandas plot function.

        Returns:
            None
        """
        assert len(self.scenarios) > 1
        _, axs = plt.subplots(
            len(self.scenarios) - 1, 1, sharex=True, tight_layout=True, figsize=figsize
        )
        # if there are only two scenarios axs will be just an ax object
        # convert to a list to comply with other cases
        try:
            len(axs)
        except TypeError:
            axs = [axs]
        for ax, (idx, scenario) in zip(axs, enumerate(self.scenarios[1:], 1)):
            plot_jaccard_map(
                ax,
                scenario,
                self.jaccard_indices[idx],
                similarity=similarity,
                shrink=shrink,
                **kwargs,
            )
        plt.show()

    def plot_all_trade_communities(
        self, figsize: tuple[float, float] | None = None
    ) -> None:
        """
        Plots the trade communities in each of the scenarios.

        Arguments:
            figsize (tuple[float, float] | None, optional): the composite figure
                size as expected by the matplotlib subplots routine.

        Returns:
            None
        """
        _, axs = plt.subplots(
            len(self.scenarios), 1, sharex=True, tight_layout=True, figsize=figsize
        )
        for ax, sc in zip(axs, self.scenarios):
            sc._plot_trade_communities(ax)
        plt.show()

    def report(self) -> None:
        """
        TODO
        """
        raise NotImplementedError
