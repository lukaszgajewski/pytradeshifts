from typing import Iterable, Any
from geopy import distance
import numpy as np
import geopandas as gpd
import pandas as pd
import country_converter as coco
import os
from itertools import groupby
import networkx as nx
from matplotlib.axes import Axes
from geopy.distance import geodesic
from scipy.spatial.distance import squareform, pdist


def plot_winkel_tripel_map(ax):
    """
    Helper function to plot a Winkel Tripel map with a border.
    """
    border_geojson = gpd.read_file(
        "https://raw.githubusercontent.com/ALLFED/ALLFED-map-border/main/border.geojson"
    )
    border_geojson.plot(ax=ax, edgecolor="black", linewidth=0.1, facecolor="none")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])


def prepare_centroids(trade_matrix_index: pd.Index) -> pd.DataFrame:
    """
    Prepares a DataFrame containing coordinates of centroids for each region.
    A centroid is the geometric centre of a region
    (also the centre of mass assuming uniform mass distribution).
    This method is used internally in the PyTradeShifts class for the
    computation of the distance matrix between regions.

    Arguments:
        trade_matrix_index (pd.Index): The index of the trade matrix.

    Returns:
        pd.DataFrame: Data frame with columns ["longitude", "latitude", "name"]
            containing centroid coordinates for each region.
    """
    # https://github.com/gavinr/world-countries-centroids
    centroids_a = pd.read_csv(
        "."
        + os.sep
        + "data"
        + os.sep
        + "geospatial_references"
        + os.sep
        + "country_centroid_locations.csv"
    )
    # convert whatever names data sets have to one format so that we can
    # merge all data convieniently later
    centroids_a["name"] = [
        str(c) for c in coco.convert(centroids_a["COUNTRY"], to="name_short")
    ]
    # we need this for Taiwan
    centroids_b = gpd.read_file(
        "."
        + os.sep
        + "data"
        + os.sep
        + "geospatial_references"
        + os.sep
        + "ne_110m_admin_0_countries"
    )
    centroids_b["name"] = [
        str(c) for c in coco.convert(centroids_b["NAME_LONG"], to="name_short")
    ]
    # we use natural earth data to compute centroids
    # https://gis.stackexchange.com/questions/372564/userwarning-when-trying-to-get-centroid-from-a-polygon-geopandas
    centroids_b["centroid"] = centroids_b.to_crs("+proj=cea").centroid.to_crs(
        centroids_b.crs
    )
    centroids_b["lon"] = centroids_b["centroid"].x
    centroids_b["lat"] = centroids_b["centroid"].y
    # filter out unneeded columns
    centroids_b = centroids_b[["lon", "lat", "name"]]
    # need to add some regions manually; values taken from Google maps
    centroids_b = pd.concat(
        [
            centroids_b,
            pd.DataFrame(
                [
                    [113.54474590904003, 22.198228394900145, "Macau"],
                    [114.17506394778538, 22.32788145032773, "Hong Kong"],
                ],
                columns=["lon", "lat", "name"],
            ),
        ]
    )
    # merge the centroid data sets
    centroids = centroids_a.merge(centroids_b, how="outer", on="name")
    # filter out the regions that aren't in the trade matrix or in centroids
    trade_matrix_corrected_index = [
        str(c) for c in coco.convert(trade_matrix_index, to="name_short")
    ]
    centroids = (
        centroids.merge(
            pd.DataFrame(
                trade_matrix_corrected_index,
                columns=["name"],
            ),
            how="inner",
            on="name",
        )
        .sort_values("name")
        .reset_index(drop=True)
    )
    # fill missing values from data set 'a' with values from data set 'b'
    centroids.loc[:, "longitude"].fillna(centroids["lon"], inplace=True)
    centroids.loc[:, "latitude"].fillna(centroids["lat"], inplace=True)
    # filter out unneeded columns and remove duplicates
    centroids = centroids[["longitude", "latitude", "name"]]
    centroids = centroids.drop_duplicates()
    trade_matrix_index_restoration_dict = {
        c: cc
        for c, cc in zip(trade_matrix_corrected_index, trade_matrix_index)
        if c in centroids.index
    }
    centroids["name"].replace(trade_matrix_index_restoration_dict, inplace=True)
    centroids.sort_values("name", inplace=True)
    return centroids


def all_equal(iterable: Iterable):
    """
    Checks if all the elements are equal to each other.
    source: https://docs.python.org/3/library/itertools.html#itertools-recipes

    Arguments:
        iterable (Iterable): a list-like object containing elements to compare.

    Returns:
        bool: True if all elements are equal, False otherwise.
    """
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def jaccard_index(iterable_A: Iterable, iterable_B: Iterable) -> float:
    """
    Computes the Jaccard similarity beween two iterables.
    https://en.wikipedia.org/wiki/Jaccard_index
    Note: the iterables shall be cast to sets.

    Arguments:
        iterable_A (Iterable): a list-like object
        iterable_B (Iterable): a list-like object

    Returns:
        float: The Jaccard index (similarity) of iterable_A and iterable_B
    """
    A = set(iterable_A)
    B = set(iterable_B)
    return len(A.intersection(B)) / len(A.union(B))


def get_degree_centrality(graph: nx.DiGraph, out=False) -> dict:
    """
    TODO
    """
    if out:
        degrees = list(graph.out_degree(weight="weight"))
    else:
        degrees = list(graph.in_degree(weight="weight"))
    total_degrees = sum(map(lambda t: t[1], degrees))
    return dict(map(lambda t: (t[0], t[1] / total_degrees), degrees))


def prepare_world() -> gpd.GeoDataFrame:
    """
    Prepares the geospatial Natural Earth (NE) data (to be presumebly used in plotting).

    Arguments:
        None

    Returns:
        gpd.GeoDataFrame: NE data projected to Winke Tripel and with converted country names.
    """
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
    # Change projection to Winkel Tripel
    world = world.to_crs("+proj=wintri")

    cc = coco.CountryConverter()
    world["names_short"] = cc.pandas_convert(pd.Series(world["ADMIN"]), to="name_short")
    return world


def plot_node_metric_map(
    ax: Axes, scenario, metric: dict, metric_name: str, shrink=1.0, **kwargs
) -> None:
    """
    Plots world map with countries coloured by the specified metric.

    Arguments:
        ax (Axes): axis on which to plot.
        scenario (PyTradeShifts): a PyTradeShifts instance.
        metric (dict): dictionary containing the mapping: country->value
        metric_name (str): the name of the metric
        shrink (float, optional): colour bar shrink parameter
        **kwargs (optional): any additional keyworded arguments recognised
            by geopandas plot function.

    Returns:
        None
    """
    assert scenario.trade_communities is not None
    world = prepare_world()

    # Join the country_community dictionary to the world dataframe
    world[metric_name] = world["names_short"].map(metric)

    world.plot(
        ax=ax,
        column=metric_name,
        missing_kwds={"color": "lightgrey"},
        legend=True,
        legend_kwds={
            "label": metric_name,
            "shrink": shrink,
        },
        **kwargs,
    )

    plot_winkel_tripel_map(ax)

    # Add a title with self.scenario_name if applicable
    ax.set_title(
        f"{metric_name} for {scenario.crop} with base year {scenario.base_year[1:]}"
        + (
            f" in scenario: {scenario.scenario_name}"
            if scenario.scenario_name is not None
            else " (no scenario)"
        )
    )


def get_right_stochastic_matrix(trade_graph: nx.Graph) -> np.ndarray:
    """
    Convert graph's adjacency matrix to a right stochastic matrix (RSM).
    https://en.wikipedia.org/wiki/Stochastic_matrix

    Arguments:
        trade_graph (networkx.Graph): the graph object

    Returns:
        numpy.ndarray: an array representing the RSM.
    """
    # extract the adjaceny matrix from the graph
    right_stochastic_matrix = nx.to_pandas_adjacency(trade_graph)
    # normalise the matrix such that each row sums up to 1
    right_stochastic_matrix = right_stochastic_matrix.div(
        right_stochastic_matrix.sum(axis=1), axis=0
    )
    right_stochastic_matrix.fillna(0, inplace=True)
    return right_stochastic_matrix.values


def get_stationary_probability_vector(
    right_stochastic_matrix: np.ndarray,
) -> np.ndarray:
    """
    Find the stationary probability distribution for a given
    right stochastic matrix (RMS).

    Arguments:
        right_stochastic_matrix (numpy.ndarray): an array represnting the RMS.

    Returns:
        numpy.ndarray: a vector representing the stationary distribution.
    """
    # get eigenvalues and (row) eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(right_stochastic_matrix.T)
    # we want eigenvector associated with eigenvalue = 1
    _, probability_vector = min(
        zip(eigenvalues, eigenvectors.T), key=lambda v: abs(v[0] - 1.0)
    )
    # normalise, this is a probability distribution
    probability_vector /= np.sum(probability_vector)
    return probability_vector


def _compute_entropy_rate(
    right_stochastic_matrix: np.ndarray, stationary_probability_vector: np.ndarray
) -> float:
    """
    Computes the entropy rate in [nat] for a Markov random walk process.

    Arguments:
        right_stochastic_matrix (numpy.ndarray): the stochastic matrix of the Markov
            random walk with each row normalised (summing up to 1)
        stationary_probability_vector: numpy.ndarray): the vector containing stationary
            probability distribution of the Markov process

    Returns:
        float: the entropy rate
    """
    # we ignore the division warning because 0 x log(0) = 0 in information theory
    with np.errstate(divide="ignore"):
        entropy_rate = np.sum(
            stationary_probability_vector
            * np.sum(
                -right_stochastic_matrix
                * np.nan_to_num(np.log(right_stochastic_matrix)),
                axis=1,
            )
        )
    return entropy_rate


def get_entropy_rate(scenario) -> float:
    """
    Compute entropy rate for a given scenario.
    https://en.wikipedia.org/wiki/Entropy_rate
    This is under the assumption that we are interested in a Markov random
    walk on the trade graph.

    Arguments:
        scenario (PyTradeShifts): a PyTradeShifts object instance.

    Returns:
        float: the entropy rate of a random walker on the scnearios' trade graph.
    """
    # get the right stochastic matrix and the stationary probabiltiy vector
    stochastic_matrix = get_right_stochastic_matrix(scenario.trade_graph)
    probability_vector = get_stationary_probability_vector(stochastic_matrix)
    entropy_rate = _compute_entropy_rate(stochastic_matrix, probability_vector)
    if np.real(entropy_rate) != np.real_if_close(entropy_rate):
        print("Warning: a significant imaginary part encountered in entropy rate:")
        print(entropy_rate)
        print("Returning real part only.")
    return np.real(entropy_rate)


def get_dict_min_max(iterable: dict) -> tuple[Any, Any, Any, Any]:
    """
    Finds minimum and maximum values in a dictionary.

    Arguments:
        iterable (dict): the dictionary in which the search is performed

    Returns:
        tuple[Any, Any, Any, Any]: a tuple containing:
            (key of the smallest value, the smallest value,
            key of the larges value, the largest value)
    """
    max_key = max(iterable, key=iterable.get)
    min_key = min(iterable, key=iterable.get)
    return min_key, iterable[min_key], max_key, iterable[max_key]


def get_graph_efficiency(graph: nx.Graph, normalisation: str | None = "weak") -> float:
    """
    TODO, https://www.nature.com/articles/s42005-021-00612-5
    also a bit slow
    """
    all_pairs_paths = dict(
        nx.all_pairs_dijkstra(
            graph,
            weight=lambda _, __, attr: (
                attr["weight"] ** -1 if attr["weight"] != 0 else np.inf
            ),
        )
    )
    cost_matrix = pd.DataFrame(
        0.0,
        index=graph.nodes(),
        columns=graph.nodes(),
    )
    flow_matrix = pd.DataFrame(
        0.0,
        index=graph.nodes(),
        columns=graph.nodes(),
    )
    for source, (distances, paths) in all_pairs_paths.items():
        for target, cost in distances.items():
            cost_matrix.loc[source, target] = cost
        for target, path in paths.items():
            flow_matrix.loc[source, target] = nx.path_weight(
                graph, path, weight="weight"
            )
    E = np.sum(1 / cost_matrix.values[cost_matrix != 0])
    match normalisation:
        case "weak":
            ideal_matrix = (flow_matrix + nx.to_pandas_adjacency(graph)) / 2
        case "strong":
            ideal_matrix = flow_matrix
        case None:
            return E
        case _:
            print("Unrecognised normalisation option, defaulting to ``weak''.")
            ideal_matrix = (flow_matrix + nx.to_pandas_adjacency(graph)) / 2
    E_ideal = np.sum(ideal_matrix.values)
    return E / E_ideal


def get_stability_index(
    index_file="data/government_PRS_stability_index_2016_normalised.csv",
) -> dict[str, float]:
    """
    TODO
    """
    stability_index = pd.read_csv(index_file, index_col=0)
    stability_index.index = coco.convert(stability_index.index, to="name_short")
    return stability_index.loc[:, stability_index.columns[-1]].to_dict()


def get_distance_matrix(index: pd.Index, columns: pd.Index) -> pd.DataFrame | None:
    """
    TODO, refactor this from model too
    """
    centroids = prepare_centroids(index)
    try:
        distance_matrix = pd.DataFrame(
            squareform(
                pdist(
                    centroids.loc[:, ["latitude", "longitude"]],
                    metric=lambda lat, lon: geodesic(lat, lon).km,
                )
            ),
            columns=columns,
            index=index,
        )
    except ValueError:
        print("Error building the distance matrix.")
        print("Cannot find centroids for these regions:")
        print(index.difference(centroids["name"]))
        return
    return distance_matrix


def get_percolation_eigenvalue(
    adjacency_matrix: np.ndarray, attack_vector: np.ndarray
) -> float:
    """
    TODO
    """
    return np.real(
        np.linalg.eigvals((adjacency_matrix.T * (1 - attack_vector)).T).max()
    )


def get_percolation_threshold(
    adjacency_matrix: np.ndarray, node_importance_list: list
) -> tuple[int, list[int], list[float]]:
    """
    TODO
    https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.100.058701
    """
    eigenvalues = []
    removed_nodes_count = []
    for node_importance in sorted(node_importance_list, reverse=True):
        attack_vector = np.fromiter(
            map(lambda x: x >= node_importance, node_importance_list),
            dtype=float,
            count=len(adjacency_matrix),
        )
        # largest eigenvalue of a matrix with elements: A_ij * (1-p_i)
        # where A is the adj. matrix, and p is the attack vector
        eigenvalues.append(get_percolation_eigenvalue(adjacency_matrix, attack_vector))
        removed_nodes_count.append(int(attack_vector.sum()))
    # eigenvalue = 1 is the percolation threshold
    threshold = removed_nodes_count[
        len(eigenvalues) - np.searchsorted(eigenvalues[::-1], 1, side="left")
    ]
    return threshold, removed_nodes_count, eigenvalues
