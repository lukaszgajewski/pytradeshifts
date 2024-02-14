from typing import Iterable, Any
import numpy as np
import geopandas as gpd
import pandas as pd
import country_converter as coco
import os
from itertools import groupby
from networkx import to_pandas_adjacency as nx_to_pandas_adjacency, Graph as nx_Graph
from matplotlib.axes import Axes


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


def plot_degree_map(
    ax: Axes, scenario, degree: dict, label: str, shrink=0.15, **kwargs
) -> None:
    """
    Plots world map with each country coloured by their degree in the trade graph.

    Arguments:
        ax (Axes): axis to which the plot is to be committed.
        scenario (PyTradeShifts): PyTradeShifts instance
        degree (dict): dictionary containing the mapping: country->value
        label (str): label to be put on the colour bar and the title (e.g., 'in-degree')
        shrink (float, optional): colour bar shrink parameter
        **kwargs (optional): any additional keyworded arguments recognised
            by geopandas' plot function.

    Returns:
        None.
    """
    assert scenario.trade_communities is not None
    world = prepare_world()

    # Join the country_community dictionary to the world dataframe
    world["degree"] = world["names_short"].map(degree)

    world.plot(
        ax=ax,
        column="degree",
        missing_kwds={"color": "lightgrey"},
        legend=True,
        legend_kwds={"label": label, "shrink": shrink},
        **kwargs,
    )

    plot_winkel_tripel_map(ax)

    # Add a title with self.scenario_name if applicable
    ax.set_title(
        f"{label} for {scenario.crop} with base year {scenario.base_year[1:]}"
        + (
            f" in scenario: {scenario.scenario_name}"
            if scenario.scenario_name is not None
            else " (no scenario)"
        )
    )


def plot_jaccard_map(ax: Axes, scenario, jaccard: dict, similarity=False) -> None:
    """
    Plots world map with countries coloured by their community's Jaccard similarity
    to their original community (in the specified scenario).

    Arguments:
        ax (Axes): axis on which to plot.
        scenario (PyTradeShifts): a PyTradeShifts instance.
        jaccard (dict): dictionary containing the mapping: country->value
        similarity (bool, optional): whether to plot Jaccard index or distance.
            If True similarity (index) will be used, if False distance (1-index).
            Defualt is False.

    Returns:
        None.
    """
    assert scenario.trade_communities is not None
    world = prepare_world()

    # Join the country_community dictionary to the world dataframe
    world["jaccard_index"] = world["names_short"].map(jaccard)
    world["jaccard_distance"] = 1 - world["jaccard_index"]

    world.plot(
        ax=ax,
        column="jaccard_distance" if not similarity else "jaccard_index",
        missing_kwds={"color": "lightgrey"},
        legend=True,
        legend_kwds={"label": "Jaccard distance"},
    )

    plot_winkel_tripel_map(ax)

    # Add a title with self.scenario_name if applicable
    value_plotted_label = "Similarity to" if similarity else "Dissimilarity to"
    ax.set_title(
        f"{value_plotted_label} base scenario for {scenario.crop} with base year {scenario.base_year[1:]}"
        + (
            f" in scenario: {scenario.scenario_name}"
            if scenario.scenario_name is not None
            else " (no scenario)"
        )
    )


def get_right_stochastic_matrix(trade_graph: nx_Graph) -> np.ndarray:
    """
    Convert graph's adjacency matrix to a right stochastic matrix (RSM).
    https://en.wikipedia.org/wiki/Stochastic_matrix

    Arguments:
        trade_graph (networkx.Graph): the graph object

    Returns:
        numpy.ndarray: an array representing the RSM.
    """
    # extract the adjaceny matrix from the graph
    right_stochastic_matrix = nx_to_pandas_adjacency(trade_graph)
    # normalise the matrix such that each row sums up to 1
    right_stochastic_matrix = right_stochastic_matrix.div(
        right_stochastic_matrix.sum(axis=0)
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


def get_entropy_rate(scenario) -> float:
    """
    Compute entropy rate for a given scenario.
    https://en.wikipedia.org/wiki/Entropy_rate
    This is under the assumption that we are interested in a markovian random
    walker on the trade graph since the entropy rate is a measure of a process,
    not structure.

    Arguments:
        scenario (PyTradeShifts): a PyTradeShifts object instance.

    Returns:
        float: the entropy rate of a random walker on the scnearios' trade graph.
    """
    # get the right stochastic matrix and the stationary probabiltiy vector
    P = get_right_stochastic_matrix(scenario.trade_graph)
    probability_vector = get_stationary_probability_vector(P)
    # compute the entropy rate in [nat]
    # we ignore the division warning because 0 x log(0) = 0 in information theory
    with np.errstate(divide="ignore"):
        entropy_rate = np.sum(
            probability_vector * np.sum(-P * np.nan_to_num(np.log(P)), axis=1)
        )
    # this should be a real value
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
