from typing import Iterable
import numpy as np
import geopandas as gpd
import pandas as pd
import country_converter as coco
import os
from itertools import groupby
from networkx import to_pandas_adjacency as nx_to_pandas_adjacency


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
    TODO: https://en.wikipedia.org/wiki/Jaccard_index
    """
    A = set(iterable_A)
    B = set(iterable_B)
    return len(A.intersection(B)) / len(A.union(B))


def prepare_world() -> gpd.GeoDataFrame:
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


def plot_jaccard_map(ax, scenario, jaccard) -> None:
    """
    TODO
    """
    assert scenario.trade_communities is not None
    world = prepare_world()

    # Join the country_community dictionary to the world dataframe
    world["jaccard_index"] = world["names_short"].map(jaccard)
    world["jaccard_distance"] = 1 - world["jaccard_index"]

    world.plot(
        ax=ax,
        column="jaccard_distance",
        missing_kwds={"color": "lightgrey"},
        legend=True,
        legend_kwds={"label": "Jaccard distance"},
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


def get_stationary_probability_vector(
    right_stochastic_matrix: np.ndarray,
) -> np.ndarray:
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
    assert scenario.trade_graph is not None
    right_stochastic_matrix = nx_to_pandas_adjacency(scenario.trade_graph)
    right_stochastic_matrix = right_stochastic_matrix.div(
        right_stochastic_matrix.sum(axis=0)
    )
    right_stochastic_matrix.fillna(0, inplace=True)
    # a helper variable
    P = right_stochastic_matrix.values
    probability_vector = get_stationary_probability_vector(P)
    # entropy rate in [nat]
    entropy_rate = np.sum(
        probability_vector * np.sum(-P * np.nan_to_num(np.log(P)), axis=1)
    )
    # this should be a real value
    if np.real(entropy_rate) != np.real_if_close(entropy_rate):
        print("Warning: a significant imaginary part encountered in entropy rate:")
        print(entropy_rate)
        print("Returning real part only.")
    return np.real(entropy_rate)
