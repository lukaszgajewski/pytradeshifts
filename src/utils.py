import geopandas as gpd
import pandas as pd
import country_converter as coco
import os


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
    # filter out the regions that aren't in the trade matrix
    centroids = (
        centroids.merge(
            pd.DataFrame(
                [str(c) for c in coco.convert(trade_matrix_index, to="name_short")],
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
    return centroids
