import pandas as pd
import geopandas as gpd
import numpy as np
import country_converter as coco
from preprocessing import rename_countries
from scipy.spatial.distance import squareform, pdist
from geopy.distance import geodesic

beta = 2
trade = pd.read_csv("data/preprocessed_data/Wheat_Y2021_Global_trade.csv", index_col=0)
trade = rename_countries(trade, "All_Data", "Trade_DetailedTradeMatrix_E", "Area Code")
trade.index = [str(c) for c in coco.convert(trade.index, to="name_short")]

# https://github.com/gavinr/world-countries-centroids
centroids_a = pd.read_csv("data/countries.csv")
centroids_a["name"] = [
    str(c) for c in coco.convert(centroids_a["COUNTRY"], to="name_short")
]

# we need this for Hong Kong and Taiwan
centroids_b = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
centroids_b["name"] = [
    str(c) for c in coco.convert(centroids_b["name"], to="name_short")
]

# https://gis.stackexchange.com/questions/372564/userwarning-when-trying-to-get-centroid-from-a-polygon-geopandas
centroids_b["centroid"] = centroids_b.to_crs("+proj=cea").centroid.to_crs(
    centroids_b.crs
)
centroids_b["lon"] = centroids_b["centroid"].x
centroids_b["lat"] = centroids_b["centroid"].y

centroids = centroids_a.merge(centroids_b, how="outer", on="name")
centroids = (
    centroids.merge(pd.DataFrame(trade.index, columns=["name"]), how="inner", on="name")
    .sort_values("name")
    .reset_index(drop=True)
)
centroids.loc[:, ["longitude", "latitude"]].fillna(
    centroids[["lon", "lat"]], inplace=True
)
centroids = centroids[["longitude", "latitude", "name"]]

assert len(trade) == len(centroids), (
    len(trade),
    len(centroids),
    set(trade.index).difference(centroids["name"]),
)

dist = pd.DataFrame(
    squareform(
        pdist(
            centroids.loc[:, ["latitude", "longitude"]],
            metric=lambda lat, lon: geodesic(lat, lon).km,
        )
    ),
    columns=centroids["name"].values,
    index=centroids["name"].values,
)
# for some reason this crashes, looks like some sort of a bug in pandas?
# trade = trade.multiply(dist.pow(-beta))
trade = pd.DataFrame(
    np.multiply(trade.values, np.power(dist.values, -beta)),
    index=trade.index,
    columns=trade.columns,
)
np.fill_diagonal(trade.values, 0)
print(trade)
