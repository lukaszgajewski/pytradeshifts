import pandas as pd
import geopandas as gpd
import numpy as np
import country_converter as coco
from preprocessing import rename_countries
from scipy.spatial.distance import squareform, pdist

beta = 2
trade = pd.read_csv("data/preprocessed_data/Wheat_Y2021_Global_trade.csv", index_col=0)
trade = rename_countries(trade, "All_Data", "Trade_DetailedTradeMatrix_E", "Area Code")
# https://github.com/gavinr/world-countries-centroids
centroids_a = pd.read_csv("data/countries.csv")
centroids_a["name"] = coco.convert(centroids_a["COUNTRY"], to="name_short")


centroids_b = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
centroids_b["name"] = [
    str(c) for c in coco.convert(centroids_b["name"], to="name_short")
]

# # https://github.com/UNFAOstatistics/gisfao
# centroids_b = pd.read_csv("data/fao_world_centroids.csv", index_col=0)
# centroids_b["name"] = [
#     str(c) for c in coco.convert(centroids_b["ADM0_NAME"], to="name_short")
# ]

centroids = centroids_a.merge(centroids_b, how="outer", on="name")
centroids = (
    centroids.merge(pd.DataFrame(trade.index, columns=["name"]), how="inner", on="name")
    .sort_values("name")
    .reset_index(drop=True)
)
# this is failing, these datasets are insufficient
assert len(trade) == len(centroids), (len(trade), len(centroids))
# this won't work because some of the lon/lat are NaNs and need to be replacd
# with x, y -- this is because of the _a,_b merge
dist = pd.DataFrame(
    squareform(pdist(centroids.loc[:, ["longitude", "latitude"]])),
    columns=centroids["name"].values,
    index=centroids["name"].values,
)

trade = trade.multiply(dist.pow(-beta))
