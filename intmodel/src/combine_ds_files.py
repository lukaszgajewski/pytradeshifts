import pandas as pd
from os import listdir
from os.path import isfile, join
from functools import reduce
import country_converter as coco

cc = coco.CountryConverter()

ds_dir = "intmodel/data/domestic_supply"
ds_files = [f for f in listdir(ds_dir) if isfile(join(ds_dir, f))]

dfs = [
    pd.read_csv(ds_dir + "/" + f, header=0, names=["Country", f[: f.find("_")]])
    for f in ds_files
]
df = reduce(lambda left, right: pd.merge(left, right, on=["Country"], how="outer"), dfs)
df = df.fillna(0)

df["iso3"] = cc.pandas_convert(pd.Series(df["Country"]), to="ISO3")

df = df.melt(
    id_vars=["iso3", "Country"],
    value_vars=[c for c in df.columns if c != "iso3" and c != "Country"],
)
df.columns = ["Area Code (ISO3)", "Area", "Item", "Value"]

df.to_csv("intmodel/data/domestic_supply_combined.csv")
