import pandas as pd
from os import listdir
from os.path import isfile, join
from functools import reduce
import country_converter as coco
from domestic_supply import get_scenarios
from tqdm import tqdm


def combine_files(ds_dir, ds_files, output_file):
    dfs = [
        pd.read_csv(ds_dir + "/" + f, header=0, names=["Country", f[: f.find("_")]])
        for f in ds_files
    ]
    df = reduce(
        lambda left, right: pd.merge(left, right, on=["Country"], how="outer"), dfs
    )
    df = df.fillna(0)
    cc = coco.CountryConverter()
    df["iso3"] = cc.pandas_convert(pd.Series(df["Country"]), to="ISO3")

    df = df.melt(
        id_vars=["iso3", "Country"],
        value_vars=[c for c in df.columns if c != "iso3" and c != "Country"],
    )
    df.columns = ["Area Code (ISO3)", "Area", "Item", "Value"]

    df.to_csv(output_file)


if __name__ == "__main__":
    ds_dir = "intmodel/data/domestic_supply"
    print("Combining no scenario domestic supply.")
    ds_files = [
        f
        for f in listdir(ds_dir)
        if isfile(join(ds_dir, f)) and "crop_reduction" not in f
    ]
    output_file = "intmodel/data/domestic_supply_combined/domestic_supply_combined.csv"
    combine_files(ds_dir, ds_files, output_file)
    print("Combining domestic supply with crop reduction.")
    scenarios = get_scenarios("intmodel/data/scenario_files")
    for scenario in tqdm(scenarios):
        ds_files = [
            f for f in listdir(ds_dir) if isfile(join(ds_dir, f)) and scenario in f
        ]
        combine_files(
            ds_dir, ds_files, f"intmodel/data/domestic_supply_combined/{scenario}"
        )
