import pandas as pd
from os import listdir
from os.path import isfile, join
from functools import reduce
import country_converter as coco
from domestic_supply import get_scenarios
from tqdm import tqdm


def combine_files(ds_dir: str, ds_files: list[str], output_file: str) -> None:
    """Combine all domestic supply files into one; per scenario.

    Arguments:
        ds_dir (str): path to the directory holding the domestic supply files.
        ds_files (list[str]): list of those files.
        output_file (str): path to the output file.

    Returns:
        None.
    """
    # read all the domestic supply csv
    dfs = [
        pd.read_csv(ds_dir + "/" + f, header=0, names=["Country", f[: f.find("_")]])
        for f in ds_files
    ]
    # merge on country
    df = reduce(
        lambda left, right: pd.merge(left, right, on=["Country"], how="outer"), dfs
    )
    # fill missing data with zeroes
    df = df.fillna(0)
    # add iso code column
    cc = coco.CountryConverter()
    df["iso3"] = cc.pandas_convert(pd.Series(df["Country"]), to="ISO3")
    # convert from wide to long format
    df = df.melt(
        id_vars=["iso3", "Country"],
        value_vars=[c for c in df.columns if c != "iso3" and c != "Country"],
    )
    # rename columns for consistency's sake
    df.columns = ["Area Code (ISO3)", "Area", "Item", "Value"]
    # save to file
    df.to_csv(output_file)


def main():
    # specify the directory holding the data
    ds_dir = "intmodel/data/domestic_supply"
    print("Combining no scenario domestic supply.")
    # get the files
    ds_files = [
        f
        for f in listdir(ds_dir)
        if isfile(join(ds_dir, f)) and "crop_reduction" not in f
    ]
    # specify output file
    output_file = "intmodel/data/domestic_supply_combined/domestic_supply_combined.csv"
    # run the function
    combine_files(ds_dir, ds_files, output_file)
    print("Combining domestic supply with crop reduction.")
    scenarios = get_scenarios("intmodel/data/scenario_files")
    # do as the above but for each of the scnearios
    for scenario in tqdm(scenarios):
        ds_files = [
            f for f in listdir(ds_dir) if isfile(join(ds_dir, f)) and scenario in f
        ]
        combine_files(
            ds_dir, ds_files, f"intmodel/data/domestic_supply_combined/{scenario}"
        )


if __name__ == "__main__":
    main()
