import pandas as pd
from os import listdir
from os.path import isfile, join
from tqdm import tqdm


def main():
    # specify the macros directory
    macros_dir = "intmodel/data/macros"
    # get only the scenario files
    macros_files = [
        f
        for f in listdir(macros_dir)
        if isfile(join(macros_dir, f)) and "crop_reduction" in f
    ]
    # get the base / no scenario file
    df = pd.read_csv(macros_dir + "/" + "macros_csv.csv", index_col=0)
    # merge
    for mf in tqdm(macros_files):
        df = df.merge(
            pd.read_csv(macros_dir + "/" + mf, index_col=0, usecols=[0, 2, 3, 4]),
            left_index=True,
            right_index=True,
            how="outer",
            suffixes=[
                None,
                "_" + mf[mf.find("year_") : mf.find(".")],
            ],
        )
    # this is just rearranging columns to my liking
    first_columns = ["country", "crop_kcals", "crop_fat", "crop_protein"]
    df = df[
        first_columns
        + sorted(
            [c for c in df.columns if c not in first_columns],
            key=lambda c: (int(c[c.find("year_") + len("year_") :])),
        )
    ]
    df = df.reset_index()
    # save to file
    df.to_csv("intmodel/data/macros_csv.csv", index=False)


if __name__ == "__main__":
    main()
