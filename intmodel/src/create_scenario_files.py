import pandas as pd


def main():
    reduction_data = pd.read_csv("intmodel/data/crop_reduction_by_year.csv")

    for ii, c in enumerate(reduction_data.columns[1:]):
        reduction_data[["iso3", c]].to_csv(
            f"intmodel/data/scenario_files/crop_reduction_year_{ii+1}.csv", index=False
        )


if __name__ == "__main__":
    main()
