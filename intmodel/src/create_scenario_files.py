import pandas as pd


def main():
    reduction_data = pd.read_csv("intmodel/data/nuclear_winter_csv.csv")
    reduction_data = reduction_data[
        ["iso3"] + [f"crop_reduction_year{ii}" for ii in range(1, 11)]
    ]
    reduction_data = reduction_data.set_index("iso3")
    reduction_data *= 100
    reduction_data[reduction_data < -100] = -100
    reduction_data = reduction_data.reset_index()

    for ii, c in enumerate(reduction_data.columns[1:]):
        reduction_data[["iso3", c]].to_csv(
            f"intmodel/data/scenario_files/crop_reduction_year_{ii+1}.csv", index=False
        )


if __name__ == "__main__":
    main()
