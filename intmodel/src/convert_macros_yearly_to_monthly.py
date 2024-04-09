import pandas as pd


def main():
    macro_data = pd.read_csv("intmodel/data/macros_csv.csv")
    macro_data = (
        macro_data.set_index(["iso3", "country"]) / 12
    )  # divide to get monthly values

    for item in ["kcals", "fat", "protein"]:
        for year in range(0, 10):
            for month in range(1, 13):
                # split each year into 12 months
                macro_data[f"crop_{item}_month_{month+12*year}"] = macro_data[
                    f"crop_{item}_year_{year+1}"
                ]
            macro_data = macro_data.drop(columns=f"crop_{item}_year_{year+1}")

    macro_data = macro_data.reset_index()
    macro_data.to_csv("intmodel/data/macros_csv_monthly_average.csv", index=False)


if __name__ == "__main__":
    main()
