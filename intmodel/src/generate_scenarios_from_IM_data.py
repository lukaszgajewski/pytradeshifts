import pandas as pd


IM_combined = pd.read_csv(
    "data/preprocessed_data/integrated_model/from_IM/computer_readable_combined.csv"
)
assert all(
    [
        len(gr[1]) == 1
        for gr in IM_combined[["iso3", "crop_reduction_year1"]].groupby("iso3")
    ]
)

for cry in [f"crop_reduction_year{y}" for y in range(1, 11)]:
    x = IM_combined[["iso3", cry]]
    x.loc[x[cry] < -1, cry] = -1  # cap at -1.0
    x.loc[:, cry] = x[cry] * 100  # convert to %
    x.to_csv(f"data/scenario_files/integrated_model/{cry}.csv", index=False)
