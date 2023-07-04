import faostat
import pandas as pd

# Download data from FAOSTAT
datasets = faostat.list_datasets_df(https_proxy=None)
print(datasets)
for label, code in zip(datasets["label"], datasets["code"]):
    print(label, code)

trade_matrix = faostat.get_data_df(
    "TM",
    show_flags=False,
    null_values=False,
    https_proxy=None,
    pars={'years': 2018}
)
trade_matrix.to_csv("trade_matrix.csv", index=False)
