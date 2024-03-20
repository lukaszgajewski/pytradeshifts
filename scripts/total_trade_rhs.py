import os
from src.preprocessing import (
    serialise_faostat_bulk,
    remove_entries_from_data,
    _unify_indices,
)
from src.model import PyTradeShifts as PTS
import pandas as pd
import numpy as np


def get_prod(region="All_Data", year="Y2020", unit="t"):
    try:
        print(f"Reading in production data for all items in {region}...")
        prod = pd.read_pickle(
            f"data{os.sep}temp_files{os.sep}Production_Crops_Livestock_E_{region}.pkl"
        )
    except FileNotFoundError:
        print(f"Pickled Data in {region} not found. Reading zip to create pickle.")
        production_zip = (
            f"data{os.sep}data_raw{os.sep}Production_Crops_Livestock_E_{region}.zip"
        )
        serialise_faostat_bulk(production_zip)
        print("Serialisation complete. Run the script again.")
        return

    prod = prod[((prod["Unit"] == unit) & (~prod[year].isna()))]
    prod = prod[["Area Code (M49)", year]]
    prod = prod.groupby(["Area Code (M49)"]).sum()
    prod = remove_entries_from_data(prod)
    return prod


def get_trad(
    region="All_Data",
    year="Y2020",
    unit="tonnes",
    element="Export Quantity",
):
    try:
        print(f"Reading in trade data for all items in {region}...")
        trad = pd.read_pickle(
            f"data{os.sep}temp_files{os.sep}Trade_DetailedTradeMatrix_E_{region}.pkl"
        )
    except FileNotFoundError:
        print(f"Pickled Data in {region} not found. Reading zip to create pickle.")
        trade_zip = (
            f"data{os.sep}data_raw{os.sep}Trade_DetailedTradeMatrix_E_{region}.zip"
        )
        serialise_faostat_bulk(trade_zip)
        print("Serialisation complete. Run the script again.")
        return

    trad = trad[
        ((trad["Unit"] == unit) & (trad["Element"] == element) & (~trad[year].isna()))
    ]
    trad = trad[["Reporter Country Code (M49)", "Partner Country Code (M49)", year]]
    print("Finished filtering trade matrix")
    print("Pivot trade matrix")
    trad = pd.pivot_table(
        trad,
        columns="Partner Country Code (M49)",
        index="Reporter Country Code (M49)",
        values=year,
        aggfunc="sum",
    )
    print("Finished pivoting trade matrix")

    # Remove entries which are not countries
    trad = remove_entries_from_data(trad)
    return trad


def compute_total_prod_trad(region="All_Data", year="Y2020"):
    p = get_prod(region=region, year=year)
    t = get_trad(region=region, year=year)
    p, t = _unify_indices(p, t)

    m49names = pd.read_csv("data/data_raw/m49.csv")
    m49names = (
        m49names[["m49", "country_name_en"]]
        .set_index("m49")
        .to_dict()["country_name_en"]
    )
    p.index = [m49names[int(c[1:])] for c in p.index]
    t.index = [m49names[int(c[1:])] for c in t.index]
    t.columns = t.index

    p.to_csv("data/preprocessed_data/integrated_model/total_food_production.csv")
    t.to_csv("data/preprocessed_data/integrated_model/total_food_export.csv")
    return p, t


class RHSOfTrade(PTS):
    def __init__(self, trade_table, production_table) -> None:
        self.trade_matrix = trade_table
        self.production_data = production_table.squeeze()
        self.no_trade_removed = False
        self.prebalanced = False
        self.reexports_corrected = False
        self.remove_net_zero_countries()
        # Prebalance the trade matrix
        self.prebalance()
        # Remove re-exports
        self.correct_reexports()
        # Set diagonal to zero
        np.fill_diagonal(self.trade_matrix.values, 0)
        self.rhs = self.get_rhs()

    def get_rhs(self):
        right_stochastic_matrix = self.trade_matrix.copy()
        right_stochastic_matrix = right_stochastic_matrix.div(
            right_stochastic_matrix.sum(axis=1), axis=0
        )
        right_stochastic_matrix.fillna(0, inplace=True)
        return right_stochastic_matrix


if __name__ == "__main__":
    try:
        print("loading prod,trade data...")
        p = pd.read_csv(
            "data/preprocessed_data/integrated_model/total_food_production.csv",
            index_col=0,
        )
        t = pd.read_csv(
            "data/preprocessed_data/integrated_model/total_food_export.csv", index_col=0
        )
    except FileNotFoundError:
        print("prod, trade data not found, computing...")
        p, t = compute_total_prod_trad()

    rot = RHSOfTrade(t, p)
    print(rot.rhs)
