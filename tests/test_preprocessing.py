import pandas as pd
from src.preprocessing import format_prod_trad_data
from src.preprocessing import rename_countries
import os


def call_format_prod_trad_data_for_testing(region):
    if region == "Oceania":
        production, trade = format_prod_trad_data(
            f"data{os.sep}temp_files{os.sep}Production_Crops_Livestock_E_Oceania.pkl",
            f"data{os.sep}temp_files{os.sep}Trade_DetailedTradeMatrix_E_Oceania.pkl",
            item="Wheat",
        )
    # elif region == "All Data":
    #     try:
    #         production = pd.read_csv(f"data{os.sep}preprocessed_data{os.sep}production.csv")
    #     except FileNotFoundError:
    #         production, trade = format_prod_trad_data(
    #             f"data{os.sep}temp_files{os.sep}Production_Crops_Livestock_E_All_Data.pkl",
    #             f"data{os.sep}temp_files{os.sep}Trade_DetailedTradeMatrix_E_All_Data.pkl",
    #             item="Wheat",
    #         )
    production_from_R = pd.read_csv(
        f"data{os.sep}validation_data_from_Hedlung_2022{os.sep}oceania{os.sep}NEW_production_wheat_2021.csv"
    )[["Area", "value"]]
    production_from_R.set_index("Area", inplace=True)
    production_from_R.sort_index(inplace=True)
    production_from_R = production_from_R.squeeze()
    trade_from_R = pd.read_csv(
        f"data{os.sep}validation_data_from_Hedlung_2022{os.sep}oceania{os.sep}NEW_trade_wheat_2021.csv"
    )
    trade_from_R.drop(columns="Unnamed: 0", inplace=True)
    trade_from_R.columns = [int(c) for c in trade_from_R.columns]
    trade_from_R.index = trade_from_R.columns
    trade_from_R = trade_from_R[sorted(trade_from_R.columns)]
    trade_from_R.sort_index(inplace=True)

    # Replace country codes with country names in the R files
    trade_from_R = rename_countries(
        trade_from_R,
        "Oceania",
        "Trade_DetailedTradeMatrix_E",
        "Area Code"
    )
    production_from_R = rename_countries(
        production_from_R,
        "Oceania",
        "Production_Crops_Livestock_E",
        "Area Code"
    )

    trade = rename_countries(
        trade,
        "Oceania",
        "Trade_DetailedTradeMatrix_E",
    )
    production = rename_countries(
        production,
        "Oceania",
        "Production_Crops_Livestock_E",
    )

    trade.sort_index(inplace=True)
    production.sort_index(inplace=True)
    trade_from_R.sort_index(inplace=True)
    production_from_R.sort_index(inplace=True)
    trade = trade[sorted(trade.columns)]
    trade_from_R = trade_from_R[sorted(trade_from_R.columns)]

    return production, trade, production_from_R, trade_from_R


def test_format_prod_trad_data_oceania():
    production, trade, production_from_R, trade_from_R = call_format_prod_trad_data_for_testing(
        "Oceania"
    )
    # Make sure the series/dataframes are the same

    assert production_from_R.shape == production.shape
    assert trade_from_R.shape == trade.shape

    assert production_from_R.sum() == production.sum()
    assert trade_from_R.sum().sum() == trade.sum().sum()

    assert production_from_R.index.equals(production.index)
    assert trade_from_R.index.equals(trade.index)
    assert trade_from_R.columns.equals(trade.columns)

    assert (production_from_R == production).all(axis=None)
    assert (trade_from_R == trade).all(axis=None)


if __name__ == "__main__":
    test_format_prod_trad_data()
