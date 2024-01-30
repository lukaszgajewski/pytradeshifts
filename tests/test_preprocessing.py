import pandas as pd
from src.preprocessing import format_prod_trad_data
from src.preprocessing import rename_countries
import os
import pytest


@pytest.mark.skipif(os.getenv('GITHUB_ACTIONS') == 'true', reason='Skipping test for Github Actions')
def test_format_prod_trad_data_oceania():
    region = "Oceania"
    production, trade = format_prod_trad_data(
        f"data{os.sep}temp_files{os.sep}Production_Crops_Livestock_E_Oceania.pkl",
        f"data{os.sep}temp_files{os.sep}Trade_DetailedTradeMatrix_E_Oceania.pkl",
        item="Wheat",
    )
    production_from_R = pd.read_csv(
        f"data{os.sep}validation_data_from_Hedlung_2022{os.sep}"
        f"{region}{os.sep}NEW_production_wheat_2018.csv"
    )[["Area", "value"]]

    production_from_R.set_index("Area", inplace=True)
    production_from_R.sort_index(inplace=True)
    production_from_R = production_from_R.squeeze()

    trade_from_R = pd.read_csv(
        f"data{os.sep}validation_data_from_Hedlung_2022{os.sep}"
        f"{region}{os.sep}NEW_trade_wheat_2018.csv"
    )

    trade_from_R.drop(columns="Unnamed: 0", inplace=True)
    trade_from_R.columns = [int(c) for c in trade_from_R.columns]
    trade_from_R.index = trade_from_R.columns
    trade_from_R = trade_from_R[sorted(trade_from_R.columns)]
    trade_from_R.sort_index(inplace=True)

    # Replace country codes with country names in the R files
    trade_from_R = rename_countries(
        trade_from_R, region, "Trade_DetailedTradeMatrix_E", "Area Code"
    )
    production_from_R = rename_countries(
        production_from_R, region, "Production_Crops_Livestock_E", "Area Code"
    )
    trade = rename_countries(
        trade,
        region,
        "Trade_DetailedTradeMatrix_E",
    )
    production = rename_countries(
        production,
        region,
        "Production_Crops_Livestock_E",
    )

    trade.sort_index(inplace=True)
    production.sort_index(inplace=True)
    trade_from_R.sort_index(inplace=True)
    production_from_R.sort_index(inplace=True)
    trade = trade[sorted(trade.columns)]
    trade_from_R = trade_from_R[sorted(trade_from_R.columns)]

    print(production_from_R.shape)
    print(production.shape)

    assert production_from_R.shape == production.shape
    assert trade_from_R.shape == trade.shape

    assert production_from_R.sum() == production.sum()
    assert trade_from_R.sum().sum() == trade.sum().sum()

    assert production_from_R.index.equals(production.index)
    assert trade_from_R.index.equals(trade.index)
    assert trade_from_R.columns.equals(trade.columns)

    assert (production_from_R == production).all(axis=None)
    assert (trade_from_R == trade).all(axis=None)


@pytest.mark.skipif(os.getenv('GITHUB_ACTIONS') == 'true', reason='Skipping test for Github Actions')
def test_format_prod_trad_data_global():
    region = "All_Data"

    try:
        production = pd.read_csv(
            f"data{os.sep}preprocessed_data{os.sep}Wheat_Y2018_Global_production.csv",
            index_col=0,
        ).squeeze()
        trade = pd.read_csv(
            f"data{os.sep}preprocessed_data{os.sep}Wheat_Y2018_Global_trade.csv",
            index_col=0,
        )
    except FileNotFoundError:
        print("Files not found, loading from pickle files to redo preprocessing")
        production, trade = format_prod_trad_data(
            f"data{os.sep}temp_files{os.sep}Production_Crops_Livestock_E_All_Data.pkl",
            f"data{os.sep}temp_files{os.sep}Trade_DetailedTradeMatrix_E_All_Data.pkl",
            item="Wheat",
        )

    production_from_R = pd.read_csv(
        f"data{os.sep}validation_data_from_Hedlung_2022{os.sep}"
        f"{region}{os.sep}NEW_production_wheat_2018.csv"
    )[["Area", "value"]]

    production_from_R.set_index("Area", inplace=True)
    production_from_R.sort_index(inplace=True)
    production_from_R = production_from_R.squeeze()

    trade_from_R = pd.read_csv(
        f"data{os.sep}validation_data_from_Hedlung_2022{os.sep}"
        f"{region}{os.sep}NEW_trade_wheat_2018.csv"
    )
    trade_from_R.drop(columns="Unnamed: 0", inplace=True)
    trade_from_R.columns = [int(c) for c in trade_from_R.columns]
    trade_from_R.index = trade_from_R.columns
    trade_from_R = trade_from_R[sorted(trade_from_R.columns)]
    trade_from_R.sort_index(inplace=True)

    # Replace country codes with country names in the R files
    trade_from_R = rename_countries(
        trade_from_R, region, "Trade_DetailedTradeMatrix_E", "Area Code"
    )
    production_from_R = rename_countries(
        production_from_R, region, "Production_Crops_Livestock_E", "Area Code"
    )

    # print all the countries that only exist in one of the dataframes
    for c in production_from_R.index:
        if c not in production.index:
            print("This country is present in R, but not in Python:")
            print(c)

    # and now the other way around
    for c in production.index:
        if c not in production_from_R.index:
            print("This country is present in Python, but not in R:")
            print(c)

    # Sort the Python and R files to make sure they are in the same order
    production.sort_index(inplace=True)
    trade.sort_index(inplace=True)
    trade_from_R.sort_index(inplace=True)
    production_from_R.sort_index(inplace=True)

    # Also sort the columns of the trade dataframes
    trade = trade[sorted(trade.columns)]
    trade_from_R = trade_from_R[sorted(trade_from_R.columns)]

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
    test_format_prod_trad_data_oceania()
    test_format_prod_trad_data_global()
