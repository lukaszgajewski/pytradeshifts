import pandas as pd
import os


def read_in_raw_trade_data(testing=False):
    """
    Reads in the raw trade matrix from the data folder and returns it as a pandas dataframe.
    The raw trade matrix can be found at: https://www.fao.org/faostat/en/#data/TM
    Select "All Data" for download.

    Arguments:
        testing (bool): Checks to only use a subset of the data for testing purposes.

    Returns:
        trade_data (pd.DataFrame): The raw trade matrix as a pandas dataframe.
    """
    # Only use subset for testing purposes
    if testing:
        trade_data = pd.read_csv(
            "." + os.sep +
            "data" + os.sep +
            "Trade_DetailedTradeMatrix_E_Oceania" +
            os.sep +
            "Trade_DetailedTradeMatrix_E_Oceania.csv",
            encoding="latin-1",
            low_memory=False)
    else:
        trade_data = pd.read_csv(
            "." + os.sep +
            "data" + os.sep +
            "Trade_DetailedTradeMatrix_E_All_Data" +
            os.sep +
            "Trade_DetailedTradeMatrix_E_All_Data.csv",
            encoding="latin-1",
            low_memory=False)
    return trade_data


def read_in_raw_production_data():
    """
    Reads in the raw food production to be used later for the
    re-export algorithm.

    Returns:
        pd.DataFrame: The raw food production data.
    """
    production_data = pd.read_csv(
        "." + os.sep +
        "data" + os.sep +
        "Production_Crops_Livestock_E_All_Data" +
        os.sep +
        "Production_Crops_Livestock_E_All_Data_NOFLAG.csv",
        encoding="latin-1",
        low_memory=False)
    return production_data


def extract_relevant_trade_data(trade_data, items, year=2021):
    """
    Extracts only the relevant data needed for building the trade model.

    Args:
        trade_data (pd.DataFrame): The raw trade matrix.
        year (int): The year to extract data for.
        items (list): The items of interest, i.e., trade goods.

    Returns:
        pd.DataFrame: The cleaned trade matrix.
    """
    # Select only relevant columns
    relevant_columns = [
        "Reporter Countries",
        "Partner Countries",
        "Element",
        "Item",
        "Unit",
        f"Y{year}"
    ]

    trade_data = trade_data[relevant_columns]

    # Filter items of interest
    trade_data = trade_data[trade_data["Item"].isin(items)]

    # Filter for export quantities only
    trade_data = trade_data[trade_data["Element"] == "Export Quantity"]

    # Drop rows with NaN values for the given year
    trade_data = trade_data.dropna(subset=[f"Y{year}"])

    # Rename specific items for readability
    trade_data["Item"] = trade_data["Item"].apply(rename_item)

    # Rename year column to "Quantity"
    trade_data = trade_data.rename(columns={f"Y{year}": "Quantity"})

    # Save the trade matrix to a CSV file in the data folder
    file_name = f"data{os.sep}trade_data_only_relevant_{year}.csv"
    trade_data.to_csv(file_name, index=False)

    return trade_data


def extract_relevant_production_data(production_data, items, year=2021):
    """
    Extracts only the relevant data for the re-export algorithm.

    Args:
        production_data (pd.DataFrame): The raw production data.
        items (list): The items of interest, i.e., trade goods.
        year (int): The year to extract data for.

    Returns:
        pd.DataFrame: The cleaned production data.
    """
    # Select only relevant columns
    relevant_columns = [
        "Area",
        "Item",
        "Element",
        "Unit",
        f"Y{year}"
    ]

    production_data = production_data[relevant_columns]

    # Filter items of interest
    production_data = production_data[production_data["Item"].isin(items)]

    # Filter for quantities only
    production_data = production_data[production_data["Element"] == "Production"]

    # Drop rows with NaN values for the given year
    production_data = production_data.dropna(subset=[f"Y{year}"])

    # Rename specific items for readability
    production_data["Item"] = production_data["Item"].apply(rename_item)

    # Rename year column to "Production"
    production_data = production_data.rename(columns={f"Y{year}": "Production"})

    # Save the production data to a CSV file in the data folder
    file_name = f"data{os.sep}production_data_only_relevant_{year}.csv"
    production_data.to_csv(file_name, index=False)

    return production_data


def rename_item(item):
    """
    Renames specific item entries for readability.

    Args:
        item (str): The item name.

    Returns:
        str: The renamed item name.
    """
    item_renames = {
        "Maize (corn)": "Maize",
        "Rice, paddy (rice milled equivalent)": "Rice"
    }
    return item_renames.get(item, item)


if __name__ == "__main__":
    # Define values
    year = 2018
    items_trade = ["Maize (corn)", "Wheat", "Rice, paddy (rice milled equivalent)"]
    items_production = ["Maize (corn)", "Wheat", "Rice"]

    # Read in raw trade data
    trade_data = read_in_raw_trade_data(testing=False)
    trade_data = extract_relevant_trade_data(trade_data, items_trade, year=year)

    print(trade_data.head())
    print(trade_data.columns)

    # Read in raw production data
    production_data = read_in_raw_production_data()
    production_data = extract_relevant_production_data(production_data, items_production, year=year)

    print(production_data.head())
    print(production_data.columns)
