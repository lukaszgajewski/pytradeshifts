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
    if testing:
        trade_data = pd.read_csv(
            "." + os.sep +
            "data" + os.sep +
            "oceania_only_for_testing" +
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



def extract_relevant_data(trade_data, year=2021, items=["Maize (corn)", "Wheat", "Rice, paddy (rice milled equivalent)"]):
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
    
    # Save the trade matrix to a CSV file in the data folder
    file_name = f"data{os.sep}trade_data_only_relevant_{year}.csv"
    trade_data.to_csv(file_name, index=False)
    
    return trade_data

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
    trade_data = read_in_raw_trade_data(testing=False)
    trade_data = extract_relevant_data(trade_data, year=2018)

    print(trade_data.head())
    print(trade_data.columns)
