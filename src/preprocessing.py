import pandas as pd
import os


def read_in_raw_trade_matrix(testing=False):
    """
    Reads in the raw trade matrix from the data folder and returns it as a pandas dataframe.
    The raw trade matrix can be found at: https://www.fao.org/faostat/en/#data/TM
    Select "All Data" for download.

    Arguments:
        testing (bool): Checks to only use a subset of the data for testing purposes.

    Returns:
        trade_matrix (pd.DataFrame): The raw trade matrix as a pandas dataframe.
    """
    if testing:
        trade_matrix = pd.read_csv(
            "." + os.sep +
            "data" + os.sep +
            "oceania_only_for_testing" +
            os.sep +
            "Trade_DetailedTradeMatrix_E_Oceania.csv",
            encoding="latin-1",
            low_memory=False)
    else:
        trade_matrix = pd.read_csv(
            "." + os.sep +
            "data" + os.sep +
            "Trade_DetailedTradeMatrix_E_All_Data" +
            os.sep +
            "Trade_DetailedTradeMatrix_E_All_Data.csv",
            encoding="latin-1",
            low_memory=False)
    return trade_matrix


def extract_relevant_data(
        trade_matrix,
        year=2021,
        items=["Maize (corn)", "Wheat", "Rice, paddy (rice milled equivalent)"]
):
    """
    Extracts only the relevant things we need for building the trade model.

    Arguments:
        trade_matrix (pd.DataFrame): The raw trade matrix.
        year (int): The year to extract data for.
        items (list): The items to extract data for. These are the trade goods we
            are interested in. You can change these if you are interested in
            other commodities.

    Returns:
        trade_matrix (pd.DataFrame): The cleaned trade matrix.
    """
    # Filter out the items we are interested in.
    relevant_columns = [
        "Reporter Countries",
        "Partner Countries",
        "Element",
        "Item",
        "Unit",
        "Y" + str(year)
    ]
    trade_matrix = trade_matrix[relevant_columns]
    trade_matrix = trade_matrix[trade_matrix["Item"].isin(items)]
    # only use imports and exports quantities
    trade_matrix = trade_matrix[
        trade_matrix["Element"].isin(["Import Quantity", "Export Quantity"])
    ]
    # drop all the are nan in the given year
    trade_matrix = trade_matrix.dropna(subset=["Y" + str(year)])

    # Save the trade matrix to a csv file to the data folder.
    trade_matrix.to_csv(
        "." + os.sep +
        "data" + os.sep +
        "trade_matrix_only_relevant_" + str(year) + ".csv",
        index=False)

    return trade_matrix


if __name__ == "__main__":
    trade_matrix = read_in_raw_trade_matrix(testing=False)
    trade_matrix = extract_relevant_data(trade_matrix)

    print(trade_matrix.head())
    print(trade_matrix.columns)
