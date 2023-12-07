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
        
    print("In test mode.= " + str(testing))
    print("Finished reading in raw trade data.")
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

    print("Finished reading in raw production data.")
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

    print("Finished extracting relevant trade data.")

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

    print("Finished extracting relevant production data.")

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


def read_faostat_bulk(faostat_zip: str) -> pd.DataFrame:
    """https://rdrr.io/cran/FAOSTAT/src/R/faostat_bulk_download.R#sym-read_faostat_bulk"""
    zip_file = ZipFile(faostat_zip)
    return pd.read_csv(
        zip_file.open(faostat_zip[faostat_zip.rfind("/") + 1 :].replace("zip", "csv")),
        encoding="latin1",
        low_memory=False,
    )


def serialise_faostat_bulk(faostat_zip: str) -> None:
    data = read_faostat_bulk(faostat_zip)
    data.to_pickle(faostat_zip.replace("zip", "pkl"))
    return None


def _melt_year_cols(data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    data = data[
        [c for c in data.columns if c[0] != "Y" or (c[-1] != "F" and c[-1] != "N")]
    ]
    return data.melt(
        id_vars=[c for c in data.columns if c[0] != "Y"],
        var_name="Year",
        value_name="Value",
    ).dropna(subset="Value")


def _prep_trad_mat(
    trad_pkl: str, item="Wheat", unit="tonnes", element="Export Quantity", year="Y2021"
) -> pd.DataFrame:
    trad = pd.read_pickle(trad_pkl)
    trad = _melt_year_cols(trad)
    trad = trad[
        (
            (trad["Item"] == item)
            & (trad["Unit"] == unit)
            & (trad["Element"] == element)
            & (trad["Year"] == year)
        )
    ]
    trad = trad[["Reporter Country Code", "Partner Country Code", "Value"]]
    trad = trad.pivot(
        columns="Partner Country Code", index="Reporter Country Code", values="Value"
    )
    return trad


def _prep_prod_vec(prod_pkl: str, item="Wheat", unit="t", year="Y2021") -> pd.DataFrame:
    prod = pd.read_pickle(prod_pkl)
    prod = _melt_year_cols(prod)
    prod = prod[
        ((prod["Item"] == item) & (prod["Unit"] == unit) & (prod["Year"] == year))
    ]
    prod = prod[["Area Code", "Value"]]
    prod = prod.set_index("Area Code")
    return prod


def _unify_indices(
    prod_vec: pd.DataFrame, trad_mat: pd.DataFrame
) -> tuple[pd.Series, pd.DataFrame]:
    index = trad_mat.index.union(trad_mat.columns).union(prod_vec.index)
    index = index.sort_values()
    trad_mat = trad_mat.reindex(index=index, columns=index).fillna(0)
    prod_vec = prod_vec.reindex(index=index).fillna(0)
    prod_vec = prod_vec.squeeze()
    return prod_vec, trad_mat


def format_prod_trad_data(
    prod_pkl: str,
    trad_pkl: str,
    item="Wheat",
    prod_unit="t",
    trad_unit="tonnes",
    element="Export Quantity",
    year="Y2021",
) -> tuple[pd.Series, pd.DataFrame]:
    prod_vec = _prep_prod_vec(prod_pkl, item, prod_unit, year)
    trad_mat = _prep_trad_mat(trad_pkl, item, trad_unit, element, year)
    return _unify_indices(prod_vec, trad_mat)




if __name__ == "__main__":
    # Define values
    year = 2018
    items_trade = ["Maize (corn)", "Wheat", "Rice, paddy (rice milled equivalent)"]
    items_production = ["Maize (corn)", "Wheat", "Rice"]

    # Read in raw trade data
    trade_data = read_in_raw_trade_data(testing=False)
    trade_data = extract_relevant_trade_data(trade_data, items_trade, year=year)

    # Read in raw production data
    production_data = read_in_raw_production_data()
    production_data = extract_relevant_production_data(production_data, items_production, year=year)
