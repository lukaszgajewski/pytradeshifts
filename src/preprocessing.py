import pandas as pd
import os
from zipfile import ZipFile

"""
Data for this project is downloaded from FAOSTAT.
Trade data is downloaded from:
http://www.fao.org/faostat/en/#data/TM
Production data is downloaded from:
http://www.fao.org/faostat/en/#data/QC
"""


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
    """
    Return pandas.DataFrame containing FAOSTAT data extracted from a bulk
    download zip file.
    This is based on the following R implementation:
    https://rdrr.io/cran/FAOSTAT/src/R/faostat_bulk_download.R#sym-read_faostat_bulk

    Arguments:
        faostat_zip (str): Path to the FAOSTAT zip file.

    Returns:
        pd.DataFrame: The FAOSTAT data.
    """
    zip_file = ZipFile(faostat_zip)
    return pd.read_csv(
        zip_file.open(faostat_zip[faostat_zip.rfind("/") + 1:].replace("zip", "csv")),
        encoding="latin1",
        low_memory=False,
    )


def serialise_faostat_bulk(faostat_zip: str) -> None:
    """
    Read FAOSTAT data from a bulk download zip file as a pandas.DataFrame,
    and save it as a pickle to allow for faster loading in the future.

    Arguments:
        faostat_zip (str): Path to the FAOSTAT zip file.

    Returns:
        None
    """
    data = read_faostat_bulk(faostat_zip)
    data.to_pickle(
        f"data{os.sep}temp_files{os.sep}{faostat_zip[faostat_zip.rfind('/') + 1:].replace(
            'zip', 'pkl'
        )}"
    )
    return None


def _melt_year_cols(data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """
    Filter out unnecessary columns from the data and melt the year columns.

    Arguments:
        data (pd.Series | pd.DataFrame): The data to be melted.

    Returns:
        pd.Series | pd.DataFrame: The melted data.
    """
    # there are columns of format: Y2021N, Y2021F, Y2021
    # where 2021 can be any year. We only want to keep of format Y2021
    data = data[
        [c for c in data.columns if c[0] != "Y" or (c[-1] != "F" and c[-1] != "N")]
    ]
    # and then we want to melt all those year columns (Y2019, Y2020, Y2021 etc.)
    # so that we have a "Year" and "Value" columns
    # there are other ways of handling this but this is consistent with Croft et al.
    return data.melt(
        id_vars=[c for c in data.columns if c[0] != "Y"],
        var_name="Year",
        value_name="Value",
    ).dropna(subset="Value")


def _prep_trade_matrix(
    trade_pkl: str, item: str, unit="tonnes", element="Export Quantity", year="Y2021"
) -> pd.DataFrame:
    """
    Return properly formatted trade matrix.

    Arguments:
        trade_pkl (str): Path to the trade matrix pickle file.
        item (str): Item to filter for.
        unit (str): Unit to filter for.
        element (str): Element to filter for.
        year (str): Year to filter for.

    Returns:
        pd.DataFrame: The trade matrix.

    Notes:
        The optional arguments must be determined semi-manually as their allowed values
        depend on particular datasets. E.g., unit can be "tonnes" in one file and "t"
        in another.
    """
    trad = pd.read_pickle(trade_pkl)
    trad = _melt_year_cols(trad)
    trad = trad[
        (
            (trad["Item"] == item)
            & (trad["Unit"] == unit)
            & (trad["Element"] == element)
            & (trad["Year"] == year)
        )
    ]
    trad = trad[["Reporter Country Code (M49)", "Partner Country Code (M49)", "Value"]]
    trad = trad.pivot(
        columns="Partner Country Code (M49)", index="Reporter Country Code (M49)", values="Value"
    )
    # Remoev the ' from the index and columns
    trad.index = trad.index.str.replace("'", "")
    trad.columns = trad.columns.str.replace("'", "")

    return trad


def _prep_production_vector(
        production_pkl: str, item="Wheat", unit="t", year="Y2021"
) -> pd.DataFrame:
    """
    Return properly formatted production vector.

    Arguments:
        production_pkl (str): Path to the production vector pickle file.
        item (str): Item to filter for.
        unit (str): Unit to filter for.
        year (str): Year to filter for.

    Returns:
        pd.DataFrame: The production vector.

    Notes:
        The optional arguments must be determined semi-manually as their allowed values
        depend on particular datasets. E.g., unit can be "tonnes" in one file and "t"
        in another.
    """
    prod = pd.read_pickle(production_pkl)
    prod = _melt_year_cols(prod)
    prod = prod[
        ((prod["Item"] == item) & (prod["Unit"] == unit) & (prod["Year"] == year))
    ]
    prod = prod[["Area Code (M49)", "Value"]]
    prod = prod.set_index("Area Code (M49)")
    # Remoev the ' from the index
    prod.index = prod.index.str.replace("'", "")
    return prod


def _unify_indices(
    production_vector: pd.DataFrame, trade_matrix: pd.DataFrame
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Return the production (as a Series) and trade matrix (DataFrame) with
    unified (i.e., such that they match each other),
    and sorted indices/columns.
    Missing values are replaced by 0.

    Arguments:
        production_vector (pd.DataFrame): The production vector.
        trade_matrix (pd.DataFrame): The trade matrix.

    Returns:
        tuple[pd.Series, pd.DataFrame]: The production vector and trade matrix
            with unified indices/columns.
    """
    index = trade_matrix.index.union(trade_matrix.columns).union(production_vector.index)
    index = index.sort_values()
    trade_matrix = trade_matrix.reindex(index=index, columns=index).fillna(0)
    production_vector = production_vector.reindex(index=index).fillna(0)
    production_vector = production_vector.squeeze()
    return (production_vector, trade_matrix)


def format_prod_trad_data(
    production_pkl: str,
    trade_pkl: str,
    item: str,
    production_unit="t",
    trade_unit="tonnes",
    element="Export Quantity",
    year="Y2021",
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Return properly formatted production vector (as a Series),
    and trade matrix (DataFrame).

    Arguments:
        production_pkl (str): Path to the production vector pickle file.
        trade_pkl (str): Path to the trade matrix pickle file.
        item (str): Item to filter for.
        production_unit (str): Unit to filter for in the production vector.
        trade_unit (str): Unit to filter for in the trade matrix.
        element (str): Element to filter for in the trade matrix.
        year (str): Year to filter for.

    Returns:
        tuple[pd.Series, pd.DataFrame]: The production vector and trade matrix.

    Notes:
        The optional arguments must be determined semi-manually as their allowed values
        depend on particular datasets. E.g., unit can be "tonnes" in one file and "t"
        in another.
    """
    production_vector = _prep_production_vector(production_pkl, item, production_unit, year)
    trade_matrix = _prep_trade_matrix(trade_pkl, item, trade_unit, element, year)
    return _unify_indices(production_vector, trade_matrix)


def main(
    region: str,
    item: str,
    production_unit="t",
    trade_unit="tonnes",
    element="Export Quantity",
    year="Y2021",
) -> pd.DataFrame:
    try:
        print(f"Reading in data for {item} in {region}...")
        production_pkl = f"data{os.sep}temp_files{os.sep}Production_Crops_Livestock_E_{region}.pkl"
        trade_pkl = f"data{os.sep}temp_files{os.sep}Trade_DetailedTradeMatrix_E_{region}.pkl"
        production, trade_matrix = format_prod_trad_data(
            production_pkl,
            trade_pkl,
            item,
            production_unit,
            trade_unit,
            element,
            year,
        )
    except FileNotFoundError:
        print(
            f"Data for {item} in {region} in pickle format not found. Reading zip to create pickle"
        )
        production_zip = f"data{os.sep}data_raw{os.sep}Production_Crops_Livestock_E_{region}.zip"
        trade_zip = f"data{os.sep}data_raw{os.sep}Trade_DetailedTradeMatrix_E_{region}.zip"
        serialise_faostat_bulk(production_zip)
        serialise_faostat_bulk(trade_zip)
        print(f"Pickles created. Reading in data for {item} in {region}...")
        production_pkl = production_zip.replace("zip", "pkl")
        trade_pkl = trade_zip.replace("zip", "pkl")
        production, trade_matrix = format_prod_trad_data(
            production_pkl,
            trade_pkl,
            item,
            production_unit,
            trade_unit,
            element,
            year,
        )
    # Replace the area codes with country names. This is based on the M49 codes
    # https://data.apps.fao.org/catalog/dataset/m49-code-list-global-region-country
    codes = pd.read_csv(
        f"data{os.sep}supplemental_data{os.sep}m49.csv"
    )
    # Create a dictionary of country codes and names with the m49 column as the key
    # and the country_names_en column as the value. This also adds two leading zeros
    # to the codes with only one character and one leading zero to the codes with
    # two characters. This is to match the codes in the production and trade matrix
    # which have two leading zeros. This is not the case for the codes in the m49
    # file for unknown reasons. Would be too much to ask for the UN to be consistent.
    # Convert the codes to strings to allow for the leading zeros
    codes_dict = {
        str(code).zfill(3): country_name
        for code, country_name in zip(codes["m49"], codes["country_name_en"])
    }

    # Go through the index of production and index/columns of trade matrix
    # and replace the codes with country names
    for code in production.index:
        production.rename(index={code: codes_dict[code]}, inplace=True)
    for code in trade_matrix.index:
        trade_matrix.rename(index={code: codes_dict[code]}, inplace=True)
        trade_matrix.rename(columns={code: codes_dict[code]}, inplace=True)

    # Rename the item for readability
    item = rename_item(item)

    # Save to CSV
    production.to_csv(f"data{os.sep}preprocessed_data{os.sep}{item}_{year}_production.csv")
    trade_matrix.to_csv(f"data{os.sep}preprocessed_data{os.sep}{item}_{year}_trade.csv")


if __name__ == "__main__":
    # Define values
    year = 2018
    items_trade = ["Maize (corn)", "Wheat", "Rice, paddy (rice milled equivalent)"]
    items_production = ["Maize (corn)", "Wheat", "Rice"]
    # Define regions for which the data is processed
    region = "Oceania"

    for item in items_trade:
        main(
            region,
            item,
        )
