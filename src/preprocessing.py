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
    print("Unzipping file")
    zip_file = ZipFile(faostat_zip)
    print("Finished unzipping file")
    print("Reading csv from zip")
    df = pd.read_csv(
        zip_file.open(faostat_zip[faostat_zip.rfind("/") + 1:].replace("zip", "csv")),
        encoding="latin1",
        low_memory=False,
    )
    print("Finished reading csv from zip")
    return df


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
    print("Starting to convert zip to pickle")
    data.to_pickle(
        f"data{os.sep}temp_files{os.sep}{faostat_zip[faostat_zip.rfind('/') + 1:].replace(
            'zip', 'pkl'
        )}"
    )
    print("Finished converting zip to pickle")
    return None


def _melt_year_cols(data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """
    Filter out unnecessary columns from the data and melt the year columns.

    Arguments:
        data (pd.Series | pd.DataFrame): The data to be melted.

    Returns:
        pd.Series | pd.DataFrame: The melted data.
    """
    print("Melt year columns")
    # there are columns of format: Y2021N, Y2021F, Y2021
    # where 2021 can be any year. We only want to keep of format Y2021
    data = data[
        [c for c in data.columns if c[0] != "Y" or (c[-1] != "F" and c[-1] != "N")]
    ]
    # and then we want to melt all those year columns (Y2019, Y2020, Y2021 etc.)
    # so that we have a "Year" and "Value" columns
    # there are other ways of handling this but this is consistent with Croft et al.
    # Though this implementation floods the RAM, sorry about that.
    melted = data.melt(
        id_vars=[c for c in data.columns if c[0] != "Y"],
        var_name="Year",
        value_name="Value",
    ).dropna(subset="Value")
    print("Finished melting year columns")
    return melted


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
    print("Filter trade matrix")
    trad = trad[
        (
            (trad["Item"] == item)
            & (trad["Unit"] == unit)
            & (trad["Element"] == element)
            & (trad["Year"] == year)
        )
    ]
    trad = trad[["Reporter Country Code (M49)", "Partner Country Code (M49)", "Value"]]
    print("Finished filtering trade matrix")
    print("Pivot trade matrix")
    trad = trad.pivot(
        columns="Partner Country Code (M49)", index="Reporter Country Code (M49)", values="Value"
    )
    print("Finished pivoting trade matrix")

    # Remove entries which are not countries
    trad = remove_entries_from_data(trad)

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
    print("Filter production vector")
    prod = prod[
        ((prod["Item"] == item) & (prod["Unit"] == unit) & (prod["Year"] == year))
    ]
    prod = prod[["Area Code (M49)", "Value"]]
    print("Finished filtering production vector")
    prod = prod.set_index("Area Code (M49)")

    # Remove entries which are not countries
    prod = remove_entries_from_data(prod)

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
    print("Unify indices")
    index = trade_matrix.index.union(trade_matrix.columns).union(production_vector.index)
    index = index.sort_values()
    trade_matrix = trade_matrix.reindex(index=index, columns=index).fillna(0)
    production_vector = production_vector.reindex(index=index).fillna(0)
    production_vector = production_vector.squeeze()
    print("Finished unifying indices")
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
    print("Started to format production data")
    production_vector = _prep_production_vector(production_pkl, item, production_unit, year)
    print("Finished formatting production data\n")
    print("Started to format trade data")
    trade_matrix = _prep_trade_matrix(trade_pkl, item, trade_unit, element, year)
    print("Finished formatting trade data\n")
    return _unify_indices(production_vector, trade_matrix)


def rename_countries(
        data: pd.Series | pd.DataFrame,
        region: str,
        filename: str,
        code_type: str = "M49 Code",
) -> pd.DataFrame:
    """
    Rename country codes with country names in either production or trade data.

    Arguments:
        data (pd.DataFrame): The data to be renamed.
        region (str): The region of the data.
        filename (str): The filename for the country codes CSV file.
        code_type (str): The type of country code to be used.
        after_union (bool): Whether the index/columns of the data are already unified.

    Returns:
        pd.DataFrame: The data with country codes replaced by country names.
    """
    # Read in the country codes from the zip file
    faostat_zip = f"data{os.sep}data_raw{os.sep}{filename}_{region}.zip"
    zip_file = ZipFile(faostat_zip)
    # Open the csv file in the zip Production_Crops_Livestock_E_AreaCodes.csv
    # and read it into a dataframe
    codes = pd.read_csv(
        zip_file.open(filename + "_AreaCodes.csv"),
        encoding="latin1",
        low_memory=False,
    )
    # Create a dictionary with the country codes as keys and country names as values
    codes_dict = dict(zip(codes[code_type], codes["Area"]))

    print(f"Replacing country codes with country names in {filename.split('_')[0]} data")
    for code in data.index:
        data.rename(index={code: codes_dict[code]}, inplace=True)

    if isinstance(data, pd.DataFrame):
        for code in data.columns:
            data.rename(columns={code: codes_dict[code]}, inplace=True)

    return data


def remove_entries_from_data(data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """
    Removes a bunch of entries from the data, which do not actually represent countries
    or where no trade data is available.

    Arguments:
        data (pd.Series | pd.DataFrame): The data to be filtered.

    Returns:
        pd.Series | pd.DataFrame: The filtered data.
    """
    entries_to_remove = {
        # Groups of countries by region
        "World": "'001",
        "Africa": "'002",
        "South America": "'005",
        "Oceania": "'009",
        "Western Africa": "'011",
        "Central America": "'013",
        "Eastern Africa": "'014",
        "Northern Africa": "'015",
        "Middle Africa": "'017",
        "Southern Africa": "'018",
        "Americas": "'019",
        "Northern America": "'021",
        "Eastern Asia": "'030",
        "Southern Asia": "'034",
        "South-eastern Asia": "'035",
        "Southern Europe": "'039",
        "Australia and New Zealand": "'053",
        "Melanesia": "'054",
        "European Union (27)": "'097",
        "Asia": "'142",
        "Central Asia": "'143",
        "Western Asia": "'145",
        "Europe": "'150",
        "Eastern Europe": "'151",
        "Northern Europe": "'154",
        "Western Europe": "'155",
        # Groups of countries by property
        "Least Developed Countries": "'199",
        "Land Locked Developing Countries": "'432",
        "Small Island Developing States": "'722",
        "Low Income Food Deficit Countries": "'901",
        "Net Food Importing Developing Countries": "'902",
        # We want to look at China and Taiwan seperately, so this is not needed
        # as 159 refers to China incl. Taiwan
        "China": "'159",
        # No reliable data available, therefore they are excluded
        # This assessment is based on the fact that all of these countries
        # don't show up as a partner country in the trade data of the FAO
        "Democratic People's Republic of Korea": "'408",
        "Chad": "'148",
        "South Sudan": "'728",
    }

    # Remove the entries
    data = data[~data.index.isin(entries_to_remove.values())]
    if isinstance(data, pd.DataFrame):
        data = data.loc[:, ~data.columns.isin(entries_to_remove.values())]

    return data


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
        # Replace the zip with the pickle and link to the temp files folder
        production_pkl = production_zip.replace("zip", "pkl")
        trade_pkl = trade_zip.replace("zip", "pkl")
        production_pkl = production_pkl.replace("data_raw", "temp_files")
        trade_pkl = trade_pkl.replace("data_raw", "temp_files")

        production, trade_matrix = format_prod_trad_data(
            production_pkl,
            trade_pkl,
            item,
            production_unit,
            trade_unit,
            element,
            year,
        )

    # Replace country codes with country names
    trade_matrix = rename_countries(trade_matrix, region, "Trade_DetailedTradeMatrix_E")
    production = rename_countries(production, region, "Production_Crops_Livestock_E")

    # Rename the item for readability
    item = rename_item(item)

    # Make sure that production index and trade matrix index/columns are the same
    assert production.index.equals(trade_matrix.index)
    assert production.index.equals(trade_matrix.columns)

    # Replace "All_Data" with "global" for readability
    if region == "All_Data":
        region = "Global"

    # Save to CSV
    production.to_csv(f"data{os.sep}preprocessed_data{os.sep}{item}_{year}_{region}_production.csv")
    trade_matrix.to_csv(f"data{os.sep}preprocessed_data{os.sep}{item}_{year}_{region}_trade.csv")


if __name__ == "__main__":
    # Define values
    year = "Y2021"
    items_trade = ["Maize (corn)", "Wheat", "Rice, paddy (rice milled equivalent)"]
    # Define regions for which the data is processed
    # "Oceania" is used for testing, as it has the least amount of countries
    # to run with all data use: "All_Data" for region
    region = "Oceania"
    print("\n")
    for item in items_trade:
        main(
            region,
            item,
            year=year,
        )
        print("\n\n")
