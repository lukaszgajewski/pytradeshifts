import pandas as pd
import os
import country_converter as coco
import logging
from zipfile import ZipFile
from time import time
from tqdm import tqdm

# This is just to keep the output clean, as otherwise the coco packages notifies
# you about every regex match that did not work
coco_logger = coco.logging.getLogger()
coco_logger.setLevel(logging.CRITICAL)

"""
Data for this project is downloaded from FAOSTAT.
Trade data is downloaded from:
http://www.fao.org/faostat/en/#data/TM
Production data is downloaded from:
http://www.fao.org/faostat/en/#data/QC
"""


def rename_item(item: str) -> str:
    """
    Renames specific item entries for readability.

    Args:
        item (str): The item name.

    Returns:
        str: The renamed item name.
    """
    item_renames = {
        # "Maize (corn)": "Maize",
        # "Rice, paddy (rice milled equivalent)": "Rice",
        "Swine / pigs": "Pig",
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
        zip_file.open(faostat_zip[faostat_zip.rfind("/") + 1 :].replace("zip", "csv")),
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
    base_path = "data" + os.sep + "temp_files" + os.sep
    formatted_filename = faostat_zip[faostat_zip.rfind("/") + 1 :].replace("zip", "pkl")
    full_path = f"{base_path}{formatted_filename}"

    data.to_pickle(full_path)
    print("Finished converting zip to pickle")
    return None


def _prep_trade_matrix(
    trade_pkl: str, item: str, unit="tonnes", element="Export Quantity", year="Y2018"
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
    print("Filter trade matrix")
    trad = trad[
        (
            (trad["Item"] == item)
            & (trad["Unit"] == unit)
            & (trad["Element"] == element)
            & (~trad[year].isna())
        )
    ]
    trad = trad[["Reporter Country Code (M49)", "Partner Country Code (M49)", year]]
    print("Finished filtering trade matrix")
    print("Pivot trade matrix")
    trad = trad.pivot(
        columns="Partner Country Code (M49)",
        index="Reporter Country Code (M49)",
        values=year,
    )
    print("Finished pivoting trade matrix")

    # Remove entries which are not countries
    trad = remove_entries_from_data(trad)

    return trad


def _prep_production_vector(
    production_pkl: str, item="Wheat", unit="t", year="Y2018"
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
    print("Filter production vector")
    prod = prod[
        ((prod["Item"] == item) & (prod["Unit"] == unit) & (~prod[year].isna()))
    ]
    prod = prod[["Area Code (M49)", year]]
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
    index = trade_matrix.index.union(trade_matrix.columns).union(
        production_vector.index
    )
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
    year="Y2018",
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
    production_vector = _prep_production_vector(
        production_pkl, item, production_unit, year
    )
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
) -> pd.Series | pd.DataFrame:
    """
    Rename country codes with country names in either production or trade data.

    Arguments:
        data (pd.Series | pd.DataFrame): The data to be renamed.
        region (str): The region of the data.
        filename (str): The filename for the country codes CSV file.
        code_type (str): The type of country code to be used.

    Returns:
        pd.Series | pd.DataFrame: The data with country codes replaced by country names.
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
    # Also rename China; Taiwan Province of to Taiwan, so we don't run into
    # problems later on with the country names
    codes.loc[codes["Area"] == "China; Taiwan Province of", "Area"] = "Taiwan"

    # Create a dictionary with the country codes as keys and country names as values
    cc = coco.CountryConverter()
    codes_area_short = cc.pandas_convert(pd.Series(codes["Area"]), to="name_short")

    codes_dict = dict(zip(codes[code_type], codes_area_short))

    print(
        f"Replacing country codes with country names in {filename.split('_')[0]} data"
    )
    data.rename(index=codes_dict, inplace=True)

    if isinstance(data, pd.DataFrame):
        data.rename(columns=codes_dict, inplace=True)

    return data


def remove_entries_from_data(
    data: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
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
    }

    # Remove the entries
    data = data[~data.index.isin(entries_to_remove.values())]
    if isinstance(data, pd.DataFrame):
        data = data.loc[:, ~data.columns.isin(entries_to_remove.values())]

    return data


def get_all_item_names(production_pkl):
    prod = pd.read_pickle(production_pkl)
    return prod["Item"].unique()


def main(
    region: str,
    production_unit="t",
    trade_unit="tonnes",
    element="Export Quantity",
    year="Y2020",
) -> None:
    try:
        print(f"Reading in data for all items in {region}...")
        production_pkl = (
            f"data{os.sep}temp_files{os.sep}Production_Crops_Livestock_E_{region}.pkl"
        )
        trade_pkl = (
            f"data{os.sep}temp_files{os.sep}Trade_DetailedTradeMatrix_E_{region}.pkl"
        )
    except FileNotFoundError:
        print(f"Pickled Data in {region} not found. Reading zip to create pickle.")
        production_zip = (
            f"data{os.sep}data_raw{os.sep}Production_Crops_Livestock_E_{region}.zip"
        )
        trade_zip = (
            f"data{os.sep}data_raw{os.sep}Trade_DetailedTradeMatrix_E_{region}.zip"
        )
        serialise_faostat_bulk(production_zip)
        serialise_faostat_bulk(trade_zip)
        print("Serialisation complete. Run the script again.")
        return

    items = get_all_item_names(production_pkl)
    # Replace "All_Data" with "global" for readability
    region_label = "Global" if region == "All_Data" else region

    for item in tqdm(items):
        p_f_name = f"data{os.sep}preprocessed_data{os.sep}integrated_model{os.sep}{item}_{year}_{region_label}_production.csv"
        t_f_name = f"data{os.sep}preprocessed_data{os.sep}integrated_model{os.sep}{item}_{year}_{region_label}_trade.csv"
        if os.path.isfile(p_f_name) and os.path.isfile(t_f_name):
            print(item, "files already exist, skipping.")
            continue
        production, trade_matrix = format_prod_trad_data(
            production_pkl,
            trade_pkl,
            item,
            production_unit,
            trade_unit,
            element,
            year,
        )
        try:
            # Replace country codes with country names
            trade_matrix = rename_countries(
                trade_matrix, region, "Trade_DetailedTradeMatrix_E"
            )
            production = rename_countries(
                production, region, "Production_Crops_Livestock_E"
            )
        except AttributeError as AE:
            # TODO: this happens for Tallowtree, Jojoba seed
            # reason is: only China, Mexico produces it, and nobody buys it
            # needs to be resolved manually later on
            print(item)
            print(production)
            print(trade_matrix)
            print(AE)
            continue

        # Rename the item for readability
        if "/" in item:
            item = rename_item(item)
            p_f_name = f"data{os.sep}preprocessed_data{os.sep}integrated_model{os.sep}{item}_{year}_{region_label}_production.csv"
            t_f_name = f"data{os.sep}preprocessed_data{os.sep}integrated_model{os.sep}{item}_{year}_{region_label}_trade.csv"

        # Make sure that production index and trade matrix index/columns are the same
        if not production.index.equals(trade_matrix.index):
            print("index != index", item)
            continue
        if not production.index.equals(trade_matrix.columns):
            print("index != columns", item)
            continue

        # Save to CSV
        try:
            production.to_csv(p_f_name)
            trade_matrix.to_csv(t_f_name)
        except OSError as OSE:
            t = time()
            print(f"Something broke when saving {item} to file, saving to a temp csv")
            print("CSV ID: ", t)
            production.to_csv(f"{t}_prod.csv")
            trade_matrix.to_csv(f"{t}_trade.csv")
            print(OSE)


if __name__ == "__main__":
    # Define values
    year = "Y2020"
    # Define regions for which the data is processed
    # "Oceania" is used for testing, as it has the least amount of countries
    # to run with all data use: "All_Data" for region
    # WARNING: takes 2-3 hourse on a laptop
    region = "All_Data"
    main(
        region,
        year=year,
    )
