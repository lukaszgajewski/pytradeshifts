import pandas as pd
from input_output import data, load_fao_zip


def convert_to_ISO3(FAO_data: pd.DataFrame, country_codes_path: str) -> pd.DataFrame:
    """
    Convert the "Area Code" column in `FAO_data` to "ISO3".

    Arguments:
        FAO_data (pd.DataFrame): pandas DataFrame containing the production data from FAO.
        country_codes_path (str): path to a CSV containing country codes table.
            Note: We are not using the country converter (coco) package because it is
            unreliable with this data. The CSV should be directly from the FAO itself.

    Returns:
        pd.DataFrame: the modified pandas DataFrame.
    """
    # read in the CSV, get the columns we need and convert to a dict
    country_codes = pd.read_csv(country_codes_path)
    country_codes = country_codes[["Country Code", "ISO3 Code"]]
    country_codes = country_codes.set_index("Country Code").to_dict()["ISO3 Code"]
    # replace FAO codes with ISO3
    # Note: we're not using the DataFrame member function .map() here
    # because it is slow
    FAO_data["ISO3"] = [
        country_codes[cc] if cc in country_codes else "not found"
        for cc in FAO_data["Area Code"]
    ]
    # fix the Eswatini/Swaziland code
    FAO_data["ISO3"].replace("SWZ", "SWT", inplace=True)
    return FAO_data


def filter_data(
    FAO_data: pd.DataFrame, nutrition_data_path: str, yield_reduction_data_path: str
) -> pd.DataFrame:
    # keep only the countries that we have nuclear winter data for
    countries_of_interest = pd.read_csv(yield_reduction_data_path, index_col=0).index
    FAO_data = FAO_data[FAO_data["ISO3"].isin(countries_of_interest)]

    # all we need is production
    FAO_data = FAO_data[FAO_data["Element"] == "Production"]

    # keep only the columns we care about
    FAO_data = FAO_data[["ISO3", "Item", "Y2020"]]

    # drop NAs
    FAO_data = FAO_data.dropna()

    # keep only those items we have nutrition data for
    nutritional_data_items = pd.read_csv(nutrition_data_path)["Item"]
    FAO_data = FAO_data[FAO_data["Item"].isin(nutritional_data_items)]

    FAO_data = FAO_data.reset_index(drop=True)

    return FAO_data


def main():
    """
    Read FAO production data (zip), filter out unnecessary information,
    and serialise the result to a pickle.
    Data input/output paths are specified in the `src/input_output.py` file.

    Arguments:
        None.

    Returns:
        None.
    """
    filter_data(
        convert_to_ISO3(
            load_fao_zip(data["input"]["production"]),
            data["input"]["country_codes"],
        ),
        data["input"]["nutrition"],
        data["input"]["yield_reduction"],
    ).to_pickle(data["intermidiary"]["production"])


if __name__ == "__main__":
    main()
