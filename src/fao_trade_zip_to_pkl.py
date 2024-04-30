import pandas as pd
from input_output import data, load_fao_zip


def convert_to_ISO3(FAO_data, country_codes_path):
    country_codes = pd.read_csv(country_codes_path)
    country_codes = country_codes[["Country Code", "ISO3 Code"]]
    country_codes = country_codes.set_index("Country Code").to_dict()["ISO3 Code"]
    FAO_data["Reporter ISO3"] = [
        country_codes[cc] if cc in country_codes else "not found"
        for cc in FAO_data["Reporter Country Code"]
    ]
    FAO_data["Partner ISO3"] = [
        country_codes[cc] if cc in country_codes else "not found"
        for cc in FAO_data["Partner Country Code"]
    ]
    FAO_data["Reporter ISO3"].replace("SWZ", "SWT", inplace=True)
    FAO_data["Partner ISO3"].replace("SWZ", "SWT", inplace=True)
    return FAO_data


def filter_data(FAO_data, nutrition_data_path, nuclear_winter_data_path):
    # keep only the countries that we have nuclear winter data for
    countries_of_interest = pd.read_csv(nuclear_winter_data_path, index_col=0).index
    FAO_data = FAO_data[
        (
            (FAO_data["Reporter ISO3"].isin(countries_of_interest))
            & (FAO_data["Partner ISO3"].isin(countries_of_interest))
        )
    ]

    # all we need is exports
    FAO_data = FAO_data[FAO_data["Element"] == "Export Quantity"]

    # keep only the columns we care about
    FAO_data = FAO_data[["Reporter ISO3", "Partner ISO3", "Item", "Y2020"]]

    # drop NAs
    FAO_data = FAO_data.dropna()

    # keep only those items we have nutrition data for
    nutritional_data_items = pd.read_csv(nutrition_data_path)["Item"]
    FAO_data = FAO_data[FAO_data["Item"].isin(nutritional_data_items)]

    FAO_data = FAO_data.reset_index(drop=True)

    return FAO_data


def main():
    filter_data(
        convert_to_ISO3(
            load_fao_zip(data["input"]["trade"]),
            data["input"]["country_codes"],
        ),
        data["input"]["nutrition"],
        data["input"]["nuclear_winter"],
    ).to_pickle(data["intermidiary"]["trade"])


if __name__ == "__main__":
    main()
