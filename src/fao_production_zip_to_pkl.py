import pandas as pd
from input_output import data, load_fao_zip, serialise_fao_data


def convert_to_ISO3(FAO_data, country_codes_path):
    country_codes = pd.read_csv(country_codes_path)
    country_codes = country_codes[["Country Code", "ISO3 Code"]]
    country_codes = country_codes.set_index("Country Code").to_dict()["ISO3 Code"]
    FAO_data["ISO3"] = [
        country_codes[cc] if cc in country_codes else "not found"
        for cc in FAO_data["Area Code"]
    ]
    FAO_data["ISO3"].replace("SWZ", "SWT", inplace=True)
    return FAO_data


def filter_data(FAO_data, nutrition_data_path, nuclear_winter_data_path):
    # keep only the countries that we have nuclear winter data for
    countries_of_interest = pd.read_csv(nuclear_winter_data_path, index_col=0).index
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

    return FAO_data


def main():
    serialise_fao_data(
        filter_data(
            convert_to_ISO3(
                load_fao_zip(data["input"]["production"]),
                data["input"]["country_codes"],
            ),
            data["input"]["nutrition"],
            data["input"]["nuclear_winter"],
        ),
        data["intermidiary"]["production"],
    )


if __name__ == "__main__":
    main()
