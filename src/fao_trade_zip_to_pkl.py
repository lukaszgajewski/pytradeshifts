import pandas as pd
from zipfile import ZipFile


def load_data(FAO_zip_path="data/input/Trade_DetailedTradeMatrix_E_All_Data.zip"):
    zip_file = ZipFile(FAO_zip_path)
    FAO_data = pd.read_csv(
        zip_file.open(
            FAO_zip_path[FAO_zip_path.rfind("/") + 1 :].replace("zip", "csv")
        ),
        encoding="latin1",
        low_memory=False,
    )
    return FAO_data


def convert_to_ISO3(FAO_data, country_codes_path="data/input/country_codes.csv"):
    country_codes = pd.read_csv(country_codes_path)
    country_codes = country_codes[["Country Code", "ISO3 Code"]]
    country_codes = country_codes.set_index("Country Code").to_dict()["ISO3 Code"]
    FAO_data["Reporter ISO3"] = [
        country_codes[c] if c in country_codes else "not found"
        for c in FAO_data["Reporter Country Code"]
    ]
    FAO_data["Partner ISO3"] = [
        country_codes[c] if c in country_codes else "not found"
        for c in FAO_data["Partner Country Code"]
    ]
    FAO_data["Reporter ISO3"].replace("SWZ", "SWT", inplace=True)
    FAO_data["Partner ISO3"].replace("SWZ", "SWT", inplace=True)
    return FAO_data


def remove_unwanted_entries(
    FAO_data,
    nutrition_data_path="data/input/primary_crop_nutritional_data.csv",
    nuclear_winter_data_path="data/input/nuclear_winter_csv.csv",
):
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
    FAO_data = FAO_data[
        [
            "Reporter ISO3",
            "Partner ISO3",
            "Item",
            "Y2020",
        ]
    ]

    # drop NAs
    FAO_data = FAO_data.dropna()

    # keep only those items we have nutrition data for
    nutritional_data = pd.read_csv(nutrition_data_path)
    FAO_data = FAO_data[FAO_data["Item"].isin(nutritional_data["Item"])]
    return FAO_data


def serialise(FAO_data, output_file_path="data/intermediary/trade_data.pkl"):
    # reorganise columns
    FAO_data = FAO_data[["Reporter ISO3", "Partner ISO3", "Item", "Y2020"]]
    FAO_data = FAO_data.reset_index(drop=True)

    # serialise
    FAO_data.to_pickle(output_file_path)


def main():
    serialise(remove_unwanted_entries(convert_to_ISO3(load_data())))


if __name__ == "__main__":
    main()
