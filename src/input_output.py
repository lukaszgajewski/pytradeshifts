import pandas as pd
from zipfile import ZipFile


def load_fao_zip(FAO_zip_path):
    zip_file = ZipFile(FAO_zip_path)
    FAO_data = pd.read_csv(
        zip_file.open(
            FAO_zip_path[FAO_zip_path.rfind("/") + 1 :].replace("zip", "csv")
        ),
        encoding="latin1",
        low_memory=False,
    )
    return FAO_data


data = {
    # these files are raw inputs that must be provided for the code to work
    "input": {
        # this is the data from: https://www.fao.org/faostat/en/#data/TM
        # bulk download -> all data
        "trade": "data/input/Trade_DetailedTradeMatrix_E_All_Data.zip",
        # this data is from: https://www.fao.org/faostat/en/#data/QCL
        "production": "data/input/Production_Crops_Livestock_E_All_Data.zip",
        # this is the column name from the two datasets above
        # Y2020 means we take data from the year 2020
        "year_flag": "Y2020",
        # country codes as specified by the FAO
        # we need conversion from FAO code to ISO3
        # this file can be obtained by going to either of the datasets above
        # and clicking "Definitions and standards", then "Country/Region",
        # and finally the download button
        # there's probabyl an easier way but I haven't found it.
        "country_codes": "data/input/country_codes.csv",
        # this is a list of food items and their nutritional value per kg
        # the list provided in this repo is thanks to the hard work of Mike Hinge
        # TODO: add sources
        "nutrition": "data/input/primary_crop_nutritional_data.csv",
        # yield reduction data; here we consider a nuclear winter scenario
        # the data is from TODO: add source
        "yield_reduction": "data/input/nuclear_winter_csv.csv",
        # the fraction of yield per country at each month, also the result of the
        # hard work of Mike Hinge TODO: add sources
        "seasonality": "data/input/seasonality_csv.csv",
    },
    # these are intermidiary files created during running the whole procedure
    # the idea is that, e.g., if we change the scenario (yield reduction) file
    # we don't have to recompute everything
    "intermidiary": {
        # this is trading data of food items as per the nutrional data
        # for each country that is present in the yield reduction data
        # with all unnecessary data removed
        "trade": "data/intermediary/trade_data.pkl",
        "production": "data/intermediary/production_data.pkl",
        # this is the above converted to calories, summed up and put into a matrix
        "caloric_trade": "data/intermediary/total_caloric_trade.csv",
        # or a vecotr in this case
        "caloric_production": "data/intermediary/total_caloric_production.csv",
    },
    # the final output files
    "output": {
        "yearly": "data/output/domestic_supply_kcals.csv",
        "monthly": "data/output/domestic_supply_kcals_monthly.csv",
    },
}

if __name__ == "__main__":
    pass
