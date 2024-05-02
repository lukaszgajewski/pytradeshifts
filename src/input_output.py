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
    "input": {
        "trade": "data/input/Trade_DetailedTradeMatrix_E_All_Data.zip",
        "production": "data/input/Production_Crops_Livestock_E_All_Data.zip",
        "country_codes": "data/input/country_codes.csv",
        "nutrition": "data/input/primary_crop_nutritional_data.csv",
        "yield_reduction": "data/input/nuclear_winter_csv.csv",
        "seasonality": "data/input/seasonality_csv.csv",
    },
    "intermidiary": {
        "trade": "data/intermediary/trade_data.pkl",
        "production": "data/intermediary/production_data.pkl",
        "caloric_trade": "data/intermediary/total_caloric_trade.csv",
        "caloric_production": "data/intermediary/total_caloric_production.csv",
    },
    "output": {
        "yearly": "data/output/domestic_supply_kcals.csv",
        "monthly": "data/output/domestic_supply_kcals_monthly.csv",
    },
}

if __name__ == "__main__":
    pass
