import pandas as pd
from input_output import data


def load_data(yearly_domestic_supply_path, monthly_seasonality_path):
    yearly_domestic_supply = pd.read_csv(yearly_domestic_supply_path)
    monthly_domestic_supply = (
        yearly_domestic_supply.set_index("ISO3") / 12
    )  # divide to get monthly values

    monthly_seasonality = pd.read_csv(monthly_seasonality_path, index_col=0)
    monthly_seasonality = monthly_seasonality.sort_index()
    monthly_seasonality = monthly_seasonality.loc[
        monthly_domestic_supply.index, :
    ]  # no MNE entry in nuclear winter data so we remove it from here
    # TODO: unify data sets earlier than this
    monthly_seasonality.index.name = "ISO3"
    monthly_seasonality = monthly_seasonality[monthly_seasonality.columns[1:]]
    return monthly_domestic_supply, monthly_seasonality


def main():
    monthly_domestic_supply, monthly_seasonality = load_data(
        data["output"]["yearly"], data["input"]["seasonality"]
    )
    # bomb drops May 1st so we split year one into 8 months
    year_one = [
        monthly_domestic_supply["crop_kcals_baseline_domestic_supply_year_1"]
        .mul(monthly_seasonality[f"seasonality_m{month+4}"], axis=0)
        .rename(f"crop_kcals_baseline_domestic_supply_month_{month}")
        for month in range(1, 9)
    ]
    other_years = [
        monthly_domestic_supply[f"crop_kcals_baseline_domestic_supply_year_{year}"]
        .mul(monthly_seasonality[f"seasonality_m{month}"], axis=0)
        .rename(
            f"crop_kcals_baseline_domestic_supply_month_{((year - 1) * 12) + month + (8 - 12)}"
        )
        for year in range(2, 11)
        for month in range(1, 13)
    ]
    monthly_domestic_supply = pd.concat(
        [monthly_domestic_supply["crop_kcals_baseline_domestic_supply"]]
        + year_one
        + other_years,
        axis=1,
    )
    monthly_domestic_supply = monthly_domestic_supply.reset_index()
    monthly_domestic_supply.to_csv(data["output"]["monthly"], index=False)


if __name__ == "__main__":
    main()
