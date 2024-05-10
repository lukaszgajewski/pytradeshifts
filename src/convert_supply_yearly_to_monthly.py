import pandas as pd
from input_output import data


def load_data(
    yearly_domestic_supply_path: str, monthly_seasonality_path: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read yearly domestic supply and monthly seasonality data; return them as
    pandas DataFrames.

    Arguments:
        yearly_domestic_supply_path (str): path to file containing domestic supply
            yearly. This file is a result of running `src/compute_domestic_supply.py`.
        monthly_seasonality_path (str): path to file containing seasonality data
            for each country. A sample file is provided in `data/input/seasonality_csv.csv`.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: DataFrames containing yearly domestic supply,
            and monthly seasonality for each country.
    """
    yearly_domestic_supply = pd.read_csv(yearly_domestic_supply_path)
    # divide by 12 to get monthly values
    # this is a crude oversimplification but we simply have no access
    # to monthly data on countries' agrocultural import/exports as of right now
    # so we assume that its proportional to the seasonality values
    monthly_domestic_supply = yearly_domestic_supply.set_index("ISO3") / 12
    # get the seasonality and rename the index to match domestic supply data
    monthly_seasonality = pd.read_csv(monthly_seasonality_path, index_col=0)
    monthly_seasonality = monthly_seasonality.sort_index()
    monthly_seasonality.index.name = "ISO3"
    # make sure that the indices match
    pd.testing.assert_index_equal(
        monthly_domestic_supply.index, monthly_seasonality.index
    )
    # we do not need the country names column
    monthly_seasonality = monthly_seasonality.drop(columns="country")
    return monthly_domestic_supply, monthly_seasonality


def compute_year_one(
    monthly_domestic_supply: pd.DataFrame, monthly_seasonality: pd.DataFrame
) -> list[pd.Series]:
    """
    Compute domestic supply per month in year one. We're assuming the yield
    reduction scenario initial event happens May 1st.

    Arguments:
        monthly_domestic_supply (pd.DataFrame): domestic supply average per month.
        monthly_seasonality (pd.DataFrame): data indicating the seasonality of
            the domestic supply for each country, month.

    Returns:
        list[pd.Series]: list of domestic supply vectors (indexed by country),
            one vector per month (May-December).
    """
    return [
        monthly_domestic_supply["crop_kcals_baseline_domestic_supply_year_1"]
        .mul(monthly_seasonality[f"seasonality_m{month+4}"], axis=0)
        .rename(f"crop_kcals_baseline_domestic_supply_month_{month}")
        for month in range(1, 9)
    ]


def compute_other_years(
    monthly_domestic_supply: pd.DataFrame,
    monthly_seasonality: pd.DataFrame,
    total_years=10,
) -> list[pd.Series]:
    """
    Compute domestic supply per month in years following year one (so, year two onwards).

    Arguments:
        monthly_domestic_supply (pd.DataFrame): domestic supply average per month.
        monthly_seasonality (pd.DataFrame): data indicating the seasonality of
            the domestic supply for each country, month.
        total_years (int, optional): total number of years, inclusive with year one,
            in which yield reduction scenario initial event occurs.

    Returns:
        list[pd.Series]: list of domestic supply vectors (indexed by country),
            one vector per month (January-December x # of years less one).
    """
    return [
        monthly_domestic_supply[f"crop_kcals_baseline_domestic_supply_year_{year}"]
        .mul(monthly_seasonality[f"seasonality_m{month}"], axis=0)
        .rename(
            f"crop_kcals_baseline_domestic_supply_month_{((year - 1) * 12) + month + (8 - 12)}"
        )
        # yes, I know 8-12 is -4; this is for clarity
        # since there are 8 months in year one;
        # one could argue that it is more intuitive to think of it as
        # starting after 4 months (-4), so ¯\_(ツ)_/¯ ...
        for year in range(2, total_years + 1)
        for month in range(1, 13)
    ]


def main():
    """
    Convert yearly caloric domestic supply to monthly, using the specified
    seasonality data.
    Data input/output paths are specified in the `src/input_output.py` file.

    Arguments:
        None.

    Returns:
        None.
    """
    monthly_domestic_supply, monthly_seasonality = load_data(
        data["output"]["yearly"], data["input"]["seasonality"]
    )
    # bomb drops May 1st so we split year one into 8 months
    year_one = compute_year_one(monthly_domestic_supply, monthly_seasonality)
    # other years we treat normally, i.e., splitting them into 12 months
    other_years = compute_other_years(monthly_domestic_supply, monthly_seasonality)
    # combine into one DataFrame
    monthly_domestic_supply = pd.concat(
        [monthly_domestic_supply["crop_kcals_baseline_domestic_supply"]]
        + year_one
        + other_years,
        axis=1,
    )
    # save to file
    monthly_domestic_supply = monthly_domestic_supply.reset_index()
    monthly_domestic_supply.to_csv(data["output"]["monthly"], index=False)


if __name__ == "__main__":
    main()
