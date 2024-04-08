import pandas as pd
from tqdm import tqdm
from domestic_supply import get_scenarios


def import_nutrients_and_products(
    nutrition_xls, domestic_supply_csv
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return nutrition and production data from Excel and CSV files respectively.

    Arguments:
        None

    Returns:
        tuple: a tuple containing two dataframes: products and nutrition
    """
    nutrition = pd.read_excel(
        nutrition_xls,
        sheet_name="Nutrition data from FAOSTAT",
        usecols="A:E",
        skiprows=1,
    )
    products = pd.read_csv(domestic_supply_csv, index_col=0)
    return products, nutrition


def compute_total_nutrients(
    products: pd.DataFrame, nutrition: pd.DataFrame
) -> pd.DataFrame:
    """
    Return the sum of kcals, fat, and protein for the products passed in.

    Arguments:
        products (pandas.DataFrame): A DataFrame containing food products and their values.

    Returns:
        pandas.DataFrame: A dataframe containing the sum of kcals, fat, and protein for all countries.
    """
    # Merge dataframes
    df = pd.merge(left=nutrition, right=products, on="Item")
    # Replace NaNs with zeroes
    df["Value"].fillna(0, inplace=True)
    # Convert units
    df["Value"] = df["Value"]
    df["Calories"] = df["Calories"] * (1 / 4000)  # kcals to dry caloric kg
    # Compute nutrients
    df["crop_kcals"] = df["Calories"] * df["Value"]
    df["crop_fat"] = df["Fat"] * df["Value"]
    df["crop_protein"] = df["Protein"] * df["Value"]
    # Sum it all up
    df = df.groupby(["Area Code (ISO3)", "Area"]).sum(numeric_only=True)[
        ["crop_kcals", "crop_fat", "crop_protein"]
    ]
    # Format
    df = df.reset_index()
    df.columns = [
        "iso3",
        "country",
        "crop_kcals",
        "crop_fat",
        "crop_protein",
    ]
    return df


def main():
    print("Computing macros for no scenario domestic supply.")
    products, nutrients = import_nutrients_and_products(
        "intmodel/data/ALLFED Food consumption, supplies and balances.xlsx",
        "intmodel/data/domestic_supply_combined/domestic_supply_combined.csv",
    )
    total_nutrients = compute_total_nutrients(products, nutrients)
    # Fix Eswatini/Swaziland iso3 code
    total_nutrients["iso3"].replace("SWZ", "SWT", inplace=True)
    # Save to file
    total_nutrients.to_csv("intmodel/data/macros/macros_csv.csv", index=False)
    print("Computing domestic supply macros for all scenarios.")
    scenarios = get_scenarios("intmodel/data/scenario_files")
    for scenario in tqdm(scenarios):
        products, nutrients = import_nutrients_and_products(
            "intmodel/data/ALLFED Food consumption, supplies and balances.xlsx",
            f"intmodel/data/domestic_supply_combined/{scenario}",
        )
        total_nutrients = compute_total_nutrients(products, nutrients)
        # Fix Eswatini/Swaziland iso3 code
        total_nutrients["iso3"].replace("SWZ", "SWT", inplace=True)
        # Save to file
        total_nutrients.to_csv(
            f"intmodel/data/macros/{scenario}_macros_csv.csv", index=False
        )


if __name__ == "__main__":
    main()
