import pandas as pd
import numpy as np
from tqdm import tqdm
from input_output import data


def compute_calories(
    trade_data: pd.DataFrame, production_data: pd.DataFrame, nutrition_data_path: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert food items to their respective caloric value (in "dry caloric tonnes").

    Arguments:
        trade_data (pd.DataFrame): FAO trading data in a pandas DataFrame.
        production_data (pd.DataFrame): FAO production data in a pandas DataFrame.
        nutrition_data_path (str): path to a CSV with nutrional data;
            must have "Item" and "Calories" columns. A sample file is provieded in:
            `data/input/primary_crop_nutritional_data.csv`

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: trade and production DataFrames with
            "Dry Caloric Tonnes" column added.
    """
    # get the nutritional table and convert it to a dict
    # mapping food items to their caloric value
    nutrition_data = (
        pd.read_csv(nutrition_data_path)[["Item", "Calories"]]
        .set_index("Item", drop=True)
        .to_dict()["Calories"]
    )
    # compute the calories
    trade_data["Dry Caloric Tonnes"] = (
        trade_data["Item"].map(nutrition_data) * trade_data["Y2020"] * 1000 / 4e6
    )
    # 1000 is tonnes to kg, 4e6 is Cal (a.k.a. kcal) to dry caloric tonne
    # yes, it is a bit redundant since 1e3 cancels out
    # but I want it here like this for clarity
    # Additional note: we sue map here which can ve slow but it is convenient
    # and this data is post-filtering so speed should not be a concern anymore
    production_data["Dry Caloric Tonnes"] = (
        production_data["Item"].map(nutrition_data)
        * production_data["Y2020"]
        * 1000
        / 4e6
    )
    return (
        trade_data[["Reporter ISO3", "Partner ISO3", "Item", "Dry Caloric Tonnes"]],
        production_data[["ISO3", "Item", "Dry Caloric Tonnes"]],
    )


def reindex_trade_and_production(
    trade_matrix: pd.DataFrame, production_series: pd.Series, country_list: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Reindex trade and production data such that they match each other and the
    "ground truth" country list.

    Arguments:
        trade_matrix (pd.DataFrame): FAO trading data as a *pivoted table*.
        production_series (pd.Series): FAO production data as a Series.
        country_list (list[str]): the "ground truth" country list to which we
            match the indices.

    Returns:
        tuple[pd.DataFrame, pd.Series]: modified trade and production data with
            matching indices.
    """
    trade_matrix = trade_matrix.reindex(
        index=country_list, columns=country_list
    ).fillna(0)
    production_series = (
        production_series.reindex(index=country_list).fillna(0).squeeze()
    )
    return trade_matrix, production_series


def format_trade_and_production(
    trade_data: pd.DataFrame, production_data: pd.DataFrame, country_list: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Convert trading and production data into a format appropriate for the
    prebalancing and re-export correction algorithms.

    Arguments:
        trade_data (pd.DataFrame): the FAO trading data.
        production (pd.DataFrame): the FAO production data.
        country_list (list[str]): the "ground truth" country list to which we
            shall conform the indices.

    Returns:
        tuple[pd.DataFrame, pd.Series]: a trading marix (pivoted table) and,
            a production Series (a vector).
    """
    production_series = production_data.set_index("ISO3")[
        "Dry Caloric Tonnes"
    ].squeeze()
    trade_matrix = trade_data.pivot(
        values="Dry Caloric Tonnes", index="Reporter ISO3", columns="Partner ISO3"
    )
    return reindex_trade_and_production(trade_matrix, production_series, country_list)


def prebalance(
    trade_matrix: pd.DataFrame, production_series: pd.Series, tol=1e-3
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Pre-balance trading data to ensure that countries don't export more than
    they produce and import.
    This is from the procedure in:
    Croft, S. A., West, C. D., & Green, J. M. H. (2018).
    "Capturing the heterogeneity of sub-national production in global trade flows."
    Journal of Cleaner Production, 203, 1106–1118.
    https://doi.org/10.1016/j.jclepro.2018.08.267

    Arguments:
        trade_matrix (pd.DataFrame): FAO trading data as a *pivoted table*.
        production_series (pd.Series): FAO production data as a Series.
        tol (float, optional): tolerance amount for the domestic supply
            to be negative, i.e., the algorithm runs until domestic supply
            for each country is > -tol.

    Returns:
        tuple[pd.DataFrame, pd.Series]: trading matrix (prebalanced) and
            production vector (unmodified).
    """
    domestic_supply = (
        production_series + trade_matrix.sum(axis=0) - trade_matrix.sum(axis=1)
    )
    while (domestic_supply <= -tol).any():
        sf = (production_series + trade_matrix.sum(axis=0)) / trade_matrix.sum(axis=1)
        multiplier = np.where(domestic_supply < 0, sf, 1)
        trade_matrix = pd.DataFrame(
            np.diag(multiplier) @ trade_matrix.values,
            index=trade_matrix.index,
            columns=trade_matrix.columns,
        )
        domestic_supply = (
            production_series + trade_matrix.sum(axis=0) - trade_matrix.sum(axis=1)
        )
    return trade_matrix, production_series


def remove_net_zero_countries(
    trade_matrix: pd.DataFrame, production_series: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Return production and trading data with "all zero" countries removed.
    "All zero" countries are such states that has production = 0 and sum of rows
    in trade matrix = 0, and sum of columns = 0.

    Arguments:
        trade_matrix (pd.DataFrame): FAO trading data as a *pivoted table*.
        production_series (pd.Series): FAO production data as a Series.

    Returns:
        tuple[pd.DataFrame, pd.Series]: trading matrix and production vector.
    """
    non_zero_countries = ~(
        trade_matrix.sum(axis=1).eq(0)
        & trade_matrix.sum(axis=0).eq(0)
        & (production_series == 0)
    )
    production_series = production_series[non_zero_countries]
    trade_matrix = trade_matrix.loc[non_zero_countries, non_zero_countries]
    return trade_matrix, production_series


def correct_reexports(
    trade_matrix: pd.DataFrame, production_series: pd.Series, tol=1e-3
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Correct for re-exports in the trade matrix.
    This is from the procedure in:
    Croft, S. A., West, C. D., & Green, J. M. H. (2018).
    "Capturing the heterogeneity of sub-national production in global trade flows."
    Journal of Cleaner Production, 203, 1106–1118.
    https://doi.org/10.1016/j.jclepro.2018.08.267

    Note: the procedure can fail when one of the intermidiary matrices becomes
    un-invertible. This generally should not happen but if it does we print
    a warning, and use Moore-Penrose pseudo-inverse instead.

    Arguments:
        trade_matrix (pd.DataFrame): FAO trading matrix *after prebalancing*.
        production_series (pd.Series): FAO production data as a Series.
        tol (float, optional): tolerance amount for small values to be considered
            non-zero, i.e., values < tol shall be 0.

    Returns:
        tuple[pd.DataFrame, pd.Series]: trading matrix (corrected)
            and production vector (unmodified).
    """
    trade_matrix, production_series = remove_net_zero_countries(
        trade_matrix, production_series
    )
    trade_matrix = trade_matrix.T
    x = production_series + trade_matrix.sum(axis=1)
    try:
        y = np.linalg.inv(np.diag(x))
    except np.linalg.LinAlgError:
        print(
            "Warning: failed to solve for `y` in correct_reexports(); using the Moore-Penrose pseudo inverse."
        )
    y = np.linalg.pinv(np.diag(x))
    A = trade_matrix @ y
    try:
        R = np.linalg.inv(np.identity(len(A)) - A) @ np.diag(production_series)
    except np.linalg.LinAlgError:
        print(
            "Warning: failed to solve for `R` in correct_reexports(); using the Moore-Penrose pseudo inverse."
        )
        R = np.linalg.pinv(np.identity(len(A)) - A) @ np.diag(production_series)
    c = np.diag(y @ (x - trade_matrix.sum(axis=0)))
    R = (c @ R).T
    R[~np.isfinite(R)] = 0
    R[R < tol] = 0

    trade_matrix = pd.DataFrame(
        R, index=trade_matrix.index, columns=trade_matrix.columns
    )
    np.fill_diagonal(trade_matrix.values, 0)
    return trade_matrix, production_series


def main():
    """
    Compute total caloric trade flow between countries and total caloric
    production for each country.
    Data input/output paths are specified in the `src/input_output.py` file.

    Arguments:
        None.

    Returns:
        None.
    """
    # read the "ground truth" country list, we assume this is the countries
    # for which we have the yield reduction data
    country_list = sorted(
        set(pd.read_csv(data["input"]["yield_reduction"])["iso3"].values)
    )
    # convert food items to their caloric values
    caloric_trade, caloric_production = compute_calories(
        pd.read_pickle(data["intermidiary"]["trade"]),
        pd.read_pickle(data["intermidiary"]["production"]),
        data["input"]["nutrition"],
    )
    # compute total production and init the total trade
    # total production is finished but
    # total trade at this point is filled with 0s for now
    # we shall fill it in shortly
    total_trade, total_production = reindex_trade_and_production(
        pd.DataFrame(
            np.zeros((len(country_list), len(country_list))),
            index=country_list,
            columns=country_list,
        ),
        caloric_production.groupby(["ISO3"]).sum(numeric_only=True).squeeze(),
        country_list,
    )
    # get the longest food item name for printing niceness
    longest_item_name = len(max(caloric_production["Item"], key=len))
    # we need to add up all trading matrices for each food item
    for item, trade_data in (pbar := tqdm(caloric_trade.groupby("Item"))):
        # this shows which item is currently being considered in the
        # progress bar
        pbar.set_description(item + " " * (longest_item_name - len(item)))
        # prepare data -> prebalance -> correct re-exports -> format nicely
        # we ignore production series as we have that computed already
        trade_matrix, _ = reindex_trade_and_production(
            *correct_reexports(
                *prebalance(
                    *format_trade_and_production(
                        trade_data,
                        caloric_production[caloric_production["Item"] == item],
                        country_list,
                    )
                )
            ),
            country_list,
        )
        # add up
        total_trade += trade_matrix
    # save results to file
    total_trade.to_csv(data["intermidiary"]["caloric_trade"])
    total_production.to_csv(data["intermidiary"]["caloric_production"])


if __name__ == "__main__":
    main()
