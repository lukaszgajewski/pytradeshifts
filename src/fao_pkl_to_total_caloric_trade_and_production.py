import pandas as pd
import numpy as np
from tqdm import tqdm
from input_output import data


def compute_calories(trade_data, production_data, nutrition_data_path):
    nutrition_data = (
        pd.read_csv(nutrition_data_path)[["Item", "Calories"]]
        .set_index("Item", drop=True)
        .to_dict()["Calories"]
    )
    trade_data["Dry Caloric Tonnes"] = (
        trade_data["Item"].map(nutrition_data) * trade_data["Y2020"] * 1000 / 4e6
    )
    # 1000 is tonnes to kg, 4e6 is Cal (a.k.a. kcal) to dry caloric tonne
    # yes, it is a bit redundant since 1e3 cancels out
    # but I want it here like this for clarity
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


def reindex_trade_and_production(trade_matrix, production_series, country_list):
    trade_matrix = trade_matrix.reindex(
        index=country_list, columns=country_list
    ).fillna(0)
    production_series = (
        production_series.reindex(index=country_list).fillna(0).squeeze()
    )
    return trade_matrix, production_series


def format_trade_and_production(trade_data, production_data, country_list):
    production_series = production_data.set_index("ISO3")[
        "Dry Caloric Tonnes"
    ].squeeze()
    trade_matrix = trade_data.pivot(
        values="Dry Caloric Tonnes", index="Reporter ISO3", columns="Partner ISO3"
    )
    return reindex_trade_and_production(trade_matrix, production_series, country_list)


def prebalance(trade_matrix, production_series, precision=1e-3):
    """
    This implementation also includes pre-balancing to ensure that countries don't
    export more than they produce and import.
    From Croft et al.

    Arguments:
        precision (float, optional): Specifies precision of the prebalancing.

    Returns:
        None
    """
    domestic_supply = (
        production_series + trade_matrix.sum(axis=0) - trade_matrix.sum(axis=1)
    )
    while (domestic_supply <= -precision).any():
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


def remove_net_zero_countries(trade_matrix, production_series):
    """
    Return production and trading data with "all zero" countries removed.
    "All zero" countries are such states that has production = 0 and sum of rows
    in trade matrix = 0, and sum of columns = 0.

    Arguments:
        None

    Returns:
        None
    """
    non_zero_countries = ~(
        trade_matrix.sum(axis=1).eq(0)
        & trade_matrix.sum(axis=0).eq(0)
        & (production_series == 0)
    )
    production_series = production_series[non_zero_countries]
    trade_matrix = trade_matrix.loc[non_zero_countries, non_zero_countries]
    return trade_matrix, production_series


def correct_reexports(trade_matrix, production_series, precision=1e-3, n_tries=10):
    """
    Removes re-exports from the trade matrix.
    This is a Python implementation of the R/Matlab code from:
    Croft, S. A., West, C. D., & Green, J. M. H. (2018).
    "Capturing the heterogeneity of sub-national production
    in global trade flows."

    Journal of Cleaner Production, 203, 1106–1118.

    https://doi.org/10.1016/j.jclepro.2018.08.267


    Input to this function should be prebalanced and have countries with all zeroes
    removed.

    Arguments:
        None

    Returns:
        None
    """
    trade_matrix, production_series = remove_net_zero_countries(
        trade_matrix, production_series
    )
    trade_matrix = trade_matrix.T
    x = production_series + trade_matrix.sum(axis=1)
    for try_count in range(n_tries):
        try:
            y = np.linalg.inv(np.diag(x))
            break
        except np.linalg.LinAlgError:
            print(
                f"""
                Warning: Determinant=0 encountered in correct_reexports() when solving for `y`.
                Re-applying remove_net_zero_countries(),
                and attempting to invert the matrix again;
                re-try attempt number {try_count+1}.
                """
            )
            trade_matrix, production_series = remove_net_zero_countries(
                trade_matrix, production_series
            )
            x = production_series + trade_matrix.sum(axis=1)
            y = np.linalg.inv(np.diag(x))
    else:
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
    R[R < precision] = 0

    trade_matrix = pd.DataFrame(
        R, index=trade_matrix.index, columns=trade_matrix.columns
    )
    np.fill_diagonal(trade_matrix.values, 0)
    return trade_matrix, production_series


def main():
    country_list = sorted(
        set(pd.read_csv(data["input"]["nuclear_winter"])["iso3"].values)
    )
    caloric_trade, caloric_production = compute_calories(
        pd.read_pickle(data["intermidiary"]["trade"]),
        pd.read_pickle(data["intermidiary"]["production"]),
        data["input"]["nutrition"],
    )
    total_trade, total_production = reindex_trade_and_production(
        pd.DataFrame(
            np.zeros((len(country_list), len(country_list))),
            index=country_list,
            columns=country_list,
        ),
        caloric_production.groupby(["ISO3"]).sum(numeric_only=True).squeeze(),
        country_list,
    )
    longest_item_name = len(max(caloric_production["Item"], key=len))
    for item, trade_data in (pbar := tqdm(caloric_trade.groupby("Item"))):
        pbar.set_description(item + " " * (longest_item_name - len(item)))
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
        total_trade += trade_matrix
    total_trade.to_csv(data["intermidiary"]["caloric_trade"])
    total_production.to_csv(data["intermidiary"]["caloric_production"])


if __name__ == "__main__":
    main()
