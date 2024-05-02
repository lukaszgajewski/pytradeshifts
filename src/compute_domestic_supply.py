import pandas as pd
from input_output import data


def load_data(
    total_caloric_trade_path: str,
    total_caloric_production_path: str,
    yearly_reduction_path: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Read total caloric trade, production data and yield reduction values;
    return in an appropriate format (DataFrames) for computing domestic supply

    Arguments:
        total_caloric_trade_path (str): path to file containing total caloric
            trade volume between countries. This file is the result of
            the `src/fao_pkl_to_total_caloric_trade_and_production.py` script.
        total_caloric_production_path (str): path to file containing total caloric
            trade production for each country. This file is the result of
            the `src/fao_pkl_to_total_caloric_trade_and_production.py` script.
        yearly_reduction_path (str): path to a CSV containing yield reduction data,
            e.g., in a nuclear winter scenario. A sample file is provided in:
            `data/input/nuclear_winter_csv.csv`.

    Returns:
        tuple[pd.DataFrame, pd.Series, pd.DataFrame]: caloric trade matrix,
            production vector, and yield reduction data.
    """
    # read the data
    total_trade = pd.read_csv(total_caloric_trade_path, index_col=0)
    total_production = pd.read_csv(total_caloric_production_path, index_col=0).squeeze()
    yield_reduction = pd.read_csv(yearly_reduction_path, index_col=0)
    # match the index name
    yield_reduction.index.name = "ISO3"
    # take only the crop reduction data
    yield_reduction = yield_reduction[
        [col for col in yield_reduction.columns if "crop_reduction_year" in col]
    ]
    # convert the values to a fraction
    yield_reduction = yield_reduction + 1
    # ensure we don't get negative yield
    yield_reduction[yield_reduction < 0] = 0
    yield_reduction = yield_reduction.sort_index()
    return total_trade, total_production, yield_reduction


def compute_domestic_supply(
    trade_matrix: pd.DataFrame, production_series: pd.Series
) -> pd.Series:
    """
    Compute the domestic supply = production + import - export; for each country.

    Arguments:
        trade_matrix (pd.DataFrame): trading data as a pivoted table.
        production_series (pd.Series): production data as a vector.

    Returns:
        pd.Series: domestic supply vector.
    """
    return production_series + trade_matrix.sum(axis=0) - trade_matrix.sum(axis=1)


def compute_reduced_supply_yearly(
    total_trade: pd.DataFrame,
    total_production: pd.Series,
    yield_reduction: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute the caloric domestic supply under the specified yield reduction circumstances.

    Arguments:
        total_trade (pd.DataFrame): a matrix (pivoted table) containing total
            caloric trade volume between countries.
        total_production (pd.Series): a vector with total caloric production
            of each country.
        yield_reduction (pd.DataFrame): yield reduction data for each year of the
            considered scenario (e.g., nuclear winter).

    Returns:
        pd.DataFrame: DataFrame containing domestic caloric supply in each year.
    """
    return pd.concat(
        [
            compute_domestic_supply(
                total_trade.mul(yield_reduction_vector, axis=0),
                total_production.mul(yield_reduction_vector, axis=0),
            ).rename(
                "crop_kcals_baseline_domestic_supply_year_"
                + vector_label[len("crop_reduction_year") :]
            )
            for vector_label, yield_reduction_vector in yield_reduction.items()
        ],
        axis=1,
    )


def main():
    """
    Get the current state of total caloric trade and production, and compute
    domestic caloric supply of each country for each year of a considered
    yield reduction scenario such as a nuclear winter.
    Data input/output paths are specified in the `src/input_output.py` file.

    Arguments:
        None.

    Returns:
        None.
    """
    total_trade, total_production, yield_reduction = load_data(
        data["intermidiary"]["trade"],
        data["intermidiary"]["production"],
        data["input"]["yield_reduction"],
    )
    yearly_domestic_supply = (
        compute_domestic_supply(total_trade, total_production)
        .to_frame(name="crop_kcals_baseline_domestic_supply")
        .join(
            compute_reduced_supply_yearly(
                total_trade, total_production, yield_reduction
            )
        )
    )
    yearly_domestic_supply.to_csv(data["output"]["yearly"])


if __name__ == "__main__":
    main()
