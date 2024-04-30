import pandas as pd
from input_output import data


def load_data(
    total_caloric_trade_path, total_caloric_production_path, yearly_reduction_path
):
    total_trade = pd.read_csv(total_caloric_trade_path, index_col=0)
    total_production = pd.read_csv(total_caloric_production_path, index_col=0).squeeze()
    yield_reduction = pd.read_csv(yearly_reduction_path, index_col=0)
    yield_reduction.index.name = "ISO3"
    yield_reduction = yield_reduction[
        [col for col in yield_reduction.columns if "crop_reduction_year" in col]
    ]
    yield_reduction = yield_reduction + 1  # make it a fraction
    yield_reduction[yield_reduction < 0] = 0  # ensure we don't get negative yield
    yield_reduction = yield_reduction.sort_index()
    return total_trade, total_production, yield_reduction


def compute_domestic_supply(trade_matrix, production_series):
    return production_series + trade_matrix.sum(axis=0) - trade_matrix.sum(axis=1)


def compute_reduced_supply_yearly(total_trade, total_production, yield_reduction):
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
    total_trade, total_production, yield_reduction = load_data(
        data["intermidiary"]["trade"],
        data["intermidiary"]["production"],
        data["input"]["nuclear_winter"],
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
