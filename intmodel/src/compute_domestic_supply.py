import pandas as pd


def load_data(
    total_caloric_trade_path="intmodel/data/total_caloric_trade.csv",
    total_caloric_production_path="intmodel/data/total_caloric_production.csv",
    yearly_reduction_path="intmodel/data/nuclear_winter_csv.csv",
    monthly_seasonality_path="intmodel/data/seasonality_csv.csv",
):
    total_t = pd.read_csv(total_caloric_trade_path, index_col=0)
    total_p = pd.read_csv(total_caloric_production_path, index_col=0).squeeze()
    n = pd.read_csv(yearly_reduction_path, index_col=0)
    n.index.name = "ISO3"
    n = n[[c for c in n.columns if "crop_reduction_year" in c]]
    n = n + 1  # make it a fraction
    n[n < 0] = 0  # ensure we don't get negative yield
    n = n.sort_index()
    m = pd.read_csv(monthly_seasonality_path, index_col=0)
    m.index.name = "ISO3"
    m = m[m.columns[1:]]
    return total_t, total_p, n, m


def compute_domestic_supply(trade_matrix, production_series):
    return production_series + trade_matrix.sum(axis=0) - trade_matrix.sum(axis=1)


def compute_reduced_supply_yearly(total_t, total_p, n):
    r = [
        compute_domestic_supply(
            total_t.mul(cc, axis=0), total_p.mul(cc, axis=0)
        ).rename(
            "crop_kcals_baseline_domestic_supply_year_"
            + c[len("crop_reduction_year") :]
        )
        for c, cc in n.items()
    ]
    return pd.concat(r, axis=1)


def main():
    t, p, n, m = load_data()
    yearly_domestic_supply = (
        compute_domestic_supply(t, p)
        .to_frame(name="crop_kcals_baseline_domestic_supply")
        .join(compute_reduced_supply_yearly(t, p, n))
    )
    yearly_domestic_supply.to_csv("intmodel/data/domestic_supply_kcals.csv")


if __name__ == "__main__":
    main()
