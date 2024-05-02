from time import time

from fao_trade_zip_to_pkl import main as serialise_trade_data
from fao_production_zip_to_pkl import main as serialise_production_data
from fao_pkl_to_total_caloric_trade_and_production import (
    main as compute_total_trade_and_production,
)
from compute_domestic_supply import main as compute_yearly_domestic_supply
from convert_supply_yearly_to_monthly import main as compute_monthly_domestic_supply


def main():
    t = time()
    print("Serialising trade data...")
    serialise_trade_data()
    print("Serialising production data...")
    serialise_production_data()
    print("Computing total caloric trade and production...")
    compute_total_trade_and_production()
    print("Computing domestic supply yearly...")
    compute_yearly_domestic_supply()
    print("Computing domestic supply monthly...")
    compute_monthly_domestic_supply()
    print("Fin. Elapsed time [s]: ", time() - t)


if __name__ == "__main__":
    main()
