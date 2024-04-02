import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.model import PyTradeShifts
from read_nutrition_data import read_nutrition_data


class DomesticSupply(PyTradeShifts):
    def __init__(
        self,
        item: str,
        base_year: int,
        region="Global",
    ) -> None:
        super().__init__(item, base_year, region=region, testing=True)
        # Read in the data
        self.load_data()
        # Remove countries with all zeroes in trade and production
        self.remove_net_zero_countries()
        # Prebalance the trade matrix
        self.prebalance()
        # Remove re-exports
        self.correct_reexports()
        # Set diagonal to zero
        np.fill_diagonal(self.trade_matrix.values, 0)
        if self.scenario_name is not None and self.scenario_file_name is not None:
            self.apply_scenario()

    def load_data(self) -> None:
        """
        Loads the data into a pandas dataframe and cleans it
        of countries with trade below a certain percentile.

        Arguments:
            None

        Returns:
            None
        """
        assert self.trade_matrix is None
        print(f"Attempting to load {self.crop} in {self.base_year}.")
        # Read in the data
        trade_matrix = pd.read_csv(
            f"intmodel{os.sep}data{os.sep}prod_trade{os.sep}{self.crop}_{self.base_year}_{self.region}_trade.csv",
            index_col=0,
        )

        production_data = pd.read_csv(
            f"intmodel{os.sep}data{os.sep}prod_trade{os.sep}{self.crop}_{self.base_year}_{self.region}_production.csv",
            index_col=0,
        ).squeeze()

        # ensure we do not have duplicates
        # duplicates are most likely a result of incorrect preprocessing
        # or lack thereof
        if not trade_matrix.index.equals(trade_matrix.index.unique()):
            print("Warning: trade matrix has duplicate indices")
            trade_entries_to_keep = ~trade_matrix.index.duplicated(keep="first")
            trade_matrix = trade_matrix.loc[
                trade_entries_to_keep,
                trade_entries_to_keep,
            ]
        if not production_data.index.equals(production_data.index.unique()):
            print("Warning: production has duplicate indices")
            production_data = production_data.loc[
                ~production_data.index.duplicated(keep="first")
            ]
        # remove a "not found" country
        # this would be a result of a region naming convention that
        # country_converter failed to handle
        if "not found" in trade_matrix.index:
            print("Warning: 'not found' present in trade matrix index")
            trade_matrix.drop(index="not found", inplace=True)
        if "not found" in trade_matrix.columns:
            print("Warning: 'not found' present in trade matrix columns")
            trade_matrix.drop(columns="not found", inplace=True)
        if "not found" in production_data.index:
            print("Warning: 'not found' present in production index")
            production_data.drop(index="not found", inplace=True)

        print(f"Loaded data for {self.crop} in {self.base_year}.")

        # Retain only the countries where we have production data and trade data
        countries = np.intersect1d(trade_matrix.index, production_data.index)
        trade_matrix = trade_matrix.loc[countries, countries]
        production_data = production_data.loc[countries]
        # Make sure this worked
        assert trade_matrix.shape[0] == production_data.shape[0]
        assert trade_matrix.shape[1] == production_data.shape[0]

        # Save the data
        self.trade_matrix = trade_matrix
        self.production_data = production_data

    def get_domestic_supply(self) -> pd.Series:
        """Domestic supply is production plus imports minus exports."""
        ds = (
            self.production_data
            + self.trade_matrix.sum(axis=0)  # total import
            - self.trade_matrix.sum(axis=1)  # total export
        )
        return ds.squeeze()


def get_allowed_items():
    nd = read_nutrition_data()
    return nd.index


if __name__ == "__main__":
    for item in tqdm(get_allowed_items()):
        ds_fname = f"intmodel/data/domestic_supply{os.sep}{item}_2020_Global_supply.csv"
        if os.path.isfile(ds_fname):
            print(f"{item} domestic supply file already exists, skipping.")
            continue
        try:
            DS = DomesticSupply(item, 2020)
        except FileNotFoundError:
            print(f"{item} production/trade data not found, skipping.")
            continue
        except (np.linalg.LinAlgError, np.core._exceptions._UFuncInputCastingError):
            print(f"{item} has a singular matrix problem, skipping.")
            continue
        ds = DS.get_domestic_supply()
        try:
            ds.to_csv(ds_fname)
        except AttributeError:
            print(f"{item} seems result in a single value:", ds)
            print("Domestic supply file shall not be made.")
            continue
