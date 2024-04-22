import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.model import PyTradeShifts
import country_converter as coco


class DomesticSupply(PyTradeShifts):
    def __init__(
        self,
        item: str,
        base_year: int,
        region="Global",
        scenario_name=None,
        scenario_file_name=None,
    ) -> None:
        super().__init__(
            item,
            base_year,
            region=region,
            scenario_name=scenario_name,
            scenario_file_name=scenario_file_name,
            testing=True,
        )
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

    def apply_scenario(self) -> None:
        """
        Loads the scenario files unifies the names and applies the scenario to the trade matrix.
        by multiplying the trade matrix with the scenario scalar.

        This assumes that the scenario file consists of a csv file with the country names
        as the index and the changes in production as the only column.

        Arguments:
            None

        Returns:
            None
        """
        assert self.scenario_name is not None
        assert self.scenario_file_name is not None
        assert self.scenario_run is False
        self.scenario_run = True

        # Read in the scenario data
        scenario_data = pd.read_csv(
            self.scenario_file_name,
            index_col=0,
        ).squeeze()
        # Cast all the values to float
        scenario_data = pd.to_numeric(scenario_data, errors="raise")

        # make sure that this only contains numbers
        assert scenario_data.dtype == float

        assert isinstance(scenario_data, pd.Series)

        # Make sure that all the values are above -100, as this is a percentage change
        assert scenario_data.min() >= -100

        # Convert the percentage change to a scalar, so we can multiply the trade matrix with it
        scenario_data = 1 + scenario_data / 100

        # Make sure that all the values are above 0, as yield cannot become negative
        assert scenario_data.min() >= 0

        # Drop all NaNs
        scenario_data = scenario_data.dropna()

        cc = coco.CountryConverter()
        # Convert the country names to the same format as in the trade matrix
        scenario_data.index = cc.pandas_convert(
            pd.Series(scenario_data.index), to="name_short"
        )

        if self.only_keep_scenario_countries:
            # Only keep the countries that are in the trade matrix index, trade matrix columns and
            # the scenario data
            countries = np.intersect1d(
                np.intersect1d(self.trade_matrix.index, self.trade_matrix.columns),
                scenario_data.index,
            )
            self.trade_matrix = self.trade_matrix.loc[countries, countries]
            scenario_data = scenario_data.loc[countries]

            # Sort the indices
            self.trade_matrix = self.trade_matrix.sort_index(axis=0).sort_index(axis=1)
            scenario_data = scenario_data.sort_index()

            # Make sure the indices + columns are the same
            assert self.trade_matrix.index.equals(self.trade_matrix.columns)
            assert self.trade_matrix.index.equals(scenario_data.index)

            # Multiply all the columns with the scenario data
            self.trade_matrix = self.trade_matrix.mul(scenario_data.values, axis=0)
        else:
            # Multiply the trade matrix with the scenario data, but only for the countries
            # that are in the scenario data. Still keep all the countries in the trade matrix.
            # But first remove that are in the scenario data but not in the trade matrix, as
            # we are not interested in them.

            # Filter scenario data to include only countries present in the trade matrix
            scenario_data = scenario_data[
                scenario_data.index.isin(self.trade_matrix.index)
            ]
            # Add all the countries that are in the trade matrix but not in the scenario data
            # to the scenario data with a scalar of 1 (which means their production does not change)
            scenario_data = scenario_data.reindex(self.trade_matrix.index, fill_value=1)

            # Update trade matrix values based on scenario data (masking for missing values)
            self.trade_matrix = self.trade_matrix.mul(scenario_data, axis=0)

            # Assert index consistency
            assert self.trade_matrix.index.equals(self.trade_matrix.columns)

        print(f"Applied scenario {self.scenario_name}.")


def get_allowed_items():
    nd = pd.read_csv("intmodel/data/primary_crop_nutritional_data.csv")
    assert nd["Item"].is_unique
    return nd["Item"].values


def get_scenarios(scenarios_dir):
    scenario_files = [
        f
        for f in os.listdir(scenarios_dir)
        if os.path.isfile(os.path.join(scenarios_dir, f))
    ]
    return scenario_files


def main():
    allowed_items = get_allowed_items()
    print("Computing domestic supply with no scenario.")
    for item in tqdm(allowed_items):
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
    print("Computing domestic supply with crop reduction.")
    scenarios_dir = "intmodel/data/scenario_files"
    scenarios = get_scenarios(scenarios_dir)
    for item in tqdm(allowed_items):
        for scenario in scenarios:
            ds_fname = f"intmodel/data/domestic_supply{os.sep}{item}_2020_Global_supply_{scenario}"
            if os.path.isfile(ds_fname):
                print(f"{item} domestic supply file already exists, skipping.")
                continue
            try:
                DS = DomesticSupply(
                    item,
                    2020,
                    scenario_name=scenario,
                    scenario_file_name=os.path.join(scenarios_dir, scenario),
                )
            except FileNotFoundError:
                print(
                    f"{item} production/trade or {scenario} data not found, skipping."
                )
                continue
            except (np.linalg.LinAlgError, np.core._exceptions._UFuncInputCastingError):
                print(
                    f"{item} in scenario:{scenario}, has a singular matrix problem, skipping."
                )
                continue
            ds = DS.get_domestic_supply()
            try:
                ds.to_csv(ds_fname)
            except AttributeError:
                print(f"{item} seems result in a single value:", ds)
                print("Domestic supply file shall not be made.")
                continue


if __name__ == "__main__":
    main()
