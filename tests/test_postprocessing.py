from src.model import PyTradeShifts
from src.postprocessing import Postprocessing
import pandas as pd


def test_init():
    # TODO: replace this with a fixture
    ISIMIP = pd.read_csv(
        "." + "/data/scenario_files/ISIMIP_climate/ISIMIP_wheat_Hedlung.csv",
        index_col=0,
    )
    nan_indices = ISIMIP.index[ISIMIP.iloc[:, 0].isnull()].tolist()
    Wheat2018 = PyTradeShifts(
        "Wheat",
        2018,
        region="Global",
        testing=False,
        countries_to_remove=nan_indices,
        cd_kwargs={"seed": 2},
        make_plot=False,
    )
    ISIMIP = PyTradeShifts(
        crop="Wheat",
        base_year=2018,
        scenario_file_name="ISIMIP_climate/ISIMIP_wheat_Hedlung.csv",
        scenario_name="ISIMIP",
        countries_to_remove=nan_indices,
        cd_kwargs={"seed": 2},
        make_plot=False,
    )
    pp = Postprocessing(
        [Wheat2018, Wheat2018, ISIMIP], anchor_countries=["China", "Russia"]
    )
    pp.print_distance_metrics()


if __name__ == "__main__":
    test_init()
