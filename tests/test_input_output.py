import os.path
import warnings
from src.input_output import data


def test_input_data_files_exist():
    for k, v in data["input"].items():
        if k == "year_flag" or k == "from_raw":
            continue
        try:
            assert os.path.isfile(v), f"File not found: {v}"
        except AssertionError as AE:
            if k == "trade" or k == "production":
                warnings.warn(
                    UserWarning(
                        f"FAO zip file -- {v} -- is missing, which is fine if you're OK with the precomputed pickle."
                    )
                )
                assert (
                    data["input"]["from_raw"] is False
                ), 'With FAO zip missing \'data["input"]["from_raw"]\' must be set to False.'
            else:
                raise AE


def test_intermidiary_data_files_exist():
    for v in data["intermidiary"].values():
        assert os.path.isfile(
            v
        ), f'File not found: {v}; please run the main script with \'data["input"]["from_raw"]\' set to True.'


def test_output_data_files_exist():
    for v in data["output"].values():
        assert os.path.isfile(v), f"File not found: {v}; please run the main script."
