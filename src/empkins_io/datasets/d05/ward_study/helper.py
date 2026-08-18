from pathlib import Path
import pandas as pd


def get_raw_data_file(base_path: Path):

    file_path = base_path.joinpath("raw", "EmpkinSD05WardStudy_DATA_2026-08-18_1848.csv")
    data = pd.read_csv(file_path)

    return data

def get_ipos_external(base_path: Path, cols):

    data = get_raw_data_file(base_path)
    data = data.loc[data.redcap_repeat_instrument == "ground_truth_fremderfassung_ipos_0208", cols]

    return data