import pandas as pd

from empkins_io.utils.exceptions import CleanedOpenposeDataNotFoundException, OpenposeDataNotFoundException


def get_uncleaned_openpose_data(file_path):
    """Returns the openpose data for a single recording (= single subject and condition)."""
    if not file_path.exists():
        raise OpenposeDataNotFoundException(f"Openpose data not found in {file_path}.")
    df = pd.read_csv(file_path, header=[0, 1, 2], index_col=0)
    return df


def get_cleaned_openpose_data(file_path, phase_start=None, phase_end=None):
    """Returns the cleaned openpose data at path file_path. If phase_start and phase_end are given,
    the data is cut to the specified timestamps."""
    if not file_path.exists():
        raise CleanedOpenposeDataNotFoundException(f"Cleaned Openpose data not found in {file_path}.")
    df = pd.read_csv(file_path, header=[0, 1, 2, 3], index_col=0)
    if phase_start is None or phase_end is None:
        return df
    return df[phase_start:phase_end]
