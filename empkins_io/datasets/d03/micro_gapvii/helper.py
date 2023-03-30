from pathlib import Path
import os
from typing import Dict, Optional, Tuple
import pandas as pd

from biopsykit.io.biopac import BiopacDataset
from biopsykit.io import load_atimelogger_file
from empkins_micro.utils._types import path_t
from empkins_io.utils.exceptions import TimelogNotFoundException, SamplingRateMismatchException


def _build_data_path(base_path: path_t, participant_id: str, study_protocol: str) -> Path:
    data_path = base_path.joinpath(f"data_per_subject/{participant_id}/{study_protocol}")
    assert data_path.exists()
    return data_path


def _load_biopac_data(base_path: path_t, participant_id: str, study_protocol: str) -> Tuple[pd.DataFrame, int]:
    biopac_dir_path = _build_data_path(base_path, participant_id=participant_id, study_protocol=study_protocol).joinpath("biopac/raw")

    biopac_file_path = biopac_dir_path.joinpath(f"biopac_data_{participant_id}_{study_protocol}.acq")

    dataset_biopac = BiopacDataset.from_acq_file(biopac_file_path)
    biopac_df = dataset_biopac.data_as_df(index="local_datetime")
    fs = dataset_biopac._sampling_rate

    # check if biopac sampling rate is the same for each channel
    sampling_rates = set(fs.values())
    if len(sampling_rates) > 1:
        raise SamplingRateMismatchException(f"Biopac sampling rates are not the same for every channel! Found sampling rates: {sampling_rates}")

    fs = list(sampling_rates)[0]
    return biopac_df, fs

def _load_timelog(base_path: path_t, participant_id: str, study_protocol: str) -> pd.DataFrame:
    timelog_dir_path = _build_data_path(base_path, participant_id=participant_id, study_protocol=study_protocol).joinpath("timelog/cleaned")
    timelog_file_path = timelog_dir_path.joinpath(f"{participant_id}_{study_protocol}_cleaned_timelog.csv")
    if timelog_file_path.exists():
        timelog = load_atimelogger_file(timelog_file_path, timezone="Europe/Berlin")
        return timelog
    raise TimelogNotFoundException(f"No cleaned timelog file was found for {participant_id}! "
                                   "Run the 'notebooks/clean_timelog.ipynb' notebook first!")

