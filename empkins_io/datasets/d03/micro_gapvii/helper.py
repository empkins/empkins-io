from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
from biopsykit.io import load_atimelogger_file
from biopsykit.io.biopac import BiopacDataset

from empkins_io.sensors.emrad import EmradDataset
from empkins_io.utils._types import path_t
from empkins_io.utils.exceptions import SamplingRateMismatchException, TimelogNotFoundException


def _build_data_path(base_path: path_t, participant_id: str, condition: str) -> Path:
    data_path = base_path.joinpath(f"data_per_subject/{participant_id}/{condition}")
    assert data_path.exists()
    return data_path


def _load_biopac_data(base_path: path_t, participant_id: str, condition: str) -> Tuple[pd.DataFrame, int]:
    biopac_dir_path = _build_data_path(base_path, participant_id=participant_id, condition=condition).joinpath(
        "biopac/raw"
    )

    biopac_file_path = biopac_dir_path.joinpath(f"biopac_data_{participant_id}_{condition}.acq")

    dataset_biopac = BiopacDataset.from_acq_file(biopac_file_path)
    biopac_df = dataset_biopac.data_as_df(index="local_datetime")
    fs = dataset_biopac._sampling_rate

    # check if biopac sampling rate is the same for each channel
    sampling_rates = set(fs.values())
    if len(sampling_rates) > 1:
        raise SamplingRateMismatchException(
            f"Biopac sampling rates are not the same for every channel! Found sampling rates: {sampling_rates}"
        )

    fs = list(sampling_rates)[0]
    return biopac_df, fs

def _load_emrad_data(base_path: path_t, participant_id: str, condition: str) -> EmradDataset:
    emrad_dir = _build_data_path(base_path, participant_id=participant_id, condition=condition).joinpath("emrad/raw")

    emrad_file = emrad_dir.joinpath(f"emrad_data_{participant_id}_{condition}.hdf5")

    return EmradDataset.from_hd5_file(emrad_file)

def _load_timelog(base_path: path_t, participant_id: str, condition: str, phase: str, phase_fine: bool) -> pd.DataFrame:
    timelog_dir_path = _build_data_path(base_path, participant_id=participant_id, condition=condition).joinpath(
        "timelog/cleaned"
    )
    timelog_file_path = timelog_dir_path.joinpath(f"{participant_id}_{condition}_processed_phases_timelog.csv")
    if timelog_file_path.exists():
        timelog = load_atimelogger_file(timelog_file_path, timezone="Europe/Berlin")
        if (phase == "all") & phase_fine:
            timelog_fine = timelog.drop('Talk', axis=1, level=0)
            timelog_fine = timelog_fine.drop('Math', axis=1, level=0)
            return timelog_fine
        elif (phase == "all") & (not phase_fine):
            timelog_coarse = timelog.drop('Talk_1', axis=1, level=0)
            timelog_coarse = timelog_coarse.drop('Talk_2', axis=1, level=0)
            timelog_coarse = timelog_coarse.drop('Math_1', axis=1, level=0)
            timelog_coarse = timelog_coarse.drop('Math_2', axis=1, level=0)
            return timelog_coarse
        else:
            timelog = timelog.iloc[:, timelog.columns.get_level_values(0) == phase]
            return timelog
    raise TimelogNotFoundException(
        f"No cleaned timelog file was found for {participant_id}! "
        "Run the 'notebooks/clean_timelog.ipynb' notebook first!"
    )
