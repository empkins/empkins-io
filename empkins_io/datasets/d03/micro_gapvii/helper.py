from pathlib import Path
from typing import Tuple, Dict

import pandas as pd
import numpy as np
from biopsykit.io import load_atimelogger_file
from biopsykit.io.biopac import BiopacDataset
from biopsykit.io.nilspod import _handle_counter_inconsistencies_session
from nilspodlib.exceptions import SynchronisationError, SessionValidationError, InvalidInputFileError
from pandas import DataFrame

from empkins_io.datasets.d03.micro_gapvii._custom_synced_session import CustomSyncedSession
from empkins_io.sensors.emrad import EmradDataset
from empkins_io.utils._types import path_t
from empkins_io.utils.exceptions import (
    SamplingRateMismatchException,
    TimelogNotFoundException,
    NilsPodDataNotFoundException,
    NilsPodDataLoadException,
)


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


def _load_radar_data(base_path: path_t, participant_id: str, condition: str) -> tuple[DataFrame, float]:
    radar_dir_path = _build_data_path(base_path, participant_id=participant_id, condition=condition).joinpath(
        "emrad/raw"
    )
    radar_file_path = radar_dir_path.joinpath(f"emrad_data_{participant_id}_{condition}.h5")

    dataset_radar = EmradDataset.from_hd5_file(radar_file_path)
    radar_df = dataset_radar.data_as_df(index="local_datetime")
    # radar_df.index.name = "time"
    fs = dataset_radar.sampling_rate_hz
    return radar_df, fs


def _load_timelog(base_path: path_t, participant_id: str, condition: str, phase: str, phase_fine: bool) -> pd.DataFrame:
    timelog_dir_path = _build_data_path(base_path, participant_id=participant_id, condition=condition).joinpath(
        "timelog/cleaned"
    )
    timelog_file_path = timelog_dir_path.joinpath(f"{participant_id}_{condition}_processed_phases_timelog.csv")
    if timelog_file_path.exists():
        timelog = load_atimelogger_file(timelog_file_path, timezone="Europe/Berlin")
        if (phase == "all") & phase_fine:
            timelog_fine = timelog.drop("Talk", axis=1, level=0)
            timelog_fine = timelog_fine.drop("Math", axis=1, level=0)
            return timelog_fine
        elif (phase == "all") & (not phase_fine):
            timelog_coarse = timelog.drop("Talk_1", axis=1, level=0)
            timelog_coarse = timelog_coarse.drop("Talk_2", axis=1, level=0)
            timelog_coarse = timelog_coarse.drop("Math_1", axis=1, level=0)
            timelog_coarse = timelog_coarse.drop("Math_2", axis=1, level=0)
            return timelog_coarse
        else:
            timelog = timelog.iloc[:, timelog.columns.get_level_values(0) == phase]
            return timelog
    raise TimelogNotFoundException(
        f"No cleaned timelog file was found for {participant_id}! "
        "Run the 'notebooks/clean_timelog.ipynb' notebook first!"
    )


def _load_nilspod_session(base_path: path_t, participant_id: str, condition: str) -> Tuple[pd.DataFrame, float]:
    data_path = _build_data_path(base_path, participant_id=participant_id, condition=condition)
    data_path = data_path.joinpath("nilspod/raw")

    nilspod_files = sorted(data_path.glob("NilsPodX-*.bin"))
    if len(nilspod_files) == 0:
        raise NilsPodDataNotFoundException("No NilsPod files found in directory!")

    try:
        session = CustomSyncedSession.from_folder_path(data_path)

        # fix for "classical nilspod bug" where last sample counter is corrupted
        session.cut(stop=-10, inplace=True)
        session.align_to_syncregion(inplace=True)
    except (ZeroDivisionError, SynchronisationError, SessionValidationError, InvalidInputFileError, KeyError) as e:
        raise NilsPodDataLoadException("Cannot load NilsPod data!") from e

    _handle_counter_inconsistencies_session(session, handle_counter_inconsistency="ignore")

    # convert dataset to dataframe and localize timestamp
    df = session.data_as_df(index="local_datetime", concat_df=True)
    df.index.name = "time"
    fs = session.info.sampling_rate_hz[0]
    return df, fs
