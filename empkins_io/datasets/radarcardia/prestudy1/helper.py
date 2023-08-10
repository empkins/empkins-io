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
)


def _build_data_path(base_path: path_t, participant_id: str) -> Path:
    data_path = base_path.joinpath(f"data_per_subject/{participant_id}")
    assert data_path.exists()
    return data_path


def _load_biopac_data(base_path: path_t, participant_id: str, channel_mapping: dict) -> Tuple[pd.DataFrame, int]:
    biopac_dir_path = _build_data_path(base_path, participant_id=participant_id).joinpath(
        "biopac/raw"
    )
    biopac_file_path = biopac_dir_path.joinpath(f"biopac_data_{participant_id}.acq")

    dataset_biopac = BiopacDataset.from_acq_file(biopac_file_path, channel_mapping=channel_mapping)

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


def _load_radar_data(base_path: path_t, participant_id: str, fs: float) -> tuple[DataFrame, float]:
    radar_dir_path = _build_data_path(base_path, participant_id=participant_id).joinpath(
        "emrad/raw"
    )
    radar_file_path = radar_dir_path.joinpath(f"emrad_data_{participant_id}.h5")

    dataset_radar = EmradDataset.from_hd5_file(radar_file_path, sampling_rate_hz=fs)
    radar_df = dataset_radar.data_as_df(index="local_datetime", add_sync_out=True)["rad1"]
    fs = dataset_radar.sampling_rate_hz
    return radar_df, fs


def _load_timelog(base_path: path_t, participant_id: str, exclude_fail: bool) -> pd.DataFrame:
    timelog_dir_path = _build_data_path(base_path, participant_id=participant_id).joinpath(
        "timelog/raw"
    )
    timelog_file_path = timelog_dir_path.joinpath(f"timelog_{participant_id}.csv")
    if timelog_file_path.exists():
        # timelog = load_atimelogger_file(timelog_file_path, timezone="Europe/Berlin")
        timelog = _test_atime(timelog_file_path, timezone="Europe/Berlin", exclude_fail=exclude_fail)
        return timelog
    raise TimelogNotFoundException(
        f"No timelog file was found for {participant_id}!"
    )


def _test_atime(file_path, timezone, exclude_fail):
    file_path = Path(file_path)
    timelog = pd.read_csv(file_path)

    # find out if file is german or english and get the right column names
    if "Aktivitätstyp" in timelog.columns:
        phase_col = "Aktivitätstyp"
        time_cols = ["Von", "Bis"]
        fail_col = ["Kommentar"]
    elif "Activity type" in timelog.columns:
        phase_col = "Activity type"
        time_cols = ["From", "To"]
    else:
        phase_col = "phase"
        time_cols = ["start", "end"]

    fail_col = timelog[fail_col].isna()

    timelog = timelog.set_index(phase_col)
    timelog = timelog[time_cols]



    # timelog = timelog.rename(columns={time_cols[0]: "start", time_cols[1]: "end"})
    # timelog.index.name = "phase"
    # timelog.columns.name = "start_end"

    # return timelog

    # timelog = timelog.apply(pd.to_datetime, axis=1).applymap(lambda val: val.tz_localize(timezone))
    # timelog = pd.DataFrame(timelog.T.unstack(), columns=["time"])
    # timelog = timelog[::-1].reindex(["start", "end"], level="start_end")
    # timelog = timelog.T
    # return timelog
