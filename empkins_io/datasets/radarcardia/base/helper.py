import datetime
from pathlib import Path
from typing import Dict, Literal, Optional, Sequence, Tuple, Union

import pandas as pd
import numpy as np
from biopsykit.io.biopac import BiopacDataset
from biopsykit.utils.time import tz
from pandas import DataFrame

from empkins_io.sensors.emrad import EmradDataset
from empkins_io.utils._types import path_t
from empkins_io.utils.exceptions import (
    SamplingRateMismatchException,
    TimelogNotFoundException,
)

import bioread


def _build_data_path(base_path: path_t, participant_id: str) -> Path:
    data_path = base_path.joinpath(f"data_per_subject/{participant_id}")
    assert data_path.exists()
    return data_path


def _build_timelog_path(base_path: path_t, participant_id: str) -> Path:
    timelog_dir_path = _build_data_path(base_path=base_path, participant_id=participant_id).joinpath(
        "timelog/processed"
    )
    timelog_file_path = timelog_dir_path.joinpath(f"timelog_{participant_id}.csv")
    return timelog_file_path


def _load_biopac_data(base_path: path_t, participant_id: str, channel_mapping: dict, state: str) -> Tuple[
    pd.DataFrame, int]:
    if state == "raw":
        return _load_biopac_raw_data(base_path=base_path, participant_id=participant_id,
                                     channel_mapping=channel_mapping)

    if state == "aligned":
        return _load_biopac_aligned_data(base_path=base_path, participant_id=participant_id)


def _load_biopac_raw_data(base_path: path_t, participant_id: str, channel_mapping: dict) -> Tuple[pd.DataFrame, int]:
    biopac_dir_path = _build_data_path(base_path, participant_id=participant_id).joinpath(
        "biopac/raw"
    )
    biopac_file_path = biopac_dir_path.joinpath(f"biopac_data_{participant_id}.acq")

    dataset_biopac = BiopacDataset.from_acq_file(biopac_file_path, channel_mapping=channel_mapping)
    biopac_df = dataset_biopac.data_as_df(index="local_datetime")
    # biopac_df = dataset_biopac.data_as_df(index="time")
    fs = dataset_biopac._sampling_rate

    # check if biopac sampling rate is the same for each channel
    sampling_rates = set(fs.values())
    if len(sampling_rates) > 1:
        raise SamplingRateMismatchException(
            f"Biopac sampling rates are not the same for every channel! Found sampling rates: {sampling_rates}"
        )

    fs = list(sampling_rates)[0]
    return biopac_df, fs


def _load_biopac_aligned_data(base_path: path_t, participant_id: str) -> DataFrame:
    data_path = _build_data_path(base_path=base_path, participant_id=participant_id)
    data_path = data_path.joinpath(f"biopac/processed/biopac_data_pp_{participant_id}.h5")

    biopac_data = pd.read_hdf(data_path, key="biopac_data")
    return biopac_data, None


def _load_radar_data(base_path: path_t, participant_id: str, fs: float, state: str) -> tuple[DataFrame, float]:
    if state == "raw":
        return _load_radar_raw_data(base_path=base_path, participant_id=participant_id, fs=fs)

    if state == "aligned":
        return _load_radar_aligned_data(base_path=base_path, participant_id=participant_id)


def _load_radar_raw_data(base_path: path_t, participant_id: str, fs: float) -> tuple[DataFrame, float]:
    radar_dir_path = _build_data_path(base_path, participant_id=participant_id).joinpath(
        "emrad/raw"
    )
    radar_file_path = radar_dir_path.joinpath(f"emrad_data_{participant_id}.h5")

    dataset_radar = EmradDataset.from_hd5_file(radar_file_path, sampling_rate_hz=fs)
    radar_df = dataset_radar.data_as_df(index="local_datetime", add_sync_out=True)["rad1"]
    # radar_df = dataset_radar.data_as_df(index="time", add_sync_out=True)["rad1"]
    fs = dataset_radar.sampling_rate_hz
    return radar_df, fs


def _load_radar_aligned_data(base_path: path_t, participant_id: str) -> DataFrame:
    data_path = _build_data_path(base_path=base_path, participant_id=participant_id)
    data_path = data_path.joinpath(f"emrad/processed/emrad_data_pp_{participant_id}.h5")

    radar_data = pd.read_hdf(data_path, key="emrad_data")
    return radar_data, None


def _load_timelog(base_path: path_t, participant_id: str) -> pd.DataFrame:
    timelog_file_path = _build_timelog_path(base_path=base_path, participant_id=participant_id)
    if timelog_file_path.exists():
        timelog = _load_atimelogger_file(timelog_file_path, timezone="Europe/Berlin")
        return timelog
    raise TimelogNotFoundException(
        f"No timelog file was found for {participant_id}!"
    )


def _load_atimelogger_file(file_path: path_t, timezone: Optional[Union[datetime.tzinfo, str]] = None) -> pd.DataFrame:
    """Load time log file exported from the aTimeLogger app.

    The resulting dataframe will have one row and start and end times of the single phases as columns.

    Parameters
    ----------
    file_path : :class:`~pathlib.Path` or str
        path to time log file. Must a csv file
    timezone : str or :class:`datetime.tzinfo`, optional
        timezone of the time logs, either as string or as `tzinfo` object.
        Default: 'Europe/Berlin'

    Returns
    -------
    :class:`~pandas.DataFrame`
        time log dataframe

    See Also
    --------
    :func:`~biopsykit.utils.io.convert_time_log_datetime`
        convert timelog dataframe into dictionary
    `aTimeLogger app <https://play.google.com/store/apps/details?id=com.aloggers.atimeloggerapp>`_

    """
    # ensure pathlib
    file_path = Path(file_path)

    if timezone is None:
        timezone = tz
    timelog = pd.read_csv(file_path)
    # find out if file is german or english and get the right column names
    if "Aktivitätstyp" in timelog.columns:
        phase_col = "Aktivitätstyp"
        time_cols = ["Von", "Bis"]
    elif "Activity type" in timelog.columns:
        phase_col = "Activity type"
        time_cols = ["From", "To"]
    else:
        phase_col = "phase"
        time_cols = ["start", "end"]

    timelog = timelog.set_index(phase_col)
    timelog = timelog[time_cols]

    timelog = timelog.rename(columns={time_cols[0]: "start", time_cols[1]: "end"})
    timelog.index.name = "location"
    timelog.columns.name = "start_end"

    timelog = timelog.apply(pd.to_datetime, axis=1).applymap(lambda val: val.tz_localize(timezone))
    timelog = pd.DataFrame(timelog.T.unstack(), columns=["time"])
    timelog = timelog[::-1].reindex(["start", "end"], level="start_end")
    timelog = timelog.T

    return timelog


def _save_aligned_data(base_path: path_t, participant_id: str, biopac: pd.DataFrame, radar: pd.DataFrame):
    data_path = _build_data_path(base_path=base_path, participant_id=participant_id)
    biopac_path = data_path.joinpath(f"biopac/processed/biopac_data_pp_{participant_id}.h5")
    radar_path = data_path.joinpath(f"emrad/processed/emrad_data_pp_{participant_id}.h5")

    biopac_path.parent.mkdir(exist_ok=True)
    radar_path.parent.mkdir(exist_ok=True)

    biopac.to_hdf(biopac_path, mode="w", key="biopac_data", index=True)
    radar.to_hdf(radar_path, mode="w", key="emrad_data", index=True)


def _save_data_to_location_h5(
        base_path: path_t,
        participant_id: str,
        data: pd.DataFrame,
        biopac: bool,
        radar: bool,
        location: str,
        file_name: str,
):
    data_path = _build_data_path(base_path=base_path, participant_id=participant_id)
    if biopac and radar:
        raise NotImplementedError("Dataframe can not be saved to radar and biopac directory")
    elif biopac:
        data_path = data_path.joinpath(f"biopac/")
    elif radar:
        data_path = data_path.joinpath(f"emrad/")
    else:
        raise ValueError("Specify in which directory the dataframe should be saved")

    data_path = data_path.joinpath(f"processed/data_per_location/{location}/{file_name}_{participant_id}.h5")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_hdf(data_path, mode="w", key="data", index=True)


def _load_data_from_location_h5(
        base_path: path_t,
        participant_id: str,
        biopac: bool,
        radar: bool,
        location: str,
        file_name: str,
):
    data_path = _build_data_path(base_path=base_path, participant_id=participant_id)
    if biopac and radar:
        raise NotImplementedError("Dataframe can not be loaded from radar and biopac directory")
    elif biopac:
        data_path = data_path.joinpath(f"biopac/")
    elif radar:
        data_path = data_path.joinpath(f"emrad/")
    else:
        raise ValueError("Specify from which directory the dataframe should be loaded")

    data_path = data_path.joinpath(f"processed/data_per_location/{location}/{file_name}_{participant_id}.h5")
    data = pd.read_hdf(data_path, key="data")

    return data


def _calc_biopac_timelog_shift(base_path: path_t, participant_id: str):

    # Biopac Sync Event Marker
    biopac_dir_path = _build_data_path(base_path, participant_id=participant_id).joinpath(
        "biopac/raw"
    )
    biopac_file_path = biopac_dir_path.joinpath(f"biopac_data_{participant_id}.acq")

    biopac_data: bioread.reader.Datafile = bioread.read(str(biopac_file_path))
    event_marker = biopac_data.event_markers
    sync_events = []
    for event in event_marker:
        if event.type == "User Type 1":
            sync_events.append(event)
    if len(sync_events) == 1:
        sync_event = sync_events[0]
    elif len(sync_events) == 0:
        raise Exception("Sync Event Marker in Biopac Data File is missing")
    else:
        raise NotImplementedError(
            "Two or more Biopac Event Markers have been detected. Handling is not implemented.")

    sync_event_time = sync_event.date_created_utc

    # Timelog Sync Entry
    timelog = _load_timelog(base_path=base_path, participant_id=participant_id)
    timelog_sync_start_time = timelog["sync"]["start"].time

    shift = sync_event_time - timelog_sync_start_time
    return shift

