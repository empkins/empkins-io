import datetime
import json
from pathlib import Path
from typing import Literal

import bioread
import numpy as np
import pandas as pd
from biopsykit.io.biopac import BiopacDataset
from biopsykit.utils.time import tz

from empkins_io.sensors.emrad import EmradDataset
from empkins_io.sync import SyncedDataset
from empkins_io.utils._types import path_t
from empkins_io.utils.exceptions import SynchronizationError, SamplingRateMismatchError


def _get_locations_from_index(index: pd.DataFrame):
    locations = index.drop(columns="subject").values.tolist()
    locations = ["_".join(i) for i in locations]
    return locations


def _load_atimelogger_file(file_path: path_t, timezone: datetime.tzinfo | str | None = None) -> pd.DataFrame:
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

    timelog = timelog.apply(pd.to_datetime, axis=1)  # Convert rows to datetime
    timelog = timelog.apply(lambda col: col.map(lambda val: val.tz_localize(timezone)))
    timelog = pd.DataFrame(timelog.T.unstack(), columns=["time"])
    timelog = timelog[::-1].reindex(["start", "end"], level="start_end")
    timelog = timelog.T

    return timelog


def _calc_biopac_timelog_shift(base_path: path_t, participant_id: str):
    biopac_file_path = base_path.joinpath(
        f"data_per_subject/{participant_id}/biopac/raw/{participant_id}_biopac_data.acq"
    )

    biopac_data: bioread.reader.Datafile = bioread.read(str(biopac_file_path))
    event_marker = biopac_data.event_markers
    sync_events = []
    for event in event_marker:
        if event.type == "User Type 1":
            sync_events.append(event)
    if len(sync_events) == 1:
        sync_event = sync_events[0]
    elif len(sync_events) == 0:
        raise SynchronizationError("Sync Event Marker in Biopac Data File is missing")
    else:
        raise NotImplementedError("Two or more Biopac Event Markers have been detected. Handling is not implemented.")

    sync_event_time = sync_event.date_created_utc

    timelog_file_path = base_path.joinpath(
        f"data_per_subject/{participant_id}/timelog/processed/{participant_id}_timelog.csv"
    )
    timelog = _load_atimelogger_file(timelog_file_path, timezone="Europe/Berlin")
    timelog_sync_start_time = timelog["sync"]["start"].time

    # calculate time shift between biopac and timelog
    shift = sync_event_time - timelog_sync_start_time
    shift = np.floor(shift.total_seconds())

    # save shift parameters to json file
    shift_dict = {
        "sync_event_time_biopac": sync_event_time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "sync_entry_time_timelog": timelog_sync_start_time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "biopac_timelog_shift": shift,
    }
    shift_path = base_path.joinpath(
        f"data_per_subject/{participant_id}/timelog/processed/{participant_id}_timelog_shift.json"
    )
    with shift_path.open("w", encoding="utf-8") as f:
        json.dump(shift_dict, f, indent=4)

    shift = pd.Timedelta(seconds=shift)
    return shift


def _load_radar_raw(base_path: path_t, participant_id: str, fs: float) -> pd.DataFrame:
    radar_path = base_path.joinpath(f"data_per_subject/{participant_id}/emrad/cleaned/{participant_id}_emrad_data.h5")
    if radar_path.exists():
        radar_df = pd.read_hdf(radar_path, key="emrad_data")
    else:
        radar_file_path = base_path.joinpath(
            f"data_per_subject/{participant_id}/emrad/raw/{participant_id}_emrad_data.h5"
        )
        dataset_radar = EmradDataset.from_hd5_file(radar_file_path, sampling_rate_hz=fs)
        radar_df = dataset_radar.data_as_df(index="local_datetime", add_sync_out=True)["rad2"]
        radar_path.parent.mkdir(parents=True, exist_ok=True)
        radar_df.to_hdf(radar_path, mode="w", key="emrad_data", index=True)
    return radar_df


def _load_biopac_raw(base_path: path_t, participant_id: str, channel_mapping: dict):
    biopac_path = base_path.joinpath(
        f"data_per_subject/{participant_id}/biopac/cleaned/{participant_id}_biopac_data.h5"
    )
    if biopac_path.exists():
        biopac_df = pd.read_hdf(biopac_path, key="biopac_data")
    else:
        biopac_file_path = base_path.joinpath(
            f"data_per_subject/{participant_id}/biopac/raw/{participant_id}_biopac_data.acq"
        )
        dataset_biopac = BiopacDataset.from_acq_file(biopac_file_path, channel_mapping=channel_mapping)
        biopac_df = dataset_biopac.data_as_df(index="local_datetime")
        fs = dataset_biopac._sampling_rate
        sampling_rates = set(fs.values())
        if len(sampling_rates) > 1:
            raise SamplingRateMismatchError(
                f"Biopac sampling rates are not the same for every channel! Found sampling rates: {sampling_rates}"
            )
        biopac_path.parent.mkdir(parents=True, exist_ok=True)
        biopac_df.to_hdf(biopac_path, mode="w", key="biopac_data", index=True)
    return biopac_df


def _load_radar_synced(base_path, subject, BIOPAC_CHANNEL_MAPPING, _SAMPLING_RATES, index):
    data_path = base_path.joinpath(f"data_per_subject/{subject}/emrad/processed/{subject}_emrad_data.h5")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_synced(base_path, subject, BIOPAC_CHANNEL_MAPPING, _SAMPLING_RATES, index, resample=True)
    data = pd.read_hdf(data_path, key=f"emrad_data")
    return data


def _load_biopac_synced(base_path, subject, BIOPAC_CHANNEL_MAPPING, _SAMPLING_RATES, index):
    data_path = base_path.joinpath(f"data_per_subject/{subject}/biopac/processed/{subject}_biopac_data.h5")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_synced(base_path, subject, BIOPAC_CHANNEL_MAPPING, _SAMPLING_RATES, index, resample=True)
    data = pd.read_hdf(data_path, key=f"biopac_data")
    return data


def _sync_datasets(
    base_path: path_t, participant_id: str, channel_mapping: dict, fs: dict, index, location: str, resample: bool
) -> SyncedDataset:
    if resample:
        biopac_data = _load_biopac_raw(
            base_path=base_path, participant_id=participant_id, channel_mapping=channel_mapping
        )
        radar_data = _load_radar_raw(base_path=base_path, participant_id=participant_id, fs=fs["radar_original"])
    else:
        radar_data = _load_radar_synced(base_path, participant_id, channel_mapping, fs, index)
        biopac_data = _load_biopac_synced(base_path, participant_id, channel_mapping, fs, index)

    index_start = max(biopac_data.index[0], radar_data.index[0])
    index_end = min(biopac_data.index[-1], radar_data.index[-1])
    biopac_data = biopac_data.loc[index_start:index_end]
    radar_data = radar_data.loc[index_start:index_end]

    synced_dataset = SyncedDataset(sync_type="m-sequence")
    if resample:
        synced_dataset.add_dataset(
            "biopac", data=biopac_data, sync_channel_name="sync", sampling_rate=fs["biopac_original"]
        )
        synced_dataset.add_dataset(
            "radar", data=radar_data, sync_channel_name="Sync_Out", sampling_rate=fs["radar_original"]
        )
        synced_dataset.resample_datasets(fs_out=fs["resampled"], method="dynamic", wave_frequency=10)
    else:
        synced_dataset.add_dataset("biopac", data=biopac_data, sync_channel_name="sync", sampling_rate=fs["resampled"])
        synced_dataset.add_dataset(
            "radar", data=radar_data, sync_channel_name="Sync_Out", sampling_rate=fs["resampled"]
        )
    synced_dataset.align_and_cut_m_sequence(primary="radar", reset_time_axis=True, cut_to_shortest=True)
    return synced_dataset


def _ensure_synced(
    base_path, subject, BIOPAC_CHANNEL_MAPPING, _SAMPLING_RATES, index, resample: bool, location=""
) -> None:
    if resample:
        radar_path = base_path.joinpath(f"data_per_subject/{subject}/emrad/processed/{subject}_emrad_data.h5")
        radar_path.parent.mkdir(parents=True, exist_ok=True)
        biopac_path = base_path.joinpath(f"data_per_subject/{subject}/biopac/processed/{subject}_biopac_data.h5")
        biopac_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        radar_path = base_path.joinpath(
            f"data_per_subject/{subject}/data_per_location/{location}/{subject}_emrad_data.h5"
        )
        biopac_path = base_path.joinpath(
            f"data_per_subject/{subject}/data_per_location/{location}/{subject}_biopac_data.h5"
        )
    if radar_path.exists() and biopac_path.exists():
        return
    else:
        synced_datasets = _sync_datasets(
            base_path,
            participant_id=subject,
            channel_mapping=BIOPAC_CHANNEL_MAPPING,
            fs=_SAMPLING_RATES,
            index=index,
            location=_get_locations_from_index(index)[0],
            resample=resample,
        )
        if not resample:
            base_path1 = base_path.joinpath(f"data_per_subject/{subject}/data_per_location/{location}")
            base_path1.mkdir(parents=True, exist_ok=True)
        synced_datasets.datasets_aligned["radar_aligned_"].to_hdf(radar_path, mode="w", key="emrad_data", index=True)
        synced_datasets.datasets_aligned["biopac_aligned_"].to_hdf(biopac_path, mode="w", key="biopac_data", index=True)
