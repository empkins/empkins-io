import datetime
from pathlib import Path
from typing import Dict, Literal, Optional, Sequence, Tuple, Union

import pandas as pd
import numpy as np
from biopsykit.io.biopac import BiopacDataset
from biopsykit.utils.time import tz
from pandas import DataFrame
from empkins_io.sync import SyncedDataset
from empkins_io.sensors.emrad import EmradDataset
from empkins_io.utils._types import path_t
from empkins_io.utils.exceptions import (
    SamplingRateMismatchException,
    TimelogNotFoundException,
)

import bioread
import json
import os


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


def _build_protocol_path(base_path: path_t, participant_id: str) -> Path:
    protocol_dir_path = _build_data_path(base_path=base_path, participant_id=participant_id).joinpath(
        "protocol/processed"
    )
    protocol_file_path = protocol_dir_path.joinpath(f"protocol_{participant_id}.csv")
    return protocol_file_path


def _load_biopac_data(
        base_path: path_t,
        participant_id: str,
        fs: dict,
        channel_mapping: dict,
        state: str,
        trigger_extraction: bool,
        location: str
) -> pd.DataFrame:

    if state == "raw_unsynced":
        biopac = _load_biopac_raw_unsynced_data(
            base_path=base_path,
            participant_id=participant_id,
            channel_mapping=channel_mapping,
            trigger_extraction=trigger_extraction
        )
        return biopac

    if state == "raw_synced":
        biopac = _load_biopac_raw_synced_data(
            base_path=base_path,
            participant_id=participant_id,
            fs=fs,
            channel_mapping=channel_mapping,
            trigger_extraction=trigger_extraction
        )
        return biopac

    if state == "location_synced":
        biopac = _load_biopac_location_synced_data(
            base_path=base_path,
            participant_id=participant_id,
            fs=fs,
            channel_mapping=channel_mapping,
            trigger_extraction=trigger_extraction,
            location=location
        )
        return biopac
    else:
        raise ValueError("Invalid Biopac data processing state")


def _load_biopac_raw_unsynced_data(
        base_path: path_t, participant_id: str, channel_mapping: dict, trigger_extraction: bool
) -> pd.DataFrame:
    biopac_path = _build_data_path(base_path=base_path, participant_id=participant_id).joinpath(
        f"biopac/cleaned/biopac_data_{participant_id}.h5"
    )

    if biopac_path.exists() and not trigger_extraction:
        biopac_df = pd.read_hdf(biopac_path, key="biopac_data")
    else:
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

        biopac_df.to_hdf(biopac_path, mode="w", key="biopac_data", index=True)

    return biopac_df


def _load_biopac_raw_synced_data(
        base_path: path_t, participant_id: str, fs: dict, channel_mapping: dict, trigger_extraction: bool
) -> DataFrame:
    biopac_path = _build_data_path(base_path=base_path, participant_id=participant_id).joinpath(
        f"biopac/processed/biopac_data_{participant_id}.h5"
    )
    if not biopac_path.exists() or trigger_extraction:
        synced_dataset = _sync_datasets(
            base_path=base_path,
            participant_id=participant_id,
            channel_mapping=channel_mapping,
            fs=fs,
            trigger_extraction=trigger_extraction
        )
        _save_raw_synced_data(base_path=base_path, participant_id=participant_id, synced_dataset=synced_dataset)

    biopac_data = pd.read_hdf(biopac_path, key="biopac_data")
    return biopac_data


def _load_biopac_location_synced_data(
        base_path: path_t,
        participant_id: str,
        fs: dict,
        channel_mapping: dict,
        trigger_extraction: bool,
        location: str
) -> DataFrame:
    biopac_path = _build_data_path(base_path=base_path, participant_id=participant_id).joinpath(
        f"biopac/processed/data_per_location/{location}/biopac_data_{participant_id}.h5"
    )
    if not biopac_path.exists() or trigger_extraction:
        synced_dataset = _sync_datasets(
            base_path=base_path,
            participant_id=participant_id,
            channel_mapping=channel_mapping,
            fs=fs,
            trigger_extraction=trigger_extraction,
            location=location
        )
        _save_location_synced_data(
            base_path=base_path, participant_id=participant_id, synced_dataset=synced_dataset, location=location
        )

    biopac_data = pd.read_hdf(biopac_path, key="biopac_data")
    return biopac_data


def _load_radar_data(
        base_path: path_t,
        participant_id: str,
        fs: dict,
        channel_mapping: dict,
        state: str,
        trigger_extraction: bool,
        location: str
) -> DataFrame:

    if state == "raw_unsynced":
        radar = _load_radar_raw_unsynced_data(
            base_path=base_path,
            participant_id=participant_id,
            fs=fs["radar_original"],
            trigger_extraction=trigger_extraction
        )
        return radar

    if state == "raw_synced":
        radar = _load_radar_raw_synced_data(
            base_path=base_path,
            participant_id=participant_id,
            fs=fs,
            channel_mapping=channel_mapping,
            trigger_extraction=trigger_extraction
        )
        return radar

    if state == "location_synced":
        radar = _load_radar_location_synced_data(
            base_path=base_path,
            participant_id=participant_id,
            fs=fs,
            channel_mapping=channel_mapping,
            trigger_extraction=trigger_extraction,
            location=location
        )
        return radar
    else:
        raise ValueError("Invalid Biopac data processing state")


def _load_radar_raw_unsynced_data(
        base_path: path_t, participant_id: str, fs: float, trigger_extraction: bool
) -> DataFrame:
    radar_path = _build_data_path(base_path=base_path, participant_id=participant_id).joinpath(
        f"emrad/cleaned/emrad_data_{participant_id}.h5"
    )

    if radar_path.exists() and not trigger_extraction:
        radar_df = pd.read_hdf(radar_path, key="emrad_data")
    else:
        radar_dir_path = _build_data_path(base_path, participant_id=participant_id).joinpath(
            "emrad/raw"
        )
        radar_file_path = radar_dir_path.joinpath(f"emrad_data_{participant_id}.h5")

        dataset_radar = EmradDataset.from_hd5_file(radar_file_path, sampling_rate_hz=fs)
        radar_df = dataset_radar.data_as_df(index="local_datetime", add_sync_out=True)["rad1"]

        radar_df.to_hdf(radar_path, mode="w", key="emrad_data", index=True)

    return radar_df


def _load_radar_raw_synced_data(
        base_path: path_t, participant_id: str, fs: dict, channel_mapping: dict, trigger_extraction: bool
) -> DataFrame:
    radar_path = _build_data_path(base_path=base_path, participant_id=participant_id).joinpath(
        f"emrad/processed/emrad_data_{participant_id}.h5"
    )

    if not radar_path.exists() or trigger_extraction:
        synced_dataset = _sync_datasets(
            base_path=base_path,
            participant_id=participant_id,
            channel_mapping=channel_mapping,
            fs=fs,
            trigger_extraction=trigger_extraction
        )
        _save_raw_synced_data(base_path=base_path, participant_id=participant_id, synced_dataset=synced_dataset)

    radar_data = pd.read_hdf(radar_path, key="emrad_data")
    return radar_data


def _load_radar_location_synced_data(
        base_path: path_t,
        participant_id: str,
        fs: dict,
        channel_mapping: dict,
        trigger_extraction: bool,
        location: str
) -> DataFrame:
    radar_path = _build_data_path(base_path=base_path, participant_id=participant_id).joinpath(
        f"emrad/processed/data_per_location/{location}/emrad_data_{participant_id}.h5"
    )
    if not radar_path.exists() or trigger_extraction:
        synced_dataset = _sync_datasets(
            base_path=base_path,
            participant_id=participant_id,
            channel_mapping=channel_mapping,
            fs=fs,
            trigger_extraction=trigger_extraction,
            location=location
        )
        _save_location_synced_data(
            base_path=base_path, participant_id=participant_id, synced_dataset=synced_dataset, location=location
        )

    radar_data = pd.read_hdf(radar_path, key="emrad_data")
    return radar_data


def _sync_datasets(
        base_path: path_t,
        participant_id: str,
        channel_mapping: dict,
        fs: dict,
        trigger_extraction: bool,
        location: Optional[str] = None
):
    biopac_data = _load_biopac_raw_unsynced_data(
        base_path=base_path,
        participant_id=participant_id,
        channel_mapping=channel_mapping,
        trigger_extraction=trigger_extraction
    )
    radar_data = _load_radar_raw_unsynced_data(
        base_path=base_path,
        participant_id=participant_id,
        fs=fs["radar_original"],
        trigger_extraction=trigger_extraction
    )

    if location is not None:
        timelog = _load_timelog(base_path=base_path, participant_id=participant_id)
        shift = _get_biopac_timelog_shift(
            base_path=base_path, participant_id=participant_id, trigger_extraction=trigger_extraction
        )
        start = timelog[location]["start"][0] + shift - pd.Timedelta(seconds=15)
        end = timelog[location]["end"][0] + shift + pd.Timedelta(seconds=15)

        biopac_data = biopac_data.loc[start:end]
        radar_data = radar_data.loc[start:end]

    synced_dataset = SyncedDataset(sync_type="m-sequence")
    synced_dataset.add_dataset("biopac", data=biopac_data, sync_channel_name="sync",
                               sampling_rate=fs["biopac_original"])
    synced_dataset.add_dataset("radar", data=radar_data, sync_channel_name="Sync_Out",
                               sampling_rate=fs["radar_original"])
    synced_dataset.resample_datasets(fs_out=fs["resampled"], method="dynamic", wave_frequency=10)
    synced_dataset.align_and_cut_m_sequence(
        primary="radar", reset_time_axis=True, cut_to_shortest=True
    )
    return synced_dataset


def _save_raw_synced_data(base_path: path_t, participant_id: str, synced_dataset: SyncedDataset):
    data_path = _build_data_path(base_path=base_path, participant_id=participant_id)
    biopac_path = data_path.joinpath(f"biopac/processed/biopac_data_{participant_id}.h5")
    radar_path = data_path.joinpath(f"emrad/processed/emrad_data_{participant_id}.h5")

    synced_dataset.datasets_aligned["biopac_aligned_"].to_hdf(biopac_path, mode="w", key="biopac_data", index=True)
    synced_dataset.datasets_aligned["radar_aligned_"].to_hdf(radar_path, mode="w", key="emrad_data", index=True)


def _save_location_synced_data(base_path: path_t, participant_id: str, synced_dataset: SyncedDataset, location: str):
    data_path = _build_data_path(base_path=base_path, participant_id=participant_id)
    biopac_path = data_path.joinpath(f"biopac/processed/data_per_location/{location}/biopac_data_{participant_id}.h5")
    radar_path = data_path.joinpath(f"emrad/processed/data_per_location/{location}/emrad_data_{participant_id}.h5")

    biopac_path.parent.mkdir(parents=True, exist_ok=True)
    radar_path.parent.mkdir(parents=True, exist_ok=True)

    synced_dataset.datasets_aligned["biopac_aligned_"].to_hdf(biopac_path, mode="w", key="biopac_data", index=True)
    synced_dataset.datasets_aligned["radar_aligned_"].to_hdf(radar_path, mode="w", key="emrad_data", index=True)
    #synced_dataset.

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


def _get_biopac_timelog_shift(base_path: path_t, participant_id: str, trigger_extraction: bool) -> pd.Timedelta:
    shift_path = _build_data_path(base_path, participant_id).joinpath(
        f"timelog/processed/timelog_shift_{participant_id}.json"
    )
    if shift_path.exists() and not trigger_extraction:
        shift_dict = json.load(shift_path.open(encoding="utf-8"))
        shift = pd.Timedelta(seconds=shift_dict["biopac_timelog_shift"])
    else:
        shift = _calc_biopac_timelog_shift(base_path, participant_id)

    return shift


def _calc_biopac_timelog_shift(base_path: path_t, participant_id: str):

    # biopac sync event marker
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

    # timelog sync entry
    timelog = _load_timelog(base_path=base_path, participant_id=participant_id)
    timelog_sync_start_time = timelog["sync"]["start"].time

    # calculate time shift between biopac and timelog
    shift = sync_event_time - timelog_sync_start_time
    shift = np.floor(shift.total_seconds())

    # save shift parameters to json file
    shift_dict = {
        "sync_event_time_biopac": sync_event_time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "sync_entry_time_timelog": timelog_sync_start_time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "biopac_timelog_shift": shift
    }
    shift_path = _build_data_path(base_path, participant_id).joinpath(
        f"timelog/processed/timelog_shift_{participant_id}.json"
    )
    with open(shift_path, "w", encoding="utf-8") as f:
        json.dump(shift_dict, f, indent=4)

    shift = pd.Timedelta(seconds=shift)
    return shift


def _load_protocol(base_path: path_t, participant_id: str) -> pd.DataFrame:
    protocol_file_path = _build_protocol_path(base_path=base_path, participant_id=participant_id)
    if protocol_file_path.exists():
        protocol = pd.read_csv(protocol_file_path, index_col=0)
        return protocol
    raise FileNotFoundError("Processed Protocol file was not found.")

