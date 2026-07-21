import datetime
import json
from pathlib import Path
from typing import Literal

import bioread
import numpy as np
import pandas as pd
from biopsykit.io.biopac import BiopacDataset
from biopsykit.utils.time import tz
from pandas import DataFrame

from empkins_io.sensors.emrad import EmradDataset
from empkins_io.sync import SyncedDataset
from empkins_io.utils._types import path_t
from empkins_io.utils.exceptions import (
    SamplingRateMismatchError,
    SynchronizationError,
    TimelogNotFoundError,
)

<<<<<<<< HEAD:src/empkins_io/datasets/radarcardia/study/helper.py
import bioread
import json
import h5py
import re

========
>>>>>>>> origin/main:src/empkins_io/datasets/radarcardia/base/helper.py

def _build_data_path(base_path: path_t, participant_id: str) -> Path:
    data_path = base_path.joinpath(f"data_per_subject/{participant_id}")
    assert data_path.exists()
    return data_path


def _build_location_data_path(
        base_path: path_t,
        participant_id: str,
        location: str,
        sub_dir: Optional[str],
        file_name: str
) -> Path:
    if sub_dir is None:
        data_path = _build_data_path(base_path=base_path, participant_id=participant_id).joinpath(
            f"data_per_location/{location}/{participant_id}_{file_name}.h5"
        )
    else:
        data_path = _build_data_path(base_path=base_path, participant_id=participant_id).joinpath(
            f"data_per_location/{location}/{sub_dir}/{participant_id}_{file_name}.h5"
        )

    return data_path


def _build_timelog_path(base_path: path_t, participant_id: str) -> Path:
    timelog_dir_path = _build_data_path(base_path=base_path, participant_id=participant_id).joinpath(
        "timelog/processed"
    )
    timelog_file_path = timelog_dir_path.joinpath(f"{participant_id}_timelog.csv")
    return timelog_file_path


def _build_protocol_path(base_path: path_t, participant_id: str) -> Path:
    protocol_dir_path = _build_data_path(base_path=base_path, participant_id=participant_id).joinpath(
        "protocol/processed"
    )
    protocol_file_path = protocol_dir_path.joinpath(f"{participant_id}_protocol.csv")
    return protocol_file_path


def _load_biopac_data(
<<<<<<<< HEAD:src/empkins_io/datasets/radarcardia/study/helper.py
        base_path: path_t,
        participant_id: str,
        fs: dict,
        channel_mapping: dict,
        state: str,
        trigger_extraction: dict[str, bool],
        location: str
========
    base_path: path_t,
    participant_id: str,
    fs: dict,
    channel_mapping: dict,
    state: str,
    trigger_extraction: bool,
    location: str,
>>>>>>>> origin/main:src/empkins_io/datasets/radarcardia/base/helper.py
) -> pd.DataFrame:
    if state == "raw_unsynced":
        biopac = _load_biopac_raw_unsynced_data(
            base_path=base_path,
            participant_id=participant_id,
            channel_mapping=channel_mapping,
            trigger_extraction=trigger_extraction,
        )
        return biopac

    if state == "raw_synced":
        biopac = _load_raw_synced_data(
            base_path=base_path,
            participant_id=participant_id,
            fs=fs,
            channel_mapping=channel_mapping,
            trigger_extraction=trigger_extraction,
            data_type="biopac",
        )
        return biopac

    if state == "location_synced":
        biopac = _load_location_synced_data(
            base_path=base_path,
            participant_id=participant_id,
            fs=fs,
            channel_mapping=channel_mapping,
            trigger_extraction=trigger_extraction,
            location=location,
            data_type="biopac",
        )
        return biopac
    raise ValueError("Invalid Biopac data processing state")


def _load_radar_data(
<<<<<<<< HEAD:src/empkins_io/datasets/radarcardia/study/helper.py
        base_path: path_t,
        participant_id: str,
        fs: dict,
        channel_mapping: dict,
        state: str,
        trigger_extraction: dict[str, bool],
        location: str
========
    base_path: path_t,
    participant_id: str,
    fs: dict,
    channel_mapping: dict,
    state: str,
    trigger_extraction: bool,
    location: str,
>>>>>>>> origin/main:src/empkins_io/datasets/radarcardia/base/helper.py
) -> DataFrame:
    if state == "raw_unsynced":
        radar = _load_radar_raw_unsynced_data(
            base_path=base_path,
            participant_id=participant_id,
            fs=fs["radar_original"],
            trigger_extraction=trigger_extraction,
        )
        return radar

    if state == "raw_synced":
        radar = _load_raw_synced_data(
            base_path=base_path,
            participant_id=participant_id,
            fs=fs,
            channel_mapping=channel_mapping,
            trigger_extraction=trigger_extraction,
            data_type="emrad",
        )
        return radar

    if state == "location_synced":
        radar = _load_location_synced_data(
            base_path=base_path,
            participant_id=participant_id,
            fs=fs,
            channel_mapping=channel_mapping,
            trigger_extraction=trigger_extraction,
            location=location,
            data_type="emrad",
        )
        return radar
    raise ValueError("Invalid Biopac data processing state")


def _load_biopac_raw_unsynced_data(
<<<<<<<< HEAD:src/empkins_io/datasets/radarcardia/study/helper.py
        base_path: path_t, participant_id: str, channel_mapping: dict, trigger_extraction: dict[str, bool]
========
    base_path: path_t, participant_id: str, channel_mapping: dict, trigger_extraction: bool
>>>>>>>> origin/main:src/empkins_io/datasets/radarcardia/base/helper.py
) -> pd.DataFrame:
    biopac_path = _build_data_path(base_path=base_path, participant_id=participant_id).joinpath(
        f"biopac/cleaned/{participant_id}_biopac_data.h5"
    )

    if biopac_path.exists() and not trigger_extraction["raw_unsynced"]:
        biopac_df = pd.read_hdf(biopac_path, key="biopac_data")
    else:
<<<<<<<< HEAD:src/empkins_io/datasets/radarcardia/study/helper.py
        print("Extracting Biopac Raw Data")
        biopac_dir_path = _build_data_path(base_path, participant_id=participant_id).joinpath(
            "biopac/raw"
        )
        biopac_file_path = biopac_dir_path.joinpath(f"{participant_id}_biopac_data.acq")
========
        biopac_dir_path = _build_data_path(base_path, participant_id=participant_id).joinpath("biopac/raw")
        biopac_file_path = biopac_dir_path.joinpath(f"biopac_data_{participant_id}.acq")
>>>>>>>> origin/main:src/empkins_io/datasets/radarcardia/base/helper.py
        dataset_biopac = BiopacDataset.from_acq_file(biopac_file_path, channel_mapping=channel_mapping)
        biopac_df = dataset_biopac.data_as_df(index="local_datetime")
        fs = dataset_biopac._sampling_rate

        # check if biopac sampling rate is the same for each channel
        sampling_rates = set(fs.values())
        if len(sampling_rates) > 1:
            raise SamplingRateMismatchError(
                f"Biopac sampling rates are not the same for every channel! Found sampling rates: {sampling_rates}"
            )

        biopac_df.to_hdf(biopac_path, mode="w", key="biopac_data", index=True)

    return biopac_df


def _load_radar_raw_unsynced_data(
<<<<<<<< HEAD:src/empkins_io/datasets/radarcardia/study/helper.py
        base_path: path_t, participant_id: str, fs: float, trigger_extraction: dict[str, bool]
========
    base_path: path_t, participant_id: str, fs: float, trigger_extraction: bool
>>>>>>>> origin/main:src/empkins_io/datasets/radarcardia/base/helper.py
) -> DataFrame:
    radar_path = _build_data_path(base_path=base_path, participant_id=participant_id).joinpath(
        f"emrad/cleaned/{participant_id}_emrad_data.h5"
    )

    if radar_path.exists() and not trigger_extraction["raw_unsynced"]:
        radar_df = pd.read_hdf(radar_path, key="emrad_data")
    else:
<<<<<<<< HEAD:src/empkins_io/datasets/radarcardia/study/helper.py
        print("Extracting EMRAD Raw Data")
        radar_dir_path = _build_data_path(base_path, participant_id=participant_id).joinpath(
            "emrad/raw"
        )
        radar_file_path = radar_dir_path.joinpath(f"{participant_id}_emrad_data.h5")
========
        radar_dir_path = _build_data_path(base_path, participant_id=participant_id).joinpath("emrad/raw")
        radar_file_path = radar_dir_path.joinpath(f"emrad_data_{participant_id}.h5")
>>>>>>>> origin/main:src/empkins_io/datasets/radarcardia/base/helper.py

        dataset_radar = EmradDataset.from_hd5_file(radar_file_path, sampling_rate_hz=fs)
        radar_df = dataset_radar.data_as_df(index="local_datetime", add_sync_out=True)["rad2"]

        radar_df.to_hdf(radar_path, mode="w", key="emrad_data", index=True)

    return radar_df


def _load_raw_synced_data(
<<<<<<<< HEAD:src/empkins_io/datasets/radarcardia/study/helper.py
        base_path: path_t,
        participant_id: str,
        fs: dict,
        channel_mapping: dict,
        trigger_extraction: dict[str, bool],
        data_type: Literal["biopac", "emrad"]
========
    base_path: path_t,
    participant_id: str,
    fs: dict,
    channel_mapping: dict,
    trigger_extraction: bool,
    data_type: Literal["biopac", "emrad"],
>>>>>>>> origin/main:src/empkins_io/datasets/radarcardia/base/helper.py
) -> pd.DataFrame:
    data_path = _build_data_path(base_path=base_path, participant_id=participant_id).joinpath(
        f"{data_type}/processed/{participant_id}_{data_type}_data.h5"
    )
    if not data_path.exists() or trigger_extraction["raw_synced"]:
        print("Extracting Raw Synced Data")
        synced_dataset = _sync_datasets(
            base_path=base_path,
            participant_id=participant_id,
            channel_mapping=channel_mapping,
            fs=fs,
            trigger_extraction=trigger_extraction,
        )
        _save_raw_synced_data(base_path=base_path, participant_id=participant_id, synced_dataset=synced_dataset)

    data = pd.read_hdf(data_path, key=f"{data_type}_data")
    return data


def _load_location_synced_data(
<<<<<<<< HEAD:src/empkins_io/datasets/radarcardia/study/helper.py
        base_path: path_t,
        participant_id: str,
        fs: dict,
        channel_mapping: dict,
        trigger_extraction: dict[str, bool],
        location: str,
        data_type: Literal["biopac", "emrad"]
========
    base_path: path_t,
    participant_id: str,
    fs: dict,
    channel_mapping: dict,
    trigger_extraction: bool,
    location: str,
    data_type: Literal["biopac", "emrad"],
>>>>>>>> origin/main:src/empkins_io/datasets/radarcardia/base/helper.py
) -> DataFrame:
    data_path = _build_data_path(base_path=base_path, participant_id=participant_id).joinpath(
        f"data_per_location/{location}/raw/{participant_id}_{data_type}_data.h5"
    )
    # for location syncing resampling is not necessary as it was already applied on the whole data recording
    if not data_path.exists() or trigger_extraction["location_synced"]:
        print("Extracting Location Synced Data")
        synced_dataset = _sync_datasets_without_resample(
            base_path=base_path,
            participant_id=participant_id,
            channel_mapping=channel_mapping,
            fs=fs,
            trigger_extraction=trigger_extraction,
            location=location,
        )
        _save_location_synced_data(
            base_path=base_path, participant_id=participant_id, synced_dataset=synced_dataset, location=location
        )

    data = pd.read_hdf(data_path, key=f"{data_type}_data")
    return data


def _sync_datasets(
<<<<<<<< HEAD:src/empkins_io/datasets/radarcardia/study/helper.py
        base_path: path_t,
        participant_id: str,
        channel_mapping: dict,
        fs: dict,
        trigger_extraction: dict[str, bool],
========
    base_path: path_t,
    participant_id: str,
    channel_mapping: dict,
    fs: dict,
    trigger_extraction: bool,
>>>>>>>> origin/main:src/empkins_io/datasets/radarcardia/base/helper.py
):
    biopac_data = _load_biopac_raw_unsynced_data(
        base_path=base_path,
        participant_id=participant_id,
        channel_mapping=channel_mapping,
        trigger_extraction=trigger_extraction,
    )
    radar_data = _load_radar_raw_unsynced_data(
        base_path=base_path,
        participant_id=participant_id,
        fs=fs["radar_original"],
        trigger_extraction=trigger_extraction,
    )

    # cut to overlapping time range
    index_start = max(biopac_data.index[0], radar_data.index[0])
    index_end = min(biopac_data.index[-1], radar_data.index[-1])
    biopac_data = biopac_data.loc[index_start:index_end]
    radar_data = radar_data.loc[index_start:index_end]

    # sync and resample entire data recording
    synced_dataset = SyncedDataset(sync_type="m-sequence")
    synced_dataset.add_dataset(
        "biopac", data=biopac_data, sync_channel_name="sync", sampling_rate=fs["biopac_original"]
    )
    synced_dataset.add_dataset(
        "radar", data=radar_data, sync_channel_name="Sync_Out", sampling_rate=fs["radar_original"]
    )
    synced_dataset.resample_datasets(fs_out=fs["resampled"], method="dynamic", wave_frequency=10)
    synced_dataset.align_and_cut_m_sequence(primary="radar", reset_time_axis=True, cut_to_shortest=True)
    return synced_dataset


def _sync_datasets_without_resample(
<<<<<<<< HEAD:src/empkins_io/datasets/radarcardia/study/helper.py
        base_path: path_t,
        participant_id: str,
        channel_mapping: dict,
        fs: dict,
        trigger_extraction: dict[str, bool],
        location: str
========
    base_path: path_t, participant_id: str, channel_mapping: dict, fs: dict, trigger_extraction: bool, location: str
>>>>>>>> origin/main:src/empkins_io/datasets/radarcardia/base/helper.py
):
    biopac_data = _load_raw_synced_data(
        base_path=base_path,
        participant_id=participant_id,
        fs=fs,
        channel_mapping=channel_mapping,
        trigger_extraction=trigger_extraction,
        data_type="biopac",
    )
    radar_data = _load_raw_synced_data(
        base_path=base_path,
        participant_id=participant_id,
        fs=fs,
        channel_mapping=channel_mapping,
        trigger_extraction=trigger_extraction,
        data_type="emrad",
    )
    # cut measurement from current location
    timelog = _load_timelog(base_path=base_path, participant_id=participant_id)
    shift = _get_biopac_timelog_shift(
        base_path=base_path, participant_id=participant_id
    )
    start = timelog[location]["start"][0] + shift - pd.Timedelta(seconds=5)
    end = timelog[location]["end"][0] + shift + pd.Timedelta(seconds=5)

    biopac_data = biopac_data.loc[start:end]
    radar_data = radar_data.loc[start:end]

    # sync radar and biopac without resampling
    synced_dataset = SyncedDataset(sync_type="m-sequence")
    synced_dataset.add_dataset("biopac", data=biopac_data, sync_channel_name="sync", sampling_rate=fs["resampled"])
    synced_dataset.add_dataset("radar", data=radar_data, sync_channel_name="Sync_Out", sampling_rate=fs["resampled"])
    synced_dataset.align_and_cut_m_sequence(primary="radar", reset_time_axis=True, cut_to_shortest=True)
    return synced_dataset


def _save_raw_synced_data(base_path: path_t, participant_id: str, synced_dataset: SyncedDataset):
    data_path = _build_data_path(base_path=base_path, participant_id=participant_id)
    biopac_path = data_path.joinpath(f"biopac/processed/{participant_id}_biopac_data.h5")
    radar_path = data_path.joinpath(f"emrad/processed/{participant_id}_emrad_data.h5")

    synced_dataset.datasets_aligned["biopac_aligned_"].to_hdf(biopac_path, mode="w", key="biopac_data", index=True)
    synced_dataset.datasets_aligned["radar_aligned_"].to_hdf(radar_path, mode="w", key="emrad_data", index=True)


def _save_location_synced_data(base_path: path_t, participant_id: str, synced_dataset: SyncedDataset, location: str):
    data_path = _build_data_path(base_path=base_path, participant_id=participant_id).joinpath(
        f"data_per_location/{location}/raw"
    )
    biopac_path = data_path.joinpath(f"{participant_id}_biopac_data.h5")
    radar_path = data_path.joinpath(f"{participant_id}_emrad_data.h5")

    biopac_path.parent.mkdir(parents=True, exist_ok=True)
    radar_path.parent.mkdir(parents=True, exist_ok=True)

    synced_dataset.datasets_aligned["biopac_aligned_"].to_hdf(biopac_path, mode="w", key="biopac_data", index=True)
    synced_dataset.datasets_aligned["radar_aligned_"].to_hdf(radar_path, mode="w", key="emrad_data", index=True)


def _load_timelog(base_path: path_t, participant_id: str) -> pd.DataFrame:
    timelog_file_path = _build_timelog_path(base_path=base_path, participant_id=participant_id)
    if timelog_file_path.exists():
        timelog = _load_atimelogger_file(timelog_file_path, timezone="Europe/Berlin")
        return timelog
    raise TimelogNotFoundError(f"No timelog file was found for {participant_id}!")


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

    timelog = timelog.apply(pd.to_datetime, axis=1).applymap(lambda val: val.tz_localize(timezone))
    timelog = pd.DataFrame(timelog.T.unstack(), columns=["time"])
    timelog = timelog[::-1].reindex(["start", "end"], level="start_end")
    timelog = timelog.T

    return timelog


def _save_data_to_location_h5(
    base_path: path_t, participant_id: str, data: pd.DataFrame, location: str, file_name: str, sub_dir: str
):
    data_path = _build_location_data_path(
        base_path=base_path, participant_id=participant_id, location=location, sub_dir=sub_dir, file_name=file_name
    )
    data_path.parent.mkdir(parents=True, exist_ok=True)

    data.to_hdf(data_path, mode="w", key="data", index=True)


<<<<<<<< HEAD:src/empkins_io/datasets/radarcardia/study/helper.py
def _save_data_to_location_np_array(
        base_path: path_t,
        participant_id: str,
        data: np.ndarray,
        location: str,
        file_name: str,
        sub_dir: str
):
    data_path = _build_location_data_path(
        base_path=base_path, participant_id=participant_id, location=location, sub_dir=sub_dir, file_name=file_name
    )
    data_path = data_path.with_suffix(".npy")
    data_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(data_path, data)


def _save_data_to_location_dict(
        base_path: path_t,
        participant_id: str,
        data: dict,
        location: str,
        file_name: str,
        sub_dir: str
):
    data_path = _build_location_data_path(
        base_path=base_path, participant_id=participant_id, location=location, sub_dir=sub_dir, file_name=file_name
    )
    data_path = data_path.with_suffix(".json")
    data_path.parent.mkdir(parents=True, exist_ok=True)

    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def _save_data_to_location_list_h5(
        base_path: path_t,
        participant_id: str,
        data: list,
        location: str,
        file_name: str,
        sub_dir: str
):
    data_path = _build_location_data_path(
        base_path=base_path, participant_id=participant_id, location=location, sub_dir=sub_dir, file_name=file_name
    )
    data_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(data_path, 'w') as f:
        for i, seg in enumerate(data):
            f.create_dataset(f'Data_{i}', data=seg)


def _load_data_from_location_h5(
        base_path: path_t,
        participant_id: str,
        location: str,
        file_name: str,
        sub_dir: str
):
    data_path = _build_location_data_path(
        base_path=base_path, participant_id=participant_id, location=location, sub_dir=sub_dir, file_name=file_name
    )
========
def _load_data_from_location_h5(base_path: path_t, participant_id: str, location: str, file_name: str, sub_dir: str):
    if sub_dir is None:
        data_path = _build_data_path(base_path=base_path, participant_id=participant_id).joinpath(
            f"data_per_location/{location}/{file_name}_{participant_id}.h5"
        )
    else:
        data_path = _build_data_path(base_path=base_path, participant_id=participant_id).joinpath(
            f"data_per_location/{location}/{sub_dir}/{file_name}_{participant_id}.h5"
        )
>>>>>>>> origin/main:src/empkins_io/datasets/radarcardia/base/helper.py
    data = pd.read_hdf(data_path, key="data")

    return data


def _load_data_from_location_np_array(
        base_path: path_t,
        participant_id: str,
        location: str,
        file_name: str,
        sub_dir: str
):
    data_path = _build_location_data_path(
        base_path=base_path, participant_id=participant_id, location=location, sub_dir=sub_dir, file_name=file_name
    )
    data_path = data_path.with_suffix(".npy")
    data = np.load(data_path)

    return data


def _load_data_from_location_dict(
        base_path: path_t,
        participant_id: str,
        location: str,
        file_name: str,
        sub_dir: str
):
    data_path = _build_location_data_path(
        base_path=base_path, participant_id=participant_id, location=location, sub_dir=sub_dir, file_name=file_name
    )
    data_path = data_path.with_suffix(".json")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data

def _load_data_from_location_list_h5(
        base_path: path_t,
        participant_id: str,
        location: str,
        file_name: str,
        sub_dir: str
):
    data_path = _build_location_data_path(
        base_path=base_path, participant_id=participant_id, location=location, sub_dir=sub_dir, file_name=file_name
    )

    data_loaded = []
    with h5py.File(data_path, 'r') as f:
        keys = sorted(f.keys(), key=lambda x: int(re.findall(r'\d+', x)[0]))
        for key in keys:
            print(key)
            data = f[key][()]  # read the dataset as a NumPy array
            data_loaded.append(data)

    return data_loaded


def _get_biopac_timelog_shift(base_path: path_t, participant_id: str) -> pd.Timedelta:
    shift_path = _build_data_path(base_path, participant_id).joinpath(
        f"timelog/processed/{participant_id}_timelog_shift.json"
    )
    if shift_path.exists():
        shift_dict = json.load(shift_path.open(encoding="utf-8"))
        shift = pd.Timedelta(seconds=shift_dict["biopac_timelog_shift"])
    else:
        print("Extracting Biopac Timelog Shift")
        shift = _calc_biopac_timelog_shift(base_path, participant_id)

    return shift


def _calc_biopac_timelog_shift(base_path: path_t, participant_id: str):
<<<<<<<< HEAD:src/empkins_io/datasets/radarcardia/study/helper.py

    # extract sync event marker from biopac file
    biopac_dir_path = _build_data_path(base_path, participant_id=participant_id).joinpath(
        "biopac/raw"
    )
    biopac_file_path = biopac_dir_path.joinpath(f"{participant_id}_biopac_data.acq")
========
    # biopac sync event marker
    biopac_dir_path = _build_data_path(base_path, participant_id=participant_id).joinpath("biopac/raw")
    biopac_file_path = biopac_dir_path.joinpath(f"biopac_data_{participant_id}.acq")
>>>>>>>> origin/main:src/empkins_io/datasets/radarcardia/base/helper.py

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

    # get the start time from timelog sync entry
    timelog = _load_timelog(base_path=base_path, participant_id=participant_id)
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
    shift_path = _build_data_path(base_path, participant_id).joinpath(
        f"timelog/processed/{participant_id}_timelog_shift.json"
    )
    with shift_path.open("w", encoding="utf-8") as f:
        json.dump(shift_dict, f, indent=4)

    shift = pd.Timedelta(seconds=shift)
    return shift


def _load_protocol(base_path: path_t, participant_id: str) -> pd.DataFrame:
    protocol_file_path = _build_protocol_path(base_path=base_path, participant_id=participant_id)
    if protocol_file_path.exists():
        protocol = pd.read_csv(protocol_file_path, index_col=0)
        return protocol
    raise FileNotFoundError("Processed Protocol file was not found.")


def _load_apnea_segmentation(base_path: path_t, participant_id: str) -> dict:
    apnea_seg_file_path = _build_data_path(base_path=base_path, participant_id=participant_id).joinpath(
        f"biopac/processed/{participant_id}_apnea_segmentation.json"
    )
    if apnea_seg_file_path.exists():
        apnea_seg = json.load(apnea_seg_file_path.open(encoding="utf-8"))
        return apnea_seg
    raise FileNotFoundError("Apnea segmentation file was not found.")


def _load_visual_segmentation(base_path: path_t, participant_id: str) -> pd.DataFrame:
    file_path = _build_data_path(base_path=base_path, participant_id=participant_id).joinpath(
        f"visual_segmentation/{participant_id}_visual_segmentation.xlsx"
    )
    if file_path.exists():
        seg = pd.read_excel(file_path, index_col=0)
        return seg
    raise FileNotFoundError("Visual segmentation file was not found.")

<<<<<<<< HEAD:src/empkins_io/datasets/radarcardia/study/helper.py
========

def _load_flipping(base_path: path_t, modality: str) -> pd.DataFrame:
    file_path = base_path.joinpath(f"flipping_total/{modality}_flipping_total.xlsx")
    if file_path.exists():
        data = pd.read_excel(file_path, index_col=0)
        return data
    raise FileNotFoundError("Flipping file was not found.")
>>>>>>>> origin/main:src/empkins_io/datasets/radarcardia/base/helper.py
