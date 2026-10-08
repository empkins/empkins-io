from typing import Callable, Union

from empkins_io.utils._types import path_t

from biopsykit.io.psg import PSGDataset
from empkins_io.sensors.emrad import EmradDataset
from biopsykit.io.biopac import BiopacDataset

from empkins_io.sync import SyncedDataset, SyncedDatasetMSequence

import pandas as pd

import matplotlib.pyplot as plt

ACTIVITIES = [
    "empty_bed",
    "enter_bed",
    "leave_bed",
    "sit_up_propping",
    "sit_up_trapeze",
    "incremental_turning",
    "turn_right",
    "turn_left",
    "turn_back",
    "reach_for_cup",
    "talking",
    "others",
]

BREATHING_PATTERNS = [
    "normal",
    "fast",
    "slow",
    "chest",
    "belly",
    "shallow",
    "deep",
    "inhalation_apnea_50",
    "inhalation_apnea_100",
    "exhalation_apnea_50",
    "exhalation_apnea_100",
    "four_phase",
    "coughing",
    "cheyne_stokes"
]


def build_data_path(base_path: path_t, subject: str) -> path_t:

    return base_path.joinpath("data_per_subject", subject)

def build_intermediate_path(base_path: path_t) -> path_t:

    return base_path.joinpath("data_intermediate")


def load_timelog(base_path: path_t, subject: str, data_type: str) -> pd.DataFrame:
    data_path = build_data_path(base_path, subject)
    data_path = data_path.joinpath("timelog", "cleaned", f"{subject}_study_log.csv")

    data = pd.read_csv(data_path)

    data["start_time"] = pd.to_datetime(
        data["start_time"], format="%Y-%m-%d %H-%M-%S:%f").dt.tz_localize("Europe/Berlin")
    data["end_time"] = pd.to_datetime(
        data["end_time"], format="%Y-%m-%d %H-%M-%S:%f").dt.tz_localize("Europe/Berlin")

    if data_type == "all":
        return data
    elif data_type in ("respiration", "breathing"):
        return data.loc[data.pattern.isin(BREATHING_PATTERNS)].sort_values(by="start_unix_ms").reset_index(drop=True)
    elif data_type == "activity":
        return data.loc[data.pattern.isin(ACTIVITIES)].sort_values(by="start_unix_ms").reset_index(drop=True)
    else:
        raise ValueError(
            f"data_type must be one of ['all', 'respiration', 'breathing', 'activity'], but got {data_type}"
        )


def load_raw_emrad_data(base_path: path_t, subject: str, data_type: str, fs: float) -> pd.DataFrame:
    data_path = build_data_path(base_path, subject)
    data_path = data_path.joinpath("emrad", "raw", f"{subject}_emrad_data_{data_type}.h5")

    emrad = EmradDataset.from_hd5_file(data_path, sampling_rate_hz=fs)
    emrad_data = emrad.data_as_df(index="local_datetime", add_sync_in=True)

    return emrad_data


def load_raw_biopac_data(
        base_path: path_t,
        subject: str,
        channel_mapping: dict,
        start_time: Union[pd.Timestamp, Callable[[], pd.Timestamp], None] = None,
) -> pd.DataFrame:
    """Load the raw BIOPAC data of one subject.

    Parameters
    ----------
    base_path : path
        Base path of the dataset.
    subject : str
        Subject id.
    channel_mapping : dict
        Mapping of the channel names in the .acq file to the channel names used in the dataset.
    start_time : :class:`~pandas.Timestamp` or callable, optional
        Start time of the recording, used if the .acq file does not contain one. The start time of a recording
        is derived from the first event marker of the .acq file. For some recordings this marker does not carry
        a date, in which case the start time of the radar recording is used instead. A callable is only
        evaluated if the start time is actually needed, so that the radar data is not loaded unnecessarily.

    """
    data_path = build_data_path(base_path, subject)
    data_path = data_path.joinpath("biopac", "raw", f"{subject}_biopac_data.acq")

    biopac = BiopacDataset.from_acq_file(
        path=data_path,
        channel_mapping=channel_mapping,
        tz="Europe/Berlin"
    )

    # the start time is missing for some recordings, because the first event marker of the .acq file does not
    # carry a date. In that case, the start time of the radar recording is used instead.
    start_time_biopac = biopac.start_time_unix
    if start_time_biopac is None or pd.isna(start_time_biopac):
        if callable(start_time):
            start_time = start_time()
        if start_time is None:
            raise ValueError(
                f"The BIOPAC file of subject '{subject}' does not contain a start time. Please provide one via "
                f"the 'start_time' parameter, e.g., the start time of the radar recording."
            )
        biopac_data = biopac.data_as_df(index="local_datetime", start_time=start_time)
    else:
        biopac_data = biopac.data_as_df(index="local_datetime")

    # binarize sync channel
    biopac_data["sync"] = biopac_data["sync"].apply(lambda x: 1 if x > 2.5 else 0)

    return biopac_data


def load_raw_psg_data(base_path: path_t, subject: str, datastreams: list) -> pd.DataFrame:
    data_path = build_data_path(base_path, subject)
    data_path = data_path.joinpath("psg", "raw", f"{subject}_psg_data.edf")

    psg_sync = PSGDataset.from_edf_file(data_path, datastreams=datastreams)
    psg_data = psg_sync.data_as_df(index="local_datetime")
    psg_data = psg_data.rename(
        columns={
            "Flow": "flow",
            "RIP Abdom": "rip_abd",
            "EKG II": "ecg",
            "SpO2": "spo2",
            "Pulse": "pulse",
            "Pleth": "pleth",
            "AUX": "sync"
        }
    )

    # binarize sync signal
    psg_data["sync"] = psg_data["sync"].apply(lambda x: 1 if x > 500 else 0)

    return psg_data

def get_respiration_data_synced(
        base_path: path_t,
        subject: str,
        sync_sequence_length_s: float,
        fs: dict,
        biopac_channel_mapping: dict,
        psg_datastreams: list,
        perform_sync: bool = False,
) -> pd.DataFrame:
    data_path = build_intermediate_path(base_path)
    data_path = data_path.joinpath("breathing_data_synced", f"{subject}_breathing_data_synced.h5")

    if perform_sync or not data_path.exists():
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df = sync_biopac_emrad_psg(base_path, subject, sync_sequence_length_s, fs, biopac_channel_mapping, psg_datastreams)
        df.to_hdf(data_path, key="data", mode="w")
    else:
        df = pd.read_hdf(data_path, key="data")

    return df

def sync_biopac_emrad_psg(
        base_path: path_t,
        subject: str,
        sync_sequence_length_s: float,
        fs: dict,
        biopac_channel_mapping: dict,
        psg_datastreams: list
) -> pd.DataFrame:

    emrad_data = load_raw_emrad_data(base_path, subject, "breathing", fs["emrad"])
    # fall back to the start time of the radar recording if the BIOPAC file does not contain one
    biopac_data = load_raw_biopac_data(
        base_path, subject, biopac_channel_mapping, start_time=emrad_data.index[0]
    )
    psg_data = load_raw_psg_data(base_path, subject, psg_datastreams)

    # resampling
    sync = SyncedDatasetMSequence()
    sync.add_dataset("biopac", biopac_data, sampling_rate=fs["biopac"], sync_channel_name="sync")
    sync.add_dataset("psg", psg_data, sampling_rate=fs["psg"], sync_channel_name="sync")
    sync.add_dataset("rad1", emrad_data.rad1, sampling_rate=fs["emrad"], sync_channel_name="Sync_In")
    sync.add_dataset("rad2", emrad_data.rad2, sampling_rate=fs["emrad"], sync_channel_name="Sync_In")
    sync.add_dataset("rad3", emrad_data.rad3, sampling_rate=fs["emrad"], sync_channel_name="Sync_In")
    sync.add_dataset("rad4", emrad_data.rad4, sampling_rate=fs["emrad"], sync_channel_name="Sync_In")

    sync.resample_datasets(fs_out=fs["resampled"], method="dynamic", wave_frequency=29)

    # align start
    sync.align_and_cut_start_m_sequence(
        primary="rad1",
        cut_to_shortest=True,
        reset_time_axis=True,
        sync_params={"sync_region_samples": (0, int(sync_sequence_length_s * fs["resampled"]))}
    )

    # align end
    sync.align_and_cut_end_m_sequence(
        primary="rad1",
        cut_to_shortest=True,
        sync_params={"sync_region_samples": (int(-sync_sequence_length_s * fs["resampled"]), None)}
    )

    data = {
        "psg": sync.psg_synced_,
        "biopac": sync.biopac_synced_,
        "rad1": sync.rad1_synced_,
        "rad2": sync.rad2_synced_,
        "rad3": sync.rad3_synced_,
        "rad4": sync.rad4_synced_,
    }

    df = pd.concat(data, axis=1, names=["type"])

    return df





