import datetime
import warnings
from pathlib import Path
from typing import Optional, Sequence, Union

import mne
import pandas as pd
from biopsykit.utils._datatype_validation_helper import _assert_file_extension, _assert_is_dir
from typing_extensions import Literal

from empkins_io.sensors.psg.psg_channels import PSG_CHANNELS_MESA, PSG_CHANNELS_SOMNO
from empkins_io.utils._types import path_t

__all__ = ["load_data", "load_data_raw", "load_data_folder"]

DATASTREAMS = Literal[PSG_CHANNELS_SOMNO, PSG_CHANNELS_MESA]


def load_data_folder(
    folder_path: path_t,
    datastreams: Optional[Union[DATASTREAMS, Sequence[DATASTREAMS]]] = None,
    timezone: Optional[Union[datetime.tzinfo, str]] = None,
):
    _assert_is_dir(folder_path)

    # look for all PSG .edf files in the folder
    dataset_list = list(sorted(folder_path.glob("*.edf")))
    if len(dataset_list) == 0:
        raise ValueError(f"No PSG files found in folder {folder_path}!")
    if len(dataset_list) > 1:
        raise ValueError(
            f"More than one PSG files found in folder {folder_path}! This function only supports one recording per folder!"
        )

    result_dict, fs = load_data(path_t.joinpath(dataset_list[0]), datastreams, timezone)

    return result_dict, fs


def load_data(
    path: path_t,
    datastreams: Optional[Union[DATASTREAMS, Sequence[DATASTREAMS]]] = None,
    timezone: Optional[Union[datetime.tzinfo, str]] = None,
):
    data_psg, fs = load_data_raw(path, timezone)

    if datastreams is None:
        datastreams = data_psg.ch_names
    if isinstance(datastreams, str):
        datastreams = [datastreams]

    result_dict = {}

    for datastream in datastreams:
        try:
            time, epochs = _create_datetime_index(data_psg.info["meas_date"], times_array=data_psg.times)
            psg_datastream = data_psg.copy().pick_channels([datastream]).get_data()[0, :]
            result_dict[datastream] = pd.DataFrame(psg_datastream, index=time, columns=["signal"])
        except ValueError:
            raise ValueError(
                "Not all channels match the selected datastreams - Following Datastreams are available: "
                + str(data_psg.ch_names)
            )

    sleep_phases = _load_ground_truth(path.parents[1])
    result_dict["sleep_phases"] = sleep_phases

    return result_dict, fs


def load_data_raw(
    path: path_t, timezone: Optional[Union[datetime.tzinfo, str]] = None,
):
    # ensure pathlib
    path = Path(path)
    _assert_file_extension(path, ".edf")

    if timezone is None:
        timezone = "Europe/Berlin"
        # Not implemented yet # TODO

    edf = mne.io.read_raw_edf(path)

    fs = edf.info["sfreq"]

    return edf, fs


def _create_datetime_index(starttime, times_array):
    starttime_s = starttime.timestamp()
    times_array = times_array + starttime_s
    datetime_index = pd.to_datetime(times_array, unit="s")
    epochs = _generate_epochs(datetime_index)
    return datetime_index, epochs


def _generate_epochs(datetime_index):
    start_time = datetime_index[0]
    epochs_30s = datetime_index.round("30s")

    epochs_clear = (epochs_30s - start_time).total_seconds()

    epochs_clear = epochs_clear / 30
    epochs = epochs_clear.astype(int)
    return epochs


def _load_ground_truth(path: path_t,):
    file_path = path.joinpath("labels/PSG_analyse.xlsx")
    try:
        sleep_phases = pd.read_excel(file_path, sheet_name="Schlafprofil", header=7, index_col=0)
    except FileNotFoundError:
        warnings.warn("No ground truth found")
        return pd.DataFrame()

    return sleep_phases
    # TODO: Read in excel or txt files with sleep labels
