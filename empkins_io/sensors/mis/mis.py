import datetime
import re
from pathlib import Path
from typing import Optional, Union, Sequence
from typing_extensions import Literal

import pandas as pd
import numpy as np
import neurokit2 as nk
from scipy.io import loadmat

from biopsykit.utils._datatype_validation_helper import _assert_is_dir, _assert_file_extension

from empkins_io.utils._types import path_t

__all__ = ["load_data", "load_data_raw", "load_data_folder"]

DATASTREAMS = Literal["hr", "resp"]


def load_data_folder(
    folder_path: path_t,
    phase_names: Optional[Sequence[str]] = None,
    datastreams: Optional[Union[DATASTREAMS, Sequence[DATASTREAMS]]] = None,
    timezone: Optional[Union[datetime.tzinfo, str]] = None,
):
    """

    Parameters
    ----------
    folder_path :
    phase_names :
    datastreams :
    timezone :

    Returns
    -------

    """
    _assert_is_dir(folder_path)

    # look for all MIS .mat files in the folder
    dataset_list = list(sorted(folder_path.glob("*.mat")))
    if len(dataset_list) == 0:
        raise ValueError(f"No MIS files found in folder {folder_path}!")
    if phase_names is None:
        phase_names = [f"Part{i}" for i in range(len(dataset_list))]

    if len(phase_names) != len(dataset_list):
        raise ValueError(
            f"Number of phases does not match number of datasets in the folder! "
            f"Expected {len(dataset_list)}, got {len(phase_names)}."
        )

    dataset_list = [
        load_data(path=dataset_path, datastreams=datastreams, timezone=timezone) for dataset_path in dataset_list
    ]
    fs_list = [fs for df, fs in dataset_list]

    if len(set(fs_list)) > 1:
        raise ValueError("Datasets have different sampling rates! Got: {}.".format(fs_list))
    fs = fs_list[0]

    dataset_dict = {phase: df for phase, (df, fs) in zip(phase_names, dataset_list)}
    return dataset_dict, fs


def load_data(
    path: path_t,
    datastreams: Optional[Union[DATASTREAMS, Sequence[DATASTREAMS]]] = None,
    timezone: Optional[Union[datetime.tzinfo, str]] = None,
):
    """Load radar data from the A04 "cardiovascular pulmonary microwave interferometer" sensor.

    Parameters
    ----------
    path : :class:`pathlib.Path`
        path to exported ".mat" file
    datastreams : {"hr", "resp"} or a list of such
        string (or list of strings) specifying which datastreams to load or ``None`` to use default datastream
        (corresponds to "hr").
    timezone : str or :class:`datetime.tzinfo`, optional
        timezone of the acquired data, either as string of as tzinfo object.
        Default: ``None`` (corresponds to "Europe/Berlin")


    Returns
    -------
    result_dict : dict
        dictionary with pandas dataframes containing instantaneous heart rate (if ``datastreams`` contains "hr")
        and respiration phases (if ``datastreams`` contains "resp")
    fs : float
        sampling rate of original A04 radar signal

    """
    data_radar, fs = load_data_raw(path, timezone)

    if datastreams is None:
        datastreams = "hr"
    if isinstance(datastreams, str):
        datastreams = [datastreams]

    result_dict = {}
    if "hr" in datastreams:
        result_dict["hr"] = _load_hr(data_radar, fs)

    if "resp" in datastreams:
        result_dict["resp"] = _load_resp(data_radar, fs)

    return result_dict, fs


def load_data_raw(
    path: path_t,
    timezone: Optional[Union[datetime.tzinfo, str]] = None,
):
    """Load raw radar data from the A04 "cardiovascular pulmonary microwave interferometer" sensor.

    Parameters
    ----------
    path : :class:`pathlib.Path`
        path to exported ".mat" file
    timezone : str or :class:`datetime.tzinfo`, optional
        timezone of the acquired data, either as string of as tzinfo object.
        Default: ``None`` (corresponds to "Europe/Berlin")

    Returns
    -------
    result_dict : :class:`~pandas.DataFrame`
        dataframe with radar data
    fs : float
        sampling rate of original A04 radar signal

    """
    # ensure pathlib
    path = Path(path)
    _assert_file_extension(path, ".mat")

    if timezone is None:
        timezone = "Europe/Berlin"

    start_date = datetime.datetime.strptime(
        re.findall(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", path.name)[0],
        "%Y-%m-%d_%H-%M-%S",
    )

    dict_radar = loadmat(path)
    fs = float(np.squeeze(dict_radar["fs"][0]))

    data_radar = pd.DataFrame(
        {
            k: np.squeeze(dict_radar[k])
            for k in [
                "ch1",
                "ch2",
                "resp",
                "pulse",
                "respStates",
                "heartbeats",
                "heartsoundQuality",
            ]
        }
    )

    data_radar.index = (
        pd.TimedeltaIndex(np.arange(0, len(data_radar)) / fs, unit="s", name="time") + start_date
    ).tz_localize(timezone)
    return data_radar, fs


def _load_hr(data: pd.DataFrame, fs: float) -> pd.DataFrame:
    loc_beat = np.where(data["heartbeats"] == 1)[0]
    rri = np.ediff1d(loc_beat) / fs
    hr_radar_raw = 60 / rri

    loc_beat = loc_beat[1:]
    hr_index = data.index[loc_beat]
    hs_quality = data["heartsoundQuality"][hr_index]
    hr_radar_raw = pd.DataFrame(
        {
            "Heart_Rate": hr_radar_raw,
            "Heartsound_Quality": hs_quality,
            "RR_Interval": rri,
            "R_Peak_Idx": loc_beat,
        }
    )

    return hr_radar_raw


def _load_resp(data: pd.DataFrame, fs: float, fs_out: Optional[float] = 100) -> pd.DataFrame:
    ds_factor = int(fs // fs_out)
    in_cols = ["resp", "respStates"]
    out_cols = ["Respiration", "Respiration_State"]
    resp_raw = data[in_cols]
    index_old = resp_raw.index.view(int)
    index_new = index_old[::ds_factor]
    resp_raw = pd.DataFrame(
        {
            out_col: nk.signal_interpolate(index_old, resp_raw[in_col].values, index_new)
            for out_col, in_col in zip(out_cols, in_cols)
        },
        index=index_new,
    )
    resp_raw.index = pd.DatetimeIndex(resp_raw.index)
    return resp_raw
