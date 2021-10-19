import datetime
import re
from datetime import datetime as dt
from pathlib import Path
from typing import Dict, Literal, Optional, Sequence, Tuple, Union

import neurokit2 as nk
import numpy as np
import pandas as pd
from biopsykit.utils._datatype_validation_helper import _assert_file_extension
from biopsykit.utils._types import path_t
from biopsykit.utils.time import tz
from scipy.io import loadmat

DATASTREAMS = Literal["hr", "resp"]


def load_data(
    path: path_t,
    datastreams: Optional[Union[DATASTREAMS, Sequence[DATASTREAMS]]] = "hr",
    timezone: Optional[Union[datetime.tzinfo, str]] = None,
) -> Tuple[Dict[str, pd.DataFrame], float]:
    """Load radar data from the A04 "cardiovascular pulmonary microwave interferometer" sensor.

    Parameters
    ----------
    path : :class:`pathlib.Path`
        path to exported ".mat" file
    datastreams : {"hr", "resp"} or a list of such
        string (or list of strings) specifying which datastreams to load
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

    if isinstance(datastreams, str):
        datastreams = [datastreams]

    result_dict = {}
    if "hr" in datastreams:
        result_dict["hr"] = _load_hr(data_radar, fs)

    if "resp" in datastreams:
        result_dict["resp"] = _load_resp(data_radar, fs)

    return result_dict, fs


def load_data_raw(path: path_t, timezone: Optional[Union[datetime.tzinfo, str]] = None) -> Tuple[pd.DataFrame, float]:
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
        timezone = tz

    start_date = dt.strptime(
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
