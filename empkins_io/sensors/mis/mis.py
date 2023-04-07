import datetime
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from biopsykit.utils._datatype_validation_helper import _assert_file_extension
from biopsykit.utils.array_handling import downsample
from scipy.io import loadmat
from typing_extensions import Literal

from empkins_io.utils._types import path_t

__all__ = ["MisDataset"]


class MisDataset:
    """Class representing a measurement of the MIS sensor.

    Attributes
    ----------
    radar_data : dict
        Dictionary with data of all radar nodes
    timezone : str
        Timezone of recording (if available)
    sampling_rate_hz : int
        Sampling rate of recording in Hz

    """

    DATASTREAMS = Literal["hr", "resp"]

    _start_time_unix: int
    _tz: str
    _sampling_rate_hz: int
    _radar_data: pd.DataFrame

    def __init__(self, radar_data: pd.DataFrame, start_time: int, sampling_rate_hz: int, tz: Optional[str] = None):
        """Get new Dataset instance.

        .. note::
            Usually you shouldn't use this init directly.
            Use the provided `from_mat_file` constructor to handle loading recorded MIS Radar Sessions.

        Parameters
        ----------
        radar_data : :class:`~pandas.DataFrame`
            Dataframe with radar data
        start_time : int
            Start time of recording in unix time
        sampling_rate_hz : int
            Sampling rate of the radar sensor in Hz
        tz : str, optional
            Timezone of recording (if available)

        """
        self._radar_data = radar_data
        self._start_time_unix = start_time
        self._sampling_rate_hz = sampling_rate_hz
        self._tz = tz

    @property
    def sampling_rate_hz(self):
        """Sampling rate of the radar sensor in Hz."""
        return self._sampling_rate_hz

    @property
    def start_time(self):
        """Start time of the recording in unix timestamp."""
        return self._start_time_unix

    @property
    def timezone(self):
        """Timezone of the recording."""
        return self._tz

    def _add_index(self, data: pd.DataFrame, index: str, fs: int) -> pd.DataFrame:
        index_names = {
            None: "n_samples",
            "time": "t",
            "utc": "utc",
            "utc_datetime": "date",
            "local_datetime": f"date ({self.timezone})",
        }
        if index and index not in index_names:
            raise ValueError(f"Supplied value for index ({index}) is not allowed. Allowed values: {index_names.keys()}")
        index_name = index_names[index]
        data.index.name = index_name

        if index is None:
            return data
        if index == "time":
            data.index -= data.index[0]
            data.index /= fs
            return data

        data.index /= fs
        data.index += self._start_time_unix

        if index == "utc_datetime":
            data.index = pd.to_datetime(data.index, unit="s")
            data.index = data.index.tz_localize("UTC")
        if index == "local_datetime":
            data.index = pd.to_datetime(data.index, unit="s")
            data.index = data.index.tz_localize("UTC").tz_convert(self.timezone)

        return data

    @classmethod
    def from_mat_file(
        cls,
        path: path_t,
        tz: Optional[str] = "Europe/Berlin",
    ):
        """Create a new Dataset from a valid .mat file.

        Parameters
        ----------
        path : :class:`pathlib.Path`
            path to exported ".mat" file
        tz : str, optional
            Timezone str of the recording.
            This can be used to localize the start and end time.
            Note, this should not be the timezone of your current PC, but the timezone relevant for the specific
            recording.

        Returns
        -------
        result_dict : :class:`~pandas.DataFrame`
            dataframe with radar data

        """
        # ensure pathlib
        path = Path(path)
        _assert_file_extension(path, ".mat")

        start_date = datetime.datetime.strptime(
            re.findall(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", path.name)[0],
            "%Y-%m-%d_%H-%M-%S",
        )
        # convert time to unix timestamp
        start_date = int(start_date.timestamp())

        dict_radar = loadmat(str(path))
        fs = int(np.squeeze(dict_radar["fs"][0]))

        data = pd.DataFrame(
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
        data = data.rename(
            {
                "ch1": "I",
                "ch2": "Q",
                "resp": "Respiration",
                "pulse": "Pulse",
                "respStates": "Respiration_State",
                "heartbeats": "Heartbeats",
                "heartsoundQuality": "Heartsound_Quality",
            },
            axis=1,
        )
        return cls(data, start_date, fs, tz)

    def data_as_df(
        self,
        index: Optional[str] = None,
    ) -> pd.DataFrame:
        """Export the dataset into a pandas DataFrame.

        Parameters
        ----------
        index : str, optional
            Specify which index should be used for the dataset. The options are:
            * "time": For the time in seconds since the first sample
            * "utc": For the utc time stamp of each sample
            * "utc_datetime": for a pandas DateTime index in UTC time
            * "local_datetime": for a pandas DateTime index in the timezone set for the session
            * None: For a simple index (0...N)

        Raises
        ------
        ValueError
            If any other than the allowed `index` values are used.

        """
        data = self._radar_data.copy()
        data = self._add_index(data, index, self.sampling_rate_hz)
        return data

    def heart_rate(self, index: Optional[str] = None) -> pd.DataFrame:
        """Return the heart rate extracted from the MIS data as pandas DataFrame.

        Parameters
        ----------
        index : str, optional
            Specify which index should be used for the dataset. The options are:
            "time": For the time in seconds since the first sample
            "utc": For the utc time stamp of each sample
            "utc_datetime": for a pandas DateTime index in UTC time
            "local_datetime": for a pandas DateTime index in the timezone set for the session
            None: For a simple index (0...N)


        Returns
        -------
        :class:`~pandas.DataFrame`
            dataframe with heart rate data

        """
        data = self.data_as_df(index=index)
        loc_beat = np.where(data["Heartbeats"] == 1)[0]
        rri = np.ediff1d(loc_beat) / self.sampling_rate_hz
        hr_radar_raw = 60 / rri

        loc_beat = loc_beat[1:]
        hr_index = data.index[loc_beat]
        hs_quality = data["Heartsound_Quality"][hr_index]
        hr_radar_raw = pd.DataFrame(
            {
                "Heart_Rate": hr_radar_raw,
                "Heartsound_Quality": hs_quality,
                "RR_Interval": rri,
                "R_Peak_Idx": loc_beat,
            }
        )

        return hr_radar_raw

    def respiration(self, fs_out: int, index: Optional[str] = None) -> pd.DataFrame:
        """Return the respiration signal extracted from the MIS data as pandas DataFrame.

        Parameters
        ----------
        fs_out : int
            Sampling rate of the output respiration signal
        index : str, optional
            Specify which index should be used for the dataset. The options are:
            "time": For the time in seconds since the first sample
            "utc": For the utc time stamp of each sample
            "utc_datetime": for a pandas DateTime index in UTC time
            "local_datetime": for a pandas DateTime index in the timezone set for the session
            None: For a simple index (0...N)


        Returns
        -------
        :class:`~pandas.DataFrame`
            dataframe with heart rate data

        """
        data = self.data_as_df()
        cols = ["Respiration", "Respiration_State"]
        resp_raw = data[cols].copy()
        out = downsample(resp_raw, self.sampling_rate_hz, fs_out)
        out = pd.DataFrame(out, columns=cols)
        out["Respiration_State"] = np.around(out["Respiration_State"])
        out = self._add_index(out, index, fs_out)
        return out
