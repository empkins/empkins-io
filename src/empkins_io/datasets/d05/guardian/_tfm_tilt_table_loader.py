import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
from biopsykit.utils.array_handling import downsample
from scipy.io import loadmat

from empkins_io.utils._types import path_t


class TFMTiltTableLoader:
    """Class for loading and processing Task Force Monitor (TFM) data from the Guardian Study."""

    SAMPLING_RATES_IN = {"ecg_1": 1000, "ecg_2": 1000, "icg_der": 500}
    SAMPLING_RATE_OUT = 500
    CHANNEL_MAPPING = {"ecg_1": "rawECG1", "ecg_2": "rawECG2", "icg_der": "rawICG"}
    PHASES = ["BeginRecording", "Pause", "Valsalva", "HoldingBreath", "TiltUp", "TiltDown"]
    ORIGINAL_NAMES = {
        "BeginRecording": "Beginn der Aufzeichnung",
        "Pause": "Ruhe",
        "Valsalva": "Valsalva",
        "HoldingBreath": "AtmungAnhalten",
        "TiltUp": "TiltUp",
        "TiltDown": "TiltDown",
    }
    _start_time_unix: pd.Timestamp
    _tz: str
    _start_time_dict: Dict[str, pd.Timestamp]

    def __init__(
        self,
        data_dict: Dict[str, pd.DataFrame],
        sampling_rate_dict: Dict[str, float],
        start_time_dict: Dict[str, pd.Timestamp],
        tz: Optional[str] = None,
        start_time: Optional[pd.Timestamp] = None,
    ):
        """Initialize a TFM dataset.

        Parameters
        ----------
        data_dict : dict
            Dictionary containing TFM data. Keys are channel names, values are dataframes with the TFM data.
        tz : str, optional
            Timezone of the data. Default: None

        """
        self._data = data_dict
        self._sampling_rate = sampling_rate_dict
        for name, data in data_dict.items():
            setattr(self, name, data)
        for name, sampling_rate in sampling_rate_dict.items():
            setattr(self, f"sampling_rate_hz_{name}", sampling_rate)
        setattr(self, "channels", list(self._data.keys()))
        self._start_time_dict = start_time_dict
        # self._start_time_dict = {phase: pd.Timestamp.today(tz="UTC") for phase in self.PHASES}
        self._tz = tz

    @classmethod
    def from_mat_file(
        cls,
        base_path: path_t,
        file_path: path_t,
        tz: Optional[str] = "Europe/Berlin",
    ):
        total_path = base_path.joinpath(file_path)
        data = loadmat(total_path, struct_as_record=False, squeeze_me=True)

        nur_gdn = os.path.basename(os.path.dirname(os.path.dirname(total_path)))

        data_raw = data["RAW_SIGNALS"]
        names = data["IV"].Name
        times = data["IV"].AbsTime
        dates = pd.read_csv(base_path.joinpath("meassuring_dates.csv"), sep=";")

        measuring_date = dates.loc[dates["participant"] == nur_gdn, "meassuring_date"].values[0]

        data_dict_tmp = {key: getattr(data_raw, value) for key, value in cls.CHANNEL_MAPPING.items()}

        data_dict = {}

        for key, value in data_dict_tmp.items():
            data_dict[key] = {}

            for key1, value1 in cls.ORIGINAL_NAMES.items():
                if value1 not in names:
                    data_dict[key][key1] = np.array([])

                else:
                    index = list(names).index(value1)
                    data_dict[key][key1] = value[index]

        for ecg_key, ecg_value in data_dict.items():
            if cls.SAMPLING_RATES_IN[ecg_key] != cls.SAMPLING_RATE_OUT:
                for recording_key, recording_value in ecg_value.items():
                    if isinstance(recording_value, np.ndarray):
                        if len(recording_value) == 0:
                            continue
                        data_dict[ecg_key][recording_key] = downsample(
                            recording_value, fs_in=cls.SAMPLING_RATES_IN[ecg_key], fs_out=cls.SAMPLING_RATE_OUT
                        )

        new_data = {}

        for ecg_key, ecg_value in data_dict.items():
            for recording_key, recording_value in ecg_value.items():
                if recording_key not in new_data:
                    new_data[recording_key] = {}

                new_data[recording_key][ecg_key] = recording_value

        for key2, value2 in new_data.items():
            new_data[key2] = pd.concat(
                [
                    pd.Series(new_data[key2]["ecg_1"]),
                    pd.Series(new_data[key2]["ecg_2"]),
                    pd.Series(new_data[key2]["icg_der"]),
                ],
                axis=1,
            )
            new_data[key2].columns = cls.CHANNEL_MAPPING.keys()
            new_data[key2] = new_data[key2].dropna(axis=0)

        start_time_dict = {}

        for key1, value1 in cls.ORIGINAL_NAMES.items():
            if value1 not in names:
                start_time_dict[key1] = pd.Timestamp("2000-01-01 00:00:00", tz="EUROPE/BERLIN")

            else:
                index = list(names).index(value1)
                start_time_dict[key1] = pd.Timestamp(measuring_date + " " + times[index], tz="EUROPE/BERLIN")

        return cls(data_dict=new_data, tz=tz, sampling_rate_dict={}, start_time_dict=start_time_dict)

    @property
    def start_time_unix(self) -> Optional[pd.Timestamp]:
        """Start time of the recording in UTC time."""
        return self._start_time_unix

    @property
    def timezone(self) -> str:
        """Timezone the dataset was recorded in."""
        return self._tz

    def data_as_dict(self, index: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        return {
            key: self._add_index(val, index=index, start_time=start_time)
            for (key, val), start_time in zip(self._data.items(), self._start_time_dict.values())
        }

    def _add_index(self, data: pd.DataFrame, index: str, start_time: Optional[pd.Timestamp] = None) -> pd.DataFrame:
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
            data = data.reset_index(drop=True)
            data.index.name = index_name
            return data

        data.index = data.index / self.SAMPLING_RATE_OUT

        if index == "time":
            return data

        if index == "utc":
            # convert counter to utc timestamps
            time_object = pd.to_datetime(start_time)
            data.index += time_object.timestamp()
            return data

        if start_time is None:
            start_time = self.start_time_unix

        if start_time is None:
            raise ValueError(
                "No start time available - can't convert to datetime index! "
                "Use a different index representation or provide a custom start time using the 'start_time' parameter."
            )

        # convert counter to pandas datetime index
        data.index = pd.to_timedelta(data.index, unit="s")
        start_time = pd.to_datetime(start_time)
        data.index += start_time
        data.index = pd.to_datetime(data.index, unit="s").tz_convert("UTC")

        if index == "local_datetime":
            data.index = pd.to_datetime(data.index, unit="s").tz_convert(self.timezone)

        return data
