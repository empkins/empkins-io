from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
from avro.datafile import DataFileReader
from avro.io import DatumReader

from empkins_io.utils._types import path_t


class EmpaticaDataset:
    path: path_t
    _raw_data: dict

    _index_type: str
    _tz: str

    _sensor_dict = {
        "accelerometer": ["x", "y", "z"],
        "gyroscope": ["x", "y", "z"],
        "eda": ["values"],
        "temperature": ["values"],
        # "tags": [],
        "bvp": ["values"],
        # "systolicPeaks": [],
        "steps": ["values"],
        # TODO: add all sensors
    }

    _sampling_rates_hz = {
        "accelerometer": 64.0,
        "eda": 4.0,
        "temperature": 1.0,
    }

    def __init__(
        self,
        path: path_t,
        index_type: Optional[str] = None,
        tz: Optional[str] = None,
    ):
        self.path = path
        if path.is_dir():
            self._raw_data = _from_folder(path)
        else:
            self._raw_data = _from_file(path)
        self._index_type = index_type
        self._tz = tz

    @property
    def timezone(self):
        """Timezone of the recording."""
        return self._tz

    @property
    def acc(self) -> pd.DataFrame:
        """Get pandas DataFrame for accelerometer."""
        return self.data_as_df("accelerometer")

    @property
    def gyro(self) -> pd.DataFrame:
        raise ValueError("Probably there is no gyro data in the Empatica dataset.")  # TODO check this
        return self.data_as_df("gyroscope")

    @property
    def eda(self) -> pd.DataFrame:
        """Get pandas DataFrame for electrodermal activity."""
        return self.data_as_df("eda")

    @property
    def temperature(self) -> pd.DataFrame:
        """Get pandas DataFrame for temperature."""
        return self.data_as_df("temperature")

    def data_as_df(self, sensor: str) -> pd.DataFrame:
        """Get pandas DataFrame for a specific sensor."""
        if sensor not in self._sensor_dict:
            raise ValueError(f"Supplied sensor ({sensor}) is not allowed. Allowed values: {self._sensor_dict.keys()}")

        if self.path.is_file():
            return self._data_as_df_single_file(sensor)

        return self._data_as_df_folder(sensor)

    def _data_as_df_folder(self, sensor: str) -> pd.DataFrame:
        """Get pandas DataFrame for a specific sensor."""
        out = {}
        prev_last_timestamp = None
        sampling_frequencies = []

        start_time_unix = self._raw_data[list(self._raw_data.keys())[0]]["rawData"][sensor]["timestampStart"]

        for file in self._raw_data:
            sensor_dict = self._raw_data[file]["rawData"][sensor]
            df = pd.DataFrame({f"{sensor}_{channel}": sensor_dict[channel] for channel in self._sensor_dict[sensor]})

            df = self._add_index_for_stitching(df, sensor_dict["samplingFrequency"], sensor_dict["timestampStart"])

            # check for gaps between files
            if prev_last_timestamp is not None:
                # if the difference between the last timestamp of the previous file and the first timestamp of the current file is larger than twice sampling distance, we assume that there is a gap between the two files
                if (df.index[0] - prev_last_timestamp) > (pd.Timedelta(seconds=2 / sensor_dict["samplingFrequency"])):
                    raise ValueError(f"Gap between files detected. Please check the files in {self.path}.")
            prev_last_timestamp = df.index[-1]
            sampling_frequencies.append(sensor_dict["samplingFrequency"])

            out[file] = df
        out_df = pd.concat(out).droplevel(0)
        out_df.reset_index(inplace=True, drop=True)

        return self._add_index(
            out_df,
            self._index_type,
            np.mean(sampling_frequencies),
            start_time_unix,
        )

    def _data_as_df_single_file(self, sensor: str) -> pd.DataFrame:
        """Get pandas DataFrame for a specific sensor."""
        sensor_dict = self._raw_data["rawData"][sensor]

        df = pd.DataFrame({f"{sensor}_{channel}": sensor_dict[channel] for channel in self._sensor_dict[sensor]})

        return self._add_index(
            df,
            self._index_type,
            sensor_dict["samplingFrequency"],
            sensor_dict["timestampStart"],
        )

    def _add_index(
        self,
        data: pd.DataFrame,
        index: str,
        sampling_rate_hz: float,
        start_time_unix: int,
    ) -> pd.DataFrame:
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

        if index is None:
            return data
        if index == "time":
            data.index -= data.index[0]
            data.index /= sampling_rate_hz
            return data

        data.index = [round(start_time_unix + i * (1e6 / sampling_rate_hz)) for i in range(len(data.index))]
        data.index.name = index_name

        if index == "utc_datetime":
            # convert unix timestamps to datetime
            data.index = pd.to_datetime(data.index, unit="us")
            data.index = data.index.tz_localize("UTC")
        if index == "local_datetime":
            data.index = pd.to_datetime(data.index, unit="us")
            data.index = data.index.tz_localize("UTC").tz_convert(self.timezone)

        return data

    def _add_index_for_stitching(
        self, data: pd.DataFrame, sampling_rate_hz: float, start_time_unix: int
    ) -> pd.DataFrame:
        return self._add_index(data, "utc_datetime", sampling_rate_hz, start_time_unix)


@lru_cache(maxsize=2)
def _from_folder(path: path_t) -> dict:
    # this expects multiple .avro files, that belong to the same recording

    # list all .avro files
    files = sorted(path.glob("*.avro"))

    dict_out = {}

    # write all dicts in dict_out
    for file in files:
        dict_out[file.name] = _from_file(file)

    return dict_out


@lru_cache(maxsize=2)
def _from_file(path: path_t) -> dict:
    # check if path is a .avro file
    if path.suffix != ".avro":
        raise ValueError(f"Supplied path ({path}) is not a .avro file.")

    reader = DataFileReader(open(path, "rb"), DatumReader())
    data = next(reader)

    return data
