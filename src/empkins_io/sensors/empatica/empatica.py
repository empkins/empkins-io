from collections.abc import Sequence
from functools import lru_cache
from typing import ClassVar

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

    _sensor_dict: ClassVar[dict[str, Sequence[str]]] = {
        "accelerometer": ["x", "y", "z"],
        "gyroscope": ["x", "y", "z"],
        "eda": ["values"],  # electrodermal activity
        "temperature": ["values"],
        "tags": ["tagsTimeMicros"],
        "bvp": ["values"],  # blood volume pulse
        "systolicPeaks": ["peaksTimeNanos"],
        "steps": ["values"],
    }

    _sampling_rates_hz: ClassVar[dict[str, float]] = {
        "accelerometer": 64.0,
        "eda": 4.0,
        "temperature": 1.0,
        "bvp": 64.0,
        "steps": 2.0,
    }

    _sensor_unit: ClassVar[dict[str]] = {
        "accelerometer": "g",
        "gyroscope": "deg/s",
        "eda": r"$\mu S$",
        "temperature": "°C",
        "bvp": "a.u.",
        "tags": "event",
        "steps": "count",
        "systolicPeaks": "event",
    }

    _sensor_name: ClassVar[dict[str, str]] = {
        "accelerometer": "Accelerometer",
        "gyroscope": "Gyroscope",
        "eda": "Electrodermal Activity",
        "temperature": "Temperature",
        "tags": "Tag Events",
        "bvp": "Blood Volume Pulse",
        "systolicPeaks": "Systolic Peaks",
        "steps": "Steps",
    }

    _index_names: ClassVar[dict[str | None, str]] = {
        None: "n_samples",
        "time": "t (s)",
        "utc": "utc",
        "utc_datetime": "date",
        "local_datetime": "date",
    }

    def __init__(
        self,
        path: path_t,
        index_type: str | None = None,  # can be e.g. local_datetime, utc_datetime, time, None
        tz: str | None = None,  # can be e.g. "Europe/Berlin"
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
        """Get pandas DataFrame for accelerometer. Values are converted from ADC in g (gravitational acceleration)."""
        imuParams = self._raw_data["rawData"]["accelerometer"]["imuParams"]
        delta_physical = imuParams["physicalMax"] - imuParams["physicalMin"]
        delta_digital = imuParams["digitalMax"] - imuParams["digitalMin"]

        df = self.data_as_df("accelerometer")
        df["accelerometer_x_g"] = [val * delta_physical / delta_digital for val in df["accelerometer_x"]]
        df["accelerometer_y_g"] = [val * delta_physical / delta_digital for val in df["accelerometer_y"]]
        df["accelerometer_z_g"] = [val * delta_physical / delta_digital for val in df["accelerometer_z"]]
        return df

    @property
    def gyro(self) -> pd.DataFrame:
        return self.data_as_df("gyroscope")

    @property
    def eda(self) -> pd.DataFrame:
        """Get pandas DataFrame for electrodermal activity."""
        return self.data_as_df("eda")

    @property
    def temperature(self) -> pd.DataFrame:
        """Get pandas DataFrame for temperature."""
        return self.data_as_df("temperature")

    @property
    def bvp(self) -> pd.DataFrame:
        """Get pandas DataFrame for blood volumne pulse."""
        return self.data_as_df("bvp")

    @property
    def steps(self) -> pd.DataFrame:
        """Get pandas DataFrame for steps."""
        return self.data_as_df("steps")

    def systolic_peaks(self, series=False) -> pd.DataFrame:
        """Get pandas Dataframe for systolic peaks (Event Data)."""
        if series:
            return pd.DataFrame(self.data_as_df("systolicPeaks").index)
        else:
            return self.data_as_df("systolicPeaks")

    def tag_events(self, series=True) -> pd.DataFrame:
        """Get pandas Dataframe for Tagging Events."""
        if series:
            return pd.DataFrame(self.data_as_df("tags").index)
        else:
            return self.data_as_df("tags")

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

        start_time_unix = self._raw_data[next(iter(self._raw_data.keys()))]["rawData"][sensor]["timestampStart"]

        for file in self._raw_data:
            sensor_dict = self._raw_data[file]["rawData"][sensor]
            df = pd.DataFrame({f"{sensor}_{channel}": sensor_dict[channel] for channel in self._sensor_dict[sensor]})

            df = self._add_index_for_stitching(df, sensor_dict["samplingFrequency"], sensor_dict["timestampStart"])

            # TODO check if that is the right way to do.
            # check for gaps between files
            if prev_last_timestamp is not None and (df.index[0] - prev_last_timestamp) > (
                pd.Timedelta(seconds=2 / sensor_dict["samplingFrequency"])
            ):
                # if the difference between the last timestamp of the previous file and the first timestamp of the
                # current file is larger than twice sampling distance, we assume that there is a gap between the
                # two files
                raise ValueError(f"Gap between files detected. Please check the files in {self.path}.")
            prev_last_timestamp = df.index[-1]
            sampling_frequencies.append(sensor_dict["samplingFrequency"])

            out[file] = df
        out_df = pd.concat(out).droplevel(0)
        out_df = out_df.reset_index(drop=True)

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

        if df.empty:
            raise ValueError(f"There is no {sensor} data in the Empatica dataset.")

        # Sesnor only contains event data
        if "samplingFrequency" not in sensor_dict:
            return self._add_index(df, self._index_type)

        # Sensor contains a sampled signal
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
        sampling_rate_hz: float = None,
        start_time_unix: int = None,
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

        # standard time unit is micro seconds
        units = "us"
        factor = 1e6
        if any("TimeNanos" in col for col in data.columns):
            units = "ns"
            factor = 1e9

        if index is None:
            return data
        if index == "time":
            if sampling_rate_hz is None:
                data.index = data.iloc[:, 0]
                data.index.name = index_name
                data.index -= data.index[0]
                data.index /= factor
                return data
            else:
                data.index -= data.index[0]
                data.index /= sampling_rate_hz
                data.index.name = index_name
                return data

        if sampling_rate_hz is None:
            data.index = data.iloc[:, 0]
        else:
            data.index = [round(start_time_unix + i * (factor / sampling_rate_hz)) for i in range(len(data.index))]

        data.index.name = index_name

        if index == "utc_datetime":
            # convert unix timestamps to datetime
            data.index = pd.to_datetime(data.index, unit=units)
            data.index = data.index.tz_localize("UTC")
        if index == "local_datetime":
            data.index = pd.to_datetime(data.index, unit=units)
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

    reader = DataFileReader(path.open("rb"), DatumReader())
    data = next(reader)

    return data
