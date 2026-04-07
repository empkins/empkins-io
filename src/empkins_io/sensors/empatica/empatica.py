"""Class for reading data from empatica. Can be used for single files and folders containing multiple -avro files."""
from __future__ import annotations

import warnings
from collections.abc import Sequence
from functools import lru_cache
from typing import ClassVar

import pandas as pd
from avro.datafile import DataFileReader
from avro.io import DatumReader
from matplotlib import dates as mdates
from matplotlib import pyplot as plt

from empkins_io.utils._types import path_t


class EmpaticaDataset:
    """Reader for Empatica avro recordings stored as single files or folders."""

    path: path_t
    _raw_data: dict
    _accelerometer_specs: dict

    _index_type: str
    _tz: str

    _sensor_specs: ClassVar[dict[str, dict[str, object]]] = {
        "accelerometer": {
            "channels": ("x", "y", "z"),
            "name": "Accelerometer",
            "unit": "g",
            "kind": "signal",
            "default_sampling_rate_hz": 64.0,
        },
        "gyroscope": {
            "channels": ("x", "y", "z"),
            "name": "Gyroscope",
            "unit": "deg/s",
            "kind": "signal",
            "default_sampling_rate_hz": 64.0,
        },
        "eda": {
            "channels": ("values",),
            "name": "Electrodermal Activity",
            "unit": r"$\mu S$",
            "kind": "signal",
            "default_sampling_rate_hz": 4.0,
        },
        "temperature": {
            "channels": ("values",),
            "name": "Temperature",
            "unit": "°C",
            "kind": "signal",
            "default_sampling_rate_hz": 1.0,
        },
        "tags": {
            "channels": ("tagsTimeMicros",),
            "name": "Tag Events",
            "unit": "event",
            "kind": "event",
        },
        "bvp": {
            "channels": ("values",),
            "name": "Blood Volume Pulse",
            "unit": "a.u.",
            "kind": "signal",
            "default_sampling_rate_hz": 64.0,
        },
        "systolicPeaks": {
            "channels": ("peaksTimeNanos",),
            "name": "Systolic Peaks",
            "unit": "event",
            "kind": "event",
        },
        "steps": {
            "channels": ("values",),
            "name": "Steps",
            "unit": "count",
            "kind": "signal",
            "default_sampling_rate_hz": 2.0,
        },
    }

    _index_names: ClassVar[dict[str | None, str]] = {
        None: "n_samples",
        "time": "t",
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
        """
        Initialize the dataset and load raw Empatica data from file or folder.

        Parameters
        ----------
        path : path_t
            Path to a single Empatica avro file or to a folder containing multiple avro files.
        index_type : str | None, optional
            Type of index that should be added to the returned data.
        tz : str | None, optional
            Timezone used when ``index_type="local_datetime"``.
        """
        self.path = path
        if path.is_dir():
            self._raw_data = _from_folder(path)
        else:
            self._raw_data = _from_file(path)
        self._accelerometer_specs = self._get_accelerometer_specs()
        self._index_type = index_type
        self._tz = tz

    @property
    def timezone(self):
        """Timezone of the recording."""
        return self._tz

    @property
    def acc(self) -> pd.DataFrame:
        """Get pandas DataFrame for accelerometer. Values are converted from ADC in g (gravitational acceleration)."""
        df = self.data_as_df("accelerometer")
        conversion_factor = self._accelerometer_specs["delta_physical"] / self._accelerometer_specs["delta_digital"]
        df["accelerometer_x_g"] = [val * conversion_factor for val in df["accelerometer_x"]]
        df["accelerometer_y_g"] = [val * conversion_factor for val in df["accelerometer_y"]]
        df["accelerometer_z_g"] = [val * conversion_factor for val in df["accelerometer_z"]]
        return df

    @property
    def gyro(self) -> pd.DataFrame:
        """Get pandas DataFrame for gyroscope data."""
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
        """Get pandas DataFrame for blood volume pulse."""
        return self.data_as_df("bvp")

    @property
    def steps(self) -> pd.DataFrame:
        """Get pandas DataFrame for steps."""
        return self.data_as_df("steps")

    def systolic_peaks(self, series=False) -> pd.DataFrame:
        """
        Get systolic peak event data.

        Parameters
        ----------
        series : bool, optional
            If ``True``, return only the timestamps as a one-column DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the systolic peak events.
        """
        if series:
            return pd.DataFrame(self.data_as_df("systolicPeaks").index)
        else:
            return self.data_as_df("systolicPeaks")

    def tag_events(self, series=True) -> pd.DataFrame:
        """
        Get tag event data.

        Parameters
        ----------
        series : bool, optional
            If ``True``, return only the timestamps as a one-column DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the tag events.
        """
        if series:
            return pd.DataFrame(self.data_as_df("tags").index)
        else:
            return self.data_as_df("tags")

    def data_as_df(self, sensor: str) -> pd.DataFrame:
        """
        Get a pandas DataFrame for a specific sensor.

        Parameters
        ----------
        sensor : str
            Name of the sensor to load.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the requested sensor data.

        Raises
        ------
        ValueError
            If the supplied sensor is not supported.
        """
        if sensor not in self._sensor_specs:
            raise ValueError(f"Supplied sensor ({sensor}) is not allowed. Allowed values: {self._sensor_specs.keys()}")

        if self.path.is_file():
            return self._data_as_df_single_file(sensor)

        return self._data_as_df_folder(sensor)

    def plot_empatica(self, sensor: str) -> None:
        """
        Plot a single Empatica sensor using the configured index type.

        Parameters
        ----------
        sensor : str
            Name of the sensor to plot.
        """
        if sensor == "accelerometer":
            data = self.acc[["accelerometer_x_g", "accelerometer_y_g", "accelerometer_z_g"]]
        else:
            data = self.data_as_df(sensor)

        sensor_spec = self._sensor_specs[sensor]
        sensor_name = str(sensor_spec["name"])
        sensor_unit = sensor_spec.get("unit")
        y_label = sensor_name if not sensor_unit else f"{sensor_name} [{sensor_unit}]"
        x_label = self._index_names.get(self._index_type, "index")

        fig, ax = plt.subplots()
        is_event_sensor = sensor_spec["kind"] == "event"
        if is_event_sensor:
            event_x = data.index
            default_color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
            ax.eventplot(
                [event_x.to_numpy()],
                orientation="horizontal",
                lineoffsets=0,
                linelengths=0.8,
                linewidths=1.2,
                colors=default_color,
            )
            ax.set_ylim(-0.75, 0.75)
            ax.set_yticks([0])
            ax.set_yticklabels(["events"])
        else:
            for column in data.columns:
                label = column.removeprefix(f"{sensor}_").replace("_", " ").title()
                ax.plot(data.index, data[column], label=label)
            if len(data.columns) > 1:
                ax.legend()

        if isinstance(data.index, pd.DatetimeIndex):
            timezone = data.index.tz if data.index.tz is not None else None
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=timezone))
            fig.autofmt_xdate()
            if self._index_type == "local_datetime" and self.timezone:
                x_label = f"date ({self.timezone})"

        ax.set_xlabel(x_label)

        ax.set_title(sensor_name)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()

    def _data_as_df_folder(self, sensor: str) -> pd.DataFrame:
        """
        Load, index, and concatenate all files of one sensor from a recording folder.

        Parameters
        ----------
        sensor : str
            Name of the sensor to load.

        Returns
        -------
        pd.DataFrame
            Concatenated sensor data from all files in the folder.
        """
        out = {}
        for file in self._raw_data:
            sensor_dict = self._raw_data[file]["rawData"][sensor]
            df = pd.DataFrame(
                {f"{sensor}_{channel}": sensor_dict[channel] for channel in self._sensor_channels(sensor)}
            )
            if df.empty:
                warnings.warn(f"Warning: {file} contains no data for {sensor}. Continuation without this file. ")
                continue

            timestamp_key, timestamp_unit = type(self)._get_explicit_timestamp_info(sensor_dict)
            out[file] = self._add_index(
                df,
                self._index_type,
                self._get_sampling_rate(sensor, sensor_dict),
                sensor_dict.get("timestampStart"),
                explicit_timestamps=sensor_dict.get(timestamp_key) if timestamp_key else None,
                explicit_timestamp_unit=timestamp_unit,
            )

        df_out = pd.concat(out).droplevel(0)

        if sensor == "tags" or sensor == "systolicPeaks":
            return df_out
        # fill dataframe gaps with nans
        intervall_seconds = 1 / self._get_sampling_rate(sensor, sensor_dict)
        freq_str = f"{intervall_seconds}s"

        target_index = pd.date_range(
            start=df_out.index.min().round(freq_str), end=df_out.index.max().round(freq_str), freq=freq_str
        )
        # define tolerance for index since the sampling frequency is slightly varying
        tolerance_val = pd.Timedelta(seconds=intervall_seconds / 2)

        df_fixed = df_out.reindex(target_index, method="nearest", tolerance=tolerance_val)

        return df_fixed

    def _data_as_df_single_file(self, sensor: str) -> pd.DataFrame:
        """
        Load and index one sensor from a single Empatica avro file.

        Parameters
        ----------
        sensor : str
            Name of the sensor to load.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the indexed sensor data.

        Raises
        ------
        ValueError
            If the requested sensor contains no data.
        """
        sensor_dict = self._raw_data["rawData"][sensor]
        df = pd.DataFrame({f"{sensor}_{channel}": sensor_dict[channel] for channel in self._sensor_channels(sensor)})

        if df.empty:
            raise ValueError(f"There is no {sensor} data in the Empatica dataset.")

        timestamp_key, timestamp_unit = type(self)._get_explicit_timestamp_info(sensor_dict)

        return self._add_index(
            df,
            self._index_type,
            self._get_sampling_rate(sensor, sensor_dict),
            sensor_dict.get("timestampStart"),
            explicit_timestamps=sensor_dict.get(timestamp_key) if timestamp_key else None,
            explicit_timestamp_unit=timestamp_unit,
        )

    def _add_index(
        self,
        data: pd.DataFrame,
        index: str,
        sampling_rate_hz: float | None = None,
        start_time_unix: int | None = None,
        explicit_timestamps: Sequence[int] | None = None,
        explicit_timestamp_unit: str | None = None,
    ) -> pd.DataFrame:
        """
        Add the requested index representation to sampled or event-based sensor data.

        Parameters
        ----------
        data : pd.DataFrame
            Sensor data without a processed index.
        index : str
            Requested index type.
        sampling_rate_hz : float | None, optional
            Sampling rate of regularly sampled data.
        start_time_unix : int | None, optional
            Start timestamp of regularly sampled data in Unix microseconds.
        explicit_timestamps : Sequence[int] | None, optional
            Explicit timestamps for event-based data.
        explicit_timestamp_unit : str | None, optional
            Unit of ``explicit_timestamps``.

        Returns
        -------
        pd.DataFrame
            Data with the requested index.
        """
        index_names = self._index_names | {"local_datetime": f"date ({self.timezone})"}
        if index and index not in index_names:
            raise ValueError(f"Supplied value for index ({index}) is not allowed. Allowed values: {index_names.keys()}")
        index_name = index_names[index]

        if index is None:
            return data
        if index == "time":
            if explicit_timestamps is not None:
                data.index = self._timestamps_to_seconds(explicit_timestamps, explicit_timestamp_unit)
            else:
                if sampling_rate_hz is None:
                    raise ValueError("sampling_rate_hz must be provided for regularly sampled data.")
                data.index -= data.index[0]
                data.index /= sampling_rate_hz
            data.index.name = index_name
            return data

        if explicit_timestamps is not None:
            units = "us"
            data.index = self._timestamps_to_microseconds(explicit_timestamps, explicit_timestamp_unit)
        else:
            units = "us"
            if sampling_rate_hz is None or start_time_unix is None:
                raise ValueError("sampling_rate_hz and start_time_unix must be provided for regularly sampled data.")
            data.index = [round(start_time_unix + i * (1e6 / sampling_rate_hz)) for i in range(len(data.index))]

        data.index.name = index_name

        if index == "utc_datetime":
            # convert unix timestamps to datetime
            data.index = pd.to_datetime(data.index, unit=units)
            data.index = data.index.tz_localize("UTC")
        if index == "local_datetime":
            data.index = pd.to_datetime(data.index, unit=units)
            data.index = data.index.tz_localize("UTC").tz_convert(self.timezone)

        return data

    def _sensor_channels(self, sensor: str) -> Sequence[str]:
        """
        Return the raw channel names defined for a sensor.

        Parameters
        ----------
        sensor : str
            Name of the sensor.

        Returns
        -------
        Sequence[str]
            Raw channel names of the sensor.
        """
        return self._sensor_specs[sensor]["channels"]  # type: ignore[return-value]

    def _get_sampling_rate(self, sensor: str, sensor_dict: dict) -> int | None:
        """
        Return the sampling rate for a sensor.

        Parameters
        ----------
        sensor : str
            Name of the sensor.
        sensor_dict : dict
            Raw sensor dictionary from the avro file.

        Returns
        -------
        int | None
            Sampling rate of the sensor or ``None`` for event-based data.
        """
        sampling_rate = sensor_dict.get("samplingFrequency")
        if sampling_rate is None:
            return None
        if sampling_rate == 0:
            fallback_sampling_rate = self._sensor_specs[sensor].get("default_sampling_rate_hz")
            if fallback_sampling_rate is None:
                raise ValueError(f"Sampling frequency for {sensor} is 0. Please check the data in {self.path}.")
            return round(fallback_sampling_rate)
        return round(sampling_rate)

    @staticmethod
    def _get_explicit_timestamp_info(sensor_dict: dict) -> tuple[str | None, str | None]:
        """
        Detect explicit event timestamp fields and return their key and unit.

        Parameters
        ----------
        sensor_dict : dict
            Raw sensor dictionary from the avro file.

        Returns
        -------
        tuple[str | None, str | None]
            Timestamp key and timestamp unit.
        """
        for key in sensor_dict:
            if key.endswith("TimeNanos"):
                return key, "ns"
            if key.endswith("TimeMicros"):
                return key, "us"
        return None, None

    @staticmethod
    def _timestamps_to_microseconds(timestamps: Sequence[int], unit: str | None) -> pd.Index:
        """
        Convert explicit timestamps in microseconds or nanoseconds to microseconds.

        Parameters
        ----------
        timestamps : Sequence[int]
            Sequence of timestamps.
        unit : str | None
            Unit of the timestamps.

        Returns
        -------
        pd.Index
            Timestamp index in microseconds.
        """
        if unit == "ns":
            return pd.Index([round(timestamp / 1000) for timestamp in timestamps], dtype="int64")
        if unit == "us":
            return pd.Index(timestamps, dtype="int64")
        raise ValueError(f"Unsupported timestamp unit: {unit}")

    @staticmethod
    def _timestamps_to_seconds(timestamps: Sequence[int], unit: str | None) -> pd.Index:
        """
        Convert explicit timestamps to relative seconds from the first event.

        Parameters
        ----------
        timestamps : Sequence[int]
            Sequence of timestamps.
        unit : str | None
            Unit of the timestamps.

        Returns
        -------
        pd.Index
            Relative time index in seconds.
        """
        if unit == "ns":
            divisor = 1e9
        elif unit == "us":
            divisor = 1e6
        else:
            raise ValueError(f"Unsupported timestamp unit: {unit}")

        start_time = timestamps[0]
        return pd.Index([(timestamp - start_time) / divisor for timestamp in timestamps], dtype="float64")

    def _get_accelerometer_specs(self) -> dict:
        """
        Extract accelerometer calibration values and derived conversion factors.

        Returns
        -------
        dict
            Dictionary containing accelerometer calibration values.
        """
        if self.path.is_dir():
            first_file = next(iter(self._raw_data.values()))
            imu_params = dict(first_file["rawData"]["accelerometer"]["imuParams"])
        else:
            imu_params = dict(self._raw_data["rawData"]["accelerometer"]["imuParams"])

        imu_params["delta_physical"] = imu_params["physicalMax"] - imu_params["physicalMin"]
        imu_params["delta_digital"] = imu_params["digitalMax"] - imu_params["digitalMin"]
        return imu_params


@lru_cache(maxsize=2)
def _from_folder(path: path_t) -> dict:
    """
    Load all avro files from a folder into a dict keyed by file name.

    Parameters
    ----------
    path : path_t
        Path to a folder containing Empatica avro files.

    Returns
    -------
    dict
        Dictionary containing all loaded avro files.
    """
    files = sorted(path.glob("*.avro"))

    dict_out = {}

    # write all dicts in dict_out
    for file in files:
        dict_out[file.name] = _from_file(file)

    return dict_out


@lru_cache(maxsize=2)
def _from_file(path: path_t) -> dict:
    """
    Load a single Empatica avro file into a Python dict.

    Parameters
    ----------
    path : path_t
        Path to the Empatica avro file.

    Returns
    -------
    dict
        Dictionary containing avro file data as a Python dict.

    Raises
    ------
    ValueError
        If the supplied path does not point to a .avro file.
    """
    if path.suffix != ".avro":
        raise ValueError(f"Supplied path ({path}) is not a .avro file.")

    reader = DataFileReader(path.open("rb"), DatumReader())
    data = next(reader)

    return data
