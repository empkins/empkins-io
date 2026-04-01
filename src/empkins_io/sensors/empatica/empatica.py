from collections.abc import Sequence
from functools import lru_cache
from typing import ClassVar

import numpy as np
import pandas as pd
from avro.datafile import DataFileReader
from avro.io import DatumReader
from matplotlib import dates as mdates
from matplotlib import pyplot as plt

from empkins_io.utils._types import path_t


class EmpaticaDataset:
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
        if sensor not in self._sensor_specs:
            raise ValueError(f"Supplied sensor ({sensor}) is not allowed. Allowed values: {self._sensor_specs.keys()}")

        if self.path.is_file():
            return self._data_as_df_single_file(sensor)

        return self._data_as_df_folder(sensor)

    def plot_empatica(self, sensor: str) -> None:
        """Plot empatica."""
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

    def _data_as_df_folder(self, sensor: str) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """Get pandas DataFrame for a specific sensor."""
        current_segment = {}
        segmented_out = {}
        segment_idx = 0
        prev_last_timestamp = None
        sampling_frequencies = []
        explicit_timestamps = []
        explicit_timestamp_unit = None
        current_start_time_unix = None

        first_file = next(iter(self._raw_data.keys()))
        first_sensor_dict = self._raw_data[first_file]["rawData"][sensor]
        timestamp_key, timestamp_unit = type(self)._get_explicit_timestamp_info(first_sensor_dict)
        explicit_timestamp_unit = timestamp_unit

        for file in self._raw_data:
            sensor_dict = self._raw_data[file]["rawData"][sensor]
            df = pd.DataFrame(
                {f"{sensor}_{channel}": sensor_dict[channel] for channel in self._sensor_channels(sensor)}
            )

            if timestamp_key is not None:
                file_timestamps = sensor_dict[timestamp_key]
                file_start_timestamp = file_timestamps[0]
                file_end_timestamp = file_timestamps[-1]

                if (
                    prev_last_timestamp is not None
                    and self._timestamp_gap_seconds(prev_last_timestamp, file_start_timestamp, timestamp_unit) > 2
                ):
                    segmented_out[f"{sensor}_segment_{segment_idx}"] = self._finalize_folder_segment(
                        current_segment,
                        sensor,
                        None,
                        None,
                        explicit_timestamps=explicit_timestamps,
                        explicit_timestamp_unit=explicit_timestamp_unit,
                    )
                    segment_idx += 1
                    current_segment = {}
                    explicit_timestamps = []

                explicit_timestamps.extend(file_timestamps)
                prev_last_timestamp = file_end_timestamp
            else:
                sampling_rate_hz = self._get_sampling_rate(sensor, sensor_dict)
                df = self._add_index_for_stitching(df, sampling_rate_hz, sensor_dict["timestampStart"])

                if current_start_time_unix is None:
                    current_start_time_unix = sensor_dict["timestampStart"]

                # check for gaps between files
                # if the gap is bigger than 2 seconds, a new dataframe is generated
                if prev_last_timestamp is not None and (df.index[0] - prev_last_timestamp) > (
                    pd.Timedelta(seconds=2 / sampling_rate_hz)
                ):
                    segmented_out[f"{sensor}_segment_{segment_idx}"] = self._finalize_folder_segment(
                        current_segment,
                        sensor,
                        sampling_frequencies,
                        current_start_time_unix,
                    )
                    segment_idx += 1
                    current_segment = {}
                    sampling_frequencies = []
                    current_start_time_unix = sensor_dict["timestampStart"]
                prev_last_timestamp = df.index[-1]
                sampling_frequencies.append(sampling_rate_hz)

            current_segment[file] = df

        segmented_out[f"{sensor}_segment_{segment_idx}"] = self._finalize_folder_segment(
            current_segment,
            sensor,
            sampling_frequencies if timestamp_key is None else None,
            current_start_time_unix,
            explicit_timestamps=explicit_timestamps if timestamp_key is not None else None,
            explicit_timestamp_unit=explicit_timestamp_unit,
        )

        if len(segmented_out) == 1:
            return next(iter(segmented_out.values()))

        return segmented_out

    def _data_as_df_single_file(self, sensor: str) -> pd.DataFrame:
        """Get pandas DataFrame for a specific sensor."""
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

    def _add_index_for_stitching(
        self, data: pd.DataFrame, sampling_rate_hz: float, start_time_unix: int
    ) -> pd.DataFrame:
        return self._add_index(data, "utc_datetime", sampling_rate_hz, start_time_unix)

    def _finalize_folder_segment(
        self,
        segment_data: dict[str, pd.DataFrame],
        sensor: str,
        sampling_frequencies: Sequence[float] | None,
        start_time_unix: int | None,
        explicit_timestamps: Sequence[int] | None = None,
        explicit_timestamp_unit: str | None = None,
    ) -> pd.DataFrame:
        out_df = pd.concat(segment_data).droplevel(0)
        out_df = out_df.reset_index(drop=True)

        return self._add_index(
            out_df,
            self._index_type,
            np.mean(sampling_frequencies) if sampling_frequencies else None,
            start_time_unix,
            explicit_timestamps=explicit_timestamps or None,
            explicit_timestamp_unit=explicit_timestamp_unit,
        )

    @staticmethod
    def _timestamp_gap_seconds(previous_timestamp: int, current_timestamp: int, unit: str | None) -> float:
        if unit == "ns":
            divisor = 1e9
        elif unit == "us":
            divisor = 1e6
        else:
            raise ValueError(f"Unsupported timestamp unit: {unit}")

        return (current_timestamp - previous_timestamp) / divisor

    def _sensor_channels(self, sensor: str) -> Sequence[str]:
        return self._sensor_specs[sensor]["channels"]  # type: ignore[return-value]

    def _get_sampling_rate(self, sensor: str, sensor_dict: dict) -> float | None:
        sampling_rate = sensor_dict.get("samplingFrequency")
        if sampling_rate is None:
            return None
        if sampling_rate == 0:
            fallback_sampling_rate = self._sensor_specs[sensor].get("default_sampling_rate_hz")
            if fallback_sampling_rate is None:
                raise ValueError(f"Sampling frequency for {sensor} is 0. Please check the data in {self.path}.")
            return float(fallback_sampling_rate)
        return float(sampling_rate)

    @staticmethod
    def _get_explicit_timestamp_info(sensor_dict: dict) -> tuple[str | None, str | None]:
        for key in sensor_dict:
            if key.endswith("TimeNanos"):
                return key, "ns"
            if key.endswith("TimeMicros"):
                return key, "us"
        return None, None

    @staticmethod
    def _timestamps_to_microseconds(timestamps: Sequence[int], unit: str | None) -> pd.Index:
        if unit == "ns":
            return pd.Index([round(timestamp / 1000) for timestamp in timestamps], dtype="int64")
        if unit == "us":
            return pd.Index(timestamps, dtype="int64")
        raise ValueError(f"Unsupported timestamp unit: {unit}")

    @staticmethod
    def _timestamps_to_seconds(timestamps: Sequence[int], unit: str | None) -> pd.Index:
        if unit == "ns":
            divisor = 1e9
        elif unit == "us":
            divisor = 1e6
        else:
            raise ValueError(f"Unsupported timestamp unit: {unit}")

        start_time = timestamps[0]
        return pd.Index([(timestamp - start_time) / divisor for timestamp in timestamps], dtype="float64")

    def _get_accelerometer_specs(self) -> dict:
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
