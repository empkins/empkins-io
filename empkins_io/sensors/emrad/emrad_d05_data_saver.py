from pathlib import Path
from typing import Dict, Optional, Sequence

import h5py
import pandas as pd
from biopsykit.utils._datatype_validation_helper import _assert_file_extension
from typing_extensions import Self

from empkins_io.utils._types import path_t
from empkins_io.utils.exceptions import InvalidFileFormatError


class EmradD05DataSaver:
    """Class representing a measurement of the Emrad sensor.

    Attributes
    ----------
    radar_data : dict
        Dictionary with data of all radar nodes
    start_time : float
        unix start time of the recording
    sampling_rate_hz : int
        Sampling rate of the radar nodes in Hz

    """

    _RADAR_NODES = ["rad1", "rad2", "rad3", "rad4"]
    _radar_data: Dict[str, pd.DataFrame]
    _start_time_unix: int
    _stop_time_unix: int
    _measurement_id: str
    _comment: str
    _sensor_id: int
    _file_name: str
    _sampling_rate_hz: float

    def __init__(
        self,
        radar_data: Dict[str, pd.DataFrame],
        start_time: int,
        stop_time: int,
        measurement_id: str,
        comment: str,
        sensor_id: int,
        file_name: str,
        sampling_rate_hz: float = 976.5625,
    ):
        """Get new Dataset instance.

        .. note::
            Usually you shouldn't use this init directly.
            Use the provided `from_hd5_file` constructor to handle loading recorded Emrad Radar Sessions.

        Parameters
        ----------
        radar_data :
            Dictionary with names of the radar node and sensor data as pandas Dataframe

        """
        self._radar_data = radar_data
        self._start_time_unix = start_time
        self._stop_time_unix = stop_time
        self._measurement_id = measurement_id
        self._comment = comment
        self._sensor_id = sensor_id
        self._file_name = file_name
        self._sampling_rate_hz = sampling_rate_hz

    @classmethod
    def from_hd5_file(
        cls, path: path_t, sampling_rate_hz: Optional[float] = 976.5625
    ) -> Self:
        """Create a new Dataset from a valid .hd5 file.

        Parameters
        ----------
        path : :class:`pathlib.Path` or str
            Path to the file
        sampling_rate_hz : float, optional
            Sampling rate of the radar sensor in Hz. Default: 1953.125 Hz

        """
        path = Path(path)
        _assert_file_extension(path, (".h5", ".hd5"))

        file: h5py.File = h5py.File(path, mode="r")
        if "Radar" not in file.keys():
            raise InvalidFileFormatError(
                f"Invalid file format! Expected HDF5 file with 'Radar' as key. Got {list(file.keys())}."
            )
        file = file["Radar"]
        start_time = file.attrs["start"]
        stop_time = file.attrs["stop"]
        measurement_id = file.attrs["measurement_id"]
        comment = file.attrs["comment"]
        sensor_id = file.attrs["sensor_id"]
        file_name = path.stem

        data = {
            key: pd.DataFrame(file[key], columns=pd.Index(["I", "Q", "Sync_In", "Sync_Out"], name="channel"))
            for key in file
        }

        return cls(data, start_time, stop_time, measurement_id, comment, sensor_id, file_name, sampling_rate_hz)

    def save_data_as_1_h_files(self):
        pass

    def save_data_as_h5(self, data, start_time):
        pass

