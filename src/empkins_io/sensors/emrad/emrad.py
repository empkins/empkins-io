from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

import h5py
import pandas as pd
from biopsykit.utils._datatype_validation_helper import _assert_file_extension
from typing_extensions import Self

from empkins_io.utils._types import path_t
from empkins_io.utils.exceptions import InvalidFileFormatError


class EmradDataset:
    """Class representing a measurement of the Emrad sensor.

    Attributes
    ----------
    radar_data : dict
        Dictionary with data of all radar nodes
    timezone : str
        Timezone of recording (if available)
    sampling_rate_hz : int
        Sampling rate of the radar nodes in Hz

    """

    _RADAR_NODES: ClassVar[Sequence[str]] = ["rad1", "rad2", "rad3", "rad4"]
    _sampling_rate_hz: float
    _start_time_unix: int

    _tz: str

    _radar_data: dict[str, pd.DataFrame]

    def __init__(
        self,
        radar_data: dict[str, pd.DataFrame],
        start_time: int,
        tz: str | None = None,
        sampling_rate_hz: float = 1953.125,
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
        self._tz = tz
        self._sampling_rate_hz = sampling_rate_hz

    @property
    def sampling_rate_hz(self) -> float:
        """Sampling rate of the radar sensor in Hz."""
        return self._sampling_rate_hz

    @property
    def timezone(self):
        """Timezone of the recording."""
        return self._tz

    @property
    def radar_data(self) -> dict[str, pd.DataFrame]:
        """Dictionary with names of the radar node and sensor data as pandas Dataframe."""
        return self._radar_data

    def data_as_df(
        self,
        nodes: Sequence[str] | None = None,
        index: str | None = None,
        add_sync_in: bool | None = False,
        add_sync_out: bool | None = False,
    ) -> pd.DataFrame:
        """Export the nodes of the dataset in a single pandas DataFrame.

        Parameters
        ----------
        nodes : list of str, optional
            List of radar node names, if only specific ones should be included. Nodes that
            are not part of the current dataset will be silently ignored. Default: ``None``.
        index : str, optional
            Specify which index should be used for the dataset. The options are:
            * "time": For the time in seconds since the first sample
            * "utc": For the utc time stamp of each sample
            * "utc_datetime": for a pandas DateTime index in UTC time
            * "local_datetime": for a pandas DateTime index in the timezone set for the session
            * None: For a simple index (0...N)
        add_sync_in : bool, optional
            ``True`` to include the "Sync_In" channel for each radar node. Default: ``False``
        add_sync_out : bool, optional
            ``True`` to include the "Sync_Out" channel for each radar node. Default: ``False``

        Raises
        ------
        ValueError
            If any other than the allowed `index` values are used.

        """
        if nodes is None:
            nodes = self._RADAR_NODES
        if isinstance(nodes, str):
            nodes = [nodes]

        radar_data = {node: self.radar_data[node] for node in nodes if not self.radar_data[node].empty}
        data = pd.concat(radar_data, axis=1, names=["node"])

        if not add_sync_in:
            data = data.drop(columns="Sync_In", level="channel")
        if not add_sync_out:
            data = data.drop(columns="Sync_Out", level="channel")

        data = self._add_index(data, index)

        return data

    def _add_index(self, data: pd.DataFrame, index: str) -> pd.DataFrame:
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
            data.index /= self.sampling_rate_hz
            return data

        # convert to utc timestamps => for index_type "utc"
        data.index /= self.sampling_rate_hz
        data.index += self._start_time_unix

        if index == "utc_datetime":
            data.index = pd.to_datetime(data.index, unit="s")
            data.index = data.index.tz_localize("UTC")
        if index == "local_datetime":
            data.index = pd.to_datetime(data.index, unit="s")
            data.index = data.index.tz_localize("UTC").tz_convert(self.timezone)

        return data

    @classmethod
    def from_hd5_file(
        cls, path: path_t, tz: str | None = "Europe/Berlin", sampling_rate_hz: float | None = 1953.125
    ) -> Self:
        """Create a new Dataset from a valid .hd5 file.

        Parameters
        ----------
        path : :class:`pathlib.Path` or str
            Path to the file
        tz : str, optional
            Timezone str of the recording. This can be used to localize the start and end time.
            Note, this should not be the timezone of your current PC, but the timezone relevant for the specific
            recording.
        sampling_rate_hz : float, optional
            Sampling rate of the radar sensor in Hz. Default: 1953.125 Hz

        """
        path = Path(path)
        _assert_file_extension(path, (".h5", ".hd5", ".hdf5"))

        file: h5py.File = h5py.File(path, mode="r")
        if "Radar" not in file:
            raise InvalidFileFormatError(
                f"Invalid file format! Expected HDF5 file with 'Radar' as key. Got {list(file.keys())}."
            )
        file = file["Radar"]
        start_time = file.attrs["start"]

        data = {
            key: pd.DataFrame(file[key], columns=pd.Index(["I", "Q", "Sync_In", "Sync_Out"], name="channel"))
            for key in file
        }

        return cls(data, start_time, tz, sampling_rate_hz)
