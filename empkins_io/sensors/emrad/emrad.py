from pathlib import Path
from typing import Dict, Optional, Sequence

import h5py
import pandas as pd
from biopsykit.utils._datatype_validation_helper import _assert_file_extension
from typing_extensions import Self

from empkins_io.utils._types import path_t
from empkins_io.utils.exceptions import InvalidFileFormatException


class EmradDataset:
    """Class representing a measurement of the Emrad sensor.

    Attributes
    ----------
    radar_data :
        Dictionary with data of all radar nodes
    timezone :
        Timezone of recording (if available)

    """

    _RADAR_NODES = ["rad1", "rad2", "rad3", "rad4"]
    _SAMPLING_RATE_HZ: float = 1952.0
    _start_time_unix: int

    _tz: str

    _radar_data: Dict[str, pd.DataFrame]

    def __init__(self, radar_data: Dict[str, pd.DataFrame], start_time: int, tz: Optional[str] = None):
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

    @property
    def timezone(self):
        return self._tz

    @property
    def radar_data(self) -> Dict[str, pd.DataFrame]:
        return self._radar_data

    def data_as_df(
        self,
        nodes: Optional[Sequence[str]] = None,
        drop_empty_nodes: Optional[bool] = True,
        index: Optional[str] = None,
        add_sync_in: Optional[bool] = False,
        add_sync_out: Optional[bool] = False,
    ) -> pd.DataFrame:
        """Export the nodes of the dataset in a single pandas DataFrame.

        Parameters
        ----------
        nodes : list of str, optional
            List of radar node names, if only specific ones should be included. Nodes that
            are not part of the current dataset will be silently ignored. Default: ``None``.
        index : str, optional
            Specify which index should be used for the dataset. The options are:
            "time": For the time in seconds since the first sample
            "utc": For the utc time stamp of each sample
            "utc_datetime": for a pandas DateTime index in UTC time
            "local_datetime": for a pandas DateTime index in the timezone set for the session
            None: For a simple index (0...N)
        drop_empty_nodes : bool, optional
            ``True`` to exclude empty radar nodes in the dataframe. Default: ``True``
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

        radar_data = {node: self.radar_data[node] for node in nodes}

        data = pd.concat(radar_data, axis=1, names=["node"])

        if drop_empty_nodes:
            data = data.dropna(axis=1, how="all")
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

        if index == "time":
            data.index -= data.index[0]
            data.index /= self._SAMPLING_RATE_HZ
            return data

        data.index /= self._SAMPLING_RATE_HZ
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
        cls,
        path: path_t,
        tz: Optional[str] = "Europe/Berlin",
    ) -> Self:
        """Create a new Dataset from a valid .hd5 file.

        Parameters
        ----------
        path :
            Path to the file
        tz
            Optional timezone str of the recording.
            This can be used to localize the start and end time.
            Note, this should not be the timezone of your current PC, but the timezone relevant for the specific
            recording.

        """
        path = Path(path)
        _assert_file_extension(path, (".h5", ".hd5"))

        file: h5py.File = h5py.File(path, mode="r")
        if "Radar" not in file.keys():
            raise InvalidFileFormatException(
                f"Invalid file format! Expected HDF5 file with 'Radar' as key. Got {list(file.keys())}."
            )
        file = file["Radar"]
        start_time = file.attrs["start"]

        data = {
            key: pd.DataFrame(file[key], columns=pd.Index(["I", "Q", "Sync_In", "Sync_Out"], name="channel"))
            for key in file.keys()
        }

        return cls(data, start_time, tz)