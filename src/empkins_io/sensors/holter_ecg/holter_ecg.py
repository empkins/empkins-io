from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from empkins_io.utils._types import path_t

from biopsykit.utils._datatype_validation_helper import _assert_file_extension


class HolterEcgLoader:
    """Class representing a measurement of CONTEC TLC9803 Holter ECG.

    Attributes
    ----------
    _ecg_data: np.ndarray
        ECG data recorded with the Holter ECG device
    _sampling_rate_hz: float
        Sampling rate of ecg_data

    """

    _ecg_data: np.array
    _sampling_rate_hz: float

    def __init__(
        self,
        ecg_data: np.array,
        sampling_rate_hz: float = 500,
    ):
        """Get new data loader instance.

        .. note::
            Usually you shouldn't use this init directly.
            Use the provided `from_ecg_file` constructor to handle loading recorded ECG Sessions.

        Parameters
        ----------
        ecg_data : np.ndarray
            ECG data recorded with the Holter ECG device
        sampling_rate_hz : float
            Sampling rate of ecg_data

        """
        self._ecg_data = ecg_data
        self._sampling_rate_hz = sampling_rate_hz

    @property
    def sampling_rate_hz(self) -> float:
        return self._sampling_rate_hz

    @property
    def ecg_data(self) -> np.ndarray:
        """Array of ECG values."""
        return self._ecg_data

    def data_as_df(
        self,
        index: Optional[str] = None,
    ) -> pd.DataFrame:
        """Export ecg data as pandas DataFrame.

        Parameters
        ----------
        index : str, optional
            Specify which index should be used for the data loader. The options are:
            * "time": For the time in seconds since the first sample
            * None: For a simple index (0...N)

        Raises
        ------
        ValueError
            If any other than the allowed `index` values are used.

        """

        ecg_data = pd.DataFrame(self.ecg_data)
        data = self._add_index(ecg_data, index)

        return data

    def _add_index(self, data: pd.DataFrame, index: str) -> pd.DataFrame:
        index_names = {
            None: "n_samples",
            "time": "t"
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

    @classmethod
    def from_ecg_file(
            cls,
            path: path_t,
            sampling_rate_hz: Optional[float] = 500
    ):
        """Create a new data loader from a valid .ecg file.

        Parameters
        ----------
        path : :class:`pathlib.Path` or str
            Path to the file

        sampling_rate_hz : float, optional
            Sampling rate of the Holter ECG device. Default: 500 Hz

        """

        path = Path(path)
        _assert_file_extension(path, (".ecg"))

        with open(path, "rb") as file:
            data = np.fromfile(file, dtype=np.int16)

        return cls(data, sampling_rate_hz)
