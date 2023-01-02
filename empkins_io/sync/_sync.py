import re
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, get_args

import numpy as np
import pandas as pd
from biopsykit.utils._datatype_validation_helper import _assert_is_dtype
from matplotlib import pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d

from empkins_io.utils.exceptions import SynchronizationError, ValidationError

SYNC_TYPE = Literal["peak", "rect", "square-wave", "m-sequence"]


class SyncedDataset:

    _VALID_INDEX_NAMES = (r"t", r"utc", r"date", r"date \(.*\)")

    datasets: Dict[str, Dict[str, Any]]
    datasets_synced_: Dict[str, Dict[str, Any]]
    sync_type: SYNC_TYPE

    def __init__(self, sync_type: SYNC_TYPE = "peak"):
        # TODO fix this
        if sync_type not in get_args(SYNC_TYPE):
            raise ValueError(f"Sync type {sync_type} not valid. Mus be one of {get_args(SYNC_TYPE)}.")
        self.sync_type = sync_type
        self.datasets = {}

    def add_dataset(self, name: str, data: pd.DataFrame, sync_channel_name: str, sampling_rate: int):
        # ensure that data is a pandas DataFrame
        _assert_is_dtype(data, pd.DataFrame)
        # ensure that data has a column with the name sync_channel
        if sync_channel_name not in data.columns:
            raise ValidationError(f"Sync channel {sync_channel_name} not found in data columns.")
        # ensure that sampling_rate is a valid sampling rate
        if not isinstance(sampling_rate, (int, float)) or sampling_rate <= 0:
            raise ValidationError(f"Sampling rate {sampling_rate} is not a valid sampling rate.")

        # check if index of the dataset has same type as the previously added datasets
        if len(self.datasets) > 0:
            # check valid index
            self._check_valid_index(data)

        setattr(self, name, data)
        self.datasets[name] = {}
        self.datasets[name]["data"] = getattr(self, name)
        self.datasets[name]["name"] = name
        self.datasets[name]["sync_channel"] = sync_channel_name
        self.datasets[name]["sampling_rate"] = sampling_rate

    def plot_sync_channels(self, **kwargs) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
        """Plot all sync channels in a single figure."""
        fig, axs = plt.subplots(nrows=len(self.datasets), sharex=True, squeeze=False, **kwargs)
        axs = axs.flatten()
        for name, ax in zip(self.datasets, axs):
            dataset = self.datasets[name]
            data = dataset["data"]
            sync_channel = dataset["sync_channel"]
            # set title
            ax.plot(data[sync_channel], label=name)
            ax.set_title(name)

        return fig, axs

    def cut_to_sync_start(self, sync_params: Optional[Dict[str, Any]] = None):
        """Cut all datasets to the region where all datasets are synced."""
        if sync_params is None:
            sync_params = {}
        for name in self.datasets:
            dataset = self.datasets[name]
            params = sync_params.get(name, {})
            data_cut = self._cut_dataset_to_sync_start(dataset, sync_params=params)
            setattr(self, f"{name}_cut_", data_cut)

    def _cut_dataset_to_sync_start(self, dataset: Dict[str, Any], sync_params: Dict[str, Any]) -> pd.DataFrame:
        sync_channel = dataset["sync_channel"]
        data = dataset["data"]

        # if self.sync_type == "m-sequence":
        #     # get the highest sampling rate of all datasets
        #     sampling_rate = max(dataset["sampling_rate"] for dataset in self.datasets.values())
        #     sync_params["sampling_rate"] = sampling_rate
        #     sync_params["sampling_rate_old"] = dataset["sampling_rate"]
        #     start = SyncedDataset._find_sync_cross_correlation(data[sync_channel], sync_params)
        #     data_cut = data.iloc[start:]
        # else:
        if self.sync_type == "peak":
            # sync_type is "peak"
            sync_data = data[sync_channel]
            sync_params["max_expected_peaks"] = 2
        elif self.sync_type == "rect":
            # sync_type is "rect"
            sync_data = np.abs(np.ediff1d(data[sync_channel]))
            sync_params["max_expected_peaks"] = 2
        elif self.sync_type == "square-wave":
            # sync_type is "square-wave"
            sync_data = np.abs(np.ediff1d(data[sync_channel]))
            # max_expected_peaks is two times the wave frequency per second, because we compute the derivative
            if sync_params.get("wave_frequency"):
                sync_params["max_expected_peaks"] = 2 + sync_params.get("wave_frequency") * (
                    len(data) / dataset["sampling_rate"]
                )
        elif self.sync_type == "m-sequence":
            # sync_type is "m-sequence"
            sync_data = np.abs(np.ediff1d(data[sync_channel]))
            sync_params["expected_peaks"] = len(sync_data)
        else:
            raise AttributeError("This should never happen.")

        peaks = SyncedDataset._find_sync_peaks(sync_data, sync_params)
        # cut data to region between first and last peak
        if len(peaks) == 1:
            data_cut = data.iloc[peaks[0] :]
        else:
            data_cut = data.iloc[peaks[0] : peaks[-1]]

        return data_cut

    def align_datasets(
        self, primary: str, cut_to_shortest: Optional[bool] = False, reset_time_axis: Optional[bool] = False
    ) -> None:
        """Align all datasets to the primary dataset.

        This function aligns the index of the secondary datasets to the primary dataset.

        Parameters
        ----------
        primary : str
            Name of the primary dataset.
        cut_to_shortest : bool, optional
            If True, cut all datasets to the length of the shortest dataset. Default: ``False``
        reset_time_axis : bool, optional
            If True, reset the time axis of all datasets to start at 0.
            note:: This is only applied if the time index is not a :class:``~pandas.DatetimeIndex``.
            Default: ``False``

        """
        if getattr(self, f"{primary}_cut_", None) is None:
            raise SynchronizationError(f"Datasets were not cut to sync start yet. Call 'cut_to_sync_start' first!")
        data_primary_cut = getattr(self, f"{primary}_cut_")
        start_time_primary = data_primary_cut.index[0]

        for name in self.datasets:
            if name == primary:
                continue
            data_cut = getattr(self, f"{name}_cut_")
            data_cut = data_cut.copy()
            start_time = data_cut.index[0]
            # align index
            data_cut.index -= start_time - start_time_primary
            setattr(self, f"{name}_aligned_", data_cut)

        setattr(self, f"{primary}_aligned_", data_primary_cut)

        if cut_to_shortest:
            # get the length of the shortest dataset
            shortest_length = min([getattr(self, f"{name}_aligned_").index[-1] for name in self.datasets])
            for name in self.datasets:
                data_aligned = getattr(self, f"{name}_aligned_")
                data_aligned = data_aligned.loc[:shortest_length]
                setattr(self, f"{name}_aligned_", data_aligned)

        if reset_time_axis:
            for name in self.datasets:
                data_aligned = getattr(self, f"{name}_aligned_")
                data_aligned.index -= data_aligned.index[0]
                setattr(self, f"{name}_aligned_", data_aligned)

    @property
    def datasets_cut(self):
        # get all datasets that were cut to sync start
        return {attr: getattr(self, attr) for attr in dir(self) if attr.endswith("cut_")}

    @property
    def datasets_aligned(self):
        # get all datasets that were cut to sync start
        return {attr: getattr(self, attr) for attr in dir(self) if attr.endswith("aligned_")}

    @staticmethod
    def _find_sync_peaks(data: np.ndarray, sync_params: Dict[str, Any]) -> np.ndarray:
        max_expected_peaks = sync_params.get("max_expected_peaks", None)
        search_region_samples = sync_params.get("search_region_samples", None)
        distance = sync_params.get("distance", None)
        height = sync_params.get("height", 0.1)
        width = sync_params.get("width", None)
        prominence = sync_params.get("prominence", None)

        # normalize data between 0 and 1
        data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        offset = 0
        if search_region_samples is not None:  # if search region is defined
            data_norm = data_norm[search_region_samples[0] : search_region_samples[1]]
            offset = search_region_samples[0]

        peaks, _ = signal.find_peaks(data_norm, distance=distance, height=height, width=width, prominence=prominence)
        # add offset to peaks
        peaks += offset
        if len(peaks) == 0:
            raise SynchronizationError("No peaks found in sync channel.")
        if max_expected_peaks is not None and len(peaks) > max_expected_peaks:
            raise SynchronizationError(
                f"Found more than '{max_expected_peaks}' sync peaks. Check your sync channels for correct input"
            )

        return peaks

    @staticmethod
    def _find_sync_cross_correlation(data: np.ndarray, sync_params: Dict[str, Any]) -> int:
        # sampling_rate_old = sync_params["sampling_rate_old"]
        # sampling_rate = sync_params["sampling_rate"]
        # interpolate data to sampling rate
        # x_old = np.arange(len(data)) / sampling_rate_old
        # x_new = np.arange(len(data)) / sampling_rate
        # interpol_f = interp1d(x_old, data, kind="nearest")
        # data = interpol_f(x_new)

        #     len_primary = len(data_prim)
        #     len_secondary = len(data_sec)
        #
        #     cross_corr = signal.correlate(data_prim[sync_ch_prim], data_sec[sync_ch_sec], mode="same")
        #     auto_corr_prim = signal.correlate(data_prim[sync_ch_prim], data_prim[sync_ch_prim], mode="same")
        #     auto_corr_sec = signal.correlate(data_sec[sync_ch_sec], data_sec[sync_ch_sec], mode="same")
        #
        #     cross_corr = cross_corr / np.sqrt(auto_corr_prim[int(len_primary / 2)] * auto_corr_sec[int(len_secondary / 2)])
        #
        #     delay_arr = np.linspace(-0.5 * len_primary / fs_prim, 0.5 * len_primary / fs_prim, len_primary)
        #     delay_sec = delay_arr[np.argmax(cross_corr)]
        #     delay_samples = int(np.around(delay_sec * fs_sec))

        raise NotImplementedError()

    def _check_valid_index(self, data: pd.DataFrame):
        index_type = list(set(type(dataset["data"].index) for dataset in self.datasets.values()))[0]
        new_index_type = type(data.index)
        if index_type != new_index_type:
            raise ValidationError(
                f"Indices of all added datasets must be of the same type. Got {[index_type, new_index_type]}."
            )
        # check that the index names of the datasets are the same
        index_name = list(set(dataset["data"].index.name for dataset in self.datasets.values()))[0]
        new_index_name = data.index.name
        if index_name != new_index_name:
            raise ValidationError(
                f"Index names of all added datasets must be the same. Got {[index_name, new_index_name]}."
            )

        for valid_index_name in self._VALID_INDEX_NAMES:
            # check if index name regex matches one of the valid index names
            if re.match(rf"^{valid_index_name}$", index_name):
                return

        raise ValidationError(
            f"Index name '{index_name}' is not valid. "
            f"Regular expressions for valid index names are {self._VALID_INDEX_NAMES}, where: \n"
            f"* 'n_samples': For a simple index (0...N)"
            f"* 't': For the time in seconds since the first sample\n"
            f"* 'utc': For the utc time stamp of each sample\n"
            f"* 'date': for a pandas DateTime index in UTC time\n"
            f"* 'date (<timezone>)': for a pandas DateTime index in the timezone set for the session\n"
        )
