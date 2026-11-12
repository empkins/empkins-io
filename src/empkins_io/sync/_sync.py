"""Synchronization of multiple datasets.

This module provides three classes:

* :class:`SyncBase`: base class holding everything that is shared between all synchronization types
  (dataset handling, resampling, plotting, index validation, and generic signal helpers).
* :class:`SyncedDataset`: synchronization based on trigger/edge/clock sync signals
  (i.e., all sync types *except* ``"m-sequence"``).
* :class:`SyncedDatasetMSequence`: synchronization based on an m-sequence sync signal.

"""

import re
import warnings
from collections.abc import Sequence
from typing import Any, Literal, get_args

import numpy as np
import pandas as pd
import resampy
from biopsykit.utils._datatype_validation_helper import _assert_is_dtype
from matplotlib import pyplot as plt
from scipy import signal
from scipy.signal import find_peaks, periodogram

from empkins_io.utils.exceptions import SynchronizationError, ValidationError

SYNC_TYPE = Literal[
    "peak",
    "rect",
    "square-wave",
    "falling-trigger",
    "rising-trigger",
    "falling-edge",
    "rising-edge",
    "falling-clock",
    "rising-clock",
    "m-sequence",
]
SYNC_TYPE_DEPRECATED = ["peak", "rect", "square-wave"]
SYNC_TYPE_ESB = [
    "falling-trigger",
    "rising-trigger",
    "falling-edge",
    "rising-edge",
    "falling-clock",
    "rising-clock",
    "m-sequence",
]
SYNC_TYPE_M_SEQUENCE = "m-sequence"

_RESULT_SUFFIXES = ("_aligned_", "_resampled_", "_synced_", "_cut_")


class SyncBase:
    """Base class for synchronizing multiple datasets.

    This class implements everything that does not depend on the used sync signal type:
    adding datasets, resampling them, plotting the sync channels, accessing the intermediate results,
    and a set of generic signal-processing helpers.

    Use one of the subclasses :class:`SyncedDataset` or :class:`SyncedDatasetMSequence`.

    """

    _VALID_INDEX_NAMES = (r"t", r"utc", r"date", r"date \(.*\)")

    datasets: dict[str, dict[str, Any]]
    datasets_synced_: dict[str, pd.DataFrame]
    sync_type: SYNC_TYPE

    def __init__(self, sync_type: SYNC_TYPE = "rising-trigger"):
        if sync_type not in get_args(SYNC_TYPE):
            raise ValueError(f"Sync type {sync_type} not valid. Must be one of {get_args(SYNC_TYPE)}.")
        if sync_type in SYNC_TYPE_DEPRECATED:
            warnings.warn(
                f"Sync type {sync_type} is deprecated. Please use one of {get_args(SYNC_TYPE)}.", DeprecationWarning
            )
        self.sync_type = sync_type
        self.datasets = {}
        self.datasets_synced_ = {}

    # --------------------------------------------------------------------------------------------------
    # dataset handling
    # --------------------------------------------------------------------------------------------------

    def add_dataset(self, name: str, data: pd.DataFrame, sync_channel_name: str, sampling_rate: int):
        """Add a dataset to be synchronized.

        Parameters
        ----------
        name : str
            Name of the dataset.
        data : :class:`~pandas.DataFrame`
            Data of the dataset.
        sync_channel_name : str
            Name of the column containing the sync signal.
        sampling_rate : int
            Sampling rate of the dataset in Hz.

        """
        # ensure that data is a pandas DataFrame
        _assert_is_dtype(data, pd.DataFrame)
        # ensure that data has a column with the name sync_channel
        if sync_channel_name not in data.columns:
            raise ValidationError(f"Sync channel {sync_channel_name} not found in data columns.")
        # ensure that sampling_rate is a valid sampling rate
        if not isinstance(sampling_rate, int | float) or sampling_rate <= 0:
            raise ValidationError(f"Sampling rate {sampling_rate} is not a valid sampling rate.")

        # ensure that the sync channel does not contain any missing values
        if data[sync_channel_name].isna().any():
            raise ValidationError(
                f"Sync channel {sync_channel_name} contains missing values. "
                f"Please check the sync channel of dataset '{name}' for gaps or dropouts."
            )

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

    def plot_sync_channels(self, **kwargs) -> tuple[plt.Figure, Sequence[plt.Axes]]:
        """Plot all sync channels in a single figure."""
        fig, axs = plt.subplots(nrows=len(self.datasets), sharex=True, squeeze=False, **kwargs)
        axs = axs.flatten()
        for name, ax in zip(self.datasets, axs, strict=False):
            dataset = self.datasets[name]
            data = dataset["data"]
            sync_channel = dataset["sync_channel"]
            # set title
            ax.plot(data[sync_channel], label=name)
            ax.set_title(name)

        return fig, axs

    def resample_datasets(self, fs_out: float, method: Literal["static", "dynamic"], **kwargs) -> None:
        """Resample all datasets to a common sampling rate.

        Parameters
        ----------
        fs_out : float
            Target sampling rate in Hz.
        method : {"static", "dynamic"}
            "static" uses the sampling rate provided when adding the dataset, "dynamic" estimates the actual
            sampling rate from the sync signal (see :meth:`_determine_actual_sampling_rate`).

        """
        # results of this and all following stages are invalidated by resampling
        self._clear_results("_resampled_", "_cut_", "_aligned_", "_synced_")

        for name in self.datasets:
            dataset = self.datasets[name]
            if method == "static":
                fs_in = dataset["sampling_rate"]
            elif method == "dynamic":
                fs_in = self._determine_actual_sampling_rate(dataset, **kwargs)

            else:
                # this should never happen
                raise ValueError(f"Method '{method}' not supported.")

            data_resample = resampy.resample(
                dataset["data"].values, sr_orig=fs_in, sr_new=fs_out, axis=0, parallel=True
            )

            index_resample = pd.date_range(
                start=dataset["data"].index[0],
                periods=len(data_resample),
                freq=f"{1/fs_out*1000}ms",
                tz=dataset["data"].index.tz,
            )
            index_resample.name = dataset["data"].index.name
            data_resample = pd.DataFrame(
                data_resample,
                columns=dataset["data"].columns,
                index=index_resample,
            )
            dataset["sampling_rate_resampled"] = fs_out
            dataset["data_resampled"] = data_resample
            setattr(self, f"{name}_resampled_", data_resample)

    # --------------------------------------------------------------------------------------------------
    # properties
    # --------------------------------------------------------------------------------------------------

    @property
    def datasets_resampled(self):
        # get all datasets that were resampled
        return {attr: getattr(self, attr) for attr in dir(self) if attr.endswith("resampled_")}

    @property
    def datasets_cut(self):
        # get all datasets that were cut to sync region
        return {attr: getattr(self, attr) for attr in dir(self) if attr.endswith("cut_")}

    @property
    def datasets_aligned(self):
        # get all datasets that were aligned
        return {attr: getattr(self, attr) for attr in dir(self) if attr.endswith("aligned_")}

    @property
    def datasets_synced(self):
        """Return the fully synchronized datasets, i.e., the final result of the sync pipeline.

        Like ``datasets_resampled`` / ``datasets_aligned`` / ``datasets_cut``, the dictionary is keyed by the
        attribute names of the results (``"radar_synced_"``). In contrast to those properties, it returns the
        explicitly filled ``datasets_synced_`` dictionary instead of scanning the instance attributes.

        """
        return self.datasets_synced_

    # --------------------------------------------------------------------------------------------------
    # name / validation helper
    # --------------------------------------------------------------------------------------------------

    @staticmethod
    def _strip_result_suffix(name: str) -> str:
        """Return the plain dataset name for an attribute name of an intermediate result.

        ``"radar_aligned_"`` becomes ``"radar"``; a name without such a suffix is returned unchanged.

        """
        for suffix in _RESULT_SUFFIXES:
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name

    def _get_sync_channel(self, name: str) -> str:
        """Return the sync channel name of a dataset.

        Accepts both a plain dataset name (``"radar"``) and the attribute names of intermediate results
        (``"radar_aligned_"``, ``"radar_resampled_"``, ``"radar_synced_"``, ``"radar_cut_"``), as used by the
        ``datasets_aligned`` / ``datasets_resampled`` / ``datasets_synced`` / ``datasets_cut`` properties.

        """
        base_name = self._strip_result_suffix(name)
        if base_name in self.datasets:
            return self.datasets[base_name]["sync_channel"]

        raise ValidationError(f"No dataset found for '{name}'. Cannot determine the sync channel.")

    def _check_valid_index(self, data: pd.DataFrame):
        index_type = next(iter({type(dataset["data"].index) for dataset in self.datasets.values()}))
        new_index_type = type(data.index)
        if index_type != new_index_type:
            raise ValidationError(
                f"Indices of all added datasets must be of the same type. Got {[index_type, new_index_type]}."
            )
        # check that the index names of the datasets are the same
        index_name = next(iter({dataset["data"].index.name for dataset in self.datasets.values()}))
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

    def _get_common_sampling_rate(self) -> float:
        """Return the sampling rate shared by all datasets.

        Falls back to the resampled sampling rates if the original sampling rates differ.

        """
        sampling_rates = {dataset["sampling_rate"] for dataset in self.datasets.values()}
        if len(sampling_rates) != 1:
            # check if there are resampled datasets
            if all("sampling_rate_resampled" in dataset for dataset in self.datasets.values()):
                sampling_rates = {dataset["sampling_rate_resampled"] for dataset in self.datasets.values()}
            else:
                raise ValueError(
                    "Sampling rates of datasets are not equal. Please resample all datasets to a "
                    "common sampling rate using `SyncedDataset.resample_datasets()`."
                )
        return next(iter(sampling_rates))

    def _determine_actual_sampling_rate(self, dataset: dict[str, Any], **kwargs) -> float:
        wave_frequency = kwargs.get("wave_frequency")
        data = dataset["data"]
        sync_channel = dataset["sync_channel"]
        fs = dataset["sampling_rate"]
        sync_abs = np.abs(np.ediff1d(data[sync_channel]))
        fft_sync, psd_sync = periodogram(sync_abs, fs=fs, window="hamming")
        psd_sync = self._normalize_signal(psd_sync)

        idx_peak = find_peaks(psd_sync, height=0.5)[0][0]
        freq_sync = fft_sync[idx_peak]

        fs_measured = (wave_frequency / freq_sync) * fs
        print(f"Measured sampling rate: {fs_measured}")
        print(f"Sync frequency: {freq_sync}")

        return fs_measured

    # --------------------------------------------------------------------------------------------------
    # generic signal helper
    # --------------------------------------------------------------------------------------------------

    def _find_sync_cross_correlation(
        self,
        primary: np.ndarray | pd.DataFrame,
        secondary: np.ndarray | pd.DataFrame,
        fs: float | None = None,  # noqa: ARG002
        plot_crosscorr: bool = False,
    ) -> int:
        """Return the lag between two binarized sync signals in samples.

        A positive lag means that the secondary signal is ahead of the primary signal, i.e., its sync sequence
        starts earlier. The lag is computed in samples and does not depend on the sampling rate.

        """
        # remove the DC offset of the binarized signals, otherwise it dominates the cross-correlation
        primary = primary - 0.5
        secondary = secondary - 0.5

        crosscorr = signal.correlate(primary, secondary, mode="full")

        if plot_crosscorr:
            _fig, axs = plt.subplots()
            axs.plot(crosscorr)

        # for "full" correlation, the zero-lag position is at index len(secondary) - 1. Deriving the lag from
        # this position keeps the result correct even if both signals have different lengths.
        lag_samples = int(np.argmax(crosscorr) - (len(secondary) - 1))
        return lag_samples

    @classmethod
    def _normalize_signal(cls, data: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    @classmethod
    def _binarize_signal(cls, data: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Binarize a signal to 0 and 1, using its mean as threshold.

        Resampling interpolates the sync signal, so its values are not binary anymore. Samples equal to the
        mean are mapped to 0, i.e., the result only contains the values 0 and 1.

        """
        return (data > np.mean(data)).astype(float)

    @classmethod
    def _pad_signal(cls, data: pd.DataFrame, padlen: int, start: bool, fs: float) -> pd.DataFrame:
        # pad with 0.5, i.e., the midpoint of the binarized sync signal
        if start:
            pad_width = ((padlen, 0), (0, 0))
            constant_values = ((0.5, None), (None, None))
        else:
            pad_width = ((0, padlen), (0, 0))
            constant_values = ((None, 0.5), (None, None))

        data_pad = np.pad(data, pad_width=pad_width, mode="constant", constant_values=constant_values)
        data_pad = pd.DataFrame(data_pad, columns=data.columns)

        if isinstance(data.index, pd.DatetimeIndex):
            data_pad.index /= fs
            if start:
                data_pad -= data_pad.index[-1]
            data_pad.index = pd.to_timedelta(data_pad.index, unit="s")
            if start:
                data_pad.index += data.index[-1]
            else:
                data_pad.index += data.index[0]
            data_pad.index.name = data.index.name
        return data_pad

    def _clear_results(self, *suffixes: str) -> None:
        """Delete the results of the given pipeline stages.

        This is called whenever a stage is (re-)run, so that results of the *following* stages cannot survive
        as stale data from an earlier run.

        """
        for suffix in suffixes:
            for attr in list(self.__dict__):
                # never delete the "datasets_*" dictionaries themselves
                if attr.startswith("datasets"):
                    continue
                if attr.endswith(suffix):
                    delattr(self, attr)
            if suffix == "_synced_":
                self.datasets_synced_.clear()

    @staticmethod
    def _cut_to_search_region(data: pd.Series, sync_region_samples: tuple[int, int] | None) -> pd.Series:
        """Cut a sync channel to the region that is used to search for the sync sequence."""
        if sync_region_samples is None:
            return data
        return data.iloc[sync_region_samples[0] : sync_region_samples[1]]

    def _reset_time_axis(self, primary: str, suffix: str) -> None:
        """Set the start of the time axis of all secondary datasets to the start of the primary dataset."""
        data_primary = getattr(self, f"{primary}{suffix}")
        for name in self.datasets:
            if name == primary:
                continue
            data = getattr(self, f"{name}{suffix}")
            data = data.copy()
            data.index = data.index - data.index[0] + data_primary.index[0]
            setattr(self, f"{name}{suffix}", data)

    def _cut_to_shortest(self, suffix: str) -> None:
        """Cut all datasets to the length of the shortest dataset."""
        shortest_length = min(len(getattr(self, f"{name}{suffix}")) for name in self.datasets)
        for name in self.datasets:
            data = getattr(self, f"{name}{suffix}")
            setattr(self, f"{name}{suffix}", data.iloc[:shortest_length])

    @staticmethod
    def _index_for_length(index: pd.Index, length: int) -> pd.Index:
        """Return an index of ``length`` samples, based on ``index``.

        The index is truncated if it is longer than ``length`` and extended with the same sampling period if it
        is shorter. Extending requires a regularly sampled index with at least two samples.

        """
        if length <= len(index):
            return index[:length]

        if len(index) < 2:
            raise SynchronizationError(
                "Cannot extend an index with less than two samples. Please provide a longer primary dataset."
            )

        step = index[1] - index[0]
        if isinstance(index, pd.DatetimeIndex):
            index_out = pd.date_range(start=index[0], periods=length, freq=step, tz=index.tz)
        else:
            index_out = pd.Index(index[0] + step * np.arange(length))
        index_out.name = index.name
        return index_out

    @staticmethod
    def _reset_and_shift(data: pd.DataFrame, shift_idx: int) -> pd.DataFrame:
        data = data.reset_index()
        data = data.iloc[shift_idx:].reset_index(drop=True)
        return data.set_index(data.columns[0])


class SyncedDataset(SyncBase):
    """Synchronize datasets using trigger-, edge-, or clock-based sync signals.

    This class covers all sync types except ``"m-sequence"``. For m-sequence sync signals use
    :class:`SyncedDatasetMSequence`.

    """

    def __init__(self, sync_type: SYNC_TYPE = "rising-trigger"):
        if sync_type == SYNC_TYPE_M_SEQUENCE:
            raise ValueError(
                f"Sync type '{SYNC_TYPE_M_SEQUENCE}' is not supported by 'SyncedDataset'. "
                f"Please use 'SyncedDatasetMSequence' instead."
            )
        super().__init__(sync_type=sync_type)

    # --------------------------------------------------------------------------------------------------
    # cutting to the sync region
    # --------------------------------------------------------------------------------------------------

    def cut_to_sync_start(self, sync_params: dict[str, Any] | None = None):
        warnings.warn(
            "cut_to_sync_start is deprecated and will be removed in the future. Use cut_to_sync_region instead.",
            DeprecationWarning,
        )
        return self.cut_to_sync_region(sync_params=sync_params)

    def cut_to_sync_region(self, sync_params: dict[str, Any] | None = None):
        """Cut all datasets to the region where all datasets are synced."""
        if sync_params is None:
            sync_params = {}
        for name in self.datasets:
            dataset = self.datasets[name]
            params = sync_params.get(name, {})
            data_cut = self._cut_dataset_to_sync_region(dataset, sync_params=params)
            setattr(self, f"{name}_cut_", data_cut)

    def _cut_dataset_to_sync_region(self, dataset: dict[str, Any], sync_params: dict[str, Any]) -> pd.DataFrame:
        data = dataset["data"]
        sync_channel = dataset["sync_channel"]
        # deprecated sync types
        if "peak" in self.sync_type:
            warnings.warn(
                "Sync type 'trigger' was renamed to 'rising-trigger'. Please update your code.", DeprecationWarning
            )
            self.sync_type = "rising-trigger"
        elif "rect" in self.sync_type:
            warnings.warn("Sync type 'rect' was renamed to 'rising-edge'. Please update your code.", DeprecationWarning)
            self.sync_type = "rising-edge"
        elif "square-wave" in self.sync_type:
            warnings.warn(
                "Sync type 'square-wave' was renamed to 'rising-clock'. Please update your code.", DeprecationWarning
            )
            self.sync_type = "rising-clock"
        # extract sync channel according to sync type
        if "trigger" in self.sync_type:
            # sync_type is "trigger"
            sync_data = data[sync_channel]
            sync_params["max_expected_peaks"] = 2
            if "falling" in self.sync_type:
                # invert sync channel to achieve rising sync signal
                sync_data = -1 * sync_data
        elif "edge" in self.sync_type:
            # sync_type is "edge"
            sync_data = np.abs(np.ediff1d(data[sync_channel]))
            sync_params["max_expected_peaks"] = 2
        elif "clock" in self.sync_type:
            # sync_type is "clock"
            sync_data = np.abs(np.ediff1d(data[sync_channel]))
            # max_expected_peaks is two times the wave frequency per second, because we compute the derivative
            if sync_params.get("wave_frequency"):
                sync_params["max_expected_peaks"] = 2 + sync_params.get("wave_frequency") * (
                    len(data) / dataset["sampling_rate"]
                )
        else:
            raise AttributeError("This should never happen.")

        peaks = self._find_sync_peaks(sync_data, sync_params)
        # cut data to region between first and last peak
        data_cut = data.iloc[peaks[0] :] if len(peaks) == 1 else data.iloc[peaks[0] : peaks[-1]]

        return data_cut

    @staticmethod
    def _find_sync_peaks(data: np.ndarray, sync_params: dict[str, Any]) -> np.ndarray:
        max_expected_peaks = sync_params.get("max_expected_peaks")
        search_region_samples = sync_params.get("search_region_samples")
        distance = sync_params.get("distance")
        height = sync_params.get("height", 0.1)
        width = sync_params.get("width")
        prominence = sync_params.get("prominence")

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

    # --------------------------------------------------------------------------------------------------
    # aligning
    # --------------------------------------------------------------------------------------------------

    def align_datasets(
        self, primary: str, cut_to_shortest: bool | None = False, reset_time_axis: bool | None = False
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
            raise SynchronizationError("Datasets were not cut to sync region yet. Call 'cut_to_sync_region' first!")
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


class SyncedDatasetMSequence(SyncBase):
    """Synchronize datasets using an m-sequence sync signal.

    The synchronization consists of three steps, which are applied in this order:

    1. :meth:`~SyncBase.resample_datasets`: resample all datasets to a common sampling rate
       (results: ``{name}_resampled_``, :attr:`~SyncBase.datasets_resampled`).
    2. :meth:`align_and_cut_start_m_sequence`: align the *start* of all datasets to the primary dataset
       (results: ``{name}_aligned_``, :attr:`~SyncBase.datasets_aligned`).
    3. :meth:`align_and_cut_end_m_sequence`: align the *end* of all datasets, i.e., compensate the clock drift
       (results: ``{name}_synced_``, :attr:`~SyncBase.datasets_synced`).

    """

    def __init__(self, sync_type: SYNC_TYPE = SYNC_TYPE_M_SEQUENCE):
        if sync_type != SYNC_TYPE_M_SEQUENCE:
            raise ValueError(
                f"'SyncedDatasetMSequence' only supports the sync type '{SYNC_TYPE_M_SEQUENCE}'. Got '{sync_type}'."
            )
        super().__init__(sync_type=sync_type)

    # --------------------------------------------------------------------------------------------------
    # step 2: aligning the start
    # --------------------------------------------------------------------------------------------------

    def align_and_cut_start_m_sequence(
        self,
        primary: str,
        cut_to_shortest: bool | None = False,
        reset_time_axis: bool | None = False,
        sync_params: dict[str, Any] | None = None,
    ):
        """Align the start of all datasets to the primary dataset using the m-sequence sync channel.

        The sync channel of every dataset is binarized first and checked to only contain the values 0 and 1.
        Each dataset is then cut to the first sample at which its sync signal is 1, which coarsely aligns the
        datasets. The remaining lag between the primary and each secondary dataset is determined by
        cross-correlating the sync channels within the search region. Finally, all datasets are cut at the
        beginning so that their first sample corresponds to the same point in time. Both positive and negative
        lags are handled: a dataset that starts earlier than the primary dataset is cut, and if any dataset
        starts later, the primary dataset (and all remaining datasets) are cut accordingly.

        Parameters
        ----------
        primary : str
            Name of the primary dataset.
        cut_to_shortest : bool, optional
            If True, cut all datasets to the length of the shortest dataset. Default: ``False``
        reset_time_axis : bool, optional
            If True, set the start of all secondary time axes to the start of the primary dataset.
            Default: ``False``
        sync_params : dict, optional
            ``sync_region_samples``: tuple ``(start, stop)`` restricting the region (in samples, relative to
            the start of the datasets) that is used to search for the sync sequence. Default: the whole signal.

        """
        if sync_params is None:
            sync_params = {}

        # assert that sampling rates are equal for all datasets
        sync_params["sampling_rate"] = self._get_common_sampling_rate()
        fs = sync_params["sampling_rate"]
        sync_region_samples = sync_params.get("sync_region_samples")

        # results of this and the following stage are invalidated by a new start alignment
        self._clear_results("_aligned_", "_synced_")

        # binarize the sync channel of every dataset, assert that it only contains 0 and 1, and cut every
        # dataset to the first sample at which the sync signal is 1
        dict_data = {}
        for name, dataset in self.datasets.items():
            data = self._get_sync_data(dataset)
            sync_channel = dataset["sync_channel"]
            self._assert_binary_sync_channel(data[sync_channel], name)
            dict_data[name] = self._cut_to_first_sync_value(data, sync_channel, name)

        data_primary = dict_data.pop(primary)
        sync_channel_primary = self.datasets[primary]["sync_channel"]

        # determine the lag of each secondary dataset relative to the primary dataset
        dict_lags = {}

        for name, data_secondary in dict_data.items():
            sync_channel_secondary = self.datasets[name]["sync_channel"]

            lag_samples = self._find_sync_cross_correlation(
                self._cut_to_search_region(data_primary[sync_channel_primary], sync_region_samples),
                self._cut_to_search_region(data_secondary[sync_channel_secondary], sync_region_samples),
                fs,
            )
            print("Start Shift: " + name + " " + str(lag_samples))

            dict_lags[name] = lag_samples

        # a positive lag means that the dataset starts earlier than the primary dataset. To align all datasets
        # without padding, the primary dataset is cut by the largest positive lag and every secondary dataset
        # by the difference between that lag and its own lag.
        max_positive_lag = max([0, *list(dict_lags.values())])

        data_primary = data_primary.iloc[max_positive_lag:]
        setattr(self, f"{primary}_aligned_", data_primary)

        for name, data in dict_data.items():
            data_aligned = data.iloc[max_positive_lag - dict_lags[name] :]
            setattr(self, f"{name}_aligned_", data_aligned)

        if reset_time_axis:
            self._reset_time_axis(primary, suffix="_aligned_")

        if cut_to_shortest:
            self._cut_to_shortest(suffix="_aligned_")

    @staticmethod
    def _assert_binary_sync_channel(sync_data: pd.Series, name: str) -> None:
        """Assert that a sync channel only contains the values 0 and 1, and no missing values."""
        values = np.unique(sync_data.to_numpy())
        if np.any(np.isnan(values)):
            raise SynchronizationError(
                f"Sync channel of dataset '{name}' contains missing values. "
                f"Please check the sync channel for gaps or dropouts."
            )
        if not np.all(np.isin(values, [0.0, 1.0])):
            raise SynchronizationError(
                f"Sync channel of dataset '{name}' is not binary after binarization. "
                f"Found the values {values[:10]}."
            )

    @staticmethod
    def _cut_to_first_sync_value(data: pd.DataFrame, sync_channel: str, name: str) -> pd.DataFrame:
        """Cut a dataset to the first sample at which the sync signal is 1."""
        idx_first = np.flatnonzero(data[sync_channel].to_numpy() == 1.0)
        if len(idx_first) == 0:
            raise SynchronizationError(
                f"Sync channel of dataset '{name}' does not contain any sample with the value 1. "
                f"Check the sync channel for correct input."
            )
        return data.iloc[idx_first[0] :]

    def _get_sync_data(self, dataset: dict[str, Any]) -> pd.DataFrame:
        """Return a copy of the (resampled) data of a dataset with a binarized sync channel."""
        data = dataset.get("data_resampled", dataset.get("data"))
        # copy the data, otherwise the binarized sync channel would overwrite the stored dataset
        data = data.copy()
        sync_channel = dataset["sync_channel"]
        data.loc[:, sync_channel] = self._binarize_signal(data[sync_channel])
        return data

    # --------------------------------------------------------------------------------------------------
    # step 3: aligning the end (clock drift compensation)
    # --------------------------------------------------------------------------------------------------

    def align_and_cut_end_m_sequence(
        self,
        primary: str,
        cut_to_shortest: bool | None = True,
        sync_params: dict[str, Any] | None = None,
    ) -> dict[str, int]:
        """Align the end of all datasets to the primary dataset, i.e., compensate the clock drift.

        The remaining sample shift at the end of the recording is determined on the datasets aligned by
        :meth:`align_and_cut_start_m_sequence`. Each secondary dataset is then resampled sample-wise by that
        shift, so that its start *and* end match the primary dataset.

        Parameters
        ----------
        primary : str
            Name of the primary dataset.
        cut_to_shortest : bool, optional
            If True, cut all datasets to the length of the shortest dataset. Default: ``True``
        sync_params : dict, optional
            ``sync_region_samples``: tuple ``(start, stop)`` restricting the region (in samples, relative to
            the start of the aligned datasets) that is used to search for the sync sequence. For the end
            alignment this region should cover the *end* of the recording. Default: the whole signal.

        Returns
        -------
        dict
            Sample shift per dataset, keyed by the attribute names of :attr:`~SyncBase.datasets_aligned`.

        Notes
        -----
        The search region should be placed at the very end of the recording and should be short compared to the
        recording length. The cross-correlation returns one lag for the whole region, which corresponds to the
        drift accumulated up to the *centre* of the region. Using that lag as a correction for the whole
        recording therefore underestimates the drift by ``drift * (region_length / 2) / recording_length``. For
        a region of a few seconds and a recording of several minutes, this bias stays below one sample. A region
        covering a large part of the recording additionally smears the cross-correlation peak, because the drift
        within the region becomes comparable to the length of one m-sequence chip.

        """
        dict_sample_shift = self._find_shift(primary, sync_params=sync_params)
        self.resample_sample_wise(primary, dict_sample_shift, cut_to_shortest=cut_to_shortest)
        return dict_sample_shift

    def _find_shift(self, primary: str, sync_params: dict[str, Any] | None = None) -> dict[str, int]:
        """Determine the remaining sample shift between the aligned datasets.

        Parameters
        ----------
        primary : str
            Name of the primary dataset.
        sync_params : dict, optional
            Additional parameters, e.g., ``sync_region_samples``.

        Returns
        -------
        dict
            Sample shift per dataset, keyed by the attribute names of :attr:`~SyncBase.datasets_aligned`.

        """
        if sync_params is None:
            sync_params = {}

        # assert that sampling rates are equal for all datasets
        sync_params["sampling_rate"] = self._get_common_sampling_rate()
        fs = sync_params["sampling_rate"]
        sync_region_samples = sync_params.get("sync_region_samples")

        dict_lags = {}

        primary_key = f"{primary}_aligned_"
        datasets_aligned = self.datasets_aligned
        if primary_key not in datasets_aligned:
            raise SynchronizationError(
                f"No aligned dataset found for the primary dataset '{primary}'. "
                f"Call 'align_and_cut_start_m_sequence' first!"
            )

        sync_channel_primary = self._get_sync_channel(primary_key)
        data_primary = datasets_aligned[primary_key].copy()
        data_primary.loc[:, sync_channel_primary] = self._binarize_signal(data_primary[sync_channel_primary])

        for name, dataset in datasets_aligned.items():
            if name == primary_key:
                continue

            # copy the data, otherwise the binarized sync channel would overwrite the aligned dataset
            data_secondary = dataset.copy()
            sync_channel_secondary = self._get_sync_channel(name)
            data_secondary.loc[:, sync_channel_secondary] = self._binarize_signal(
                data_secondary[sync_channel_secondary]
            )

            lag_samples = self._find_sync_cross_correlation(
                self._cut_to_search_region(data_primary[sync_channel_primary], sync_region_samples),
                self._cut_to_search_region(data_secondary[sync_channel_secondary], sync_region_samples),
                fs,
            )
            dict_lags[name] = lag_samples
            print("End Shift: " + name + " " + str(lag_samples))

        return dict_lags

    def resample_sample_wise(self, primary, dict_sample_shift, cut_to_shortest=True):
        """Resample the aligned datasets sample-wise to compensate the remaining sample shift.

        Parameters
        ----------
        primary : str
            Name of the primary dataset.
        dict_sample_shift : dict
            Sample shift per dataset, as returned by :meth:`_find_shift`.
        cut_to_shortest : bool, optional
            If True, cut all datasets to the length of the shortest dataset. If False, all datasets keep their
            own length and their time axis is extended accordingly. Default: ``True``

        Notes
        -----
        The results are stored in ``datasets_synced_`` (see the :attr:`~SyncBase.datasets_synced` property) and,
        for consistency with the other stages, additionally as ``{name}_synced_`` attributes. They represent the
        final result of the synchronization pipeline:
        ``resample_datasets`` (``_resampled_``) -> ``align_and_cut_start_m_sequence`` (``_aligned_``) ->
        ``align_and_cut_end_m_sequence`` (``_synced_``).

        """
        # results of this stage are invalidated by a new sample-wise resampling
        self._clear_results("_synced_")

        dict_resampled = {}
        index = None

        for name in self.datasets_aligned:
            df = self.datasets_aligned[name]

            if name == f"{primary}_aligned_":
                index = df.index
                df = df.reset_index(drop=True)
                dict_resampled[name] = df
                continue
            data_resample = self._resample_sample_wise(df, dict_sample_shift[name])
            # the resampling interpolates the binary sync signal => binarize it again
            sync_channel = self._get_sync_channel(name)
            data_resample[sync_channel] = self._binarize_signal(data_resample[sync_channel])
            dict_resampled[name] = data_resample

        if index is None:
            raise SynchronizationError(
                f"No aligned dataset found for the primary dataset '{primary}'. "
                f"Call 'align_and_cut_start_m_sequence' first!"
            )

        # cut all datasets to the length of the shortest one, if requested
        target_length = min(len(data) for data in dict_resampled.values()) if cut_to_shortest else None

        for name, data in dict_resampled.items():
            # strip the "_aligned_" suffix to get the plain dataset name
            name_out = self._strip_result_suffix(name)

            data_synced = data if target_length is None else data.iloc[:target_length]
            data_synced = data_synced.copy()
            data_synced.index = self._index_for_length(index, len(data_synced))
            self.datasets_synced_[f"{name_out}_synced_"] = data_synced
            setattr(self, f"{name_out}_synced_", data_synced)

    def _resample_sample_wise(self, df: pd.DataFrame, sample_shift: int) -> pd.DataFrame:
        """Resample a dataset from its own length to ``len(df) + sample_shift`` samples.

        The resampling uses the same method as :meth:`~SyncBase.resample_datasets`, i.e., the band-limited
        interpolation of :func:`resampy.resample`, with the sample counts as input and output "rates".

        """
        df_size = len(df)

        if sample_shift == 0:
            # nothing to resample
            return df.reset_index(drop=True)

        data_resample = resampy.resample(
            df.to_numpy(), sr_orig=df_size, sr_new=df_size + sample_shift, axis=0, parallel=True
        )

        # array should remain df, keep datetime index
        df_resample = pd.DataFrame(data_resample, columns=df.columns)

        return df_resample
