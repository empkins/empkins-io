import re
import warnings
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union, get_args

import numpy as np
import pandas as pd
import resampy
from biopsykit.utils._datatype_validation_helper import _assert_is_dtype
from matplotlib import pyplot as plt
from scipy import signal
from scipy.signal import find_peaks, periodogram, resample

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


class SyncedDataset:

    _VALID_INDEX_NAMES = (r"t", r"utc", r"date", r"date \(.*\)")

    datasets: Dict[str, Dict[str, Any]]
    datasets_synced_: Dict[str, Dict[str, Any]]
    sync_type: SYNC_TYPE

    def __init__(self, sync_type: SYNC_TYPE = "rising-trigger"):
        # TODO fix this
        if sync_type not in get_args(SYNC_TYPE):
            raise ValueError(f"Sync type {sync_type} not valid. Must be one of {get_args(SYNC_TYPE)}.")
        if sync_type in SYNC_TYPE_DEPRECATED:
            warnings.warn(
                f"Sync type {sync_type} is deprecated. Please use one of {get_args(SYNC_TYPE)}.", DeprecationWarning
            )
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

    def resample_datasets(self, fs_out: float, method: Literal["static", "dynamic"], **kwargs) -> None:
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

            # # resample data with scikit-signal
            # data_resample = signal.resample(
            #     dataset["data"].values, num=int(len(dataset["data"]) * fs_out / fs_in), axis=0
            # )

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

    def cut_to_sync_start(self, sync_params: Optional[Dict[str, Any]] = None):
        warnings.warn(
            "cut_to_sync_start is deprecated and will be removed in the future. Use cut_to_sync_region instead.",
            DeprecationWarning,
        )
        return self.cut_to_sync_region(sync_params=sync_params)

    def cut_to_sync_region(self, sync_params: Optional[Dict[str, Any]] = None):
        """Cut all datasets to the region where all datasets are synced."""
        if sync_params is None:
            sync_params = {}
        for name in self.datasets:
            dataset = self.datasets[name]
            params = sync_params.get(name, {})
            data_cut = self._cut_dataset_to_sync_region(dataset, sync_params=params)
            setattr(self, f"{name}_cut_", data_cut)

    def _cut_dataset_to_sync_region(self, dataset: Dict[str, Any], sync_params: Dict[str, Any]) -> pd.DataFrame:

        if self.sync_type == "m-sequence":
            raise NotImplementedError(
                "For cutting and aligning datasets, please use the 'cut_to_sync_start_m_sequence' method."
            )

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

        if "falling" in self.sync_type:
            # invert sync channel to achieve rising sync signal
            sync_data = -1 * data[sync_channel]
        peaks = SyncedDataset._find_sync_peaks(sync_data, sync_params)
        # cut data to region between first and last peak
        data_cut = data.iloc[peaks[0] :] if len(peaks) == 1 else data.iloc[peaks[0] : peaks[-1]]

        return data_cut

    def _find_shift(self, primary: str, sync_params: Optional[Dict[str, Any]] = None):
        if sync_params is None:
            sync_params = {}

        # assert that sampling rates are equal for all datasets
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
        dict_lags = {}

        sync_params["sampling_rate"] = list(sampling_rates)[0]

        sync_channel_primary = "Sync_Out"

        fs = sync_params["sampling_rate"]

        data_primary = self.datasets_aligned[primary]
        data_primary = data_primary.copy()
        data_primary.loc[:, sync_channel_primary] = self._binarize_signal(data_primary[sync_channel_primary])

        for name, dataset in self.datasets_aligned.items():
            if name == primary:
                continue

            data_secondary = dataset
            sync_channel_secondary = "Sync_Out"
            data_secondary.loc[:, sync_channel_secondary] = self._binarize_signal(
                data_secondary[sync_channel_secondary]
            )

            data_primary = data_primary.reset_index()
            data_secondary = data_secondary.reset_index()

            # cut to search region
            sync_region_samples = sync_params.get("sync_region_samples", (0, len(data_primary)))
            data_primary_search = data_primary.iloc[sync_region_samples[0] : sync_region_samples[1]]
            data_secondary_search = data_secondary.iloc[sync_region_samples[0] : sync_region_samples[1]]

            lag_samples = self._find_sync_cross_correlation(
                data_primary_search[sync_channel_primary], data_secondary_search[sync_channel_secondary], fs
            )
            dict_lags[name] = lag_samples
            print("Shift: " + name + " " + str(lag_samples))

        return dict_lags

    def resample_sample_wise(self, primary, dict_sample_shift, cut_to_shortest=True):
        dict_resampled = {}

        for name in self.datasets_aligned:

            df = self.datasets_aligned[name]

            if name == primary:
                index = df.index
                df = df.reset_index(drop=True)
                dict_resampled[name] = df
                continue
            data_resample = self._resample_sample_wise(df, dict_sample_shift[name])
            dict_resampled[name] = data_resample

        if cut_to_shortest:
            shortest_length = min(len(data) for name, data in dict_resampled.items())
            index = index[:shortest_length]
            for name, data in dict_resampled.items():
                # cut name after second _ to get rid of _aligned_
                name = "_".join(name.split("_")[:2])

                print(len(data))

                data_aligned = data.iloc[:shortest_length]
                data_aligned.index = index
                setattr(self, f"{name}_resampled_", data_aligned)

    def _resample_sample_wise(self, df, sample_shift):

        df_size = len(df)

        df_resample = resample(df, df_size + sample_shift)

        # array should remain df, keep datetime index
        df_resample = pd.DataFrame(df_resample, columns=df.columns)

        return df_resample

    def align_and_cut_m_sequence(
        self,
        primary: str,
        cut_to_shortest: Optional[bool] = False,
        reset_time_axis: Optional[bool] = False,
        sync_params: Optional[Dict[str, Any]] = None,
    ):
        if sync_params is None:
            sync_params = {}

        # assert that sampling rates are equal for all datasets
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

        sync_params["sampling_rate"] = list(sampling_rates)[0]
        sync_channel_primary = self.datasets[primary]["sync_channel"]

        fs = sync_params["sampling_rate"]

        dict_data_pad = {}
        dict_lags = {}

        data_primary = self.datasets[primary].get("data_resampled", self.datasets[primary].get("data"))
        data_primary = data_primary.copy()
        data_primary.loc[:, sync_channel_primary] = self._binarize_signal(data_primary[sync_channel_primary])

        for name, dataset in self.datasets.items():
            if name == primary:
                continue

            data_secondary = dataset.get("data_resampled", dataset.get("data"))
            sync_channel_secondary = dataset["sync_channel"]
            data_secondary.loc[:, sync_channel_secondary] = self._binarize_signal(
                data_secondary[sync_channel_secondary]
            )

            # zero-pad the shorter signal in the end. This is necessary for the cross-correlation
            padlen_samples = len(data_secondary) - len(data_primary)

            if padlen_samples > 0:
                # the primary signal is shorter than the secondary signal => cut the secondary signal
                data_secondary = data_secondary.iloc[:-padlen_samples]
            else:
                # the secondary signal is shorter than the primary signal => pad the secondary signal
                data_secondary = self._pad_signal(data_secondary, -padlen_samples, start=False, fs=fs)

            data_primary = data_primary.reset_index()
            data_secondary = data_secondary.reset_index()

            # cut to search region
            sync_region_samples = sync_params.get("sync_region_samples", (0, len(data_primary)))
            data_primary_search = data_primary.iloc[sync_region_samples[0] : sync_region_samples[1]]
            data_secondary_search = data_secondary.iloc[sync_region_samples[0] : sync_region_samples[1]]

            lag_samples = self._find_sync_cross_correlation(
                data_primary_search[sync_channel_primary], data_secondary_search[sync_channel_secondary], fs
            )
            print(lag_samples)

            dict_data_pad[name] = data_secondary
            dict_lags[name] = lag_samples

            data_primary = data_primary.set_index(data_primary.columns[0])

        setattr(self, f"{primary}_aligned_", data_primary)

        # align all the signals that are *behind* the primary signal by cutting the beginning
        for name, data in dict_data_pad.items():
            if dict_lags[name] < 0:
                data = data.iloc[-dict_lags[name] :].reset_index(drop=True)

            data = data.set_index(data.columns[0])
            dict_data_pad[name] = data
            setattr(self, f"{name}_aligned_", data)

        # align all the signals that are *ahead* of the primary signal by cutting the beginning of all other signals
        for name, data in dict_data_pad.items():
            if dict_lags[name] > 0:
                # shift all the others to match this one
                for name2, data2 in dict_data_pad.items():
                    if name2 == name:
                        continue
                    data2 = self._reset_and_shift(data2, dict_lags[name])

                    setattr(self, f"{name2}_aligned_", data2)

                # shift primary
                data_primary = getattr(self, f"{primary}_aligned_")
                data_primary = self._reset_and_shift(data_primary, dict_lags[name])
                setattr(self, f"{primary}_aligned_", data_primary)

        if reset_time_axis:
            data_primary = getattr(self, f"{primary}_aligned_")
            for name in self.datasets:
                if name == primary:
                    continue
                data_aligned = getattr(self, f"{name}_aligned_")
                data_aligned.index = data_aligned.index - data_aligned.index[0] + data_primary.index[0]
                setattr(self, f"{name}_aligned_", data_aligned)

        if cut_to_shortest:
            shortest_length = min(len(data) for name, data in self.datasets_aligned.items())
            for name in self.datasets:
                data_aligned = getattr(self, f"{name}_aligned_")
                data_aligned = data_aligned.iloc[:shortest_length]
                setattr(self, f"{name}_aligned_", data_aligned)

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

    def _find_sync_cross_correlation(
        self,
        primary: Union[np.ndarray, pd.DataFrame],
        secondary: Union[np.ndarray, pd.DataFrame],
        fs: float,
    ) -> int:

        # find the cross-correlation values and the index of the maximum cross-correlation
        lag_values = np.arange((-len(primary) + 1) / fs, len(primary) / fs, 1 / fs)

        crosscorr = signal.correlate(primary, secondary)
        max_crosscorr_idx = np.argmax(crosscorr)

        # find the lag at the cross-correlation maximum (t-value) and the number of timesteps corresponding to this lag
        lag_samples = int(round(lag_values[max_crosscorr_idx] * fs))
        return lag_samples

    def _check_valid_index(self, data: pd.DataFrame):
        index_type = list({type(dataset["data"].index) for dataset in self.datasets.values()})[0]
        new_index_type = type(data.index)
        if index_type != new_index_type:
            raise ValidationError(
                f"Indices of all added datasets must be of the same type. Got {[index_type, new_index_type]}."
            )
        # check that the index names of the datasets are the same
        index_name = list({dataset["data"].index.name for dataset in self.datasets.values()})[0]
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

    def _determine_actual_sampling_rate(self, dataset: Dict[str, Any], **kwargs) -> float:
        wave_frequency = kwargs.get("wave_frequency", None)
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

    @classmethod
    def _normalize_signal(cls, data: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    @classmethod
    def _binarize_signal(cls, data: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        return 0.5 * (np.sign(data - np.mean(data)) + 1)

    @classmethod
    def _pad_signal(cls, data: pd.DataFrame, padlen: int, start: bool, fs: float) -> pd.DataFrame:
        if start:
            pad_width = ((padlen, 0), (0, 0))
            constant_values = ((0, None), (None, None))
        else:
            pad_width = ((0, padlen), (0, 0))
            constant_values = ((None, 0), (None, None))
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

    @staticmethod
    def _reset_and_shift(data: pd.DataFrame, shift_idx: int) -> pd.DataFrame:
        data = data.reset_index()
        data = data.iloc[shift_idx:].reset_index(drop=True)
        return data.set_index(data.columns[0])
