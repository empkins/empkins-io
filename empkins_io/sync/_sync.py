from typing import Any, Dict, Literal, Optional

import numpy as np
import pandas as pd
from biopsykit.utils._datatype_validation_helper import _assert_is_dtype
from scipy import signal

SYNC_TYPE = Literal["peak", "square", "m-sequence"]


class SyncedDataset:

    primary: Dict[str, Any]
    secondaries: Dict[str, Dict[str, Any]]
    secondaries_synced_: Dict[str, Dict[str, Any]]
    sync_type: SYNC_TYPE

    def __init__(self, sync_type: SYNC_TYPE = "peak"):
        # assert sync_type in SYNC_TYPE, f"Sync type {sync_type} not valid. Mus be one of {SYNC_TYPE}."
        self.sync_type = sync_type
        self.primary = {}
        self.secondaries = {}

    def add_primary(self, name: str, data: pd.DataFrame, sync_channel_name: str, sampling_rate: int):
        # assert that data is a pandas DataFrame
        _assert_is_dtype(data, pd.DataFrame)
        # assert that data has a column with the name sync_channel
        assert sync_channel_name in data.columns, f"Sync channel {sync_channel_name} not in data."
        # assert that primary_sampling_rate is a valid sampling rate
        assert isinstance(sampling_rate, (int, float)), f"Sampling rate '{sampling_rate}' not valid"

        setattr(self, name, data)
        self.primary["name"] = name
        self.primary["sync_channel"] = sync_channel_name
        self.primary["sampling_rate"] = sampling_rate

    def add_secondary(self, name: str, data: pd.DataFrame, sync_channel_name: str, sampling_rate: int):
        # assert that data is a pandas DataFrame
        _assert_is_dtype(data, pd.DataFrame)
        # assert that data has a column with the name sync_channel
        assert sync_channel_name in data.columns, f"Sync channel {sync_channel_name} not in data."
        # assert that sampling_rate is a valid sampling rate
        assert isinstance(sampling_rate, (int, float)), f"Sampling rate '{sampling_rate}' not valid."

        self.secondaries[name] = {}
        setattr(self, name, data)
        self.secondaries[name]["name"] = name
        self.secondaries[name]["sync_channel"] = sync_channel_name
        self.secondaries[name]["sampling_rate"] = sampling_rate

    def synchronize_datasets(self, cut_to_sync_region: Optional[bool] = True):
        """Synchronize all datasets to the primary dataset.

        Parameters
        ----------
        cut_to_sync_region : bool, optional
            ``True`` to cut all datasets to the region where all datasets are synced, ``False`` otherwise.
            Default: ``True``

        """
        for name in self.secondaries:
            data_synced = self._align_to_primary(self.primary, self.secondaries[name])
            if cut_to_sync_region:
                data_synced = self._cut_to_sync_region(data_synced, self.secondaries[name])
            setattr(self, f"{name}_synced", data_synced)
        primary_synced = getattr(self, self.primary["name"])
        if cut_to_sync_region:
            primary_synced = self._cut_to_sync_region(primary_synced, self.primary)
        setattr(self, f"{self.primary['name']}_synced", primary_synced)

    @property
    def synced_datasets(self):
        # get all synced attributes
        synced_attributes = {attr: getattr(self, attr) for attr in dir(self) if attr.endswith("_synced")}
        return synced_attributes

    def _align_to_primary(self, primary: Dict[str, Any], secondary: Dict[str, Any]) -> pd.DataFrame:
        """Align a secondary dataset to the primary dataset.

        Parameters
        ----------
        primary : dict
            dictionary containing the primary dataset
        secondary : dict
            dictionary containing the secondary dataset

        Returns
        -------
        pd.DataFrame
            secondary dataset aligned to the primary dataset

        """
        if self.sync_type == "m-sequence":
            delay_samples = self._align_cross_corr(primary, secondary)
        else:
            delay_samples = 0

        data = getattr(self, secondary["name"])
        return data.shift(delay_samples).dropna()

    def _align_cross_corr(self, primary: Dict[str, Any], secondary: Dict[str, Any]) -> int:
        data_prim = getattr(self, primary["name"])
        data_sec = getattr(self, secondary["name"])
        sync_ch_prim = primary["sync_channel"]
        sync_ch_sec = secondary["sync_channel"]
        fs_prim = primary["sampling_rate"]
        fs_sec = secondary["sampling_rate"]

        len_primary = len(data_prim)
        len_secondary = len(data_sec)

        cross_corr = signal.correlate(data_prim[sync_ch_prim], data_sec[sync_ch_sec], mode="same")
        auto_corr_prim = signal.correlate(data_prim[sync_ch_prim], data_prim[sync_ch_prim], mode="same")
        auto_corr_sec = signal.correlate(data_sec[sync_ch_sec], data_sec[sync_ch_sec], mode="same")

        cross_corr = cross_corr / np.sqrt(auto_corr_prim[int(len_primary / 2)] * auto_corr_sec[int(len_secondary / 2)])

        delay_arr = np.linspace(-0.5 * len_primary / fs_prim, 0.5 * len_primary / fs_prim, len_primary)
        delay_sec = delay_arr[np.argmax(cross_corr)]
        delay_samples = int(np.around(delay_sec * fs_sec))
        return delay_samples

    def _cut_to_sync_region(self, data: pd.DataFrame, secondary: Dict[str, Any]) -> pd.DataFrame:
        sync_ch_sec = secondary["sync_channel"]
        # find peaks in sync channel derivation
        peaks, _ = signal.find_peaks(np.ediff1d(data[sync_ch_sec]), height=0.5)
        # cut data to region between first and last peak
        data = data.iloc[peaks[0] : peaks[-1]]
        return data
