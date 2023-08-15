import json
from functools import cached_property, lru_cache
from itertools import product
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
import pandas as pd
from tpcp import Dataset
import pandas as pd
import warnings

from biopsykit.io import load_long_format_csv
from biopsykit.utils.dataframe_handling import multi_xs, wide_to_long
from biopsykit.utils.file_handling import get_subject_dirs

from empkins_io.datasets.radarcardia.base.helper import (
    _load_biopac_data,
    _load_radar_data,
    _load_timelog
)

from empkins_io.sync import SyncedDataset
from empkins_io.utils._types import path_t

_cached_get_biopac_data = lru_cache(maxsize=4)(_load_biopac_data)
_cached_get_radar_data = lru_cache(maxsize=4)(_load_radar_data)


class BaseDataset(Dataset):

    base_path: path_t
    use_cache: bool
    _sampling_rates: Dict[str, float] = {
        "radar_original": 8000000 / 4096,
        "biopac_original": 2000,
        "resampled": 1000
    }

    BIOPAC_CHANNEL_MAPPING: Dict[str, str] = {
        "ECG (.05 - 150 Hz)": "ecg",
        "Cardiac Output - Z": "icg",
        "Cardiac Output - dZ/dt": "icg_der",
        "Sync": "sync"
    }

    def __init__(
            self,
            base_path: path_t,
            groupby_cols: Optional[Sequence[str]] = None,
            subset_index: Optional[Sequence[str]] = None,
            use_cache: Optional[bool] = True
    ):
        self.base_path = base_path
        self.use_cache = use_cache

        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self) -> pd.DataFrame:
        raise NotImplementedError()

    @property
    def sampling_rates(self) -> Dict[str, float]:
        return self._sampling_rates

    @property
    def biopac_raw_unsynced(self) -> pd.DataFrame:
        if not self.is_single(["participant"]):
            raise ValueError("BIOPAC data can only be accessed for one single participant at once")

        participant_id = self.index["participant"][0]

        if self.is_single(None):
            location = self._get_locations_from_index()
            return self._get_biopac_data(participant_id=participant_id, location=location, state="raw")

        print(f"Complete BIOPAC Dataset for {participant_id}:")

        return self._get_biopac_data(participant_id=participant_id, location="all", state="raw")

    @property
    def emrad_raw_unsynced(self) -> pd.DataFrame:
        if not self.is_single(["participant"]):
            raise ValueError("Radar data can only be accessed for one single participant at once")

        participant_id = self.index["participant"][0]

        if self.is_single(None):
            location = self._get_locations_from_index()
            return self._get_radar_data(participant_id=participant_id, location=location, state="raw")

        print(f"Complete Radar Dataset for {participant_id}:")

        return self._get_radar_data(participant_id=participant_id, location="all", state="raw")

    @property
    def biopac_raw_synced(self) -> pd.DataFrame:
        if not self.is_single(["participant"]):
            raise ValueError("BIOPAC data can only be accessed for one single participant at once")

        participant_id = self.index["participant"][0]

        if self.is_single(None):
            location = self._get_locations_from_index()
            return self._load_and_sync_datasets(participant_id=participant_id, location=location, biopac=True, state="raw")

        return self._load_and_sync_datasets(participant_id=participant_id, location="all", biopac=True, state="raw")

    @property
    def emrad_raw_synced(self) -> pd.DataFrame:
        if not self.is_single(["participant"]):
            raise ValueError("Radar data can only be accessed for one single participant at once")

        participant_id = self.index["participant"][0]

        if self.is_single(None):
            location = self._get_locations_from_index()
            return self._load_and_sync_datasets(participant_id=participant_id, location=location, radar=True, state="raw")

        return self._load_and_sync_datasets(participant_id=participant_id, location="all", radar=True, state="raw")

    @property
    def raw_dataset_synced(self) -> pd.DataFrame:
        if not self.is_single(["participant"]):
            raise ValueError("Data can only be accessed for one single participant at once")

        participant_id = self.index["participant"][0]

        if self.is_single(None):
            location = self._get_locations_from_index()
            return self._load_and_sync_datasets(participant_id=participant_id, location=location, biopac=True, radar=True, state="raw")

        return self._load_and_sync_datasets(participant_id=participant_id, location="all", biopac=True, radar=True, state="raw")

    @property
    def timelog(self) -> pd.DataFrame:
        if not self.is_single(["participant"]):
            raise ValueError("Timelog can only be accessed for one single participant at once")
        locations = self._get_locations_from_index()
        participant_id = self.index["participant"][0]
        tl = self._get_timelog(participant_id)
        return tl[locations]

    @property
    def timelog_all(self) -> pd.DataFrame:
        if not self.is_single(["participant"]):
            raise ValueError("Timelog can only be accessed for one single participant at once")
        participant_id = self.index["participant"][0]
        return self._get_timelog(participant_id)

    def _get_locations_from_index(self):
        locations = self.index.drop(columns="participant").values.tolist()
        locations = ["_".join(i) for i in locations]
        return locations

    def _load_and_sync_datasets(self, participant_id: str, location: str, state: str, biopac: bool=False, radar:bool=False) -> pd.DataFrame:
        biopac_data = self._get_biopac_data(participant_id=participant_id, location=location, state=state)
        radar_data = self._get_radar_data(participant_id=participant_id, location=location, state=state)

        synced_dataset = SyncedDataset(sync_type="m-sequence")
        synced_dataset.add_dataset("biopac", data=biopac_data, sync_channel_name="sync",
                                   sampling_rate=self._sampling_rates["biopac_original"])
        synced_dataset.add_dataset("radar", data=radar_data, sync_channel_name="Sync_Out",
                                   sampling_rate=self._sampling_rates["radar_original"])
        synced_dataset.resample_datasets(fs_out=self._sampling_rates["resampled"], method="dynamic", wave_frequency=10)
        synced_dataset.align_and_cut_m_sequence(
            primary="radar", reset_time_axis=True, cut_to_shortest=True
        )

        if biopac and radar:
            return synced_dataset.datasets_aligned
        elif biopac:
            return synced_dataset.datasets_aligned["biopac_aligned_"]
        elif radar:
            return synced_dataset.datasets_aligned["radar_aligned_"]
        else:
            return

    def _get_biopac_data(self, participant_id: str, location: str, state: str):

        if self.use_cache:
            raise NotImplementedError("Using cached properties is not implemented yet")
        else:
            biopac, _ = _load_biopac_data(
                self.base_path,
                participant_id=participant_id,
                channel_mapping=self.BIOPAC_CHANNEL_MAPPING,
                state=state
            )

        if location == "all":
            return biopac
        else:
            tl = self.timelog
            start = tl[location[0]]["start"][0]
            end = tl[location[0]]["end"][0]
            return biopac.loc[start:end]

    def _get_radar_data(self, participant_id: str, location: str, state: str):
        radar, _ = _load_radar_data(
            self.base_path,
            participant_id=participant_id,
            fs=self._sampling_rates["radar_original"],
            state=state
        )

        if location == "all":
            return radar
        else:
            tl = self.timelog
            start = tl[location[0]]["start"][0]
            end = tl[location[0]]["end"][0]
            return radar[start:end]

    def _get_timelog(self, participant_id: str) -> pd.DataFrame:
        return _load_timelog(self.base_path, participant_id)
