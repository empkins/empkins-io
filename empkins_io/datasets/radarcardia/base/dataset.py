import json
from functools import cached_property, lru_cache
from itertools import product
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
import pandas as pd
from tpcp import Dataset
import warnings


from empkins_io.datasets.radarcardia.base.helper import (
    _load_biopac_data,
    _load_radar_data,
    _load_timelog,
    _save_aligned_data,
    _save_data_to_location_h5,
    _load_data_from_location_h5,
    _build_timelog_path,
    _calc_biopac_timelog_shift
)

from empkins_io.sync import SyncedDataset
from empkins_io.utils._types import path_t

_cached_get_biopac_data = lru_cache(maxsize=4)(_load_biopac_data)
_cached_get_radar_data = lru_cache(maxsize=4)(_load_radar_data)


class BaseDataset(Dataset):

    base_path: path_t
    use_cache: bool
    calc_shift: bool
    bp_tl_shift: pd.Timedelta

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
            use_cache: Optional[bool] = False,
            calc_biopac_timelog_shift: Optional[bool] = True
    ):
        self.base_path = base_path
        self.use_cache = use_cache
        self.calc_shift = calc_biopac_timelog_shift

        if self.calc_shift:
            self.bp_tl_shift = None
        else:
            self.bp_tl_shift = pd.Timedelta("0s")

        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self) -> pd.DataFrame:
        raise NotImplementedError()

    @property
    def sampling_rates(self) -> Dict[str, float]:
        return self._sampling_rates

    @property
    def biopac_timelog_shift(self):
        if not self.is_single(["subject"]):
            raise ValueError(
                "Shift between Timelog and BIOPAC/EMRAD data can only be accessed for one single participant at once"
            )

        if not self.bp_tl_shift:
            participant_id = self.index["subject"][0]
            self.bp_tl_shift = self._get_biopac_timelog_shift(participant_id=participant_id)

        return self.bp_tl_shift


    @property
    def biopac_raw_unsynced(self) -> pd.DataFrame:
        if not self.is_single(["subject"]):
            raise ValueError("BIOPAC data can only be accessed for one single participant at once")

        participant_id = self.index["subject"][0]

        if self.is_single(None):
            location = self._get_locations_from_index()
            return self._get_biopac_data(participant_id=participant_id, location=location, state="raw")

        print(f"Complete BIOPAC Dataset for {participant_id}:")

        return self._get_biopac_data(participant_id=participant_id, location="all", state="raw")

    @property
    def emrad_raw_unsynced(self) -> pd.DataFrame:
        if not self.is_single(["subject"]):
            raise ValueError("Radar data can only be accessed for one single participant at once")

        participant_id = self.index["subject"][0]

        if self.is_single(None):
            location = self._get_locations_from_index()
            return self._get_radar_data(participant_id=participant_id, location=location[0], state="raw")

        print(f"Complete Radar Dataset for {participant_id}:")

        return self._get_radar_data(participant_id=participant_id, location="all", state="raw")

    @property
    def biopac_raw_synced(self) -> pd.DataFrame:
        if not self.is_single(["subject"]):
            raise ValueError("BIOPAC data can only be accessed for one single participant at once")

        participant_id = self.index["subject"][0]

        if self.is_single(None):
            location = self._get_locations_from_index()
            return self._load_and_sync_datasets(
                participant_id=participant_id, location=location[0], biopac=True, state="raw"
            )

        return self._load_and_sync_datasets(participant_id=participant_id, location="all", biopac=True, state="raw")

    @property
    def emrad_raw_synced(self) -> pd.DataFrame:
        if not self.is_single(["subject"]):
            raise ValueError("Radar data can only be accessed for one single participant at once")

        participant_id = self.index["subject"][0]

        if self.is_single(None):
            location = self._get_locations_from_index()
            return self._load_and_sync_datasets(
                participant_id=participant_id, location=location[0], radar=True, state="raw"
            )

        return self._load_and_sync_datasets(participant_id=participant_id, location="all", radar=True, state="raw")

    @property
    def dataset_raw_synced(self) -> pd.DataFrame:
        if not self.is_single(["subject"]):
            raise ValueError("Data can only be accessed for one single participant at once")

        participant_id = self.index["subject"][0]

        if self.is_single(None):
            location = self._get_locations_from_index()
            return self._load_and_sync_datasets(
                participant_id=participant_id, location=location[0], biopac=True, radar=True, state="raw"
            )

        return self._load_and_sync_datasets(
            participant_id=participant_id, location="all", biopac=True, radar=True, state="raw"
        )

    @property
    def biopac_data(self) -> pd.DataFrame:
        if not self.is_single(["subject"]):
            raise ValueError("BIOPAC data can only be accessed for one single participant at once")

        participant_id = self.index["subject"][0]

        if self.is_single(None):
            location = self._get_locations_from_index()
            return self._get_biopac_data(participant_id=participant_id, location=location, state="aligned")

        print(f"Complete BIOPAC Dataset for {participant_id}:")

        return self._get_biopac_data(participant_id=participant_id, location="all", state="aligned")

    @property
    def radar_data(self) -> pd.DataFrame:
        if not self.is_single(["subject"]):
            raise ValueError("Radar data can only be accessed for one single participant at once")

        participant_id = self.index["subject"][0]

        if self.is_single(None):
            location = self._get_locations_from_index()
            return self._get_radar_data(participant_id=participant_id, location=location[0], state="aligned")

        print(f"Complete Radar Dataset for {participant_id}:")

        return self._get_radar_data(participant_id=participant_id, location="all", state="aligned")

    @property
    def timelog(self) -> pd.DataFrame:
        if not self.is_single(["subject"]):
            raise ValueError("Timelog can only be accessed for one single participant at once")
        locations = self._get_locations_from_index()
        participant_id = self.index["subject"][0]
        tl = self._get_timelog(participant_id)
        return tl[locations]

    @property
    def timelog_all(self) -> pd.DataFrame:
        if not self.is_single(["subject"]):
            raise ValueError("Timelog can only be accessed for one single participant at once")
        participant_id = self.index["subject"][0]
        return self._get_timelog(participant_id)

    @property
    def timelog_path(self):
        participant_id = self.index["subject"][0]
        timelog_path = _build_timelog_path(base_path=self.base_path, participant_id=participant_id)
        return timelog_path

    def save_aligned_data(self, biopac_data, radar_data):
        if not self.is_single(["subject"]):
            raise ValueError("BIOPAC and Radar data can only be saved for a single participant at once")
        participant_id = self.index["subject"][0]
        _save_aligned_data(
            base_path=self.base_path, participant_id=participant_id, biopac=biopac_data, radar=radar_data)

    def save_data_to_location(self, data: pd.DataFrame, file_name: str, radar: bool=False, biopac: bool=False):
        locations = self.index.drop(columns="subject").columns.tolist()
        if not self.is_single(locations):
            raise ValueError("Data can only be saved for a single location-breathing combination")

        participant_id = self.index["subject"][0]
        location = self._get_locations_from_index()[0]
        _save_data_to_location_h5(
            base_path=self.base_path,
            participant_id=participant_id,
            data=data,
            biopac=biopac,
            radar=radar,
            location=location,
            file_name=file_name
        )

    def load_data_from_location(self, file_name: str, radar: bool=False, biopac: bool=False):
        locations = self.index.drop(columns="subject").columns.tolist()
        if not self.is_single(locations):
            raise ValueError("Data can only be loaded for a single location-breathing combination")

        participant_id = self.index["subject"][0]
        location = self._get_locations_from_index()[0]
        data = _load_data_from_location_h5(
            base_path=self.base_path,
            participant_id=participant_id,
            radar=radar,
            biopac=biopac,
            location=location,
            file_name=file_name
        )
        return data

    def _get_locations_from_index(self):
        locations = self.index.drop(columns="subject").values.tolist()
        locations = ["_".join(i) for i in locations]
        return locations

    def _load_and_sync_datasets(
            self, participant_id: str, location: str, state: str, biopac: bool = False, radar: bool = False
    ) -> pd.DataFrame:
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
            # start = tl[location[0]]["start"][0]
            # end = tl[location[0]]["end"][0]

            # test
            start = tl[location[0]]["start"][0] + self.biopac_timelog_shift
            end = tl[location[0]]["end"][0] + self.biopac_timelog_shift
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
            start = tl[location]["start"][0] + self.biopac_timelog_shift
            end = tl[location]["end"][0] + self.biopac_timelog_shift
            return radar[start:end]

    def _get_timelog(self, participant_id: str) -> pd.DataFrame:
        return _load_timelog(self.base_path, participant_id)

    def _get_biopac_timelog_shift(self, participant_id: str):
        return _calc_biopac_timelog_shift(base_path=self.base_path, participant_id=participant_id)
