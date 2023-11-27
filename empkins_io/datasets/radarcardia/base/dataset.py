import json
from functools import cached_property, lru_cache
from itertools import product
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union
import pandas as pd
from tpcp import Dataset
import warnings


from empkins_io.datasets.radarcardia.base.helper import (
    _load_biopac_data,
    _load_radar_data,
    _load_timelog,
    _build_timelog_path,
    _build_protocol_path,
    _get_biopac_timelog_shift,
    _load_protocol,
    _save_data_to_location_h5,
    _load_data_from_location_h5
)

from empkins_io.utils._types import path_t

_cached_get_biopac_data = lru_cache(maxsize=4)(_load_biopac_data)
_cached_get_radar_data = lru_cache(maxsize=4)(_load_radar_data)


class BaseDataset(Dataset):

    base_path: path_t
    use_cache: bool
    calc_biopac_timelog_shift: bool
    trigger_data_extraction: bool
    bp_tl_shift: Union[pd.Timedelta, None]

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
            calc_biopac_timelog_shift: Optional[bool] = True,
            trigger_data_extraction: Optional[bool] = False
    ):
        self.base_path = base_path
        self.use_cache = use_cache
        self.calc_biopac_timelog_shift = calc_biopac_timelog_shift
        self.trigger_data_extraction = trigger_data_extraction

        if self.calc_biopac_timelog_shift:
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
    def biopac_raw_unsynced(self) -> tuple[pd.DataFrame, float]:
        if not self.is_single(["subject"]):
            raise ValueError("BIOPAC data can only be accessed for one single participant at once")

        participant_id = self.index["subject"][0]
        biopac_data = self._get_biopac_data(participant_id=participant_id, state="raw_unsynced")

        return biopac_data, self.sampling_rates["biopac_original"]

    @property
    def emrad_raw_unsynced(self) -> tuple[pd.DataFrame, float]:
        if not self.is_single(["subject"]):
            raise ValueError("Radar data can only be accessed for one single participant at once")

        participant_id = self.index["subject"][0]
        radar_data = self._get_radar_data(participant_id=participant_id, state="raw_unsynced")

        return radar_data, self.sampling_rates["radar_original"]

    @property
    def biopac_raw_synced(self) -> tuple[pd.DataFrame, float]:
        if not self.is_single(["subject"]):
            raise ValueError("BIOPAC data can only be accessed for one single participant at once")

        participant_id = self.index["subject"][0]
        biopac_data = self._get_biopac_data(participant_id=participant_id, state="raw_synced")

        return biopac_data, self.sampling_rates["resampled"]

    @property
    def emrad_raw_synced(self) -> tuple[pd.DataFrame, float]:
        if not self.is_single(["subject"]):
            raise ValueError("Radar data can only be accessed for one single participant at once")

        participant_id = self.index["subject"][0]
        radar_data = self._get_radar_data(participant_id=participant_id, state="raw_synced")

        return radar_data, self.sampling_rates["resampled"]

    @property
    def biopac_data(self) -> tuple[pd.DataFrame, float]:
        if not self.is_single(None):
            raise ValueError("BIOPAC data can only be accessed for one single location at once")

        participant_id = self.index["subject"][0]
        biopac_data = self._get_biopac_data(participant_id=participant_id, state="location_synced")

        return biopac_data, self.sampling_rates["resampled"]

    @property
    def emrad_data(self) -> tuple[pd.DataFrame, float]:
        if not self.is_single(None):
            raise ValueError("Radar data can only be accessed for one single location at once")

        participant_id = self.index["subject"][0]
        radar_data = self._get_radar_data(participant_id=participant_id, state="location_synced")

        return radar_data, self.sampling_rates["resampled"]

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
    def protocol(self) -> pd.DataFrame:
        if not self.is_single(["subject"]):
            raise ValueError("Protocol Information can only be accessed for one single participant at once")
        participant_id = self.index["subject"][0]
        return self._get_protocol(participant_id)

    @property
    def timelog_path(self):
        participant_id = self.index["subject"][0]
        timelog_path = _build_timelog_path(base_path=self.base_path, participant_id=participant_id)
        return timelog_path

    @property
    def protocol_path(self):
        participant_id = self.index["subject"][0]
        protocol_path = _build_protocol_path(base_path=self.base_path, participant_id=participant_id)
        return protocol_path

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

    def _get_biopac_data(self, participant_id: str, state: str):
        biopac = _load_biopac_data(
            self.base_path,
            participant_id=participant_id,
            fs=self._sampling_rates,
            channel_mapping=self.BIOPAC_CHANNEL_MAPPING,
            state=state,
            trigger_extraction=self.trigger_data_extraction,
            location=self._get_locations_from_index()[0]
        )

        if self.is_single(None):
            location = self._get_locations_from_index()
            if len(location) > 1:
                raise ValueError("BIOPAC data can be accessed for all or only one single location at the same time.")
            tl = self.timelog
            start = tl[location[0]]["start"][0] + self.biopac_timelog_shift
            end = tl[location[0]]["end"][0] + self.biopac_timelog_shift
            return biopac.loc[start:end]

        return biopac

    def _get_radar_data(self, participant_id: str, state: str):
        radar = _load_radar_data(
            self.base_path,
            participant_id=participant_id,
            fs=self._sampling_rates,
            channel_mapping=self.BIOPAC_CHANNEL_MAPPING,
            state=state,
            trigger_extraction=self.trigger_data_extraction,
            location=self._get_locations_from_index()[0]
        )

        if self.is_single(None):
            location = self._get_locations_from_index()
            if len(location) > 1:
                raise ValueError("Radar data can be accessed for all or only one single location at the same time.")
            tl = self.timelog
            start = tl[location[0]]["start"][0] + self.biopac_timelog_shift
            end = tl[location[0]]["end"][0] + self.biopac_timelog_shift
            return radar[start:end]

        return radar

    def _get_timelog(self, participant_id: str) -> pd.DataFrame:
        return _load_timelog(self.base_path, participant_id)

    def _get_protocol(self, participant_id: str) -> pd.DataFrame:
        return _load_protocol(self.base_path, participant_id)

    def _get_biopac_timelog_shift(self, participant_id: str):
        return _get_biopac_timelog_shift(
            base_path=self.base_path, participant_id=participant_id, trigger_extraction=self.trigger_data_extraction
        )
