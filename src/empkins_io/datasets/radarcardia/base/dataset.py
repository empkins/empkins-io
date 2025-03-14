from functools import lru_cache
from typing import Dict, Optional, Sequence, Union
import pandas as pd
from tpcp import Dataset
from pathlib import Path


from empkins_io.datasets.radarcardia.base.helper import (
    _load_biopac_data,
    _load_radar_data,
    _load_timelog,
    _build_timelog_path,
    _build_protocol_path,
    _get_biopac_timelog_shift,
    _load_protocol,
    _save_data_to_location_h5,
    _load_data_from_location_h5,
    _load_apnea_segmentation,
    _load_visual_segmentation,
    _load_flipping,
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

    _SAMPLING_RATES: Dict[str, float] = {"radar_original": 8000000 / 4096, "biopac_original": 2000, "resampled": 1000}

    BIOPAC_CHANNEL_MAPPING: Dict[str, str] = {
        "ECG (.05 - 150 Hz)": "ecg",
        "Cardiac Output - Z": "icg",
        "Cardiac Output - dZ/dt": "icg_der",
        "Sync": "sync",
    }

    def __init__(
        self,
        base_path: path_t,
        groupby_cols: Optional[Sequence[str]] = None,
        subset_index: Optional[Sequence[str]] = None,
        use_cache: Optional[bool] = False,
        calc_biopac_timelog_shift: Optional[bool] = True,
        trigger_data_extraction: Optional[bool] = False,
    ):
        """
        Creates new instance of the BaseDataset class
        Args:
            base_path: path_t (path to the base directory of the dataset)
            groupby_cols: Optional[Sequence[str]]
            subset_index: Optional[Sequence[str]]
            use_cache: Optional[bool] (currently not implemented)
            calc_biopac_timelog_shift: Optional[bool] (flag indicating whether to calculate and use the time shift between the
                BIOPAC event marker and the start of the sync timelog entry to correct start and end times)
            trigger_data_extraction: Optional[bool] (flag indicating whether to trigger the data extraction from the
                BIOPAC and radar data)
        Returns:
        """
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
        return self._SAMPLING_RATES

    @property
    def biopac_timelog_shift(self):
        """
        Returns the time shift between the BIOPAC event marker and the start of the sync timelog entry.
        This shift is necessary to synchronize the BIOPAC data with the timelog.
        Args:
        Returns:
            bp_tl_shift: pd.Timedelta
        """
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
        """
        Returns the raw BIOPAC data without any synchronization with the emrad dataset
        Args:
        Returns:
            biopac_data: pd.DataFrame
            sampling_rate: float
        """
        if not self.is_single(["subject"]):
            raise ValueError("BIOPAC data can only be accessed for one single participant at once")

        participant_id = self.index["subject"][0]
        biopac_data = self._get_biopac_data(participant_id=participant_id, state="raw_unsynced")

        return biopac_data, self.sampling_rates["biopac_original"]

    @property
    def emrad_raw_unsynced(self) -> tuple[pd.DataFrame, float]:
        """
        Returns the raw radar data without any synchronization with the BIOPAC dataset
        Args:
        Returns:
            radar_data: pd.DataFrame
            sampling_rate: float
        """
        if not self.is_single(["subject"]):
            raise ValueError("Radar data can only be accessed for one single participant at once")

        participant_id = self.index["subject"][0]
        radar_data = self._get_radar_data(participant_id=participant_id, state="raw_unsynced")

        return radar_data, self.sampling_rates["radar_original"]

    @property
    def biopac_raw_synced(self) -> tuple[pd.DataFrame, float]:
        """
        Returns the raw BIOPAC data synchronized with the emrad dataset
        Args:
        Returns:
            biopac_data: pd.DataFrame
            sampling_rate: float
        """
        if not self.is_single(["subject"]):
            raise ValueError("BIOPAC data can only be accessed for one single participant at once")

        participant_id = self.index["subject"][0]
        biopac_data = self._get_biopac_data(participant_id=participant_id, state="raw_synced")

        return biopac_data, self.sampling_rates["resampled"]

    @property
    def emrad_raw_synced(self) -> tuple[pd.DataFrame, float]:
        """
        Returns the raw radar data synchronized with the BIOPAC dataset
        Args:
        Returns:
            radar_data: pd.DataFrame
            sampling_rate: float
        """
        if not self.is_single(["subject"]):
            raise ValueError("Radar data can only be accessed for one single participant at once")

        participant_id = self.index["subject"][0]
        radar_data = self._get_radar_data(participant_id=participant_id, state="raw_synced")

        return radar_data, self.sampling_rates["resampled"]

    @property
    def biopac_data(self) -> tuple[pd.DataFrame, float]:
        """
        Returns the BIOPAC data synchronized with the emrad dataset cut for a single measurement location
        Args:
        Returns:
            biopac_data: pd.DataFrame
            sampling_rate: float
        """
        if not self.is_single(None):
            raise ValueError("BIOPAC data can only be accessed for one single location at once")

        participant_id = self.index["subject"][0]
        biopac_data = self._get_biopac_data(participant_id=participant_id, state="location_synced")

        return biopac_data, self.sampling_rates["resampled"]

    @property
    def emrad_data(self) -> tuple[pd.DataFrame, float]:
        """
        Returns the radar data synchronized with the BIOPAC dataset cut for a single measurement location
        Args:
        Returns:
            radar_data: pd.DataFrame
            sampling_rate: float
        """
        if not self.is_single(None):
            raise ValueError("Radar data can only be accessed for one single location at once")

        participant_id = self.index["subject"][0]
        radar_data = self._get_radar_data(participant_id=participant_id, state="location_synced")

        return radar_data, self.sampling_rates["resampled"]

    @property
    def timelog(self) -> pd.DataFrame:
        """
        Return the timelog data for the currently selected dataset, if only one participant is selected
        Args:
        Returns:
            timelog: pd.DataFrame
        """
        if not self.is_single(["subject"]):
            raise ValueError("Timelog can only be accessed for one single participant at once")
        locations = self._get_locations_from_index()
        participant_id = self.index["subject"][0]
        tl = self._get_timelog(participant_id)
        return tl[locations]

    @property
    def timelog_all(self) -> pd.DataFrame:
        """
        Returns the all timelog entries for the current participant
        Args:
        Returns:
            timelog: pd.DataFrame
        """
        if not self.is_single(["subject"]):
            raise ValueError("Timelog can only be accessed for one single participant at once")
        participant_id = self.index["subject"][0]
        return self._get_timelog(participant_id)

    @property
    def protocol(self) -> pd.DataFrame:
        """
        Returns the study protocol information for the current participant
        Args:
        Returns:
            protocol: pd.DataFrame
        """
        if not self.is_single(["subject"]):
            raise ValueError("Protocol Information can only be accessed for one single participant at once")
        participant_id = self.index["subject"][0]
        return self._get_protocol(participant_id)

    @property
    def apnea_segmentation(self) -> Dict:
        """
        Returns the apnea segmentation for the current participant. Apnea segmentations are only available for
        measurements at the aorta where the breath holding is performed.
        Args:
        Returns:
            apnea_seg: Dict
        """
        if not self.is_single(["subject"]):
            raise ValueError("Apnea Segmentation can only be accessed for one single participant at once")
        participant_id = self.index["subject"][0]
        apnea_seg = self._get_apnea_segmentation(participant_id)
        return apnea_seg

    @property
    def timelog_path(self) -> Path:
        """
        Returns the path to the timelog file for the current participant
        Args:
        Returns:
            timelog_path: Path
        """
        if not self.is_single(["subject"]):
            raise ValueError("Timelog path can only be accessed for one single participant at once")
        participant_id = self.index["subject"][0]
        timelog_path = _build_timelog_path(base_path=self.base_path, participant_id=participant_id)
        return timelog_path

    @property
    def protocol_path(self) -> Path:
        """
        Returns the path to the protocol file for the current participant
        Args:
        Returns:
            protocol_path: Path
        """
        if not self.is_single(["subject"]):
            raise ValueError("Protocol path can only be accessed for one single participant at once")
        participant_id = self.index["subject"][0]
        protocol_path = _build_protocol_path(base_path=self.base_path, participant_id=participant_id)
        return protocol_path

    @property
    def visual_segmentation(self) -> pd.DataFrame:
        """
        Returns the visual heart sound segmentation for the current participant
        Args:
        Returns:
            visual_seg: pd.DataFrame
        """
        if not self.is_single(["subject"]):
            raise ValueError("Visual Inspection Segmentation can only be accessed for one single participant at once")
        participant_id = self.index["subject"][0]
        data = self._get_visual_segmentation(participant_id)
        if self.is_single(None):
            loc = self._get_locations_from_index()[0]
            return data.loc[loc]
        return data

    @property
    def biopac_flipping(self) -> pd.DataFrame:
        """
        Returns the BIOPAC flipping information for the current participant
        Args:
        Returns:
            flipping_data: pd.DataFrame
        """
        flipping_data = self._get_flipping("biopac")
        if self.is_single(["subject"]):
            participant_id = self.index["subject"][0]
            return flipping_data[participant_id]
        return flipping_data

    @property
    def emrad_flipping(self) -> pd.DataFrame:
        """
        Returns the radar flipping information for the current participant
        Args:
        Returns:
            flipping_data: pd.DataFrame
        """
        flipping_data = self._get_flipping("emrad")
        if self.is_single(["subject"]):
            participant_id = self.index["subject"][0]
            return flipping_data[participant_id]
        return flipping_data

    def save_data_to_location(self, data: pd.DataFrame, file_name: str, sub_dir: Optional[str] = None):
        """
        Save a dataframe as file to "data_per_location" sub-folder for the respective participant. In this folder,
        all intermediate data can be stored, e.g. the results of the preprocessing steps.
        Args:
            data: pd.DataFrame (data frame containing the data to be saved)
            file_name: str (name of the file to be saved)
            sub_dir: Optional[str] (path to subdirectory in "data_per_location", e.g., "ensemble_averaging/all")
        Returns:
        """
        locations = self.index.drop(columns="subject").columns.tolist()
        if not self.is_single(locations):
            raise ValueError("Data can only be saved for a single location-breathing combination")

        participant_id = self.index["subject"][0]
        location = self._get_locations_from_index()[0]
        _save_data_to_location_h5(
            base_path=self.base_path,
            participant_id=participant_id,
            data=data,
            location=location,
            file_name=file_name,
            sub_dir=sub_dir,
        )

    def load_data_from_location(self, file_name: str, sub_dir: Optional[str] = None):
        """
        Load a dataframe from a file in the "data_per_location" sub-folder for the respective participant.
        Args:
            file_name: str (name of the file to be loaded)
            sub_dir: Optional[str] (path to subdirectory in "data_per_location", e.g., "ensemble_averaging/all")
        Returns:
            data: pd.DataFrame (data frame containing the loaded data)
        """
        locations = self.index.drop(columns="subject").columns.tolist()
        if not self.is_single(locations):
            raise ValueError("Data can only be loaded for a single location-breathing combination")

        participant_id = self.index["subject"][0]
        location = self._get_locations_from_index()[0]
        data = _load_data_from_location_h5(
            base_path=self.base_path,
            participant_id=participant_id,
            location=location,
            file_name=file_name,
            sub_dir=sub_dir,
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
            fs=self._SAMPLING_RATES,
            channel_mapping=self.BIOPAC_CHANNEL_MAPPING,
            state=state,
            trigger_extraction=self.trigger_data_extraction,
            location=self._get_locations_from_index()[0],
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
            fs=self._SAMPLING_RATES,
            channel_mapping=self.BIOPAC_CHANNEL_MAPPING,
            state=state,
            trigger_extraction=self.trigger_data_extraction,
            location=self._get_locations_from_index()[0],
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

    def _get_apnea_segmentation(self, participant_id: str) -> Dict:
        loc = self._get_locations_from_index()[0]

        if loc in ["aorta_prox_hold", "aorta_med_hold", "aorta_dist_hold"]:
            apnea_seg = _load_apnea_segmentation(self.base_path, participant_id)
            return apnea_seg[loc]
        else:
            raise ValueError("Apnea Segmentation is only available for hold measurements")

    def _get_visual_segmentation(self, participant_id: str) -> pd.DataFrame:
        return _load_visual_segmentation(self.base_path, participant_id)

    def _get_flipping(self, modality: str) -> pd.DataFrame:
        return _load_flipping(self.base_path, modality)
