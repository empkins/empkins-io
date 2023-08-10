import json
from functools import cached_property, lru_cache
from itertools import product
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
import pandas as pd
from tpcp import Dataset
import pandas as pd

from biopsykit.io import load_long_format_csv
from biopsykit.utils.dataframe_handling import multi_xs, wide_to_long
from biopsykit.utils.file_handling import get_subject_dirs

from empkins_io.datasets.radarcardia.study.helper import (
    _load_biopac_data,
    _load_radar_data,
    _load_timelog
)

from empkins_io.sync import SyncedDataset

from empkins_io.utils._types import path_t

class RadarCardiaStudyDataset(Dataset):

    base_path: path_t
    use_cache: bool
    _sampling_rates: Dict[str, float] = {
        "radar_original": 2400,
        "biopac_original": 2000,
        "resampled": 1000
    }
    _datasets_aligned: Dict[str, pd.DataFrame]

    BIOPAC_CHANNEL_MAPPING: Dict[str, str] = {
        "ECG (.05 - 150 Hz)": "ecg",
        "Cardiac Output - Z": "icg",
        "Cardiac Output - dZ/dt": "icg_der",
        "Digital input": "sync"
    }

    def __init__(
            self,
            base_path: path_t,
            groupby_cols: Optional[Sequence[str]] = None,
            subset_index: Optional[Sequence[str]] = None,
            use_cache: bool = True
    ):
        self.base_path = base_path
        self.use_cache = use_cache
        self._datasets_aligned = None
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self):
        participant_ids = [
            participant_dir.name for participant_dir in get_subject_dirs(self.base_path.joinpath("data_per_subject"), "VP_*")
        ]

        '''
        measurement_points = ["elbow", "wrist"]
        measurement_points_upper_body = ["jugular", "abdominal_aorta"]
        breathing = ["normal", "hold"]

        index = list(product(measurement_points, ["normal"]))
        index_upper_body = list(product(measurement_points_upper_body, breathing))

        index.extend(index_upper_body)

        index = [(participant, *i) for participant, i in product(participant_ids, index)]
        index = pd.DataFrame(index, columns=["participant", "location", "breathing"])
        '''

        # pre study

        return index

    @property
    def sampling_rates(self) -> Dict[str, float]:
        return self._sampling_rates

    @property
    def biopac_synced(self) -> pd.DataFrame:
        if self.is_single(None):
            raise NotImplementedError(
                "Accessing BIOPAC data for a single participant-location-breathing combination is not implemented yet"
            )

        if self.is_single(["participant", "location"]):
            raise ValueError("BIOPAC data can only be accessed for one single participant at once")

        if self.is_single(["participant"]):
            if self._datasets_aligned is None:
                participant_id = self.index["participant"][0]
                self._load_and_sync_datasets(participant_id)
            return self._datasets_aligned["biopac_aligned_"].copy()

        raise ValueError("BIOPAC data can only be accessed for one single participant at once")

    @property
    def emrad_synced(self) -> pd.DataFrame:
        if self.is_single(None):
            raise NotImplementedError(
                "Accessing Radar data for a single participant-location-breathing combination is not implemented yet"
            )

        if self.is_single(["participant", "location"]):
            raise ValueError("Radar data can only be accessed for one single participant at once")

        if self.is_single(["participant"]):
            if self._datasets_aligned is None:
                participant_id = self.index["participant"][0]
                self._load_and_sync_datasets(participant_id)
            return self._datasets_aligned["radar_aligned_"].copy()

        raise ValueError("Radar data can only be accessed for one single participant at once")




    @property
    def timelog(self) -> pd.DataFrame:

        print(self.is_single(["participant", "location"]))

        # if self.is_single(None):
        #     raise NotImplementedError(
        #         "Accessing Timelog for a single participant-location-breathing combination is not implemented yet"
        #     )
        #
        # if self.is_single(["participant", "location"]):
        #     raise ValueError("Timelog can only be accessed for one single participant at once")
        #
        # if self.is_single(["participant"]):
        #     participant_id = self.index["participant"][0]
        #     return self._get_timelog(participant_id, exclude_fail=True)
        #
        # raise ValueError("Timelog can only be accessed for one single participant at once")

    @property
    def timelog_all(self) -> pd.DataFrame:
        if self.is_single(None):
            raise NotImplementedError(
                "Accessing Timelog for a single participant-location-breathing combination is not implemented yet"
            )

        if self.is_single(["participant", "location"]):
            raise ValueError("Timelog can only be accessed for one single participant at once")

        if self.is_single(["participant"]):
            participant_id = self.index["participant"][0]
            return self._get_timelog(participant_id, exclude_fail=False)

        raise ValueError("Timelog can only be accessed for one single participant at once")

    def _load_and_sync_datasets(self, participant_id: str) -> pd.DataFrame:
        biopac, _ = _load_biopac_data(
            self.base_path,
            participant_id=participant_id,
            channel_mapping=self.BIOPAC_CHANNEL_MAPPING
        )

        radar, _ = _load_radar_data(
            self.base_path,
            participant_id=participant_id,
            fs=self._sampling_rates["radar_original"]
        )

        synced_dataset = SyncedDataset(sync_type="m-sequence")
        synced_dataset.add_dataset("biopac", data=biopac, sync_channel_name="sync",
                                   sampling_rate=self._sampling_rates["biopac_original"])
        synced_dataset.add_dataset("radar", data=radar, sync_channel_name="Sync_Out",
                                   sampling_rate=self._sampling_rates["radar_original"])
        synced_dataset.resample_datasets(fs_out=self._sampling_rates["resampled"], method="dynamic", wave_frequency=10)
        synced_dataset.align_and_cut_m_sequence(
            primary="radar", reset_time_axis=True, cut_to_shortest=True
        )
        self._datasets_aligned = synced_dataset.datasets_aligned

    def _get_timelog(self, participant_id: str, exclude_fail: bool) -> pd.DataFrame:
        return _load_timelog(self.base_path, participant_id, exclude_fail)
