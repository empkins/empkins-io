from typing import Dict, Optional, Sequence
import pandas as pd

from empkins_io.datasets.radarstenosis.helper import (
    _get_locations_from_index,
    _load_atimelogger_file,
    _calc_biopac_timelog_shift,
    _load_radar_raw,
    _load_biopac_raw,
    _sync_datasets,
)

from itertools import product
from biopsykit.utils.file_handling import get_subject_dirs

from empkins_io.utils._types import path_t

from tpcp import Dataset


class RadarCardiaStenosisTest(Dataset):

    base_path: path_t
    _SAMPLING_RATES: Dict[str, float] = {
        "radar_original": 8000000 / 4096 / 2,
        "biopac_original": 2000,
        "resampled": 1000,
    }
    BIOPAC_CHANNEL_MAPPING: Dict[str, str] = {
        "ecg": "ecg",
        "sync": "sync",
    }

    def __init__(
        self,
        base_path: path_t,
        groupby_cols: Optional[Sequence[str]] = None,
        subset_index: Optional[Sequence[str]] = None,
    ):
        self.base_path = base_path
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self):
        subject_ids = [
            subject_dir.name for subject_dir in get_subject_dirs(self.base_path.joinpath("data_per_subject"), "VP_*")
        ]

        measurements = [
            "tibialis_post_180",
            "tibialis_post_160",
            "tibialis_post_140",
            "tibialis_post_120",
            "tibialis_post_100",
            "tibialis_post_80",
            "tibialis_post_0",
            "tibialis_pre_140",
            "tibialis_pre_120",
            "tibialis_pre_100",
            "tibialis_pre_80",
            "tibialis_pre_0",
            "radialis_post_120",
            "radialis_post_100",
            "radialis_post_80",
            "radialis_post_0",
            "radialis_pre_140",
            "radialis_pre_120",
            "radialis_pre_100",
            "radialis_pre_80",
            "radialis_pre_0",
            "brachialis_post_120",
            "brachialis_post_100",
            "brachialis_post_80",
            "brachialis_post_0",
            "brachialis_pre_140",
            "brachialis_pre_120",
            "brachialis_pre_100",
            "brachialis_pre_80",
            "brachialis_pre_0",
        ]

        index = list(product(subject_ids, measurements))
        index = pd.DataFrame(index, columns=["subject", "measurement"])

        return index

    @property
    def subject(self) -> str:
        if not self.is_single(["subject"]):
            raise ValueError("Subject can only be accessed for one single participant at once")
        return self.index["subject"][0]

    @property
    def measurement(self) -> str:
        if not self.is_single(["measurement"]):
            raise ValueError("Measurement can only be accessed for a single measurement at once")
        return self.index["measurement"][0]

    @property
    def timelog(self) -> pd.DataFrame:
        if not self.is_single(["subject"]):
            raise ValueError("Timelog can only be accessed for one single participant at once")
        locations = _get_locations_from_index(self.index)
        participant_id = self.index["subject"][0]
        timelog_file_path = self.base_path.joinpath(
            f"data_per_subject/{participant_id}/timelog/processed/{participant_id}_timelog.csv"
        )
        timelog = _load_atimelogger_file(timelog_file_path, timezone="Europe/Berlin")
        return timelog[locations]

    @property
    def biopac_timelog_shift(self):
        shift = _calc_biopac_timelog_shift(self.base_path, self.subject)
        return shift

    @property
    def emrad_raw(self) -> pd.DataFrame:
        # radar data unsynchronized
        if not self.is_single(["subject"]):
            raise ValueError("Radar data can only be accessed for one single participant at once")

        radar = _load_radar_raw(self.base_path, self.subject, self._SAMPLING_RATES["radar_original"])
        return radar

    @property
    def biopac_raw(self) -> pd.DataFrame:
        # biopac data unsynchronized
        if not self.is_single(["subject"]):
            raise ValueError("BIOPAC data can only be accessed for one single participant at once")

        biopac = _load_biopac_raw(self.base_path, self.subject, self.BIOPAC_CHANNEL_MAPPING)
        tl = self.timelog
        return biopac

    @property
    def emrad_synced(self) -> pd.DataFrame:
        # radar data synchronized
        data_path = self.base_path.joinpath(
            f"data_per_subject/{self.subject}/emrad/processed/{self.subject}_emrad_data.h5"
        )
        if not self.is_single(["subject"]):
            raise ValueError("Radar data can only be accessed for one single participant at once")

        self._ensure_synced()
        data = pd.read_hdf(data_path, key=f"emrad_data")
        return data

    @property
    def biopac_synced(self) -> pd.DataFrame:
        # biopac data synchronized
        if not self.is_single(["subject"]):
            raise ValueError("BIOPAC data can only be accessed for one single participant at once")

        data_path = self.base_path.joinpath(
            f"data_per_subject/{self.subject}/biopac/processed/{self.subject}_biopac_data.h5"
        )
        self._ensure_synced()
        data = pd.read_hdf(data_path, key=f"biopac_data")
        return data

    def _ensure_synced(self) -> None:
        radar_path = self.base_path.joinpath(
            f"data_per_subject/{self.subject}/emrad/processed/{self.subject}_emrad_data.h5"
        )
        biopac_path = self.base_path.joinpath(
            f"data_per_subject/{self.subject}/biopac/processed/{self.subject}_biopac_data.h5"
        )
        if radar_path.exists() and biopac_path.exists():
            return
        else:
            synced_datasets = _sync_datasets(
                self.base_path, self.subject, self.BIOPAC_CHANNEL_MAPPING, self._SAMPLING_RATES
            )
            synced_datasets.datasets_aligned["radar_aligned_"].to_hdf(
                radar_path, mode="w", key="emrad_data", index=True
            )
            synced_datasets.datasets_aligned["biopac_aligned_"].to_hdf(
                biopac_path, mode="w", key="biopac_data", index=True
            )
