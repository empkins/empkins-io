from functools import lru_cache
from typing import Optional, Sequence

import pandas as pd
from biopsykit.utils.file_handling import get_subject_dirs
from tpcp import Dataset

from empkins_io.datasets.vrsr.tp_01.helper import _load_ecg_data, _load_raw_log
from empkins_io.utils._types import path_t

_cached_load_ecg_data = lru_cache(maxsize=4)(_load_ecg_data)


class VRSR_Dataset(Dataset):
    MISSING_DATA = {}

    ECG_SAMPLING_RATE = 256

    def __init__(
        self,
        base_path: path_t,
        groupby_cols: Optional[Sequence[str]] = None,
        subset_index: Optional[Sequence[str]] = None,
        exclude_without_mocap: Optional[bool] = True,
        exclude_missing_data: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ):
        # ensure pathlib
        self.base_path = base_path
        self.exclude_without_mocap = exclude_without_mocap
        self.exclude_missing_data = exclude_missing_data
        self.use_cache = use_cache
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self):
        subject_ids = [
            subject_dir.name for subject_dir in get_subject_dirs(self.base_path.joinpath("data_per_subject"), "\\d{3}")
        ]

        if self.exclude_missing_data:
            for _missing_type, sids in self.MISSING_DATA.items():
                for sid in sids:
                    if sid in subject_ids:
                        subject_ids.remove(sid)

        index = pd.DataFrame(subject_ids, columns=["subject"])
        return index

    @property
    def subject(self) -> str:
        if not self.is_single("subject"):
            raise ValueError("Dataset is not single-subject dataset!")

        return self.index["subject"].iloc[0]

    @property
    def ecg_data(self) -> pd.DataFrame:
        if not self.is_single("subject"):
            raise ValueError("Dataset is not single-subject dataset!")

        if self.use_cache:
            return _cached_load_ecg_data(self.base_path, self.subject)
        return _load_ecg_data(self.base_path, self.subject)

    @property
    def raw_log(self) -> pd.DataFrame:
        if not self.is_single("subject"):
            raise ValueError("Dataset is not single-subject dataset!")

        return _load_raw_log(self.base_path, self.subject)

    @property
    def cleaned_log(self) -> pd.DataFrame:
        if not self.is_single("subject"):
            raise ValueError("Dataset is not single-subject dataset!")

        return pd.read_csv(
            self.base_path.joinpath(
                "data_per_subject",
                self.subject,
                f"{self.subject}_timelog_cleaned.csv",
            ),
            index_col=0,
        )
