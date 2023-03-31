from functools import cached_property, lru_cache
from itertools import product
from typing import Optional, Sequence

import pandas as pd
from biopsykit.utils.file_handling import get_subject_dirs

from empkins_io.datasets.d03.macro_ap01._base_dataset import MacroBaseDataset
from empkins_io.datasets.d03.macro_ap01.helper import _get_times_for_mocap, _load_tsst_mocap_data
from empkins_io.utils._types import path_t

_cached_load_mocap_data = lru_cache(maxsize=4)(_load_tsst_mocap_data)


class MacroStudyTsstDataset(MacroBaseDataset):

    SUBJECTS_WITHOUT_MOCAP = (
        "VP_03",
        "VP_31",
    )

    conditions = ("ftsst", "tsst")
    phases = ("prep", "talk", "math")

    def __init__(
        self,
        base_path: path_t,
        groupby_cols: Optional[Sequence[str]] = None,
        subset_index: Optional[Sequence[str]] = None,
        exclude_without_mocap: Optional[bool] = True,
        use_cache: Optional[bool] = True,
    ):
        super().__init__(base_path=base_path, groupby_cols=groupby_cols, subset_index=subset_index)
        # ensure pathlib
        self.base_path = base_path
        self.exclude_without_mocap = exclude_without_mocap
        self.use_cache = use_cache

    def create_index(self):
        subject_ids = [
            subject_dir.name for subject_dir in get_subject_dirs(self.base_path.joinpath("data_per_subject"), "VP_*")
        ]

        if self.exclude_without_mocap:
            for subject_id in self.SUBJECTS_WITHOUT_MOCAP:
                if subject_id in subject_ids:
                    subject_ids.remove(subject_id)

        index = product(subject_ids, self.conditions, self.phases)

        index = pd.DataFrame(
            index,
            columns=["subject", "condition", "phase"],
        )

        return index

    @property
    def ecg(self) -> pd.DataFrame:
        return self._load_ecg_data()

    @property
    def ecg_baseline(self) -> pd.DataFrame:
        return self._load_ecg_data(True)

    @cached_property
    def mocap_data(self) -> pd.DataFrame:
        if self.is_single(None) or self.is_single(["subject", "condition"]):
            subject_id = self.index["subject"][0]
            condition = self.index["condition"][0]
            data, start_time = self._get_mocap_data(subject_id, condition)
            if self.is_single(None):
                phase = self.index["phase"].unique()[0]
            else:
                phase = "total"

            timelog = self.timelog_test
            times = _get_times_for_mocap(timelog, start_time=start_time, phase=phase)
            data_total = {}

            for idx, row in times.iterrows():
                data_total[idx] = data.loc[row["start"] : row["end"]]

            data_total = pd.concat(data_total).droplevel(0)
            return data_total
        raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")

    def _get_mocap_data(self, subject_id: str, condition: str) -> pd.DataFrame:
        if self.use_cache:
            return _cached_load_mocap_data(self.base_path, subject_id, condition)
        return _load_tsst_mocap_data(self.base_path, subject_id, condition)
