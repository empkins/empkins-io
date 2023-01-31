from functools import cached_property, lru_cache
from itertools import product
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from biopsykit.utils.file_handling import get_subject_dirs
from empkins_io.utils._types import path_t

from empkins_io.datasets.d03.macro_ap01._base_dataset import MacroBaseDataset
from empkins_io.datasets.d03.macro_ap01.helper import _load_gait_mocap_data

_cached_load_mocap_data = lru_cache(maxsize=4)(_load_gait_mocap_data)


class MacroStudyGaitDataset(MacroBaseDataset):

    SUBJECTS_WITHOUT_MOCAP = (
        "VP_01",
        "VP_02",
    )

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
        conditions = ["ftsst", "tsst"]
        trial = np.arange(0, 4)
        speed_gait = ["pref", "metro"]

        if self.exclude_without_mocap:
            for subject_id in self.SUBJECTS_WITHOUT_MOCAP:
                if subject_id in subject_ids:
                    subject_ids.remove(subject_id)

        # add tug index
        index = list(product(conditions, ["TUG"], trial, ["pref"]))

        # add gait index
        index.extend(list(product(conditions, ["GAIT"], trial, speed_gait)))

        index = [(subject, *i) for subject, i in product(subject_ids, index)]
        index = pd.DataFrame(index, columns=["subject", "condition", "test", "trial", "speed"])

        # only pref for trial 0
        index = index.drop(index[(index.trial == 0) & (index.speed == "metro")].index)
        index.reset_index(inplace=True, drop=True)

        return index

    @property
    def gait_subset(self):
        return self.get_subset(test="GAIT")

    @property
    def tug_subset(self):
        return self.get_subset(test="TUG")

    @cached_property
    def mocap_data(self) -> pd.DataFrame:
        if self.is_single(["subject", "condition"]):
            subject_id = self.index["subject"][0]
            condition = self.index["condition"][0]
            test = self.index["test"][0]
            trial = self.index["trial"][0]
            speed = self.index["speed"][0]

            data = self._get_mocap_data(subject_id, condition, test, trial, speed)

            return data
        raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")

    def _get_mocap_data(self, subject_id: str, condition: str, test: str, trial: int, speed: str) -> pd.DataFrame:
        if self.use_cache:
            return _cached_load_mocap_data(self.base_path, subject_id, condition, test, trial, speed)
        return _load_gait_mocap_data(self.base_path, subject_id, condition, test, trial, speed)
