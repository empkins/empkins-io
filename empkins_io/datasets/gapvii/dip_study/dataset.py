from functools import lru_cache
from typing import Dict, Optional, Sequence, Union, Tuple
from itertools import product
import pandas as pd
from tpcp import Dataset
from pathlib import Path
import numpy as np

from empkins_io.utils._types import path_t

from biopsykit.utils.file_handling import get_subject_dirs

from empkins_io.datasets.gapvii.dip_study.helper import _load_general_information


class DipStudyDataset(Dataset):

    base_path: path_t
    use_cache: bool

    _SAMPLING_RATES: Dict[str, int] = {
        "radar": 8000000 / 4096 / 2,
    }

    SUBJECTS_MISSING: Tuple[str] = ("VP_15", "VP_18")

    def __init__(
            self,
            base_path: path_t,
            groupby_cols: Optional[Sequence[str]] = None,
            subset_index: Optional[Sequence[str]] = None,
            use_cache: Optional[bool] = False
    ):
        self.base_path = base_path
        self.use_cache = use_cache

        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self):

        subject_ids = [
            subject_dir.name for subject_dir in get_subject_dirs(self.base_path.joinpath("data_per_subject"), "VP_*")
        ]

        for subject_id in self.SUBJECTS_MISSING:
            if subject_id in subject_ids:
                subject_ids.remove(subject_id)

        phases = ["rest_1", "cpt", "rest_2", "straw", "rest_3"]

        index = list(product(subject_ids, phases))
        index = pd.DataFrame(index, columns=["subject", "phase"])
        return index

    @property
    def sampling_rates(self) -> Dict[str, float]:
        return self._SAMPLING_RATES

    @property
    def condition_order(self):
        if self.is_single(["subject"]):
            return _load_general_information(base_path=self.base_path, column="condition_order")[self.index["subject"][0]]

        return _load_general_information(base_path=self.base_path, column="condition_order")

    @property
    def cpt_duration(self):
        if self.is_single(["subject"]):
            return _load_general_information(base_path=self.base_path, column="cpt_duration")[self.index["subject"][0]]

        return _load_general_information(base_path=self.base_path, column="cpt_duration")

    @property
    def tfm_raw_data(self) -> Dict[str, Dict[str, np.ndarray]]:
        pass
