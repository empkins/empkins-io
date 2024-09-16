from functools import lru_cache
from typing import Dict, Optional, Sequence, Union, Tuple
from itertools import product
import pandas as pd
from tpcp import Dataset
from pathlib import Path
import numpy as np

from empkins_io.sensors.tfm.tfm import TfmLoader
from empkins_io.utils._types import path_t

from biopsykit.utils.file_handling import get_subject_dirs

from empkins_io.datasets.gapvii.dip_study.helper import _load_general_information, _update_dates, _load_single_date


class DipStudyDataset(Dataset):

    base_path: path_t
    use_cache: bool

    _SAMPLING_RATES: Dict[str, int] = {
        "radar": 8000000 / 4096 / 2,
    }

    SUBJECTS_MISSING: Tuple[str] = ("VP_15", "VP_18")

    DEF_DATE = "01.01.1970"

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

    def _get_ids(self):
        subject_ids = [
            subject_dir.name for subject_dir in get_subject_dirs(self.base_path.joinpath("data_per_subject"), "VP_*")
        ]
        for subject_id in self.SUBJECTS_MISSING:
            if subject_id in subject_ids:
                subject_ids.remove(subject_id)
        return subject_ids

    def create_index(self):
        subject_ids = self._get_ids()

        phases = ["rest_1", "cpt", "rest_2", "straw", "rest_3"]
        phases_mapping = {
            "rest_1": "Ruhe 1",
            "cpt": "CPT",
            "rest_2": "Ruhe 2",
            "straw": "Atmung",
            "rest_3": "Ruhe 3"
        }

        index = list(product(subject_ids, phases))
        index = pd.DataFrame(index, columns=["subject", "phase"])
        return index


    def fill_dates(self):
        subject_ids = self._get_ids()
        subject_date_dict = {}

        # Load dates for all subjects from protocol files
        for subject in subject_ids:
            date = _load_single_date(self.base_path, subject)
            subject_date_dict[subject] = date

        # Fill dates to empkins_dip_study.xlsx
        _update_dates(self.base_path, subject_date_dict, "Sheet1")

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
    def subject(self) -> str:
        if self.is_single(["subject"]):
            return self.index["subject"][0]
        return None

    @property
    def tfm_raw_data(self) -> Dict[str, Dict[str, np.ndarray]]:
        tfm_path = self.base_path.joinpath("")
        tfm_loader = TfmLoader.from_mat_file(path=tfm_path)
        return tfm_loader

    @property
    def date(self) -> str:
        try:
            if self.is_single(["subject"]):
                return _load_general_information(base_path=self.base_path, column="date")[self.index["subject"][0]]

            return _load_general_information(base_path=self.base_path, column="date")
        except KeyError:
            # Handle the KeyError, e.g., return a default value or log an error
            print(f"Date was not found in empkins_dip_study.xlsx for subject {self.index['subject'][0]}")
            print("Please check its protocol file and run fill_dates() method.")
            return self.DEF_DATE  # or handle the error as needed
