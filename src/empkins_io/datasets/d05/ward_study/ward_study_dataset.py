from pathlib import Path
from collections.abc import Sequence

import pandas as pd
from tpcp import Dataset

from empkins_io.utils._types import path_t

from .helper import (
    get_raw_data_file,
    get_ipos_external
)

class WardStudyDataset(Dataset):
    base_path: path_t

    IPOS_E_COLS = ["record_id", "ipos_e_time", "ipos_e_pain", "ipos_e_dyspnea", "ipos_e_weakness", "ipos_e_nausea", "ipos_e_vomit",
                 "ipos_e_appetiteloss", "ipos_e_constipation", "ipos_e_drymouth", "ipos_e_sleep", "ipos_e_mobility"]

    def __init__(
        self,
        base_path: path_t,
        groupby_cols: Sequence[str] | None = None,
        subset_index: Sequence[str] | None = None,
    ):

        self.base_path = base_path

        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self) -> pd.DataFrame:

        _max_participants = 75

        subjects = []

        for i in range(1, _max_participants + 1):
            subjects.append(f"EMP_1{i:03d}")

        index = pd.DataFrame(subjects, columns=["subject"])

        return index

    @property
    def subject(self) -> str:
        if self.is_single(None):
            return self.index.iloc[0]["subject"]

    @property
    def record_id(self) -> int:
        if self.is_single(None):
            subject = self.subject.split("_")[1]
            # remove leading 1
            subject = subject[1:] if subject.startswith("1") else subject
            return int(subject)

    @property
    def raw_data(self) -> pd.DataFrame:
        return get_raw_data_file(self.base_path)

    @property
    def ipos_external(self) -> pd.DataFrame:
        data = get_ipos_external(self.base_path, self.IPOS_E_COLS)
        if self.is_single(None):
            data = data.loc[data.record_id == self.record_id].reset_index(drop=True)
            return data
        else:
            return data







