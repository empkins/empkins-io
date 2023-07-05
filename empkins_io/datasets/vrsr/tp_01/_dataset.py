from typing import Optional, Sequence

import pandas as pd
from biopsykit.utils.file_handling import get_subject_dirs
from tpcp import Dataset

from empkins_io.utils._types import path_t


class VRSR_Dataset(Dataset):

    MISSING_DATA = {}

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
            subject_dir.name for subject_dir in get_subject_dirs(self.base_path, "*")
        ]
        if self.exclude_missing_data:
            for missing_type, sids in self.MISSING_DATA.items():
                for sid in sids:
                    if sid in subject_ids:
                        subject_ids.remove(sid)

        index = pd.DataFrame(subject_ids, columns=["subject"])
        return index