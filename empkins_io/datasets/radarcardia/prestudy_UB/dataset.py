import json
from functools import cached_property, lru_cache
from itertools import product
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
import pandas as pd

from empkins_io.datasets.radarcardia.base.dataset import BaseDataset

from biopsykit.io import load_long_format_csv
from biopsykit.utils.dataframe_handling import multi_xs, wide_to_long
from biopsykit.utils.file_handling import get_subject_dirs

from empkins_io.utils._types import path_t

"""
Dataset for Radar and BIOPAC data recorded in Erlangen at 2023-08-11
Pre Study data for upper body
"""


class RadarCardiaPreStudyUBDataset(BaseDataset):

    def __init__(
            self,
            base_path: path_t,
            groupby_cols: Optional[Sequence[str]] = None,
            subset_index: Optional[Sequence[str]] = None,
            use_cache: bool = True
    ):

        super().__init__(base_path=base_path, groupby_cols=groupby_cols, subset_index=subset_index, use_cache=use_cache)

    def create_index(self):
        participant_ids = [
            participant_dir.name for participant_dir in get_subject_dirs(self.base_path.joinpath("data_per_subject"), "VP_*")
        ]

        body_parts = ["carotis_collarbone", "jugulum", "sternum", "intercostal_space", "rib", "abdominal_aorta"]
        index = list(product(body_parts, ["normal"]))

        index = [(participant, *i) for participant, i in product(participant_ids, index)]
        index = pd.DataFrame(index, columns=["participant", "location", "breathing"])

        return index
