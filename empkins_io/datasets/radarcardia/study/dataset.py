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


class RadarCardiaStudyDataset(BaseDataset):

    BIOPAC_CHANNEL_MAPPING: Dict[str, str] = {
        "ecg": "ecg",
        "icg": "icg_der",
        "ppg": "ppg",
        "pcg": "pcg",
        "sync": "sync"
    }

    def __init__(
            self,
            base_path: path_t,
            groupby_cols: Optional[Sequence[str]] = None,
            subset_index: Optional[Sequence[str]] = None,
            use_cache: Optional[bool] = False,
            calc_biopac_timelog_shift: Optional[bool] = True,
            trigger_data_extraction: Optional[bool] = False

    ):
        super().__init__(
            base_path=base_path,
            groupby_cols=groupby_cols,
            subset_index=subset_index,
            use_cache=use_cache,
            calc_biopac_timelog_shift=calc_biopac_timelog_shift,
            trigger_data_extraction=trigger_data_extraction
        )

    def create_index(self):
        subject_ids = [
            subject_dir.name for subject_dir in get_subject_dirs(self.base_path.joinpath("data_per_subject"), "VP_*")
        ]

        breathing = ["normal", "hold"]
        body_parts_1 = ["baseline", "temporalis", "carotis", "brachialis", "radialis_prox", "radialis_med",
                        "radialis_dist", "dorsalis_pedis", "poplitea"]
        body_parts_2 = ["aorta_prox", "aorta_med", "aorta_dist"]

        index = list(product(body_parts_1, ["normal"]))
        index.extend(list(product(body_parts_2, breathing)))
        index = [(subject, *i) for subject, i in product(subject_ids, index)]
        index = pd.DataFrame(index, columns=["subject", "location", "breathing"])

        return index
