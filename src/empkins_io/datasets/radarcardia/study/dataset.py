from itertools import product
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd
from biopsykit.utils.file_handling import get_subject_dirs

from empkins_io.datasets.radarcardia.base.dataset import BaseDataset
from empkins_io.utils._types import path_t


class RadarCardiaStudyDataset(BaseDataset):
    exclude_ecg_seg_failed: bool
    exclude_ecg_corrupted: bool

    BIOPAC_CHANNEL_MAPPING: Dict[str, str] = {
        "ecg": "ecg",
        "icg": "icg_der",
        "ppg": "ppg",
        "pcg": "pcg",
        "sync": "sync",
    }

    SUBJECTS_ECG_SEG_FAILED: Tuple[str] = ("VP_04",)

    SUBJECT_ECG_CORRUPTED: Tuple[Tuple] = (
        ("VP_17", "baseline", "normal"),
        ("VP_17", "temporalis", "normal"),
    )

    def __init__(
        self,
        base_path: path_t,
        groupby_cols: Optional[Sequence[str]] = None,
        subset_index: Optional[Sequence[str]] = None,
        use_cache: Optional[bool] = False,
        calc_biopac_timelog_shift: Optional[bool] = True,
        trigger_data_extraction: Optional[bool] = False,
        exclude_ecg_seg_failed: Optional[bool] = True,
        exclude_ecg_corrupted: Optional[bool] = True,
    ):
        self.exclude_ecg_seg_failed = exclude_ecg_seg_failed
        self.exclude_ecg_corrupted = exclude_ecg_corrupted

        super().__init__(
            base_path=base_path,
            groupby_cols=groupby_cols,
            subset_index=subset_index,
            use_cache=use_cache,
            calc_biopac_timelog_shift=calc_biopac_timelog_shift,
            trigger_data_extraction=trigger_data_extraction,
        )

    def create_index(self):
        subject_ids = [
            subject_dir.name for subject_dir in get_subject_dirs(self.base_path.joinpath("data_per_subject"), "VP_*")
        ]

        if self.exclude_ecg_seg_failed:
            for subject_id in self.SUBJECTS_ECG_SEG_FAILED:
                if subject_id in subject_ids:
                    subject_ids.remove(subject_id)

        breathing = ["normal", "hold"]
        body_parts_1 = [
            "baseline",
            "temporalis",
            "carotis",
            "brachialis",
            "radialis_prox",
            "radialis_med",
            "radialis_dist",
            "dorsalis_pedis",
            "poplitea",
        ]
        body_parts_2 = ["aorta_prox", "aorta_med", "aorta_dist"]

        index = list(product(body_parts_1, ["normal"]))
        index.extend(list(product(body_parts_2, breathing)))
        index = [(subject, *i) for subject, i in product(subject_ids, index)]

        if self.exclude_ecg_corrupted:
            for subject_id, body_part, breathing in self.SUBJECT_ECG_CORRUPTED:
                if (subject_id, body_part, breathing) in index:
                    index.remove((subject_id, body_part, breathing))

        index = pd.DataFrame(index, columns=["subject", "location", "breathing"])

        return index
