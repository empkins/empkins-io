from itertools import product
from typing import Dict, Optional, Sequence
import pandas as pd

from empkins_io.datasets.radarcardia.base.dataset import BaseDataset

from biopsykit.utils.file_handling import get_subject_dirs

from empkins_io.utils._types import path_t

"""
Dataset for Radar and BIOPAC data recorded in Hamburg at 2023-04-25
"""


class RadarCardiaPreStudyHDataset(BaseDataset):

    BIOPAC_CHANNEL_MAPPING: Dict[str, str] = {
        "ECG (.05 - 150 Hz)": "ecg",
        "Cardiac Output - Z": "icg",
        "Cardiac Output - dZ/dt": "icg_der",
        "Digital input": "sync",
    }

    _SAMPLING_RATES: Dict[str, float] = {"radar_original": 2400, "biopac_original": 2000, "resampled": 1000}

    def __init__(
        self,
        base_path: path_t,
        groupby_cols: Optional[Sequence[str]] = None,
        subset_index: Optional[Sequence[str]] = None,
        use_cache: bool = True,
    ):

        super().__init__(base_path=base_path, groupby_cols=groupby_cols, subset_index=subset_index, use_cache=use_cache)

    def create_index(self):
        participant_ids = [
            participant_dir.name
            for participant_dir in get_subject_dirs(self.base_path.joinpath("data_per_subject"), "VP_*")
        ]
        breathing = ["normal", "hold"]
        # front
        body_parts_1 = ["heart", "sternum", "belly_button"]
        body_parts_2 = ["thigh_left", "elbow_left"]

        index = list(product(body_parts_1, breathing))
        index.extend(list(product(body_parts_2, ["normal"])))
        index = [(pos, *i) for pos, i in product(["front"], index)]

        # back
        body_parts_1 = ["heart", "sternum", "L5"]
        body_parts_2 = ["knee", "palm"]
        index_back = list(product(body_parts_1, breathing))
        index_back.extend(list(product(body_parts_2, ["normal"])))
        index_back = [(pos, *i) for pos, i in product(["back"], index_back)]

        # both
        index.extend(index_back)
        index = [(participant, *i) for participant, i in product(participant_ids, index)]
        index = pd.DataFrame(index, columns=["participant", "position", "location", "breathing"])

        return index
