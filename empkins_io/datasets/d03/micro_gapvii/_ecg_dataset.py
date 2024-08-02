import warnings
from itertools import product

import pandas as pd
from biopsykit.utils.file_handling import get_subject_dirs

from empkins_io.datasets.d03.micro_gapvii import MicroBaseDataset


class MicroEcgDataset(MicroBaseDataset):

    PHASE_ECG = ["Baseline", "Prep", "Talk", "Math", "Recovery"]

    def create_index(self):
        # data is located in a folder named "Data" and data per participant is located in folders named "VP_xx"

        participant_ids = [
            participant_dir.name
            for participant_dir in get_subject_dirs(
                self.base_path.joinpath("data_per_subject"), "VP_[0-9]{3}"
            )
        ]

        # excludes participants where parts of data are missing
        if self.exclude_missing_data:
            for p_id in self.MISSING_DATA:
                if p_id in participant_ids:
                    participant_ids.remove(p_id)

        index = list(product(participant_ids, self.CONDITIONS, self.PHASE_ECG))
        index = pd.DataFrame(index, columns=["subject", "condition", "phase"])
        return index
