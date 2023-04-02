from functools import cached_property, lru_cache
from itertools import product
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd
from biopsykit.utils.file_handling import get_subject_dirs
from tpcp import Dataset

from empkins_io.datasets.d03.micro_gapvii.helper import _load_biopac_data, _load_timelog
from empkins_io.utils._types import path_t

_cached_get_biopac_data = lru_cache(maxsize=4)(_load_biopac_data)


class MicroBaseDataset(Dataset):

    base_path: path_t
    sync_on_load: bool
    use_cache: bool
    _sampling_rates: Dict[str, float] = {"biopac": 1000}

    # TODO sebbo: divide phases into Talk_1, Talk_2, and Talk etc.
    #  (or introduce two index levels: "phase_coarse" and "phase_fine")
    PHASES = ["Prep", "Pause_1", "Talk", "Pause_2", "Pause_3",
              "Math", "Pause_4", "Pause_5"]

    CONDITIONS = ["tsst", "ftsst"]

    MISSING_DATA: Dict[str, Sequence[str]] = {}

    def __init__(
        self,
        base_path: path_t,
        groupby_cols: Optional[Sequence[str]] = None,
        subset_index: Optional[Sequence[str]] = None,
        exclude_missing_data: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ):
        # ensure pathlib
        self.base_path = base_path
        self.exclude_missing_data = exclude_missing_data
        self.use_cache = use_cache
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self):
        # data is located in a folder named "Data" and data per participant is located in folders named "vp-xx"

        participant_ids = [
            participant_dir.name
            for participant_dir in get_subject_dirs(self.base_path.joinpath("data_per_subject"), "vp_*")
        ]

        # excludes participants where parts of data are missing
        if self.exclude_missing_data:
            for type_of_missing_data, p_ids in self.MISSING_DATA.items():
                for p_id in p_ids:
                    if p_id in participant_ids:
                        participant_ids.remove(p_id)

        phaseidx = pd.DataFrame([("Prep", "Prep"), ("Pause_1", "Pause_1"), ("Talk", "Talk_1"), ("Talk", "Talk_2"),
                                ("Pause_2", "Pause_2"), ("Pause_3", "Pause_3"), ("Math", "Math_1"), ("Math", "Math_2"),
                                ("Pause_4", "Pause_4"), ("Pause_5", "Pause_5")], columns=["phase_coarse", "phase_fine"])

        index = list(product(participant_ids, self.CONDITIONS, self.PHASES))
        idx = pd.DataFrame(index, columns=['participant', 'condition', 'phase_coarse'])
        index = pd.merge(idx, phaseidx, on='phase_coarse', how='left')
        index.sort_values(by=['participant'])
        index.reset_index()
        return index

    @property
    def sampling_rates(self) -> Dict[str, float]:
        return self._sampling_rates

    @cached_property
    def biopac(self) -> pd.DataFrame:
        if self.is_single(["participant", "condition", "phase_coarse"]):

            participant_id = self.index["participant"][0]
            condition = self.index["condition"][0]
            phase_coarse = self.index["phase_coarse"][0]

            data, fs = self._get_bioapc_data(participant_id, condition, phase_coarse)
            return data

        if self.is_single(["participant", "condition"]):
            if len(self.index["phase_coarse"]) not in [1, len(self.PHASES) + 2]:
                raise ValueError("Biopac data can only be accessed for all phases or one specific phase!")

            participant_id = self.index["participant"][0]
            condition = self.index["condition"][0]

            data, fs = self._get_bioapc_data(participant_id, condition, "all")
            return data

        if self.is_single(None):
            # get biopac data for specified participant and specified phase and study protocol
            participant_id = self.index["participant"][0]
            condition = self.index["condition"][0]

            data, fs = self._get_bioapc_data(participant_id, condition, "all")
            return data

        raise ValueError("Biopac data can only be accessed for a single participant and a single condition!")

    @property
    def timelog(self) -> pd.DataFrame:
        if self.is_single(["participant", "condition", "phase_coarse"]):
            participant_id = self.index["participant"][0]
            condition = self.index["condition"][0]
            phase_coarse = self.index['phase_coarse'][0]
            return self._get_timelog(participant_id, condition, phase_coarse)
        elif self.is_single(["participant", "condition"]):
            participant_id = self.index["participant"][0]
            condition = self.index["condition"][0]
            return self._get_timelog(participant_id, condition, "all")

        raise ValueError("Timelog can only be accessed for a single participant and a single condition!")

    def _get_bioapc_data(self, participant_id: str, condition: str, phase: str) -> Tuple[pd.DataFrame, int]:
        if self.use_cache:
            data, fs = _cached_get_biopac_data(self.base_path, participant_id, condition)
        else:
            data, fs = _load_biopac_data(self.base_path, participant_id, condition)

        if phase == "all":
            print(f"_get_biopac_data Phase: {phase}")
            return data, fs
        else:
            # cut biopac data to specified phase
            print(f"_get_biopac_data Phase: {phase}")
            timelog = self.timelog
            phase_start = timelog[phase]["start"][0]
            phase_end = timelog[phase]["end"][0]
            data = data.loc[phase_start:phase_end]
            return data, fs

    def _get_timelog(self, participant_id: str, condition: str, phase: str) -> pd.DataFrame:
        return _load_timelog(self.base_path, participant_id, condition, phase)
