import warnings
from functools import cached_property, lru_cache
from itertools import product
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
import warnings

import pandas as pd
from biopsykit.utils.file_handling import get_subject_dirs
from empkins_micro.utils._types import path_t
from empkins_io.datasets.d03.micro_gapvii.helper import (_load_biopac_data, _load_timelog)
from tpcp import Dataset

_cached_get_biopac_data = lru_cache(maxsize=4)(_load_biopac_data)

# TODO warning oder exception für Datensätze, wo was fehlt?


class MicroBaseDataset(Dataset):

    base_path: path_t
    sync_on_load: bool
    use_cache: bool
    _sampling_rates: Dict[str, float] = {"biopac": 1000}

    PHASES = ["Prep",           # underscores
              "Pause 1",
              "Talk",
              "Pause 2",
              "Pause 3",
              "Math",
              "Pause 4",
              "Pause 5"]

    STUDY_PROTOCOL = ["tsst",
                      "ftsst"]

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
            participant_dir.name for participant_dir in get_subject_dirs(self.base_path.joinpath("data_per_subject"), "vp_*")
        ]

        # excludes participants where parts of data are missing
        if self.exclude_missing_data:
            for type_of_missing_data, p_ids in self.MISSING_DATA.items():
                for p_id in p_ids:
                    if p_id in participant_ids:
                        participant_ids.remove(p_id)

        index = list(product(participant_ids, self.PHASES, self.STUDY_PROTOCOL))
        index = pd.DataFrame(index, columns=["participant", "phase", "study_protocol"])
        return index

    @property
    def sampling_rates(self) -> Dict[str, float]:
        return self._sampling_rates

    @cached_property
    def biopac(self) -> pd.DataFrame:

        if self.is_single(None):
            # get biopac data for specified participant and specified phase and study protocol
            participant_id = self.index["participant"][0]
            study_protocol = self.index["study_protocol"][0]
            phase = self.index["phase"][0]
            data, fs = self._get_bioapc_data(participant_id, study_protocol, phase)
            return data

        if self.is_single(["participant"]):
            # get biopac data for specified participant and specified study protocol for all phases
            # when >=2 phases are accessed at once, get data for all phases instead & warning
            participant_id = self.index["participant"][0]
            study_protocol = self.index["study_protocol"][0]

            if 1 < len(self.index["phase"]) < len(self.PHASES):
                warnings.warn(
                    "Returned Biopac dataset contains all phases! It can only be accessed for all phases or one specific phase!",
                    UserWarning)

            data, fs = self._get_bioapc_data(participant_id, study_protocol, "all")
            return data

        raise ValueError("Biopac data can only be accessed for a single participant!")

    @property
    def timelog(self) -> pd.DataFrame:
        # timelog data can only be accessed for a single participant
        if self.is_single(["participant"]):
            participant_id = self.index["participant"][0]
            study_protocol = self.index["study_protocol"][0]
            return self._get_timelog(participant_id, study_protocol)
        raise ValueError("Timelog can only be accessed for a single participant!")

    def _get_bioapc_data(self, participant_id: str, study_protocol: str, phase: str) -> Tuple[pd.DataFrame, int]:

        if self.use_cache:
            data, fs = _cached_get_biopac_data(self.base_path, participant_id, study_protocol)

        else:
            data, fs = _load_biopac_data(self.base_path, participant_id, study_protocol)

        if phase == "all":
            return data, fs
        else:
            # cut biopac data to specified phase
            timelog = self.timelog
            phase_start = timelog[phase]["start"][0]
            phase_end = timelog[phase]["end"][0]
            data = data.loc[phase_start: phase_end]
            return data, fs

    def _get_timelog(self, participant_id: str, study_protocol: str) -> pd.DataFrame:
        return _load_timelog(self.base_path, participant_id, study_protocol)

