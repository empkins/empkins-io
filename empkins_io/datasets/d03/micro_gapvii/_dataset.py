from functools import cached_property, lru_cache
from itertools import product
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd
from biopsykit.utils.file_handling import get_subject_dirs
from biopsykit.io.io import load_long_format_csv
from tpcp import Dataset
import warnings

from empkins_io.datasets.d03.micro_gapvii.helper import _load_biopac_data, _load_timelog, _load_emrad_data
from empkins_io.utils._types import path_t

_cached_get_biopac_data = lru_cache(maxsize=4)(_load_biopac_data)
_cached_get_emrad_data = lru_cache(maxsize=4)(_load_emrad_data)


class MicroBaseDataset(Dataset):
    base_path: path_t
    sync_on_load: bool
    use_cache: bool
    phase_fine: bool
    _sampling_rates: Dict[str, float] = {"biopac": 1000}

    PHASE_COARSE = ["Prep", "Pause_1", "Talk", "Pause_2", "Pause_3", "Math", "Pause_4", "Pause_5"]

    PHASE_FINE = ["Prep", "Pause_1", "Talk_1", "Talk_2", "Pause_2", "Pause_3", "Math_1", "Math_2", "Pause_4", "Pause_5"]

    CONDITIONS = ["tsst", "ftsst"]

    MISSING_DATA: Sequence[str] = ["VP_21", "VP_24", "VP_41", "VP_45"]  # Missing data (add participant IDs here)

    def __init__(
            self,
            base_path: path_t,
            groupby_cols: Optional[Sequence[str]] = None,
            subset_index: Optional[Sequence[str]] = None,
            exclude_missing_data: Optional[bool] = False,
            use_cache: Optional[bool] = True,
            phase_fine: Optional[bool] = False,
    ):
        # ensure pathlib
        self.base_path = base_path
        self.exclude_missing_data = exclude_missing_data
        self.use_cache = use_cache
        self.phase_fine = phase_fine
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self):
        # data is located in a folder named "Data" and data per participant is located in folders named "VP_xx"

        participant_ids = [
            participant_dir.name
            for participant_dir in get_subject_dirs(self.base_path.joinpath("data_per_subject"), "VP_[0-9]{2}")
        ]

        # excludes participants where parts of data are missing
        if self.exclude_missing_data:
            for p_id in self.MISSING_DATA:
                if p_id in participant_ids:
                    participant_ids.remove(p_id)

        if self.phase_fine:
            index = list(product(participant_ids, self.CONDITIONS, self.PHASE_FINE))
            index = pd.DataFrame(index, columns=['participant', 'condition', 'phase'])
            return index
        elif not self.phase_fine:
            index = list(product(participant_ids, self.CONDITIONS, self.PHASE_COARSE))
            index = pd.DataFrame(index, columns=['participant', 'condition', 'phase'])
            return index

    @property
    def sampling_rates(self) -> Dict[str, float]:
        return self._sampling_rates

    @property
    def cortisol(self) -> pd.DataFrame:
        cortisol_path = self.base_path.joinpath("data_tabular/saliva/cortisol/cleaned/cortisol_cleaned.csv")
        return load_long_format_csv(cortisol_path)

    @property
    def sex(self) -> pd.DataFrame:
        return load_long_format_csv(self.base_path.joinpath("data_tabular/extras/processed/sex.csv"), index_cols=["participant"])

    @property
    def condition_order(self) -> pd.DataFrame:
        return load_long_format_csv(self.base_path.joinpath("data_tabular/extras/processed/condition_order.csv"), index_cols=["participant"])

    @property
    def sit_stand(self) -> pd.DataFrame:
        return load_long_format_csv(self.base_path.joinpath("data_tabular/extras/processed/sit_stand.csv"), index_cols=["participant"])

    @property
    def condition_day_mapping(self) -> pd.DataFrame:
        return load_long_format_csv(self.base_path.joinpath("data_tabular/extras/processed/day_condition_mapping.csv"), index_cols=["participant", "day"])

    @cached_property
    def biopac(self) -> pd.DataFrame:
        if self.is_single(["participant", "condition", "phase"]):
            participant_id = self.index["participant"][0]
            condition = self.index["condition"][0]
            phase = self.index["phase"][0]

            data, fs = self._get_biopac_data(participant_id, condition, phase)
            return data

        if self.is_single(["participant", "condition"]):
            if self.phase_fine & len(self.index["phase"]) not in [1, len(self.PHASE_FINE)]:
                warnings.warn("Biopac data can only be accessed for all phases or one specific phase!")
            elif not self.phase_fine & len(self.index["phase"]) not in [1, len(self.PHASE_COARSE)]:
                warnings.warn("Biopac data can only be accessed for all phases or one specific phase!")

            participant_id = self.index["participant"][0]
            condition = self.index["condition"][0]

            data, fs = self._get_biopac_data(participant_id, condition, "all")
            return data

        if self.is_single(None):
            if self.phase_fine & len(self.index["phase"]) not in [1, len(self.PHASE_FINE)]:
                warnings.warn("Biopac data can only be accessed for all phases or one specific phase!")
            elif not self.phase_fine & len(self.index["phase"]) not in [1, len(self.PHASE_COARSE)]:
                warnings.warn("Biopac data can only be accessed for all phases or one specific phase!")

            # get biopac data for specified participant and specified phase and study protocol
            participant_id = self.index["participant"][0]
            condition = self.index["condition"][0]

            data, fs = self._get_biopac_data(participant_id, condition, "all")
            return data

        raise ValueError("Biopac data can only be accessed for all phases or one specific phase!")

    @property
    def timelog(self) -> pd.DataFrame:
        if self.is_single(["participant", "condition", "phase"]):
            participant_id = self.index["participant"][0]
            condition = self.index["condition"][0]
            phase = self.index['phase'][0]
            return self._get_timelog(participant_id, condition, phase)

        if self.is_single(["participant", "condition"]):
            if self.phase_fine & len(self.index["phase"]) not in [1, len(self.PHASE_FINE)]:
                warnings.warn("Biopac data can only be accessed for all phases or one specific phase!")
            elif not self.phase_fine & len(self.index["phase"]) not in [1, len(self.PHASE_COARSE)]:
                warnings.warn("Biopac data can only be accessed for all phases or one specific phase!")

            participant_id = self.index["participant"][0]
            condition = self.index["condition"][0]
            return self._get_timelog(participant_id, condition, "all")

        raise ValueError("Timelog can only be accessed for a single participant and a single condition!")

    @property
    def emrad(self) -> pd.DataFrame:
        if self.is_single(["participant", "condition", "phase"]):
            participant_id = self.index["participant"][0]
            condition = self.index["condition"][0]
            phase = self.index["phase"][0]

            data, fs = self._get_emrad_data(participant_id, condition, phase)
            return data

        if self.is_single(["participant", "condition"]):
            if self.phase_fine & len(self.index["phase"]) not in [1, len(self.PHASE_FINE)]:
                warnings.warn("Biopac data can only be accessed for all phases or one specific phase!")
            elif not self.phase_fine & len(self.index["phase"]) not in [1, len(self.PHASE_COARSE)]:
                warnings.warn("Biopac data can only be accessed for all phases or one specific phase!")

            participant_id = self.index["participant"][0]
            condition = self.index["condition"][0]

            data, fs = self._get_biopac_data(participant_id, condition, "all")
            return data

        if self.is_single(None):
            if self.phase_fine & len(self.index["phase"]) not in [1, len(self.PHASE_FINE)]:
                warnings.warn("Biopac data can only be accessed for all phases or one specific phase!")
            elif not self.phase_fine & len(self.index["phase"]) not in [1, len(self.PHASE_COARSE)]:
                warnings.warn("Biopac data can only be accessed for all phases or one specific phase!")

            # get biopac data for specified participant and specified phase and study protocol
            participant_id = self.index["participant"][0]
            condition = self.index["condition"][0]

            data, fs = self._get_emrad_data(participant_id, condition, "all")
            return data

        raise ValueError("Emrad data can only be accessed for all phases or one specific phase!")

    def _get_biopac_data(self, participant_id: str, condition: str, phase: str) -> Tuple[pd.DataFrame, int]:
        if self.use_cache:
            data, fs = _cached_get_biopac_data(self.base_path, participant_id, condition)
        else:
            data, fs = _load_biopac_data(self.base_path, participant_id, condition)

        if phase == "all":
            return data, fs
        else:
            # cut biopac data to specified phase
            timelog = self.timelog
            phase_start = timelog[phase]["start"][0]
            phase_end = timelog[phase]["end"][0]
            data = data.loc[phase_start:phase_end]
            return data, fs

    def _get_emrad_data(self, participant_id: str, condition: str, phase: str) -> Tuple[pd.DataFrame, int]:
        if self.use_cache:
            data, fs = _cached_get_emrad_data(self.base_path, participant_id, condition)
        else:
            data, fs = _load_emrad_data(self.base_path, participant_id, condition)

        if phase == "all":
            return data, fs
        else:
            # cut biopac data to specified phase
            timelog = self.timelog
            phase_start = timelog[phase]["start"][0]
            phase_end = timelog[phase]["end"][0]
            data = data.loc[phase_start:phase_end]
            return data, fs

    def _get_timelog(self, participant_id: str, condition: str, phase: str) -> pd.DataFrame:
        return _load_timelog(self.base_path, participant_id, condition, phase, self.phase_fine)
