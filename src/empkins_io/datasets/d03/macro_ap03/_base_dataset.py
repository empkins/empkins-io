from collections.abc import Sequence
from itertools import product
from typing import ClassVar

import pandas as pd
from biopsykit.utils.file_handling import get_subject_dirs
from statsmodels.stats.oneway import power_equivalence_oneway
from tpcp import Dataset

from empkins_io.utils._types import path_t

# _cached_load_nilspod_data = lru_cache(maxsize=4)(_load_nilspod_session)

__all__ = ["MacroBaseDataset"]


class MacroBaseDataset(Dataset):
    base_path: path_t
    use_cache: bool

    use_phases: bool
    include_prep: bool

    _sample_times_saliva: tuple[int] = (-40, -1, 15, 25, 35, 45, 60, 75)
    _sample_times_bloodspot: tuple[int] = (-40, 60)

    SAMPLING_RATE_MOCAP = 60

    SAMPLING_RATE_ZEBRIS = 60

    CONDITIONS: ClassVar[Sequence[str]] = ["ftsst", "tsst"]

    PHASES: ClassVar[Sequence[str]] = ["prep", "talk", "math"]

    SUBSETS_WITHOUT_MOCAP = ()

    SUBSETS_WITHOUT_ZEBRIS = ()

    def __init__(
        self,
        base_path: path_t,
        groupby_cols: Sequence[str] | None = None,
        subset_index: Sequence[str] | None = None,
        *,
        use_phases: bool = False,
        include_prep: bool = False,
        exclude_complete_subjects_if_error: bool = True,
        exclude_without_mocap: bool = True,
        exclude_without_zebris: bool = True,
        use_cache: bool = True,
    ):
        # ensure pathlib
        self.base_path = base_path
        self.use_phases = use_phases
        self.include_prep = include_prep
        self.exclude_complete_subjects_if_error = exclude_complete_subjects_if_error
        self.exclude_without_mocap = exclude_without_mocap
        self.exclude_without_zebris = exclude_without_zebris

        self.data_to_exclude = self._find_data_to_exclude(exclude_complete_subjects_if_error)
        self.use_cache = use_cache

        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self):
        subject_ids = [
            subject_dir.name
            for subject_dir in get_subject_dirs(self.base_path.joinpath("data_per_participant"), "VP_*")
        ]
        index_cols = ["participant", "condition"]
        index = list(product(subject_ids, self.CONDITIONS))

        index = pd.DataFrame(index, columns=index_cols)
        index = index.set_index(index_cols)
        index = index.drop(index=self.data_to_exclude).reset_index()

        return index

    def _find_data_to_exclude(self, exclude_complete_subjects_if_error: bool):
        data_to_exclude = []
        if self.exclude_without_mocap:
            data_to_exclude += self.SUBSETS_WITHOUT_MOCAP
        if self.exclude_without_zebris:
            data_to_exclude += self.SUBSETS_WITHOUT_ZEBRIS

        if exclude_complete_subjects_if_error:
            data_to_exclude = [x[0] for x in data_to_exclude]

        return data_to_exclude

    @property
    def subject(self) -> str:
        if not self.is_single("subject"):
            raise ValueError("Subject data can only be accessed for a single participant!")
        return self.index["subject"][0]

    @property
    def condition(self) -> str:
        if not self.is_single("condition"):
            raise ValueError("Condition data can only be accessed for a single condition!")
        return self.index["condition"][0]

    @property
    def sampling_rate(self) -> float:
        """Sampling rate of the MoCap system."""
        return self.SAMPLING_RATE_MOCAP

    @property
    def sample_times_saliva(self) -> Sequence[int]:
        return self._sample_times_saliva

    @property
    def sample_times_bloodspot(self) -> Sequence[int]:
        return self._sample_times_bloodspot

    @property
    def zebris(self) -> pd.DataFrame:
        # TODO: continue here
        if not self.is_single(None):
            raise ValueError(
                "Data can only be accessed for a single recording (participant, condition, phase) in the subset"
            )
        p_id = self.group_label.participant
        condition = self.group_label.condition
        phase = self.group_label.phase

        folder_path = self.base_path.joinpath("data_per_participant").joinpath(
            f"{p_id}/{condition}/zebris/export/{phase}"
        )

        return pd.DataFrame()

    @property
    def zebris_aggregated(self) -> pd.DataFrame:
        # TODO : continue here
        return pd.DataFrame()
