from collections.abc import Sequence
from functools import cached_property, lru_cache
from itertools import product

import pandas as pd
from biopsykit.utils.file_handling import get_subject_dirs

from empkins_io.datasets.d03._utils.dataset_utils import get_cleaned_openpose_data
from empkins_io.datasets.d03.macro_ap01 import MacroStudyTsstDataset
from empkins_io.datasets.d03.macro_ap01.helper import _get_times_for_mocap, _load_tsst_mocap_data
from empkins_io.utils._types import path_t, str_t

_cached_load_mocap_data = lru_cache(maxsize=4)(_load_tsst_mocap_data)


class MacroStudyTsstDatasetPerPhase(MacroStudyTsstDataset):
    """Class to conveniently access the data of the macro study dataset on a phase-level.
    If access is only required per-condition, use :class:`MacroStudyTsstDataset` instead.
    """

    PHASES = ("prep", "talk", "math")

    include_prep: bool = False

    def __init__(
        self,
        base_path: path_t,
        groupby_cols: Sequence[str] | None = None,
        subset_index: Sequence[str] | None = None,
        *,
        include_prep: bool = False,
        exclude_complete_subjects_if_error: bool = True,
        exclude_without_mocap: bool = True,
        exclude_without_openpose: bool = False,
        exclude_with_arm_errors: bool = False,
        exclude_without_prep: bool = False,
        exclude_without_gait_tests: bool = False,
        use_cache: bool = True,
        verbose: bool = True,
    ):
        self.include_prep = include_prep
        super().__init__(
            base_path=base_path,
            groupby_cols=groupby_cols,
            subset_index=subset_index,
            use_cache=use_cache,
            exclude_complete_subjects_if_error=exclude_complete_subjects_if_error,
            exclude_without_mocap=exclude_without_mocap,
            exclude_without_openpose=exclude_without_openpose,
            exclude_with_arm_errors=exclude_with_arm_errors,
            exclude_without_prep=exclude_without_prep,
            exclude_without_gait_tests=exclude_without_gait_tests,
            verbose=verbose,
        )

    def create_index(self):
        subject_ids = [
            subject_dir.name for subject_dir in get_subject_dirs(self.base_path.joinpath("data_per_subject"), "VP_*")
        ]

        phases = self.PHASES
        if not self.include_prep:
            phases = phases[1:]

        index = list(product(subject_ids, self.CONDITIONS, phases))

        index_cols = ["subject", "condition", "phase"]
        index = pd.DataFrame(index, columns=index_cols)
        index = index.set_index(index_cols)
        index = index.drop(index=self.data_to_exclude).reset_index()

        return index

    @cached_property
    def mocap_data(self) -> pd.DataFrame | dict[str, pd.DataFrame]:
        if not self.is_single(["subject", "condition"]):
            raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")

        subject_id = self.index["subject"][0]
        condition = self.index["condition"][0]
        phase = self.index["phase"][0] if self.is_single(None) else list(self.index["phase"])

        data_total = self._get_mocap_data_per_phase(subject_id, condition, phase)
        return data_total

    def _get_mocap_data_per_phase(
        self, subject_id: str, condition: str, phase: str_t, *, verbose: bool = True
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        data = self._get_mocap_data(subject_id, condition, verbose=verbose)
        timelog = self.timelog_test
        times = _get_times_for_mocap(timelog, phase)

        if isinstance(phase, str):
            return data.loc[times.loc[phase, "start"] : times.loc[phase, "end"]]

        data_total = {}
        for ph in phase:
            data_total[ph] = data.loc[times.loc[ph, "start"] : times.loc[ph, "end"]]

        return data_total

    @property
    def openpose_data(self) -> pd.DataFrame:
        """Returns the cleaned openpose data for a single subject and position.
        The data is cut to the claps specified in sync_data. Additionally, the index was set to a constant frame rate
        and the data was interpolated linearly and lowpass filtered.
        """
        if not self.is_single(["subject", "condition"]):
            raise ValueError("OpenPose data can only be accessed for a single condition of a single participant!")

        if self.is_single(None):
            raise NotImplementedError(
                "You are trying to access cleaned OpenPose data for only one phase."
                "This is theoretically possible, but not implemented yet!"
            )

        file_path = self.body_video_path.joinpath("cleaned/openpose.csv")
        openpose_cleaned_for_subject_and_condition = get_cleaned_openpose_data(file_path)

        # MISSING: crop data to single phase, if this feature is needed
        return openpose_cleaned_for_subject_and_condition
