from functools import cached_property, lru_cache
from itertools import product
from typing import Optional, Sequence

import pandas as pd
from biopsykit.utils.file_handling import get_subject_dirs

from empkins_io.datasets.d03._utils.dataset_utils import get_cleaned_openpose_data
from empkins_io.datasets.d03.macro_ap01 import MacroStudyTsstDataset
from empkins_io.datasets.d03.macro_ap01.helper import _load_tsst_mocap_data
from empkins_io.utils._types import path_t

_cached_load_mocap_data = lru_cache(maxsize=4)(_load_tsst_mocap_data)


class MacroStudyTsstDatasetPerPhase(MacroStudyTsstDataset):
    """Class to conveniently access the data of the macro study dataset on a phase-level.
    If access is only required per-condition, use :class:`MacroStudyTsstDataset` instead."""

    phases = ("prep", "talk", "math")

    def __init__(
        self,
        base_path: path_t,
        groupby_cols: Optional[Sequence[str]] = None,
        subset_index: Optional[Sequence[str]] = None,
        exclude_without_mocap: bool = True,
        exclude_without_openpose: bool = True,
        exclude_missing_data: bool = False,
        use_cache: bool = True,
    ):
        super().__init__(
            base_path=base_path,
            groupby_cols=groupby_cols,
            subset_index=subset_index,
            use_cache=use_cache,
            exclude_missing_data=exclude_missing_data,
            exclude_without_mocap=exclude_without_mocap,
            exclude_without_openpose=exclude_without_openpose,
        )

    def create_index(self):
        subject_ids = [
            subject_dir.name for subject_dir in get_subject_dirs(self.base_path.joinpath("data_per_subject"), "VP_*")
        ]

        if self.exclude_missing_data:
            for missing_type, sids in self.MISSING_DATA.items():
                for sid in sids:
                    if sid in subject_ids:
                        subject_ids.remove(sid)

        if self.exclude_without_mocap:
            for subject_id in self.SUBJECTS_WITHOUT_MOCAP:
                if subject_id in subject_ids:
                    subject_ids.remove(subject_id)

        index = list(product(subject_ids, self.conditions, self.phases))
        if self.exclude_without_openpose:
            for subject_id, condition in self.SUBSETS_WITHOUT_OPENPOSE_DATA:
                for phase in self.phases:
                    if (subject_id, condition, phase) in index:
                        index.remove((subject_id, condition, phase))

        index = pd.DataFrame(
            index,
            columns=["subject", "condition", "phase"],
        )

        return index

    @cached_property
    def mocap_data(self) -> pd.DataFrame:
        if self.is_single(["subject", "condition"]):
            subject_id = self.index["subject"][0]
            condition = self.index["condition"][0]
            if self.is_single(None):
                phase = self.index["phase"].unique()[0]
            else:
                # TODO: just because more than one phase is present, does not mean, all of them are! So it actually
                # cannot just be assumed to be "total"
                phase = "total"
            data_total = self._get_mocap_data_per_phase(subject_id, condition, phase)
            return data_total
        raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")

    @property
    def openpose_data(self) -> pd.DataFrame:
        """Returns the cleaned openpose data for a single subject and position.
        The data is cut to the claps specified in sync_data. Additionally, the index was set to a constant frame rate
        and the data was interpolated linearly and lowpass filtered."""
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
