from collections.abc import Sequence
from itertools import product
from typing import ClassVar
from functools import cached_property, lru_cache

import pandas as pd
from biopsykit.io import load_atimelogger_file
from biopsykit.utils.file_handling import get_subject_dirs
from tpcp import Dataset

__all__ = ["D07PilotStudyDataset"]

from empkins_io.utils._types import path_t
from empkins_io.datasets.d07.pilot_study.helper import _load_mocap_data

_cached_load_mocap_data = lru_cache(maxsize=4)(_load_mocap_data)


class D07PilotStudyDataset(Dataset):
    base_path: path_t
    use_cache: bool
    exclude_missing: bool

    SUBSETS_NO_MOCAP: ClassVar[str] = []

    data_to_exclude: Sequence[str]

    CONDITIONS: ClassVar[Sequence[str]] = ["Control", "Gert"]
    PHASE_MAPPER: ClassVar[dict[str, str]] = {
        "Sockentest": "sock test",
        "Sit to stand": "sit-to-stand test",
        "Langsitz Test": "long-sit test",
        "Finger-Boden-Abstand": "finger-floor distance test",
        "Aufhebe Test": "pick-up test",
        "Hebe Test": "lifting test"
    }
    PHASES: ClassVar[Sequence[str]] = PHASE_MAPPER.values()

    def __init__(
        self,
        base_path: path_t,
        groupby_cols: Sequence[str] | None = None,
        subset_index: Sequence[str] | None = None,
        *,
        exclude_missing: bool = True,
        use_cache: bool = True,
    ):
        self.base_path = base_path
        self.exclude_missing = exclude_missing
        self.use_cache = use_cache

        self.data_to_exclude = self._find_data_to_exclude()

        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self) -> pd.DataFrame:
        p_ids = [
            subject_dir.name
            for subject_dir in get_subject_dirs(self.base_path.joinpath("data_per_participant"), "VP_*")
        ]
        index_cols = ["participant", "condition", "phase"]
        index = list(product(p_ids, self.CONDITIONS, self.PHASES))

        index = pd.DataFrame(index, columns=index_cols)
        index = index.set_index(index_cols)
        index = index.drop(index=self.data_to_exclude).reset_index()
        return index

    def _find_data_to_exclude(self):
        data_to_exclude = []
        if self.exclude_missing:
            data_to_exclude += self.SUBSETS_NO_MOCAP

        return data_to_exclude

    @property
    def timelog(self):
        if not self.is_single(None):
            raise ValueError("Time logs can only be accessed for a single participant, condition and phase!")

        p_id = self.index["participant"][0]
        self.index["condition"][0]
        phases = self.index["phase"].unique()
        file_path = self.base_path.joinpath(f"data_per_participant/{p_id}/timelogs/cleaned/{p_id}_timelog.csv")

        data = load_atimelogger_file(file_path)
        data = data.rename(columns=self.PHASE_MAPPER, level="phase")
        data = data.reindex(phases, level="phase", axis=1)
        return data

    @cached_property
    def mocap(self):
        if not self.is_single(None):
            raise ValueError("Motion capture data can only be accessed for a single participant, condition and phase!")

        p_id = self.index["participant"][0]
        condition = self.index["condition"][0]
        phase = self.index["phase"][0]

        # TODO continue
        data = self._get_mocap_data(p_id)

        data_slice = data.loc[
                     self.timelog[phase]["start"].iloc[0]:self.timelog[phase]["end"].iloc[0]]
        return data_slice

    def _get_mocap_data(self, p_id: str) -> pd.DataFrame:
        if self.use_cache:
            return _cached_load_mocap_data(self.base_path, p_id)
        return _load_mocap_data(self.base_path, p_id)
