from collections.abc import Sequence
from functools import cached_property, lru_cache
from itertools import product
from pathlib import Path
from typing import ClassVar

import pandas as pd
from biopsykit.io import load_atimelogger_file
from biopsykit.utils.file_handling import get_subject_dirs
from tpcp import Dataset

__all__ = ["D07PilotStudyDataset"]

from empkins_io.datasets.d07.pilot_study.helper import _load_mocap_data
from empkins_io.utils._types import path_t

_cached_load_mocap_data = lru_cache(maxsize=4)(_load_mocap_data)


class D07PilotStudyDataset(Dataset):
    base_path: path_t
    use_cache: bool
    exclude_missing: bool

    SUBSETS_NO_MOCAP: ClassVar[str] = []

    data_to_exclude: Sequence[str]

    CONDITIONS: ClassVar[Sequence[str]] = ["Control", "Gert"]
    PHASE_MAPPER: ClassVar[dict[str, str]] = {
        "Sockentest": "sock_test",
        "Sit to stand": "sit_stand_test",
        "Langsitz Test": "long_sit_test",
        "Finger-Boden-Abstand": "finger_floor_distance_test",
        "Aufhebe Test": "pick_up_test",
        "Hebe Test": "lifting_test",
    }
    PHASES: ClassVar[Sequence[str]] = PHASE_MAPPER.values()

    CONDITION_ORDER_MAPPING = {
        "gert_first": {0: "Gert", 1: "Control"},
        "control_first": {0: "Control", 1: "Gert"},
    }

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
        if not self.is_single(["participant", "condition"]):
            raise ValueError("Time logs can only be accessed for a single participant and condition!")

        p_id = self.index["participant"][0]
        condition = self.index["condition"][0]
        phases = self.index["phase"].unique()
        file_path = self.base_path.joinpath(f"data_per_participant/{p_id}/timelogs/cleaned/{p_id}_timelog.csv")

        data = load_atimelogger_file(file_path, handle_multiple="fix")
        data = data.rename(columns=self.PHASE_MAPPER, level="phase")

        data.columns = data.columns.set_names(["phase", "condition", "start_end"])
        data.columns = data.columns.reorder_levels(["condition", "phase", "start_end"])

        t0 = data[(0, phases[0], "start")].iloc[0]
        t1 = data[(1, phases[0], "start")].iloc[0]
        if t0 > t1:  # determines which trial (0 or 1) starts earlier.
            data = data.rename(columns={0: 1, 1: 0}, level="condition")

        condition_order = self.condition_order.loc[p_id, "condition_order"]
        condition_order_map = self.CONDITION_ORDER_MAPPING[condition_order]
        data = data.rename(columns=condition_order_map, level="condition")

        data = data.reindex(phases, level="phase", axis=1)
        data = data.reindex([condition], level="condition", axis=1)
        return data

    @cached_property
    def mocap(self):
        if not self.is_single(None):
            raise ValueError("Motion capture data can only be accessed for a single participant, condition and phase!")
        p_id = self.group_label.participant
        condition = self.group_label.condition
        phase = self.group_label.phase

        # TODO continue
        condition_order = self.condition_order.loc[p_id, "condition_order"]
        condition_order_map = self.CONDITION_ORDER_MAPPING[condition_order]
        condition_key = next(i for i, cond in condition_order_map.items() if cond == condition)
        file_path = self.base_path.joinpath(
            f"data_per_participant/{p_id}/mocap/processed/VP_99-00{condition_key + 1}.mvnx"
        )
        data = self._get_mocap_data(file_path)

        # TODO: cut to selected phase by timelog
        timelog = self.timelog
        start_ts = timelog[(condition, phase, "start")].iloc[0]
        end_ts = timelog[(condition, phase, "end")].iloc[0]

        data = data.loc[start_ts:end_ts]
        return data

    def _get_mocap_data(self, file_path: Path) -> pd.DataFrame:
        if self.use_cache:
            return _cached_load_mocap_data(file_path)
        return _load_mocap_data(file_path)

    @property
    def condition_order(self):
        file_path = self.base_path.joinpath("metadata/condition_order.csv")
        data = pd.read_csv(file_path, index_col=0)

        return data.reindex(self.index["participant"].unique())
