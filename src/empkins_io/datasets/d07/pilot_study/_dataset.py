from collections.abc import Sequence
from functools import cached_property, lru_cache
from itertools import product
from pathlib import Path
from typing import ClassVar, Optional, Union

import pandas as pd
from biopsykit.io import load_atimelogger_file
from biopsykit.utils.file_handling import get_subject_dirs
from tpcp import Dataset

__all__ = ["D07PilotStudyDataset"]

from empkins_io.datasets.d07.pilot_study._helper import _load_nilspod_session
from empkins_io.datasets.d07.pilot_study.helper import _load_mocap_data
from empkins_io.utils._types import path_t

_cached_load_mocap_data = lru_cache(maxsize=4)(_load_mocap_data)
_cached_load_nilspod_session = lru_cache(maxsize=5)(_load_nilspod_session)


class D07PilotStudyDataset(Dataset):
    base_path: path_t
    use_cache: bool
    exclude_missing: bool

    SAMPLING_RATE_MOCAP = 60
    SAMPLING_RATE_ECG = 256

    NILSPOD_MAPPING: ClassVar[dict[str, str]] = {
        "chest": "b0c2",  # ecg
        "sync": "9e02",  # sync with mocap
        "board": "e76b",  # sync with video (clapper board)
    }

    SUBSETS_NO_MOCAP: ClassVar[str] = [
        ("VP_07", "Control"),
        ("VP_12", "Control"),
    ]

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
            for subject_dir in get_subject_dirs(self.base_path.joinpath("data_per_participant"), "VP_\\d+")
        ]
        index_cols = ["participant", "condition", "phase"]
        index = list(product(p_ids, self.CONDITIONS, self.PHASES))

        index = pd.DataFrame(index, columns=index_cols)
        index = index.set_index(index_cols)
        index = index.drop(index=self.data_to_exclude, errors="ignore").reset_index()
        return index

    def _find_data_to_exclude(self):
        data_to_exclude = []
        if self.exclude_missing:
            data_to_exclude += self.SUBSETS_NO_MOCAP

        return data_to_exclude

    @property
    def sampling_rate_ecg(self) -> int:
        return self.SAMPLING_RATE_ECG

    @property
    def timelog(self):
        if not self.is_single(["participant"]):
            raise ValueError("Time logs can only be accessed for a single participant!")

        p_id = self.index["participant"][0]
        conditions = self.index["condition"].unique()
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
        data = data.reindex(conditions, level="condition", axis=1)
        return data

    @cached_property
    def mocap(self):
        if not self.is_single(None):
            raise ValueError("Motion capture data can only be accessed for a single participant, condition and phase!")
        p_id = self.group_label.participant
        condition = self.group_label.condition
        phase = self.group_label.phase

        file_path = self.base_path.joinpath(f"data_per_participant/{p_id}/mocap/export/")
        mocap_files = sorted(file_path.glob(f"D07_{p_id}_{condition.lower()}*.mvnx"))
        if len(mocap_files) == 1:
            data = self._get_mocap_data(mocap_files[0])
        elif len(mocap_files) == 0:
            raise FileNotFoundError(f"No mocap file found for participant {p_id} and condition {condition}!")
        else:
            data = pd.DataFrame()
            for file in mocap_files:
                df = self._get_mocap_data(file)
                data = pd.concat([data, df])

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

    @cached_property
    def ecg(self) -> pd.DataFrame:
        """Load and return ECG data.

        The ECG data can only be loaded as a single phase or the entire recording

        Returns
        -------
        :class:`~pandas.DataFrame`
            ECG data

        """
        nilspod_data = self.nilspod_session
        ecg_data = nilspod_data[self.NILSPOD_MAPPING["chest"]][["ecg"]]
        return ecg_data

    @cached_property
    def nilspod_session(self) -> pd.DataFrame:
        participant = self.index["participant"][0]
        if self.is_single(None) or len(self.index["phase"].unique()) == len(self.PHASES):

            condition = self.index["condition"].unique()
            phase = self.index["phase"].unique()

            if not self.is_single(None) and len(condition) != len(self.CONDITIONS):
                raise ValueError(
                    "Nilspod data can only be accessed for a single participant in its entirety or for a single phase!"
                )

            if len(condition) > 1:
                condition = None
            if len(phase) > 1:
                phase = None

            data = self._load_nilspod_session(participant, condition, phase)
            return data

        raise ValueError(
            "Nilspod data can only be accessed for a single participant in its entirety or for a single phase!"
        )

    def _load_nilspod_session(
        self, participant: str, condition: Optional[str] = None, phase: Optional[Sequence[str]] = None
    ) -> Union[pd.DataFrame, dict[str, pd.DataFrame]]:
        data_path = self.base_path.joinpath(f"data_per_participant/{participant}/nilspod/raw")
        data = _cached_load_nilspod_session(data_path) if self.use_cache else _load_nilspod_session(data_path)

        if condition is None:
            # all conditions => return all data
            return data

        timelog = self.timelog

        start_ts = timelog[(condition, phase, "start")].iloc[0]
        end_ts = timelog[(condition, phase, "end")].iloc[0]

        data = data.loc[start_ts:end_ts]
        return data
